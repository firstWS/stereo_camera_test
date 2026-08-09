from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    DropoutAnchorDefinition,
    SuccessThresholds,
    compute_dropout_window,
)
from dropout_evaluation.evaluation_metrics import (  # noqa: E402
    CupObservation,
    MetricStatus,
    PoseEstimate,
    PoseReference,
    PoseTrackingState,
    RecoveryTiming,
    apply_known_rotation_offset_deg,
    apply_known_translation_offset,
    compute_availability_metrics,
    compute_cup2_world_metrics,
    compute_pose_window_metrics,
    compute_recovery_metrics,
    cup_world_position_error,
    evaluate_window,
    references_to_perfect_candidates,
    rotation_error_deg,
    translation_error,
)
def _T(translation: tuple[float, float, float], yaw_deg: float = 0.0) -> np.ndarray:
    theta = np.deg2rad(yaw_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, -s, 0.0, translation[0]],
            [s, c, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _reference(fn: int, ts: int, t=(0.0, 0.0, 0.0), yaw_deg=0.0) -> PoseReference:
    return PoseReference(
        frame_number=fn,
        device_timestamp_us=ts,
        T_world_camera=_T(t, yaw_deg),
        valid=True,
    )


def _window_for_test():
    anchor = DropoutAnchorDefinition(
        anchor_id="B_motion_start",
        start_frame=81,
        start_device_timestamp_us=1_000_000,
        convention="first_sustained_motion_frame",
        motion_class="motion_start",
    )
    from dropout_evaluation.dropout_protocol import FrameTimestamp

    frames = [
        FrameTimestamp(frame_number=81 + index, device_timestamp_us=1_000_000 + index * 33_333)
        for index in range(30)
    ]
    return compute_dropout_window(
        anchor=anchor,
        duration_sec=0.5,
        session_id="test",
        frames=frames,
    )


def test_translation_metric_known_offset() -> None:
    ref = _reference(1, 1000, t=(0.0, 0.0, 0.0))
    cand = PoseEstimate(
        frame_number=1,
        device_timestamp_us=1000,
        T_world_camera=_T((0.10, 0.0, 0.0)),
        valid=True,
    )
    assert translation_error(ref, cand) == pytest.approx(0.10, abs=1e-9)


def test_rotation_metric_so3_angle_not_euler() -> None:
    ref = _reference(1, 1000, yaw_deg=0.0)
    cand = PoseEstimate(
        frame_number=1,
        device_timestamp_us=1000,
        T_world_camera=_T((0.0, 0.0, 0.0), yaw_deg=5.0),
        valid=True,
    )
    assert rotation_error_deg(ref, cand) == pytest.approx(5.0, abs=1e-6)


def _references_for_official_frames(
    official_frames,
    frame_numbers: range | list[int],
    *,
    translation_scale: float = 0.0,
    yaw_scale: float = 0.0,
) -> list[PoseReference]:
    ts_by_fn = {frame.frame_number: frame.device_timestamp_us for frame in official_frames}
    references: list[PoseReference] = []
    for fn in frame_numbers:
        references.append(
            _reference(
                fn,
                ts_by_fn[fn],
                t=(translation_scale * fn, 0.0, 2.0),
                yaw_deg=yaw_scale * fn,
            )
        )
    return references


def test_perfect_candidate_zero_error(official_windows, official_frames) -> None:
    references = _references_for_official_frames(
        official_frames,
        range(81, 112),
        translation_scale=0.1,
        yaw_scale=0.5,
    )
    candidates = references_to_perfect_candidates(references)
    window = next(window for window in official_windows if window.window_id == "B_motion_start__1.0s")
    result = evaluate_window(
        window=window,
        references=references,
        candidates=candidates,
        algorithm_id="perfect_reference",
        thresholds=SuccessThresholds(),
    )
    assert result.pose.translation_error.median == pytest.approx(0.0, abs=1e-9)
    assert result.pose.rotation_error_deg.median == pytest.approx(0.0, abs=1e-6)
    assert result.pose.availability is not None
    assert result.pose.availability.availability_ratio == pytest.approx(1.0)


def test_known_translation_offset_candidate(official_windows, official_frames) -> None:
    references = _references_for_official_frames(official_frames, range(81, 112))
    candidates = apply_known_translation_offset(references, (0.10, 0.0, 0.0))
    window = next(window for window in official_windows if window.window_id == "B_motion_start__1.0s")
    metrics = compute_pose_window_metrics(window=window, references=references, candidates=candidates)
    assert metrics.translation_error.median == pytest.approx(0.10, abs=1e-9)
    assert metrics.rotation_error_deg.median == pytest.approx(0.0, abs=1e-6)


def test_known_rotation_offset_candidate(official_windows, official_frames) -> None:
    references = _references_for_official_frames(official_frames, range(81, 112))
    candidates = apply_known_rotation_offset_deg(references, 5.0)
    window = next(window for window in official_windows if window.window_id == "B_motion_start__1.0s")
    metrics = compute_pose_window_metrics(window=window, references=references, candidates=candidates)
    assert metrics.rotation_error_deg.median == pytest.approx(5.0, abs=1e-4)


def test_availability_and_invalid_gap_metrics() -> None:
    window = _window_for_test()
    candidates = []
    for index, fn in enumerate(range(81, 96)):
        valid = index not in {2, 3, 4}
        state = PoseTrackingState.TRACKING_LOST if index == 4 else PoseTrackingState.LOCAL_TRACKING
        candidates.append(
            PoseEstimate(
                frame_number=fn,
                device_timestamp_us=1_000_000 + (fn - 81) * 33_333,
                T_world_camera=_T((0.0, 0.0, 0.0)),
                valid=valid,
                state=state,
            )
        )
    availability = compute_availability_metrics(window=window, candidates=candidates)
    assert availability.expected_frames == 15
    assert availability.valid_pose_frames == 12
    assert availability.availability_ratio == pytest.approx(0.8)
    assert availability.longest_invalid_gap_frames == 3
    assert availability.tracking_lost_frames == 1


def test_end_of_dropout_invalid_does_not_substitute_last_valid() -> None:
    window = _window_for_test()
    references = [
        _reference(fn, 1_000_000 + (fn - 81) * 33_333)
        for fn in range(81, 96)
    ]
    candidates = []
    for fn in range(81, 96):
        valid = fn != 95
        candidates.append(
            PoseEstimate(
                frame_number=fn,
                device_timestamp_us=1_000_000 + (fn - 81) * 33_333,
                T_world_camera=_T((0.0, 0.0, 0.0)),
                valid=valid,
            )
        )
    metrics = compute_pose_window_metrics(window=window, references=references, candidates=candidates)
    assert metrics.translation_error.end_of_dropout_error is None
    assert metrics.translation_error.last_valid_before_recovery_error == pytest.approx(0.0, abs=1e-9)


def test_recovery_metrics_and_reanchor_jump() -> None:
    window = _window_for_test()
    recovery_frame = window.recovery_frame
    assert recovery_frame is not None
    frame_numbers = list(range(window.start_frame, recovery_frame + 1))
    references = [
        _reference(fn, 1_000_000 + (fn - window.start_frame) * 33_333)
        for fn in frame_numbers
    ]
    candidates = []
    for fn in frame_numbers:
        if fn < recovery_frame:
            T = _T((0.0, 0.0, 0.0))
            state = PoseTrackingState.LOCAL_TRACKING
        else:
            T = _T((0.10, 0.0, 0.0), yaw_deg=5.0)
            state = PoseTrackingState.RELOCALIZED
        candidates.append(
            PoseEstimate(
                frame_number=fn,
                device_timestamp_us=1_000_000 + (fn - window.start_frame) * 33_333,
                T_world_camera=T,
                valid=True,
                state=state,
            )
        )
    recovery = compute_recovery_metrics(window=window, references=references, candidates=candidates)
    assert recovery.recovery_requested_frame == recovery_frame
    assert recovery.recovery_actual_frame == recovery_frame
    assert recovery.recovery_latency_frames == 0
    assert recovery.reanchor_translation_jump == pytest.approx(0.10, abs=1e-9)
    assert recovery.reanchor_rotation_jump == pytest.approx(5.0, abs=1e-4)


def test_delayed_recovery_metrics_use_actual_frame() -> None:
    window = _window_for_test()
    requested = window.recovery_frame
    assert requested is not None
    actual = requested + 2
    frame_numbers = list(range(window.start_frame, actual + 1))
    references = [
        _reference(fn, 1_000_000 + (fn - window.start_frame) * 33_333)
        for fn in frame_numbers
    ]
    candidates = []
    for fn in frame_numbers:
        if fn < actual:
            T = _T((0.0, 0.0, 0.0))
            state = (
                PoseTrackingState.RELOCALIZING
                if fn >= requested
                else PoseTrackingState.LOCAL_TRACKING
            )
        else:
            T = _T((0.20, 0.0, 0.0), yaw_deg=10.0)
            state = PoseTrackingState.RELOCALIZED
        candidates.append(
            PoseEstimate(
                frame_number=fn,
                device_timestamp_us=1_000_000 + (fn - window.start_frame) * 33_333,
                valid=True,
                T_world_camera=T,
                state=state,
            )
        )
    timing = RecoveryTiming(
        recovery_requested_frame=requested,
        recovery_requested_device_timestamp_us=1_000_000 + (requested - window.start_frame) * 33_333,
        recovery_actual_frame=actual,
        recovery_actual_device_timestamp_us=1_000_000 + (actual - window.start_frame) * 33_333,
    )
    recovery = compute_recovery_metrics(
        window=window,
        references=references,
        candidates=candidates,
        recovery_timing=timing,
    )
    assert recovery.recovery_latency_frames == 2
    assert recovery.recovery_latency_sec == pytest.approx(2 * 33_333 / 1_000_000.0, abs=1e-9)
    assert recovery.reanchor_translation_jump == pytest.approx(0.20, abs=1e-9)
    assert recovery.reanchor_rotation_jump == pytest.approx(10.0, abs=1e-4)
    assert recovery.post_recovery_translation_error == pytest.approx(0.20, abs=1e-9)


def test_requested_hold_frame_is_not_treated_as_actual_recovery() -> None:
    window = _window_for_test()
    requested = window.recovery_frame
    assert requested is not None
    actual = requested + 2
    references = [
        _reference(fn, 1_000_000 + (fn - window.start_frame) * 33_333)
        for fn in range(window.start_frame, actual + 1)
    ]
    candidates = []
    for fn in range(window.start_frame, actual + 1):
        if fn < actual:
            candidates.append(
                PoseEstimate(
                    frame_number=fn,
                    device_timestamp_us=1_000_000 + (fn - window.start_frame) * 33_333,
                    T_world_camera=_T((0.0, 0.0, 0.0)),
                    valid=True,
                    state=PoseTrackingState.RELOCALIZING if fn >= requested else PoseTrackingState.LOCAL_TRACKING,
                )
            )
        else:
            candidates.append(
                PoseEstimate(
                    frame_number=fn,
                    device_timestamp_us=1_000_000 + (fn - window.start_frame) * 33_333,
                    T_world_camera=_T((0.15, 0.0, 0.0)),
                    valid=True,
                    state=PoseTrackingState.RELOCALIZED,
                )
            )
    recovery = compute_recovery_metrics(window=window, references=references, candidates=candidates)
    assert recovery.recovery_actual_frame == actual
    assert recovery.reanchor_translation_jump == pytest.approx(0.15, abs=1e-9)
    assert recovery.reanchor_translation_jump != 0.0


def test_recovery_failure_leaves_metrics_null() -> None:
    window = _window_for_test()
    references = [
        _reference(fn, 1_000_000 + (fn - window.start_frame) * 33_333)
        for fn in range(window.start_frame, window.end_frame + 1)
    ]
    candidates = [
        PoseEstimate(
            frame_number=fn,
            device_timestamp_us=1_000_000 + (fn - window.start_frame) * 33_333,
            T_world_camera=_T((0.0, 0.0, 0.0)),
            valid=True,
            state=PoseTrackingState.LOCAL_TRACKING,
        )
        for fn in range(window.start_frame, window.end_frame + 1)
    ]
    recovery = compute_recovery_metrics(window=window, references=references, candidates=candidates)
    assert recovery.recovery_requested_frame == window.recovery_frame
    assert recovery.recovery_actual_frame is None
    assert recovery.recovery_latency_frames is None
    assert recovery.reanchor_translation_jump is None
    assert recovery.post_recovery_translation_error is None


def test_b_motion_start_2s_style_delayed_recovery_jump_not_zero() -> None:
    window = _window_for_test()
    requested = 141
    actual = 143
    references = [
        _reference(fn, 1_000_000 + fn * 33_333, t=(float(fn) * 0.01, 0.0, 2.0))
        for fn in range(140, actual + 1)
    ]
    candidates = []
    for fn in range(140, actual + 1):
        if fn < actual:
            candidates.append(
                PoseEstimate(
                    frame_number=fn,
                    device_timestamp_us=1_000_000 + fn * 33_333,
                    T_world_camera=_T((0.0, 0.0, 0.0)),
                    valid=True,
                    state=PoseTrackingState.RELOCALIZING if fn >= requested else PoseTrackingState.LOCAL_TRACKING,
                )
            )
        else:
            candidates.append(
                PoseEstimate(
                    frame_number=fn,
                    device_timestamp_us=1_000_000 + fn * 33_333,
                    T_world_camera=_T((0.25, 0.0, 0.0), yaw_deg=4.0),
                    valid=True,
                    state=PoseTrackingState.RELOCALIZED,
                )
            )
    timing = RecoveryTiming(
        recovery_requested_frame=requested,
        recovery_requested_device_timestamp_us=1_000_000 + requested * 33_333,
        recovery_actual_frame=actual,
        recovery_actual_device_timestamp_us=1_000_000 + actual * 33_333,
    )
    recovery = compute_recovery_metrics(
        window=window,
        references=references,
        candidates=candidates,
        recovery_timing=timing,
    )
    assert recovery.recovery_latency_frames == 2
    assert recovery.reanchor_translation_jump == pytest.approx(0.25, abs=1e-9)
    assert recovery.reanchor_rotation_jump == pytest.approx(4.0, abs=1e-4)


def test_pose_metrics_unchanged_when_only_recovery_semantics_change() -> None:
    window = _window_for_test()
    references = [
        _reference(fn, 1_000_000 + (fn - window.start_frame) * 33_333)
        for fn in range(window.start_frame, 100)
    ]
    candidates = [
        PoseEstimate(
            frame_number=fn,
            device_timestamp_us=1_000_000 + (fn - window.start_frame) * 33_333,
            T_world_camera=_T((0.05, 0.0, 0.0)),
            valid=True,
            state=PoseTrackingState.LOCAL_TRACKING,
        )
        for fn in range(window.start_frame, window.end_frame + 1)
    ]
    pose_metrics = compute_pose_window_metrics(window=window, references=references, candidates=candidates)
    assert pose_metrics.translation_error.median == pytest.approx(0.05, abs=1e-9)
    assert pose_metrics.availability is not None
    assert pose_metrics.availability.availability_ratio == pytest.approx(1.0)


def test_cup2_world_error_geometry() -> None:
    ref = _reference(1, 1000, t=(0.0, 0.0, 0.0))
    cand = PoseEstimate(
        frame_number=1,
        device_timestamp_us=1000,
        T_world_camera=_T((0.10, 0.0, 0.0)),
        valid=True,
    )
    obs = CupObservation(
        frame_number=1,
        device_timestamp_us=1000,
        semantic_id="cup2",
        P_camera=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        valid=True,
    )
    assert cup_world_position_error(reference_pose=ref, candidate_pose=cand, observation=obs) == pytest.approx(
        0.10, abs=1e-9
    )


def test_cup2_absent_returns_not_applicable(official_windows) -> None:
    window = next(window for window in official_windows if window.window_id == "B_motion_start__0.5s")
    references = [_reference(81, 1_000_000)]
    candidates = references_to_perfect_candidates(references)
    metrics = compute_cup2_world_metrics(
        window=window,
        references=references,
        candidates=candidates,
        observations=[],
        cup2_semantic_id="cup2",
    )
    assert metrics.status == MetricStatus.NOT_APPLICABLE
    assert metrics.expected_observations == 0
    assert metrics.availability_ratio is None


def test_cup2_semantic_id_selection_only_counts_cup2() -> None:
    window = _window_for_test()
    references = [_reference(81, 1_000_000), _reference(82, 1_033_333)]
    candidates = references_to_perfect_candidates(references)
    observations = [
        CupObservation(
            frame_number=81,
            device_timestamp_us=1_000_000,
            semantic_id="cup1",
            P_camera=np.array([0.1, 0.0, 1.0]),
            valid=True,
        ),
        CupObservation(
            frame_number=82,
            device_timestamp_us=1_033_333,
            semantic_id="cup2",
            P_camera=np.array([0.1, 0.0, 1.0]),
            valid=True,
        ),
    ]
    metrics = compute_cup2_world_metrics(
        window=window,
        references=references,
        candidates=candidates,
        observations=observations,
        cup2_semantic_id="cup2",
    )
    assert metrics.status == MetricStatus.OK
    assert metrics.expected_observations == 1


def test_cup1_is_not_used_as_pose_source_contract() -> None:
  # Cup1 observations do not alter candidate pose metrics.
    window = _window_for_test()
    references = [_reference(81, 1_000_000)]
    candidates = references_to_perfect_candidates(references)
    cup1_only = [
        CupObservation(
            frame_number=81,
            device_timestamp_us=1_000_000,
            semantic_id="cup1",
            P_camera=np.array([0.5, 0.0, 1.0]),
            valid=True,
        )
    ]
    result = evaluate_window(
        window=window,
        references=references,
        candidates=candidates,
        observations=cup1_only,
    )
    assert result.cup2_world.status == MetricStatus.NOT_APPLICABLE
    assert result.pose.translation_error.median == pytest.approx(0.0, abs=1e-9)


def test_threshold_comparison_output(official_windows, official_frames) -> None:
    references = _references_for_official_frames(official_frames, range(81, 112))
    candidates = references_to_perfect_candidates(references)
    window = next(window for window in official_windows if window.window_id == "B_motion_start__1.0s")
    result = evaluate_window(
        window=window,
        references=references,
        candidates=candidates,
        thresholds=SuccessThresholds(),
    )
    assert result.threshold_comparison["pose_availability_pass"] is True
    assert result.threshold_comparison["major_tracking_lost_pass"] is True
