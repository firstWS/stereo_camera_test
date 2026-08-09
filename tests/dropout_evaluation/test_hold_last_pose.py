"""Tests for HOLD_LAST_POSE Phase 3 baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    DropoutAnchorDefinition,
    DropoutWindow,
    FrameTimestamp,
    compute_dropout_window,
    load_dropout_protocol_config,
)
from dropout_evaluation.evaluation_metrics import (  # noqa: E402
    PoseReference,
    PoseTrackingState,
    evaluate_window,
    rotation_error_deg,
    translation_error,
)
from dropout_evaluation.hold_last_pose import (  # noqa: E402
    HOLD_LAST_POSE_ALGORITHM_ID,
    HoldLastPoseStatus,
    find_recovery_runtime_pose,
    generate_hold_last_pose_candidates,
    select_pre_window_anchor,
)
from dropout_evaluation.hold_last_pose_runner import (  # noqa: E402
    evaluate_hold_last_pose_window,
    load_dropout_windows_from_manifest,
)
from dropout_evaluation.runtime_apriltag import RuntimeAprilTagPose  # noqa: E402

OFFICIAL_MANIFEST = (
    ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"
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


def _runtime_pose(fn: int, ts: int, t=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> RuntimeAprilTagPose:
    return RuntimeAprilTagPose(
        frame_number=fn,
        device_timestamp_us=ts,
        T_world_camera=_T(t, yaw_deg),
        valid=True,
    )


def _window(start_fn: int, start_ts: int, duration: float = 0.5) -> DropoutWindow:
    anchor = DropoutAnchorDefinition(
        anchor_id="test",
        start_frame=start_fn,
        start_device_timestamp_us=start_ts,
        convention="test",
        motion_class="test",
    )
    frames = [
        FrameTimestamp(frame_number=fn, device_timestamp_us=ts)
        for fn, ts in [
            (79, start_ts - 90_000),
            (80, start_ts - 30_000),
            (81, start_ts),
            (82, start_ts + 30_000),
            (83, start_ts + 60_000),
            (84, start_ts + 90_000),
            (85, start_ts + 120_000),
            (86, start_ts + 150_000),
        ]
    ]
    return compute_dropout_window(
        anchor=anchor,
        duration_sec=duration,
        session_id="test",
        frames=frames,
    )


def test_anchor_uses_latest_prior_valid_not_start_frame() -> None:
    window = _window(81, 1_000_000)
    runtime = [
        _runtime_pose(79, 900_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(80, 970_000, t=(1.0, 0.0, 0.0)),
        _runtime_pose(81, 1_000_000, t=(9.0, 0.0, 0.0)),
    ]
    anchor = select_pre_window_anchor(
        window=window,
        runtime_poses=runtime,
        max_anchor_age_frames=120,
    )
    assert anchor is not None
    assert anchor.frame_number == 80
    assert anchor.frame_number != window.start_frame


def test_anchor_skips_invalid_start_minus_one() -> None:
    window = _window(81, 1_000_000)
    runtime = [
        _runtime_pose(78, 850_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(80, 970_000, t=(1.0, 0.0, 0.0)),
    ]
    anchor = select_pre_window_anchor(
        window=window,
        runtime_poses=runtime,
        max_anchor_age_frames=120,
    )
    assert anchor is not None
    assert anchor.frame_number == 80


def test_no_prior_valid_anchor_returns_invalid() -> None:
    window = _window(81, 1_000_000)
    runtime = [_runtime_pose(81, 1_000_000)]
    result = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=[
            FrameTimestamp(81, 1_000_000),
            FrameTimestamp(82, 1_030_000),
        ],
    )
    assert result.status == HoldLastPoseStatus.INVALID
    assert result.candidates == []


def test_hold_pose_constant_during_dropout() -> None:
    window = _window(81, 1_000_000)
    runtime = [
        _runtime_pose(80, 970_000, t=(1.0, 2.0, 3.0)),
        _runtime_pose(81, 1_000_000, t=(9.0, 0.0, 0.0)),
        _runtime_pose(85, 1_120_000, t=(5.0, 0.0, 0.0)),
    ]
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_120_000),
    ]
    result = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert result.status == HoldLastPoseStatus.OK
    masked = [c for c in result.candidates if c.state == PoseTrackingState.LOCAL_TRACKING]
    assert masked
    for candidate in masked:
        assert candidate.algorithm_id == HOLD_LAST_POSE_ALGORITHM_ID
        assert np.allclose(candidate.T_world_camera, _T((1.0, 2.0, 3.0)))


def test_recovery_uses_first_valid_runtime_pose_after_boundary() -> None:
    anchor = DropoutAnchorDefinition(
        anchor_id="test",
        start_frame=81,
        start_device_timestamp_us=1_000_000,
        convention="test",
        motion_class="test",
    )
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_510_000),
        FrameTimestamp(86, 1_520_000),
        FrameTimestamp(87, 1_530_000),
    ]
    window = compute_dropout_window(
        anchor=anchor,
        duration_sec=0.5,
        session_id="test",
        frames=frames,
    )
    runtime = [
        _runtime_pose(80, 970_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(87, 1_530_000, t=(2.0, 0.0, 0.0)),
    ]
    result = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert result.provenance.recovery_requested_frame == window.recovery_frame
    assert result.provenance.recovery_actual_frame == 87
    recovered = next(c for c in result.candidates if c.frame_number == 87)
    assert recovered.state == PoseTrackingState.RELOCALIZED
    assert np.allclose(recovered.T_world_camera, _T((2.0, 0.0, 0.0)))
    relocalizing = [c for c in result.candidates if c.state == PoseTrackingState.RELOCALIZING]
    assert relocalizing
    requested_ts = window.recovery_device_timestamp_us
    assert requested_ts is not None
    assert result.provenance.recovery_latency_frames == 87 - window.recovery_frame
    assert result.provenance.recovery_latency_sec == pytest.approx(
        (1_530_000 - requested_ts) / 1_000_000.0,
        abs=1e-9,
    )
    assert result.provenance.recovery_latency_sec != pytest.approx(
        (1_530_000 - window.boundary_timestamp_us) / 1_000_000.0,
        abs=1e-12,
    )


def test_immediate_recovery_latency_is_zero() -> None:
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_510_000),
    ]
    window = compute_dropout_window(
        anchor=DropoutAnchorDefinition(
            anchor_id="test",
            start_frame=81,
            start_device_timestamp_us=1_000_000,
            convention="test",
            motion_class="test",
        ),
        duration_sec=0.5,
        session_id="test",
        frames=frames,
    )
    runtime = [
        _runtime_pose(80, 970_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(85, 1_510_000, t=(0.5, 0.0, 0.0)),
    ]
    result = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert result.provenance.recovery_requested_frame == result.provenance.recovery_actual_frame == 85
    assert result.provenance.recovery_latency_frames == 0
    assert result.provenance.recovery_latency_sec == pytest.approx(0.0, abs=1e-12)


def test_provenance_and_evaluation_recovery_latency_match() -> None:
    window = _window(81, 1_000_000)
    runtime = [
        _runtime_pose(80, 970_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(87, 1_530_000, t=(0.2, 0.0, 0.0)),
    ]
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_510_000),
        FrameTimestamp(86, 1_520_000),
        FrameTimestamp(87, 1_530_000),
    ]
    window = compute_dropout_window(
        anchor=DropoutAnchorDefinition(
            anchor_id="test",
            start_frame=81,
            start_device_timestamp_us=1_000_000,
            convention="test",
            motion_class="test",
        ),
        duration_sec=0.5,
        session_id="test",
        frames=frames,
    )
    references = [
        PoseReference(fn, ts, _T((0.0, 0.0, 0.0)), True)
        for fn, ts in [(80, 970_000), (81, 1_000_000), (84, 1_090_000), (87, 1_530_000)]
    ]
    generation, evaluation = evaluate_hold_last_pose_window(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
        thresholds=load_dropout_protocol_config(
            ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml"
        ).success_thresholds,
    )
    assert generation is not None and evaluation is not None
    prov = generation.provenance
    rec = evaluation.recovery
    assert prov.recovery_latency_frames == rec.recovery_latency_frames
    assert prov.recovery_latency_sec == pytest.approx(rec.recovery_latency_sec, abs=1e-12)
    assert rec.recovery_latency_sec == pytest.approx(
        (1_530_000 - window.recovery_device_timestamp_us) / 1_000_000.0,
        abs=1e-9,
    )


def test_find_recovery_runtime_pose_returns_null_on_failure() -> None:
    window = _window(81, 1_000_000)
    pose, latency_frames, latency_sec = find_recovery_runtime_pose(
        window=window,
        runtime_poses=[_runtime_pose(80, 970_000)],
    )
    assert pose is None
    assert latency_frames is None
    assert latency_sec is None


def test_moving_reference_increases_hold_error() -> None:
    window = _window(81, 1_000_000)
    runtime = [
        _runtime_pose(80, 970_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(85, 1_120_000, t=(5.0, 0.0, 0.0)),
    ]
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_120_000),
    ]
    references = [
        PoseReference(fn, ts, _T((float(fn - 80) * 0.1, 0.0, 0.0)), True)
        for fn, ts in [(80, 970_000), (81, 1_000_000), (82, 1_030_000), (83, 1_060_000), (84, 1_090_000)]
    ]
    generation = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    evaluation = evaluate_window(
        window=window,
        references=references,
        candidates=generation.candidates,
        algorithm_id=HOLD_LAST_POSE_ALGORITHM_ID,
    )
    assert evaluation.pose.translation_error.median is not None
    assert evaluation.pose.translation_error.median > 0.0


def test_static_scene_hold_matches_reference_zero_error() -> None:
    window = _window(81, 1_000_000)
    T = _T((0.5, -0.2, 2.0))
    runtime = [
        _runtime_pose(80, 970_000, t=(0.5, -0.2, 2.0)),
        _runtime_pose(85, 1_120_000, t=(0.5, -0.2, 2.0)),
    ]
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_120_000),
    ]
    references = [
        PoseReference(fn, ts, T.copy(), True)
        for fn, ts in [(80, 970_000), (81, 1_000_000), (82, 1_030_000), (83, 1_060_000), (84, 1_090_000)]
    ]
    generation = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    evaluation = evaluate_window(
        window=window,
        references=references,
        candidates=generation.candidates,
        algorithm_id=HOLD_LAST_POSE_ALGORITHM_ID,
    )
    assert evaluation.pose.translation_error.median == pytest.approx(0.0, abs=1e-9)
    assert evaluation.pose.rotation_error_deg.median == pytest.approx(0.0, abs=1e-6)
    assert evaluation.pose.availability is not None
    assert evaluation.pose.availability.availability_ratio == pytest.approx(1.0)


def test_reanchor_jump_measurable_on_recovery() -> None:
    anchor = DropoutAnchorDefinition(
        anchor_id="test",
        start_frame=81,
        start_device_timestamp_us=1_000_000,
        convention="test",
        motion_class="test",
    )
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_510_000),
    ]
    window = compute_dropout_window(
        anchor=anchor,
        duration_sec=0.5,
        session_id="test",
        frames=frames,
    )
    runtime = [
        _runtime_pose(80, 970_000, t=(0.0, 0.0, 0.0)),
        _runtime_pose(85, 1_510_000, t=(0.2, 0.0, 0.0)),
    ]
    references = [
        PoseReference(fn, ts, _T((0.0, 0.0, 0.0)), True)
        for fn, ts in [(80, 970_000), (81, 1_000_000), (84, 1_090_000), (85, 1_510_000)]
    ]
    generation, evaluation = evaluate_hold_last_pose_window(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
        thresholds=load_dropout_protocol_config(ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml").success_thresholds,
    )
    assert generation.status == HoldLastPoseStatus.OK
    assert evaluation is not None
    assert evaluation.recovery.reanchor_translation_jump == pytest.approx(0.2, abs=1e-9)


def test_official_manifest_load_compatible() -> None:
    if not OFFICIAL_MANIFEST.is_file():
        pytest.skip("official manifest not available")
    windows = load_dropout_windows_from_manifest(OFFICIAL_MANIFEST)
    assert len(windows) == 15
    assert windows[0].window_id.startswith("B_motion_start__")


def test_official_manifest_schema_unchanged() -> None:
    if not OFFICIAL_MANIFEST.is_file():
        pytest.skip("official manifest not available")
    payload = json.loads(OFFICIAL_MANIFEST.read_text(encoding="utf-8"))
    assert payload["windows"][0]["start_frame"] == 81
    assert "includes_cup2_appearance" in payload["windows"][0]
