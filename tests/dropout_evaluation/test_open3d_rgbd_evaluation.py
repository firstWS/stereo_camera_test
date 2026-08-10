"""Tests for Phase 4.3 official Open3D RGB-D evaluation runner."""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    DropoutAnchorDefinition,
    DropoutWindow,
    FrameTimestamp,
    SuccessThresholds,
    compute_dropout_window,
)
from dropout_evaluation.evaluation_metrics import (  # noqa: E402
    CupObservation,
    MetricStatus,
    PoseEstimate,
    PoseReference,
    PoseTrackingState,
    evaluate_window,
    references_to_perfect_candidates,
)
from dropout_evaluation.open3d_rgbd_evaluation import (  # noqa: E402
    build_hold_comparison_row,
    compute_first_masked_diagnostics,
    evaluate_open3d_rgbd_window,
    generate_window_candidates,
    load_open3d_rgbd_evaluation_config,
    sanitize_json_value,
)
from dropout_evaluation.rgbd_odometry import RGBD_ODOMETRY_ALGORITHM_ID  # noqa: E402
from dropout_evaluation.rgbd_odometry_adapter import (  # noqa: E402
    RgbdAdapterProvenance,
    RgbdAdapterStatus,
    RgbdAdapterWindowResult,
    RgbdOdometryAdapterConfig,
    WindowDiagnosticSummary,
)
from dropout_evaluation.rgbd_odometry_continuous import TrajectorySample  # noqa: E402
from dropout_evaluation.runtime_apriltag import RuntimeAprilTagPose  # noqa: E402


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


def _reference(fn: int, ts: int, t=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> PoseReference:
    return PoseReference(
        frame_number=fn,
        device_timestamp_us=ts,
        T_world_camera=_T(t, yaw_deg),
        valid=True,
    )


def _runtime(fn: int, ts: int, t=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> RuntimeAprilTagPose:
    return RuntimeAprilTagPose(
        frame_number=fn,
        device_timestamp_us=ts,
        T_world_camera=_T(t, yaw_deg),
        valid=True,
    )


def _local(
    fn: int,
    ts: int,
    *,
    valid: bool = True,
    segment_id: int = 2,
    t=(0.0, 0.0, 0.0),
    bridge_recovered: bool = False,
) -> TrajectorySample:
    if not valid:
        return TrajectorySample(
            frame_number=fn,
            device_timestamp_us=ts,
            valid=False,
            tracking_state="TRACKING_LOST",
            segment_id=segment_id,
            segment_start=False,
            continuity_from_previous_segment=True,
            tx=None,
            ty=None,
            tz=None,
            qw=None,
            qx=None,
            qy=None,
            qz=None,
            source_frame=fn - 1,
            pair_gap_frames=1,
            bridge_recovered=False,
        )
    return TrajectorySample(
        frame_number=fn,
        device_timestamp_us=ts,
        valid=True,
        tracking_state="LOCAL_TRACKING",
        segment_id=segment_id,
        segment_start=False,
        continuity_from_previous_segment=True,
        tx=float(t[0]),
        ty=float(t[1]),
        tz=float(t[2]),
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        source_frame=fn - 1,
        pair_gap_frames=1,
        bridge_recovered=bridge_recovered,
    )


def _window(start_fn: int = 81, start_ts: int = 1_000_000, duration: float = 0.5) -> DropoutWindow:
    anchor = DropoutAnchorDefinition(
        anchor_id="B_motion_start",
        start_frame=start_fn,
        start_device_timestamp_us=start_ts,
        convention="first_sustained_motion_frame",
        motion_class="motion_start",
    )
    frames = [
        FrameTimestamp(frame_number=start_fn - 2 + index, device_timestamp_us=start_ts - 60_000 + index * 30_000)
        for index in range(12)
    ]
    return compute_dropout_window(
        anchor=anchor,
        duration_sec=duration,
        session_id="test",
        frames=frames,
    )


def _write_trajectory_csv(path: Path, samples: list[TrajectorySample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(samples[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def _adapter_result(
    *,
    window_id: str,
    status: RgbdAdapterStatus,
    candidates: list[PoseEstimate],
    protocol_anchor: int | None = 80,
    alignment_frame: int | None = 79,
) -> RgbdAdapterWindowResult:
    provenance = RgbdAdapterProvenance(
        window_id=window_id,
        algorithm_id=RGBD_ODOMETRY_ALGORITHM_ID,
        protocol_runtime_anchor_frame=protocol_anchor,
        world_alignment_frame_before_dropout=alignment_frame,
        world_alignment_timestamp_us=970_000,
        alignment_age_frames=2 if alignment_frame == 79 else 1,
        alignment_age_sec=0.06,
        alignment_segment_id=2,
        segment_reset_during_dropout=False,
        local_invalid_frames_in_dropout=0,
        world_valid_frames_in_dropout=len(candidates),
        runtime_tag_used_during_dropout=False,
        reference_used_by_candidate=False,
        tag_texture_visible=True,
        recovery_requested_frame=96,
        recovery_actual_frame=96,
        recovery_latency_frames=0,
        recovery_latency_sec=0.0,
    )
    diagnostic = WindowDiagnosticSummary(
        window_id=window_id,
        dropout_start_frame=81,
        dropout_end_frame=95,
        protocol_anchor_frame=protocol_anchor,
        alignment_frame_before_dropout=alignment_frame,
        alignment_age_frames=provenance.alignment_age_frames,
        alignment_age_sec=provenance.alignment_age_sec,
        segment_id=2,
        expected_dropout_frames=15,
        world_valid_frames_in_dropout=len(candidates),
        world_invalid_frames_in_dropout=0,
        world_availability_ratio=1.0,
        segment_reset_during_dropout=False,
        first_invalid_frame=None,
        first_local_bridge_recovery_frame=None,
        recovery_requested_frame=96,
        recovery_actual_frame=96,
        recovery_latency_frames=0,
        recovery_latency_sec=0.0,
    )
    from dropout_evaluation.rgbd_odometry_adapter import SessionReplayResult

    replay = SessionReplayResult(
        frames={},
        recovery_actual_frame=96,
        recovery_actual_timestamp_us=1_200_000,
        protocol_runtime_anchor_frame=protocol_anchor,
        world_alignment_frame_before_dropout=alignment_frame,
        world_alignment_timestamp_before_dropout=970_000,
        alignment_age_frames_at_dropout=provenance.alignment_age_frames,
        alignment_age_sec_at_dropout=provenance.alignment_age_sec,
        alignment_segment_id=2,
        segment_reset_during_dropout=False,
        dropout_segment_id=2,
    )
    return RgbdAdapterWindowResult(
        window_id=window_id,
        status=status,
        provenance=provenance,
        candidates=candidates,
        diagnostic=diagnostic,
        replay=replay,
    )


def test_sanitize_json_rejects_nan_and_inf() -> None:
    payload = sanitize_json_value({"a": float("nan"), "b": float("inf"), "c": 1.0, "d": [float("nan")]})
    assert payload["a"] is None
    assert payload["b"] is None
    assert payload["c"] == 1.0
    assert payload["d"] == [None]


def test_first_masked_diagnostics_and_growth() -> None:
    window = _window()
    end_fn = window.end_frame
    end_ts = window.end_device_timestamp_us
    references = [
        _reference(81, 1_000_000, t=(0.0, 0.0, 0.0)),
        _reference(end_fn, end_ts, t=(0.0, 0.0, 0.0)),
    ]
    candidates = [
        PoseEstimate(81, 1_000_000, _T((0.0, 0.0, 0.0)), True, PoseTrackingState.LOCAL_TRACKING, RGBD_ODOMETRY_ALGORITHM_ID),
        PoseEstimate(end_fn, end_ts, _T((0.2, 0.0, 0.0)), True, PoseTrackingState.LOCAL_TRACKING, RGBD_ODOMETRY_ALGORITHM_ID),
    ]
    diag = compute_first_masked_diagnostics(window=window, references=references, candidates=candidates)
    assert diag["first_valid_masked_frame"] == 81
    assert diag["first_masked_translation_error"] == pytest.approx(0.0, abs=1e-9)
    assert diag["end_translation_error"] == pytest.approx(0.2, abs=1e-9)
    assert diag["translation_error_growth"] == pytest.approx(0.2, abs=1e-9)


def test_end_invalid_leaves_null_end_error() -> None:
    window = _window()
    references = [_reference(81, 1_000_000), _reference(95, 1_420_000)]
    candidates = [
        PoseEstimate(81, 1_000_000, _T((0.0, 0.0, 0.0)), True, PoseTrackingState.LOCAL_TRACKING, RGBD_ODOMETRY_ALGORITHM_ID),
        PoseEstimate(
            95,
            1_420_000,
            None,
            False,
            PoseTrackingState.TRACKING_LOST,
            RGBD_ODOMETRY_ALGORITHM_ID,
        ),
    ]
    diag = compute_first_masked_diagnostics(window=window, references=references, candidates=candidates)
    assert diag["end_translation_error"] is None
    assert diag["end_rotation_error"] is None
    assert diag["translation_error_growth"] is None


def test_all_valid_candidate_evaluates_with_phase3_metrics(tmp_path: Path) -> None:
    window = _window()
    start_ts = window.start_device_timestamp_us
    frames = [FrameTimestamp(fn, start_ts - 60_000 + i * 30_000) for i, fn in enumerate(range(79, 88))]
    samples = [
        _local(79, frames[0].device_timestamp_us, t=(0.79, 0.0, 0.0)),
        _local(80, frames[1].device_timestamp_us, valid=False),
        *[_local(fn, f.device_timestamp_us, t=(0.01 * fn, 0.0, 0.0)) for fn, f in zip(range(81, 88), frames[2:])],
    ]
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, samples)
    runtime = [_runtime(fn, ts, t=(0.1 * fn, 0.0, 0.0)) for fn, ts in zip(range(79, 88), [f.device_timestamp_us for f in frames])]
    references = [_reference(fn, ts, t=(0.1 * fn, 0.0, 0.0)) for fn, ts in zip(range(79, 96), range(start_ts - 60_000, start_ts + 500_000, 30_000))]

    generation, evaluation, extras = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
        thresholds=SuccessThresholds(),
    )
    assert generation.status == RgbdAdapterStatus.OK
    assert evaluation is not None
    assert evaluation.pose.availability is not None
    assert evaluation.pose.availability.availability_ratio == pytest.approx(1.0)
    assert extras["fairness"]["applies"] is True
    assert extras["fairness"]["protocol_runtime_anchor_frame"] == 80
    assert extras["fairness"]["world_alignment_frame"] == 79


def test_partial_invalid_reduces_availability(tmp_path: Path) -> None:
    window = _window()
    start_ts = window.start_device_timestamp_us
    frames = [FrameTimestamp(fn, start_ts - 60_000 + i * 30_000) for i, fn in enumerate(range(79, 88))]
    samples = []
    for fn, frame in zip(range(79, 88), frames):
        valid = fn != 84
        samples.append(_local(fn, frame.device_timestamp_us, valid=valid))
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, samples)
    runtime = [_runtime(fn, f.device_timestamp_us, t=(1.0, 0.0, 0.0)) for fn, f in zip(range(79, 88), frames)]
    references = [_reference(fn, f.device_timestamp_us) for fn, f in zip(range(79, 96), frames + [FrameTimestamp(88, start_ts + 240_000)] * 7)]

    _, evaluation, _ = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references[: len(frames)],
        cup_observations=[],
    )
    assert evaluation is not None
    assert evaluation.pose.availability is not None
    assert evaluation.pose.availability.availability_ratio < 1.0


def test_availability_below_threshold_fails_compare(tmp_path: Path) -> None:
    window = _window()
    start_ts = window.start_device_timestamp_us
    frames = [
        FrameTimestamp(79, start_ts - 60_000),
        FrameTimestamp(80, start_ts - 30_000),
        *[FrameTimestamp(fn, start_ts + i * 30_000) for i, fn in enumerate(range(81, 96))],
    ]
    samples = [
        _local(79, start_ts - 60_000),
        _local(80, start_ts - 30_000),
        *[
            _local(fn, f.device_timestamp_us, valid=(fn % 2 == 1))
            for fn, f in zip(range(81, 96), frames[2:])
        ],
    ]
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, samples)
    runtime = [_runtime(79, start_ts - 60_000), _runtime(80, start_ts - 30_000)]
    references = [_reference(f.frame_number, f.device_timestamp_us) for f in frames]

    _, evaluation, _ = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
        thresholds=SuccessThresholds(),
    )
    assert evaluation is not None
    assert evaluation.pose.availability.availability_ratio < 0.90
    assert evaluation.threshold_comparison["pose_availability_pass"] is False


def test_local_invalid_makes_cup2_world_invalid() -> None:
    window = _window()
    references = [_reference(81, 1_000_000)]
    candidates = [
        PoseEstimate(
            81,
            1_000_000,
            None,
            False,
            PoseTrackingState.TRACKING_LOST,
            RGBD_ODOMETRY_ALGORITHM_ID,
        )
    ]
    observations = [
        CupObservation(
            frame_number=81,
            device_timestamp_us=1_000_000,
            semantic_id="cup2",
            P_camera=np.array([0.1, 0.0, 1.0]),
            valid=True,
        )
    ]
    result = evaluate_window(
        window=window,
        references=references,
        candidates=candidates,
        observations=observations,
    )
    assert result.cup2_world.candidate_world_valid_count == 0


def test_cup2_not_applicable_without_masked_cup2_observations() -> None:
    window = _window()
    references = [_reference(81, 1_000_000)]
    candidates = references_to_perfect_candidates(references)
    result = evaluate_window(window=window, references=references, candidates=candidates, observations=[])
    assert result.cup2_world.status == MetricStatus.NOT_APPLICABLE


def test_b_fairness_provenance_exposes_alignment_mismatch() -> None:
    window = _window()
    references = [_reference(81, 1_000_000)]
    candidates = references_to_perfect_candidates(references)
    generation = _adapter_result(
        window_id=window.window_id,
        status=RgbdAdapterStatus.OK,
        candidates=candidates,
        protocol_anchor=80,
        alignment_frame=79,
    )
    from dropout_evaluation.open3d_rgbd_evaluation import _b_fairness_metadata

    fairness = _b_fairness_metadata(window, generation)
    assert fairness["applies"] is True
    assert fairness["protocol_runtime_anchor_frame"] == 80
    assert fairness["world_alignment_frame"] == 79
    assert fairness["warning"] is not None


def test_hold_comparison_row_structure() -> None:
    window = _window()
    references = [_reference(81, 1_000_000), _reference(95, 1_420_000)]
    candidates = references_to_perfect_candidates(references)
    evaluation = evaluate_window(window=window, references=references, candidates=candidates)
    hold_eval = evaluation.as_dict()
    row = build_hold_comparison_row(
        window_id=window.window_id,
        rgbd_evaluation=evaluation,
        hold_evaluation=hold_eval,
    )
    assert row["window_id"] == window.window_id
    assert row["rgbd"]["pose_availability"] == pytest.approx(1.0)
    assert row["hold"]["translation_median"] == pytest.approx(0.0, abs=1e-9)


def test_deterministic_in_memory_evaluation(tmp_path: Path) -> None:
    window = _window()
    start_ts = window.start_device_timestamp_us
    frames = [FrameTimestamp(fn, start_ts - 60_000 + i * 30_000) for i, fn in enumerate(range(79, 88))]
    samples = [_local(fn, f.device_timestamp_us) for fn, f in zip(range(79, 88), frames)]
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, samples)
    runtime = [_runtime(fn, f.device_timestamp_us, t=(1.0, 0.0, 0.0)) for fn, f in zip(range(79, 88), frames)]
    references = [_reference(fn, f.device_timestamp_us) for fn, f in zip(range(79, 88), frames)]

    _, eval_a, extras_a = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
    )
    _, eval_b, extras_b = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
    )
    assert eval_a is not None and eval_b is not None
    assert eval_a.as_dict() == eval_b.as_dict()
    assert extras_a == extras_b


def test_candidate_generation_has_no_reference_dependency() -> None:
    source = (ROOT / "src/dropout_evaluation/rgbd_odometry_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "evaluation_io" not in modules


def test_official_runner_candidate_wrapper_has_no_reference_import() -> None:
    source = (ROOT / "src/dropout_evaluation/open3d_rgbd_evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    candidate_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_window_candidates"
    )
    assert candidate_fn is not None


def test_phase4_config_loads_and_reuses_phase3_protocol() -> None:
    config_path = ROOT / "configs/evaluation/phase4_open3d_rgbd_scenario_a.yaml"
    config = load_open3d_rgbd_evaluation_config(config_path, repo_root=ROOT)
    assert config.protocol.session_id == "20260807_161354_scenario_a"
    assert config.paths.continuous_trajectory_csv.name == "trajectory.csv"
    assert config.paths.output_dir.name == "open3d_rgbd_odometry"


def test_recovery_delayed_when_local_invalid_at_requested_frame(tmp_path: Path) -> None:
    window = _window(duration=0.5)
    start_ts = window.start_device_timestamp_us
    recovery_ts = start_ts + 510_000
    frames = [
        FrameTimestamp(79, start_ts - 60_000),
        FrameTimestamp(80, start_ts - 30_000),
        FrameTimestamp(81, start_ts),
        FrameTimestamp(82, start_ts + 30_000),
        FrameTimestamp(83, recovery_ts),
        FrameTimestamp(84, recovery_ts + 30_000),
    ]
    samples = [
        _local(79, start_ts - 60_000),
        _local(80, start_ts - 30_000),
        _local(81, start_ts),
        _local(82, start_ts + 30_000, valid=False),
        _local(83, recovery_ts),
        _local(84, recovery_ts + 30_000),
    ]
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, samples)
    runtime = [
        _runtime(79, start_ts - 60_000, t=(1.0, 0.0, 0.0)),
        _runtime(80, start_ts - 30_000, t=(1.0, 0.0, 0.0)),
        _runtime(83, recovery_ts, t=(2.0, 0.0, 0.0)),
        _runtime(84, recovery_ts + 30_000, t=(2.0, 0.0, 0.0)),
    ]
    references = [_reference(f.frame_number, f.device_timestamp_us) for f in frames]

    generation, evaluation, extras = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
    )
    assert generation.status == RgbdAdapterStatus.OK
    assert evaluation is not None
    assert generation.provenance.recovery_actual_frame == 83
    assert extras["candidate_provenance"]["recovery_actual_frame"] == 83


def test_segment_reset_during_dropout_makes_world_invalid(tmp_path: Path) -> None:
    window = _window()
    start_ts = window.start_device_timestamp_us
    frames = [
        FrameTimestamp(79, start_ts - 60_000),
        FrameTimestamp(80, start_ts - 30_000),
        FrameTimestamp(81, start_ts),
        FrameTimestamp(82, start_ts + 30_000),
    ]
    samples = [
        _local(79, start_ts - 60_000, segment_id=2),
        _local(80, start_ts - 30_000, segment_id=2),
        _local(81, start_ts, segment_id=2),
        _local(82, start_ts + 30_000, segment_id=3),
    ]
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, samples)
    runtime = [_runtime(79, start_ts - 60_000), _runtime(80, start_ts - 30_000)]
    references = [_reference(f.frame_number, f.device_timestamp_us) for f in frames]

    generation, evaluation, extras = evaluate_open3d_rgbd_window(
        window=window,
        local_trajectory_csv=traj,
        runtime_poses=runtime,
        frame_timestamps=frames,
        references=references,
        cup_observations=[],
    )
    assert generation.replay.segment_reset_during_dropout is True
    by_frame = {c.frame_number: c for c in generation.candidates}
    assert by_frame[82].valid is False
    assert extras["segment_reset_during_dropout"] is True
