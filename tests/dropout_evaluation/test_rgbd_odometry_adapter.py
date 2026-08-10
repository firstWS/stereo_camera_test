"""Tests for Phase 4.2-C RGB-D odometry Phase 3 harness adapter."""

from __future__ import annotations

import ast
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
    is_runtime_tag_masked,
)
from dropout_evaluation.evaluation_metrics import PoseTrackingState  # noqa: E402
from dropout_evaluation.rgbd_odometry import RGBD_ODOMETRY_ALGORITHM_ID  # noqa: E402
from dropout_evaluation.rgbd_odometry_adapter import (  # noqa: E402
    RgbdAdapterStatus,
    compute_joint_alignment,
    compute_world_pose,
    generate_rgbd_odometry_candidates,
    replay_session_for_window,
    trajectory_sample_to_T_odom,
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


def _local(
    fn: int,
    ts: int,
    *,
    valid: bool = True,
    segment_id: int = 0,
    t=(0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
    bridge_recovered: bool = False,
) -> TrajectorySample:
    if valid:
        T = _T(t, yaw_deg)
        return TrajectorySample(
            frame_number=fn,
            device_timestamp_us=ts,
            valid=True,
            tracking_state="LOCAL_TRACKING",
            segment_id=segment_id,
            segment_start=fn == 1,
            continuity_from_previous_segment=True,
            tx=float(T[0, 3]),
            ty=float(T[1, 3]),
            tz=float(T[2, 3]),
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            source_frame=fn - 1 if fn > 1 else None,
            pair_gap_frames=1,
            bridge_recovered=bridge_recovered,
        )
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


def _runtime(fn: int, ts: int, t=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> RuntimeAprilTagPose:
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
            (87, start_ts + 180_000),
            (88, start_ts + 210_000),
        ]
    ]
    return compute_dropout_window(
        anchor=anchor,
        duration_sec=duration,
        session_id="test",
        frames=frames,
    )


def _candidates_by_frame(result):
    return {candidate.frame_number: candidate for candidate in result.candidates}


def test_joint_alignment_algebra() -> None:
    T_tag = _T((1.0, 2.0, 3.0), yaw_deg=10.0)
    T_odom = _T((0.1, 0.2, 0.3), yaw_deg=5.0)
    T_world_odom = compute_joint_alignment(T_tag, T_odom)
    recovered = compute_world_pose(T_world_odom, T_odom)
    assert np.allclose(recovered, T_tag, atol=1e-9)


def test_world_candidate_transform_direction() -> None:
    T_world_odom = _T((0.5, 0.0, 0.0))
    T_odom = _T((0.1, 0.0, 0.0))
    T_world = compute_world_pose(T_world_odom, T_odom)
    assert np.allclose(T_world[:3, 3], [0.6, 0.0, 0.0], atol=1e-9)


def test_tag_and_local_valid_before_dropout_create_alignment() -> None:
    window = _window(81, 1_000_000)
    local = [_local(79, 970_000), _local(80, 990_000), _local(81, 1_000_000)]
    runtime = [_runtime(79, 970_000, t=(1.0, 0.0, 0.0)), _runtime(80, 990_000, t=(1.1, 0.0, 0.0))]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=[FrameTimestamp(fn, ts) for fn, ts in [(79, 970_000), (80, 990_000), (81, 1_000_000)]],
    )
    assert result.status == RgbdAdapterStatus.OK
    assert result.provenance.world_alignment_frame_before_dropout == 80


def test_dropout_masks_runtime_tag_even_when_present() -> None:
    window = _window(81, 1_000_000)
    frames = [FrameTimestamp(fn, ts) for fn, ts in [(79, 970_000), (80, 990_000), (81, 1_000_000), (82, 1_030_000)]]
    local = [
        _local(79, 970_000),
        _local(80, 990_000),
        _local(81, 1_000_000),
        _local(82, 1_030_000),
    ]
    runtime = [
        _runtime(79, 970_000, t=(1.0, 0.0, 0.0)),
        _runtime(80, 990_000, t=(1.0, 0.0, 0.0)),
        _runtime(81, 1_000_000, t=(9.0, 0.0, 0.0)),
        _runtime(82, 1_030_000, t=(9.0, 0.0, 0.0)),
    ]
    replay = replay_session_for_window(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert replay.frames[80].joint_updated is True
    assert replay.frames[81].masked is True
    assert replay.frames[81].joint_updated is False
    assert replay.frames[82].masked is True
    assert replay.frames[82].joint_updated is False


def test_retained_alignment_when_protocol_anchor_local_invalid() -> None:
    window = _window(81, 1_000_000)
    frames = [FrameTimestamp(fn, ts) for fn, ts in [(79, 970_000), (80, 990_000), (81, 1_000_000), (82, 1_030_000)]]
    local = [
        _local(79, 970_000, t=(0.0, 0.0, 0.0)),
        _local(80, 990_000, valid=False),
        _local(81, 1_000_000, t=(0.1, 0.0, 0.0)),
        _local(82, 1_030_000, t=(0.2, 0.0, 0.0)),
    ]
    runtime = [
        _runtime(79, 970_000, t=(1.0, 0.0, 0.0)),
        _runtime(80, 990_000, t=(1.1, 0.0, 0.0)),
        _runtime(81, 1_000_000, t=(9.0, 0.0, 0.0)),
    ]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    by_frame = _candidates_by_frame(result)
    assert result.status == RgbdAdapterStatus.OK
    assert result.provenance.world_alignment_frame_before_dropout == 79
    assert result.provenance.protocol_runtime_anchor_frame == 80
    assert by_frame[80].valid is False
    assert by_frame[80].state == PoseTrackingState.TRACKING_LOST
    assert by_frame[81].valid is True
    assert by_frame[81].state == PoseTrackingState.LOCAL_TRACKING


def test_local_invalid_frame_makes_world_invalid() -> None:
    window = _window(81, 1_000_000)
    frames = [FrameTimestamp(fn, ts) for fn, ts in [(79, 970_000), (80, 990_000), (81, 1_000_000)]]
    local = [
        _local(79, 970_000),
        _local(80, 990_000),
        _local(81, 1_000_000, valid=False),
    ]
    runtime = [_runtime(79, 970_000), _runtime(80, 990_000)]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    by_frame = _candidates_by_frame(result)
    assert by_frame[81].valid is False
    assert by_frame[81].state == PoseTrackingState.TRACKING_LOST
    assert by_frame[81].T_world_camera is None


def test_bridge_recovery_restores_world_valid_same_segment() -> None:
    window = _window(81, 1_000_000, duration=0.5)
    frames = [
        FrameTimestamp(fn, ts)
        for fn, ts in [
            (79, 970_000),
            (80, 990_000),
            (81, 1_000_000),
            (82, 1_030_000),
            (83, 1_060_000),
            (84, 1_090_000),
        ]
    ]
    local = [
        _local(79, 970_000),
        _local(80, 990_000),
        _local(81, 1_000_000),
        _local(82, 1_030_000, valid=False),
        _local(83, 1_060_000, valid=False),
        _local(84, 1_090_000, t=(0.3, 0.0, 0.0), bridge_recovered=True),
    ]
    runtime = [_runtime(79, 970_000, t=(1.0, 0.0, 0.0)), _runtime(80, 990_000, t=(1.0, 0.0, 0.0))]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    by_frame = _candidates_by_frame(result)
    assert by_frame[82].valid is False
    assert by_frame[83].valid is False
    assert by_frame[84].valid is True
    assert by_frame[84].state == PoseTrackingState.LOCAL_TRACKING


def test_segment_reset_during_dropout_invalidates_world_even_if_local_valid() -> None:
    window = _window(81, 1_000_000)
    frames = [FrameTimestamp(fn, ts) for fn, ts in [(79, 970_000), (80, 990_000), (81, 1_000_000), (82, 1_030_000)]]
    local = [
        _local(79, 970_000, segment_id=1),
        _local(80, 990_000, segment_id=1),
        _local(81, 1_000_000, segment_id=1),
        _local(82, 1_030_000, segment_id=2, t=(0.0, 0.0, 0.0)),
    ]
    runtime = [_runtime(79, 970_000, t=(1.0, 0.0, 0.0)), _runtime(80, 990_000, t=(1.0, 0.0, 0.0))]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    by_frame = _candidates_by_frame(result)
    assert result.replay.segment_reset_during_dropout is True
    assert by_frame[82].valid is False


def test_boundary_joint_reanchor_enables_world_after_segment_reset() -> None:
    start_ts = 1_000_000
    window = _window(81, start_ts, duration=0.5)
    recovery_ts = start_ts + 510_000
    frames = [
        FrameTimestamp(79, start_ts - 90_000),
        FrameTimestamp(80, start_ts - 30_000),
        FrameTimestamp(81, start_ts),
        FrameTimestamp(82, start_ts + 30_000),
        FrameTimestamp(83, recovery_ts),
        FrameTimestamp(84, recovery_ts + 30_000),
    ]
    local = [
        _local(79, start_ts - 90_000, segment_id=1),
        _local(80, start_ts - 30_000, segment_id=1),
        _local(81, start_ts, segment_id=1),
        _local(82, start_ts + 30_000, segment_id=2, t=(0.0, 0.0, 0.0)),
        _local(83, recovery_ts, segment_id=2, t=(0.1, 0.0, 0.0)),
        _local(84, recovery_ts + 30_000, segment_id=2, t=(0.2, 0.0, 0.0)),
    ]
    runtime = [
        _runtime(79, start_ts - 90_000, t=(1.0, 0.0, 0.0)),
        _runtime(80, start_ts - 30_000, t=(1.0, 0.0, 0.0)),
        _runtime(83, recovery_ts, t=(2.0, 0.0, 0.0)),
        _runtime(84, recovery_ts + 30_000, t=(2.1, 0.0, 0.0)),
    ]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    by_frame = _candidates_by_frame(result)
    assert by_frame[82].valid is False
    assert by_frame[83].valid is True
    assert by_frame[83].state == PoseTrackingState.RELOCALIZED
    assert by_frame[84].state == PoseTrackingState.TAG_ANCHORED


def test_runtime_tag_not_used_during_dropout_provenance() -> None:
    window = _window(81, 1_000_000)
    frames = [FrameTimestamp(fn, ts) for fn, ts in [(79, 970_000), (80, 990_000), (81, 1_000_000)]]
    local = [_local(79, 970_000), _local(80, 990_000), _local(81, 1_000_000)]
    runtime = [_runtime(79, 970_000), _runtime(80, 990_000), _runtime(81, 1_000_000, t=(9.0, 0.0, 0.0))]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert result.provenance.runtime_tag_used_during_dropout is False
    assert result.provenance.reference_used_by_candidate is False


def test_adapter_modules_have_no_reference_dependency() -> None:
    for rel in (
        "dropout_evaluation/rgbd_odometry_adapter.py",
        "dropout_evaluation/rgbd_odometry_adapter_runner.py",
    ):
        source = (ROOT / "src" / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "evaluation_io" not in modules
        lowered = " ".join(modules).lower()
        assert "reference" not in lowered or "posthoc" in rel


def test_trajectory_sample_to_T_odom_roundtrip() -> None:
    sample = _local(5, 500_000, t=(1.0, 2.0, 3.0))
    T = trajectory_sample_to_T_odom(sample)
    assert T is not None
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_algorithm_id_mapping() -> None:
    window = _window(81, 1_000_000)
    frames = [FrameTimestamp(79, 970_000), FrameTimestamp(80, 990_000), FrameTimestamp(81, 1_000_000)]
    local = [_local(79, 970_000), _local(80, 990_000), _local(81, 1_000_000)]
    runtime = [_runtime(79, 970_000), _runtime(80, 990_000)]
    result = generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert all(candidate.algorithm_id == RGBD_ODOMETRY_ALGORITHM_ID for candidate in result.candidates)


def test_masked_interval_half_open() -> None:
    window = _window(81, 1_000_000)
    assert is_runtime_tag_masked(window.start_device_timestamp_us, window) is True
    assert is_runtime_tag_masked(window.boundary_timestamp_us, window) is False
