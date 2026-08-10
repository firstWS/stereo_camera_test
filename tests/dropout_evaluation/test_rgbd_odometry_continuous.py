from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.rgbd_odometry import PairwiseOdometryResult, transform_magnitude  # noqa: E402
from dropout_evaluation.rgbd_odometry_continuous import (  # noqa: E402
    ContinuousOdometryConfig,
    OdometryFrameInput,
    TRACKING_STATE_LOCAL,
    TRACKING_STATE_LOST,
    run_continuous_rgbd_odometry,
)


def _frame(frame_number: int, ts_us: int | None = None) -> OdometryFrameInput:
    ts = ts_us if ts_us is not None else frame_number * 33_333
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    return OdometryFrameInput(frame_number=frame_number, device_timestamp_us=ts, rgb=rgb, depth_m=depth)


def _transform(translation: tuple[float, float, float], yaw_deg: float = 0.0) -> np.ndarray:
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


def _mock_estimator_factory(motions: dict[tuple[int, int], np.ndarray | None]):
    def _estimate(
        source_rgb,
        source_depth_m,
        target_rgb,
        target_depth_m,
        rgb_intrinsics,
        *,
        config=None,
        source_frame=None,
        target_frame=None,
    ) -> PairwiseOdometryResult:
        key = (int(source_frame), int(target_frame))
        if key not in motions or motions[key] is None:
            return PairwiseOdometryResult(
                success=False,
                source_frame=source_frame,
                target_frame=target_frame,
                transform_target_source=None,
                information_matrix=None,
                input_prepare_ms=1.0,
                odometry_ms=2.0,
                total_ms=3.0,
                translation_magnitude_m=None,
                rotation_magnitude_deg=None,
                failure_reason="mock_open3d_failure",
            )
        transform = motions[key]
        translation_m, rotation_deg = transform_magnitude(transform)
        return PairwiseOdometryResult(
            success=True,
            source_frame=source_frame,
            target_frame=target_frame,
            transform_target_source=transform,
            information_matrix=np.eye(6),
            input_prepare_ms=1.0,
            odometry_ms=2.0,
            total_ms=3.0,
            translation_magnitude_m=translation_m,
            rotation_magnitude_deg=rotation_deg,
            failure_reason=None,
        )

    return _estimate


def test_normal_chain_accumulation() -> None:
    motions = {
        (1, 2): _transform((0.01, 0.0, 0.0)),
        (2, 3): _transform((0.01, 0.0, 0.0)),
        (3, 4): _transform((0.01, 0.0, 0.0)),
    }
    frames = [_frame(i) for i in range(1, 5)]
    result = run_continuous_rgbd_odometry(
        frames,
        rgb_intrinsics=None,
        config=ContinuousOdometryConfig(),
        pairwise_estimator=_mock_estimator_factory(motions),
    )
    assert all(sample.valid for sample in result.samples)
    assert result.samples[-1].tx == pytest.approx(-0.03, abs=0.01)
    assert result.summary["segment_count"] == 1


def test_bridge_recovery_after_single_reject() -> None:
    motions = {
        (1, 2): _transform((0.01, 0.0, 0.0)),
        (2, 3): _transform((2.0, 0.0, 0.0)),
        (2, 4): _transform((0.02, 0.0, 0.0)),
        (4, 5): _transform((0.01, 0.0, 0.0)),
    }
    frames = [_frame(i) for i in range(1, 6)]
    result = run_continuous_rgbd_odometry(
        frames,
        rgb_intrinsics=None,
        config=ContinuousOdometryConfig(),
        pairwise_estimator=_mock_estimator_factory(motions),
    )
    by_frame = {sample.frame_number: sample for sample in result.samples}
    assert by_frame[3].valid is False
    assert by_frame[3].tracking_state == TRACKING_STATE_LOST
    assert by_frame[4].valid is True
    assert by_frame[4].bridge_recovered is True
    assert by_frame[4].segment_id == 0


def test_segment_reset_when_bridge_gap_exceeded() -> None:
    motions = {
        (1, 2): _transform((0.01, 0.0, 0.0)),
        (2, 3): _transform((2.0, 0.0, 0.0)),
        (2, 4): _transform((2.0, 0.0, 0.0)),
        (2, 5): _transform((2.0, 0.0, 0.0)),
    }
    frames = [_frame(i) for i in range(1, 7)]
    result = run_continuous_rgbd_odometry(
        frames,
        config=ContinuousOdometryConfig(max_bridge_gap_frames=3),
        rgb_intrinsics=None,
        pairwise_estimator=_mock_estimator_factory(motions),
    )
    by_frame = {sample.frame_number: sample for sample in result.samples}
    assert by_frame[6].segment_start is True
    assert by_frame[6].continuity_from_previous_segment is False
    assert by_frame[6].valid is True
    assert result.summary["segment_count"] >= 2


def test_new_segment_starts_at_identity() -> None:
    motions = {
        (1, 2): _transform((0.001, 0.0, 0.0)),
    }
    frames = [_frame(i) for i in range(1, 7)]
    result = run_continuous_rgbd_odometry(
        frames,
        config=ContinuousOdometryConfig(max_bridge_gap_frames=1),
        rgb_intrinsics=None,
        pairwise_estimator=_mock_estimator_factory(motions),
    )
    starts = [sample for sample in result.samples if sample.segment_start]
    assert len(starts) >= 2
    assert starts[-1].tx == pytest.approx(0.0, abs=1e-9)
    assert starts[-1].ty == pytest.approx(0.0, abs=1e-9)
    assert starts[-1].tz == pytest.approx(0.0, abs=1e-9)


def test_segments_are_not_arbitrarily_linked() -> None:
    motions = {
        (1, 2): _transform((0.001, 0.0, 0.0)),
    }
    frames = [_frame(i) for i in range(1, 6)]
    result = run_continuous_rgbd_odometry(
        frames,
        config=ContinuousOdometryConfig(max_bridge_gap_frames=1),
        rgb_intrinsics=None,
        pairwise_estimator=_mock_estimator_factory(motions),
    )
    segment_ids = {sample.segment_id for sample in result.samples}
    assert len(segment_ids) >= 2


def test_invalid_frame_has_no_fake_pose() -> None:
    motions = {
        (1, 2): _transform((0.01, 0.0, 0.0)),
        (2, 3): _transform((2.0, 0.0, 0.0)),
    }
    frames = [_frame(i) for i in range(1, 4)]
    result = run_continuous_rgbd_odometry(
        frames,
        rgb_intrinsics=None,
        config=ContinuousOdometryConfig(),
        pairwise_estimator=_mock_estimator_factory(motions),
    )
    invalid = [sample for sample in result.samples if not sample.valid][0]
    assert invalid.tx is None
    assert invalid.qw is None
    assert invalid.tracking_state == TRACKING_STATE_LOST


def test_continuous_modules_have_no_reference_dependency() -> None:
    for rel in (
        "dropout_evaluation/rgbd_odometry_continuous.py",
        "dropout_evaluation/rgbd_odometry_motion.py",
        "dropout_evaluation/rgbd_odometry.py",
    ):
        source = (ROOT / "src" / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "evaluation_io" not in modules
        assert "apriltag_reference" not in modules
