from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.rgbd_odometry import PairwiseOdometryResult  # noqa: E402
from dropout_evaluation.rgbd_odometry_motion import (  # noqa: E402
    REJECT_ANGULAR_SPEED_EXCEEDED,
    REJECT_INVALID_DT,
    REJECT_INVALID_RIGID_TRANSFORM,
    REJECT_LINEAR_AND_ANGULAR_EXCEEDED,
    REJECT_LINEAR_SPEED_EXCEEDED,
    REJECT_NONFINITE_TRANSFORM,
    REJECT_OPEN3D_FAILURE,
    MotionPlausibilityConfig,
    evaluate_motion_plausibility,
    evaluate_pair_motion,
    validate_rigid_transform,
)


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


def _pair_result(
    *,
    success: bool,
    transform: np.ndarray | None = None,
) -> PairwiseOdometryResult:
    translation_m = None
    rotation_deg = None
    if success and transform is not None:
        from dropout_evaluation.rgbd_odometry import transform_magnitude

        translation_m, rotation_deg = transform_magnitude(transform)
    return PairwiseOdometryResult(
        success=success,
        source_frame=1,
        target_frame=2,
        transform_target_source=transform,
        information_matrix=np.eye(6) if success else None,
        input_prepare_ms=1.0,
        odometry_ms=2.0,
        total_ms=3.0,
        translation_magnitude_m=translation_m,
        rotation_magnitude_deg=rotation_deg,
        failure_reason=None if success else "open3d_compute_rgbd_odometry_failed",
    )


def test_normal_motion_accepted() -> None:
    accepted, reason, linear, angular = evaluate_motion_plausibility(0.01, 1.0, 0.033)
    assert accepted is True
    assert reason is None
    assert linear < 1.0
    assert angular < 60.0


def test_linear_speed_reject() -> None:
    accepted, reason, _, _ = evaluate_motion_plausibility(0.05, 0.0, 0.033)
    assert accepted is False
    assert reason == REJECT_LINEAR_SPEED_EXCEEDED


def test_angular_speed_reject() -> None:
    accepted, reason, _, _ = evaluate_motion_plausibility(0.0, 6.65, 0.033)
    assert accepted is False
    assert reason == REJECT_ANGULAR_SPEED_EXCEEDED


def test_both_speeds_reject() -> None:
    accepted, reason, _, _ = evaluate_motion_plausibility(0.05, 6.65, 0.033)
    assert accepted is False
    assert reason == REJECT_LINEAR_AND_ANGULAR_EXCEEDED


def test_dt_scaling_allows_larger_displacement_over_longer_dt() -> None:
    short_ok, _, _, _ = evaluate_motion_plausibility(0.02, 0.0, 0.033)
    long_ok, _, _, _ = evaluate_motion_plausibility(0.04, 0.0, 0.067)
    assert short_ok is True
    assert long_ok is True


def test_invalid_dt_reject() -> None:
    accepted, reason, _, _ = evaluate_motion_plausibility(0.01, 1.0, 0.0)
    assert accepted is False
    assert reason == REJECT_INVALID_DT


def test_nonfinite_transform_reject() -> None:
    T = _transform((0.0, 0.0, 0.0))
    T[0, 3] = np.nan
    ok, reason = validate_rigid_transform(T)
    assert ok is False
    assert reason == REJECT_NONFINITE_TRANSFORM


def test_bad_rotation_matrix_reject() -> None:
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, 1.0, 2.0])
    ok, reason = validate_rigid_transform(T)
    assert ok is False
    assert reason == REJECT_INVALID_RIGID_TRANSFORM


def test_open3d_failure_reject() -> None:
    evaluation = evaluate_pair_motion(_pair_result(success=False), dt_sec=0.033)
    assert evaluation.open3d_success is False
    assert evaluation.accepted is False
    assert evaluation.reject_reason == REJECT_OPEN3D_FAILURE


def test_open3d_success_with_bad_motion_reject() -> None:
    evaluation = evaluate_pair_motion(
        _pair_result(success=True, transform=_transform((2.0, 0.0, 0.0))),
        dt_sec=0.033,
    )
    assert evaluation.open3d_success is True
    assert evaluation.accepted is False
    assert evaluation.reject_reason == REJECT_LINEAR_SPEED_EXCEEDED
