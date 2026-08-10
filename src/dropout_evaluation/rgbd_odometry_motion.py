"""Candidate-only motion plausibility gate for Open3D RGB-D odometry pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rgbd_odometry import PairwiseOdometryResult, transform_magnitude

REJECT_OPEN3D_FAILURE = "OPEN3D_FAILURE"
REJECT_NONFINITE_TRANSFORM = "NONFINITE_TRANSFORM"
REJECT_INVALID_RIGID_TRANSFORM = "INVALID_RIGID_TRANSFORM"
REJECT_INVALID_DT = "INVALID_DT"
REJECT_LINEAR_SPEED_EXCEEDED = "LINEAR_SPEED_EXCEEDED"
REJECT_ANGULAR_SPEED_EXCEEDED = "ANGULAR_SPEED_EXCEEDED"
REJECT_LINEAR_AND_ANGULAR_EXCEEDED = "LINEAR_AND_ANGULAR_EXCEEDED"

DEFAULT_MAX_LINEAR_SPEED_M_S = 1.0
DEFAULT_MAX_ANGULAR_SPEED_DEG_S = 60.0
ROTATION_ORTHOGONALITY_TOLERANCE = 1e-3
ROTATION_DETERMINANT_TOLERANCE = 0.05
HOMOGENEOUS_ROW_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MotionPlausibilityConfig:
    max_linear_speed_m_s: float = DEFAULT_MAX_LINEAR_SPEED_M_S
    max_angular_speed_deg_s: float = DEFAULT_MAX_ANGULAR_SPEED_DEG_S


@dataclass(frozen=True)
class PairMotionEvaluation:
    open3d_success: bool
    accepted: bool
    reject_reason: str | None
    transform_target_source: np.ndarray | None
    translation_m: float | None
    rotation_deg: float | None
    linear_speed_m_s: float | None
    angular_speed_deg_s: float | None
    input_prepare_ms: float | None
    odometry_ms: float | None
    total_ms: float | None
    information_matrix: np.ndarray | None
    information_diagnostics: dict[str, Any]


def validate_rigid_transform(transform_4x4: np.ndarray) -> tuple[bool, str | None]:
    T = np.asarray(transform_4x4, dtype=np.float64)
    if T.shape != (4, 4):
        return False, REJECT_INVALID_RIGID_TRANSFORM
    if not np.isfinite(T).all():
        return False, REJECT_NONFINITE_TRANSFORM
    bottom = T[3, :]
    if not np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=HOMOGENEOUS_ROW_TOLERANCE):
        return False, REJECT_INVALID_RIGID_TRANSFORM
    R = T[:3, :3]
    t = T[:3, 3]
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return False, REJECT_NONFINITE_TRANSFORM
    if not np.allclose(R @ R.T, np.eye(3), atol=ROTATION_ORTHOGONALITY_TOLERANCE):
        return False, REJECT_INVALID_RIGID_TRANSFORM
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=ROTATION_DETERMINANT_TOLERANCE):
        return False, REJECT_INVALID_RIGID_TRANSFORM
    return True, None


def motion_speeds(
    translation_m: float,
    rotation_deg: float,
    dt_sec: float,
) -> tuple[float, float]:
    if dt_sec <= 0.0:
        raise ValueError("dt_sec must be positive")
    return float(translation_m / dt_sec), float(rotation_deg / dt_sec)


def evaluate_motion_plausibility(
    translation_m: float,
    rotation_deg: float,
    dt_sec: float,
    *,
    config: MotionPlausibilityConfig | None = None,
) -> tuple[bool, str | None, float, float]:
    cfg = config or MotionPlausibilityConfig()
    if dt_sec <= 0.0:
        return False, REJECT_INVALID_DT, 0.0, 0.0
    linear_speed, angular_speed = motion_speeds(translation_m, rotation_deg, dt_sec)
    linear_exceeded = linear_speed > cfg.max_linear_speed_m_s
    angular_exceeded = angular_speed > cfg.max_angular_speed_deg_s
    if linear_exceeded and angular_exceeded:
        return False, REJECT_LINEAR_AND_ANGULAR_EXCEEDED, linear_speed, angular_speed
    if linear_exceeded:
        return False, REJECT_LINEAR_SPEED_EXCEEDED, linear_speed, angular_speed
    if angular_exceeded:
        return False, REJECT_ANGULAR_SPEED_EXCEEDED, linear_speed, angular_speed
    return True, None, linear_speed, angular_speed


def evaluate_pair_motion(
    pair_result: PairwiseOdometryResult,
    *,
    dt_sec: float,
    motion_config: MotionPlausibilityConfig | None = None,
    information_diagnostics: dict[str, Any] | None = None,
) -> PairMotionEvaluation:
    info_diag = information_diagnostics or {}
    base = {
        "input_prepare_ms": pair_result.input_prepare_ms,
        "odometry_ms": pair_result.odometry_ms,
        "total_ms": pair_result.total_ms,
        "information_matrix": pair_result.information_matrix,
        "information_diagnostics": info_diag,
    }
    if not pair_result.success:
        return PairMotionEvaluation(
            open3d_success=False,
            accepted=False,
            reject_reason=REJECT_OPEN3D_FAILURE,
            transform_target_source=None,
            translation_m=None,
            rotation_deg=None,
            linear_speed_m_s=None,
            angular_speed_deg_s=None,
            **base,
        )
    transform = pair_result.transform_target_source
    rigid_ok, rigid_reason = validate_rigid_transform(transform)  # type: ignore[arg-type]
    if not rigid_ok:
        return PairMotionEvaluation(
            open3d_success=True,
            accepted=False,
            reject_reason=rigid_reason,
            transform_target_source=None,
            translation_m=pair_result.translation_magnitude_m,
            rotation_deg=pair_result.rotation_magnitude_deg,
            linear_speed_m_s=None,
            angular_speed_deg_s=None,
            **base,
        )
    translation_m = float(pair_result.translation_magnitude_m or 0.0)
    rotation_deg = float(pair_result.rotation_magnitude_deg or 0.0)
    accepted, reject_reason, linear_speed, angular_speed = evaluate_motion_plausibility(
        translation_m,
        rotation_deg,
        dt_sec,
        config=motion_config,
    )
    return PairMotionEvaluation(
        open3d_success=True,
        accepted=accepted,
        reject_reason=reject_reason,
        transform_target_source=transform if accepted else None,
        translation_m=translation_m,
        rotation_deg=rotation_deg,
        linear_speed_m_s=linear_speed,
        angular_speed_deg_s=angular_speed,
        **base,
    )
