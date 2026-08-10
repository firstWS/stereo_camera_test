"""Factory stereo + IMU calibration loader for Phase 4.5-A provisional VIO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from dataset_recorder.rgb_depth_geometry import (
    _find_extrinsic,
    _find_intrinsic,
    _intrinsic_matrix,
    extrinsic_to_homogeneous_4x4,
)


@dataclass(frozen=True)
class StereoImuCalibration:
    k_left: np.ndarray
    d_left: np.ndarray
    k_right: np.ndarray
    d_right: np.ndarray
    t_left_right: np.ndarray
    t_left_gyro: np.ndarray
    image_size: tuple[int, int]
    r1: np.ndarray
    r2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray
    calibration_quality: str = "provisional_factory"


def _distortion_vector(entry: Mapping[str, Any]) -> np.ndarray:
    dist = entry.get("distortion") or {}
    return np.array(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("p1", 0.0)),
            float(dist.get("p2", 0.0)),
            float(dist.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )


def load_stereo_imu_calibration(
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    *,
    calibration_quality: str = "provisional_factory",
) -> StereoImuCalibration:
    left_entry = _find_intrinsic(intrinsics, "LEFT_IR")
    right_entry = _find_intrinsic(intrinsics, "RIGHT_IR")
    if left_entry is None or right_entry is None:
        raise ValueError("LEFT_IR and RIGHT_IR intrinsics are required")
    lr_extrinsic = _find_extrinsic(extrinsics, "LEFT_IR", "RIGHT_IR")
    lg_extrinsic = _find_extrinsic(extrinsics, "LEFT_IR", "GYRO")
    if lr_extrinsic is None or lg_extrinsic is None:
        raise ValueError("LEFT_IR->RIGHT_IR and LEFT_IR->GYRO extrinsics are required")

    left_intr = left_entry.get("intrinsic") or {}
    width = int(left_intr.get("width", 0))
    height = int(left_intr.get("height", 0))
    k_left = _intrinsic_matrix(left_entry)
    k_right = _intrinsic_matrix(right_entry)
    d_left = _distortion_vector(left_entry)
    d_right = _distortion_vector(right_entry)
    r_lr = np.asarray(lr_extrinsic["extrinsic"]["rotation"], dtype=np.float64).reshape(3, 3)
    t_lr = np.asarray(lr_extrinsic["extrinsic"]["translation"], dtype=np.float64).reshape(3) / 1000.0
    t_left_right = extrinsic_to_homogeneous_4x4(lr_extrinsic["extrinsic"])
    t_left_gyro = extrinsic_to_homogeneous_4x4(lg_extrinsic["extrinsic"])

    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        (width, height),
        r_lr,
        t_lr,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    return StereoImuCalibration(
        k_left=k_left,
        d_left=d_left,
        k_right=k_right,
        d_right=d_right,
        t_left_right=t_left_right,
        t_left_gyro=t_left_gyro,
        image_size=(width, height),
        r1=r1,
        r2=r2,
        p1=p1,
        p2=p2,
        q=q,
        calibration_quality=calibration_quality,
    )


def calibration_fingerprint(calib: StereoImuCalibration) -> dict[str, Any]:
    return {
        "calibration_quality": calib.calibration_quality,
        "image_size": list(calib.image_size),
        "baseline_m": float(np.linalg.norm(calib.t_left_right[:3, 3])),
        "camera_imu_time_offset_known": False,
        "distortion_all_zero": bool(
            np.allclose(calib.d_left, 0.0) and np.allclose(calib.d_right, 0.0)
        ),
    }
