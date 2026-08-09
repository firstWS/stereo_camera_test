"""Runtime AprilTag pose serialization for derived observations.csv."""

from __future__ import annotations

from typing import Any

import numpy as np

from .apriltag_reference import _quaternion_to_rotation, _rotation_to_quaternion

RUNTIME_POSE_CSV_FIELDS: tuple[str, ...] = (
    "pose_valid",
    "world_tx",
    "world_ty",
    "world_tz",
    "world_qw",
    "world_qx",
    "world_qy",
    "world_qz",
)

RUNTIME_POSE_TRANSFORM_CONVENTION = "T_world_camera"
RUNTIME_QUATERNION_CONVENTION = "wxyz"


def invalid_runtime_pose_columns() -> dict[str, str]:
    return {
        "pose_valid": "False",
        "world_tx": "",
        "world_ty": "",
        "world_tz": "",
        "world_qw": "",
        "world_qx": "",
        "world_qy": "",
        "world_qz": "",
    }


def runtime_pose_columns_from_transform(T_world_camera: np.ndarray) -> dict[str, str]:
    """Serialize unsmoothed T_world_camera to observations.csv columns (meters, wxyz)."""
    T = np.asarray(T_world_camera, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_world_camera must be a 4x4 transform")
    q = _rotation_to_quaternion(T[:3, :3])
    return {
        "pose_valid": "True",
        "world_tx": str(float(T[0, 3])),
        "world_ty": str(float(T[1, 3])),
        "world_tz": str(float(T[2, 3])),
        "world_qw": str(float(q[0])),
        "world_qx": str(float(q[1])),
        "world_qy": str(float(q[2])),
        "world_qz": str(float(q[3])),
    }


def apriltag_observation_pose_columns(
    *,
    visible: bool,
    T_world_camera: np.ndarray | None,
) -> dict[str, Any]:
    if visible and T_world_camera is not None:
        return runtime_pose_columns_from_transform(T_world_camera)
    columns = invalid_runtime_pose_columns()
    columns["pose_valid"] = "False"
    return columns


def transform_from_runtime_pose_columns(row: dict[str, str]) -> np.ndarray:
    T = _quaternion_to_rotation(
        np.array(
            [
                float(row["world_qw"]),
                float(row["world_qx"]),
                float(row["world_qy"]),
                float(row["world_qz"]),
            ],
            dtype=np.float64,
        )
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = T
    transform[0, 3] = float(row["world_tx"])
    transform[1, 3] = float(row["world_ty"])
    transform[2, 3] = float(row["world_tz"])
    return transform
