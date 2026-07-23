"""Persistence boundary for future AprilTag-to-Object world-pose registration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
                0.25 * s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(R)))
        if index == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
        elif index == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
    return q / np.linalg.norm(q)


def save_world_pose_registration(
    output_path: str | Path,
    *,
    object_id: str,
    T_world_object: np.ndarray,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a measured/registered T_world_object without coupling it to live inference."""
    transform = np.asarray(T_world_object, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_world_object must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_world_object must be a homogeneous transform")

    quaternion = _rotation_matrix_to_quaternion(transform[:3, :3])
    payload = {
        "object_anchor_registration": {
            "object_id": str(object_id),
            "anchor_id": str(object_id),
            "unit": "meter",
            "source": str(source),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "translation": [float(value) for value in transform[:3, 3]],
            "registered_world_pose": {
                "position_m": {
                    "x": float(transform[0, 3]),
                    "y": float(transform[1, 3]),
                    "z": float(transform[2, 3]),
                },
                "quaternion": {
                    "x": float(quaternion[0]),
                    "y": float(quaternion[1]),
                    "z": float(quaternion[2]),
                    "w": float(quaternion[3]),
                },
            },
            "rotation_matrix": [
                [float(value) for value in row] for row in transform[:3, :3]
            ],
            "T_world_object": [[float(value) for value in row] for row in transform],
            "metadata": metadata or {},
        }
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_world_pose_registration(
    input_path: str | Path,
    *,
    expected_object_id: str | None = None,
) -> np.ndarray:
    root = yaml.safe_load(Path(input_path).read_text(encoding="utf-8")) or {}
    registration = root.get("object_anchor_registration")
    if not isinstance(registration, dict):
        raise ValueError("object_anchor_registration mapping is required")
    object_id = str(registration.get("object_id") or "")
    if expected_object_id is not None and object_id != expected_object_id:
        raise ValueError(f"registration object_id mismatch: {object_id!r}")
    transform = np.asarray(registration.get("T_world_object"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("registration T_world_object must be a finite 4x4 matrix")
    return transform
