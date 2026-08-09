"""Read-only loaders for Phase 3 evaluation inputs (not candidate generation)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .evaluation_metrics import CupObservation, PoseReference


def _parse_bool(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _quaternion_to_transform(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def load_pose_references_from_csv(reference_csv: Path) -> list[PoseReference]:
    references: list[PoseReference] = []
    with reference_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            valid = _parse_bool(row.get("reference_valid", "false"))
            ts = row.get("device_timestamp_us")
            frame_number = int(row["frame_number"])
            T = None
            if valid and row.get("ref_tx") not in (None, ""):
                T = _quaternion_to_transform(
                    float(row["ref_qw"]),
                    float(row["ref_qx"]),
                    float(row["ref_qy"]),
                    float(row["ref_qz"]),
                )
                T[0, 3] = float(row["ref_tx"])
                T[1, 3] = float(row["ref_ty"])
                T[2, 3] = float(row["ref_tz"])
            references.append(
                PoseReference(
                    frame_number=frame_number,
                    device_timestamp_us=int(ts) if ts not in (None, "") else 0,
                    T_world_camera=T,
                    valid=valid and T is not None,
                    interpolated=_parse_bool(row.get("interpolated", "false")),
                    quality=str(row.get("reference_quality", "")),
                )
            )
    return references


def load_cup_observations_from_csv(
    observations_csv: Path,
    *,
    semantic_id: str | None = None,
) -> list[CupObservation]:
    observations: list[CupObservation] = []
    with observations_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if semantic_id is not None and row.get("semantic_id") != semantic_id:
                continue
            valid = _parse_bool(row.get("depth_valid", "false"))
            P_camera = None
            if valid and row.get("camera_x") not in (None, ""):
                P_camera = np.array(
                    [float(row["camera_x"]), float(row["camera_y"]), float(row["camera_z"])],
                    dtype=np.float64,
                )
            observations.append(
                CupObservation(
                    frame_number=int(row["frame_number"]),
                    device_timestamp_us=int(row["device_timestamp_us"]),
                    semantic_id=str(row.get("semantic_id", "")),
                    P_camera=P_camera if P_camera is not None else np.zeros(3, dtype=np.float64),
                    valid=valid and P_camera is not None,
                )
            )
    return observations
