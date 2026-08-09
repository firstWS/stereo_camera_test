"""Load runtime AprilTag poses for Phase 3 candidate generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNTIME_APRILTAG_SOURCE = "runtime_apriltag"
DEFAULT_APRILTAG_OBSERVATIONS_CSV = "derived/apriltag/observations.csv"

RUNTIME_POSE_COLUMNS: tuple[str, ...] = (
    "pose_valid",
    "world_tx",
    "world_ty",
    "world_tz",
    "world_qw",
    "world_qx",
    "world_qy",
    "world_qz",
)


@dataclass(frozen=True)
class RuntimeAprilTagPose:
    frame_number: int
    device_timestamp_us: int
    T_world_camera: np.ndarray
    valid: bool
    source: str = RUNTIME_APRILTAG_SOURCE


class RuntimeAprilTagPoseUnavailableError(RuntimeError):
    """Raised when observations.csv cannot supply disambiguated runtime poses."""


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


def observations_csv_supports_runtime_pose(observations_csv: Path) -> bool:
    with observations_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        return all(column in fieldnames for column in RUNTIME_POSE_COLUMNS)


def load_runtime_apriltag_poses(*, observations_csv: Path) -> list[RuntimeAprilTagPose]:
    """Load disambiguated runtime poses from Phase 2 AprilTag observations only."""
    poses: list[RuntimeAprilTagPose] = []
    visible_without_pose_schema = 0

    with observations_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_pose_schema = all(column in fieldnames for column in RUNTIME_POSE_COLUMNS)
        for row in reader:
            if has_pose_schema:
                if not _parse_bool(row.get("pose_valid", "false")):
                    continue
            else:
                if not _parse_bool(row.get("visible", "false")):
                    continue
                visible_without_pose_schema += 1
                continue

            ts = row.get("device_timestamp_us")
            if ts in (None, ""):
                continue
            if row.get("world_tx") in (None, ""):
                continue

            frame_number = int(row["frame_number"])
            T = _quaternion_to_transform(
                float(row["world_qw"]),
                float(row["world_qx"]),
                float(row["world_qy"]),
                float(row["world_qz"]),
            )
            T[0, 3] = float(row["world_tx"])
            T[1, 3] = float(row["world_ty"])
            T[2, 3] = float(row["world_tz"])
            poses.append(
                RuntimeAprilTagPose(
                    frame_number=frame_number,
                    device_timestamp_us=int(ts),
                    T_world_camera=T,
                    valid=True,
                    source=RUNTIME_APRILTAG_SOURCE,
                )
            )

    if visible_without_pose_schema > 0:
        raise RuntimeAprilTagPoseUnavailableError(
            f"{observations_csv} has {visible_without_pose_schema} visible AprilTag rows but no "
            "runtime pose columns (pose_valid/world_*). Re-derive with runtime pose "
            "persistence enabled."
        )

    poses.sort(key=lambda item: item.device_timestamp_us)
    return poses


def load_runtime_apriltag_poses_from_session(
    session_dir: Path,
    *,
    observations_csv: str = DEFAULT_APRILTAG_OBSERVATIONS_CSV,
) -> list[RuntimeAprilTagPose]:
    return load_runtime_apriltag_poses(observations_csv=session_dir / observations_csv)
