"""Offline centered SE(3) smoothing for AprilTag evaluation reference trajectories."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .session_metadata import write_json

REFERENCE_SCHEMA_VERSION = 1
DEFAULT_METHOD = "centered_se3"
DEFAULT_WINDOW_FRAMES = 9
DEFAULT_GAP_INTERP_MAX_FRAMES = 3
DEFAULT_MIN_VALID_SAMPLES = 1
ROTATION_DETERMINANT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class AprilTagReferenceConfig:
    enabled: bool = True
    method: str = DEFAULT_METHOD
    window_frames: int = DEFAULT_WINDOW_FRAMES
    gap_interp_max_frames: int = DEFAULT_GAP_INTERP_MAX_FRAMES
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES


@dataclass
class AprilTagPoseFrame:
    frame_number: int
    device_timestamp_us: int | None
    source_valid: bool
    T_world_camera: np.ndarray | None
    reprojection_error_px: float | None = None


@dataclass
class AprilTagReferenceRow:
    frame_number: int
    device_timestamp_us: int | None
    source_valid: bool
    interpolated: bool
    reference_valid: bool
    reference_quality: str
    raw_tx: float | None
    raw_ty: float | None
    raw_tz: float | None
    raw_qw: float | None
    raw_qx: float | None
    raw_qy: float | None
    raw_qz: float | None
    ref_tx: float | None
    ref_ty: float | None
    ref_tz: float | None
    ref_qw: float | None
    ref_qx: float | None
    ref_qy: float | None
    ref_qz: float | None
    smoothing_method: str
    window_frames: int
    gap_interp_max_frames: int
    window_valid_sample_count: int
    reprojection_error_px: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "device_timestamp_us": self.device_timestamp_us,
            "source_valid": self.source_valid,
            "interpolated": self.interpolated,
            "reference_valid": self.reference_valid,
            "reference_quality": self.reference_quality,
            "raw_tx": self.raw_tx,
            "raw_ty": self.raw_ty,
            "raw_tz": self.raw_tz,
            "raw_qw": self.raw_qw,
            "raw_qx": self.raw_qx,
            "raw_qy": self.raw_qy,
            "raw_qz": self.raw_qz,
            "ref_tx": self.ref_tx,
            "ref_ty": self.ref_ty,
            "ref_tz": self.ref_tz,
            "ref_qw": self.ref_qw,
            "ref_qx": self.ref_qx,
            "ref_qy": self.ref_qy,
            "ref_qz": self.ref_qz,
            "smoothing_method": self.smoothing_method,
            "window_frames": self.window_frames,
            "gap_interp_max_frames": self.gap_interp_max_frames,
            "window_valid_sample_count": self.window_valid_sample_count,
            "reprojection_error_px": self.reprojection_error_px,
        }


def reference_config_from_mapping(mapping: Mapping[str, Any] | None) -> AprilTagReferenceConfig:
    cfg = dict(mapping or {})
    window = int(cfg.get("window_frames", DEFAULT_WINDOW_FRAMES))
    if window % 2 == 0:
        raise ValueError("apriltag_reference.window_frames must be odd")
    return AprilTagReferenceConfig(
        enabled=bool(cfg.get("enabled", True)),
        method=str(cfg.get("method", DEFAULT_METHOD)),
        window_frames=window,
        gap_interp_max_frames=int(cfg.get("gap_interp_max_frames", DEFAULT_GAP_INTERP_MAX_FRAMES)),
        min_valid_samples=int(cfg.get("min_valid_samples", DEFAULT_MIN_VALID_SAMPLES)),
    )


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(t) * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    s0 = math.sin((1.0 - t) * theta) / math.sin(theta)
    s1 = math.sin(t * theta) / math.sin(theta)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def _pose_parts(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), T[:3, :3].copy()


def _interpolate_short_gaps(
    frames: list[AprilTagPoseFrame],
    max_gap_frames: int,
) -> list[dict[str, Any]]:
    """Return working sequence with optional short-gap interpolation."""
    seq: list[dict[str, Any]] = []
    for frame in frames:
        if frame.source_valid and frame.T_world_camera is not None:
            t, R = _pose_parts(frame.T_world_camera)
            seq.append(
                {
                    "frame_number": frame.frame_number,
                    "device_timestamp_us": frame.device_timestamp_us,
                    "source_valid": True,
                    "interpolated": False,
                    "valid": True,
                    "t": t,
                    "R": R,
                    "T": frame.T_world_camera.copy(),
                    "reprojection_error_px": frame.reprojection_error_px,
                }
            )
        else:
            seq.append(
                {
                    "frame_number": frame.frame_number,
                    "device_timestamp_us": frame.device_timestamp_us,
                    "source_valid": False,
                    "interpolated": False,
                    "valid": False,
                    "t": None,
                    "R": None,
                    "T": None,
                    "reprojection_error_px": None,
                }
            )

    i = 0
    while i < len(seq):
        if seq[i]["valid"]:
            i += 1
            continue
        j = i
        while j < len(seq) and not seq[j]["valid"]:
            j += 1
        gap = j - i
        left = i - 1
        right = j
        if (
            left >= 0
            and right < len(seq)
            and seq[left]["valid"]
            and seq[right]["valid"]
            and gap <= max_gap_frames
        ):
            for k in range(i, j):
                alpha = (k - left) / (right - left)
                t = (1.0 - alpha) * seq[left]["t"] + alpha * seq[right]["t"]
                q = _slerp(
                    _rotation_to_quaternion(seq[left]["R"]),
                    _rotation_to_quaternion(seq[right]["R"]),
                    alpha,
                )
                R = _quaternion_to_rotation(q)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = t
                seq[k] = {
                    **seq[k],
                    "valid": True,
                    "interpolated": True,
                    "t": t,
                    "R": R,
                    "T": T,
                    "reprojection_error_px": None,
                }
        i = j
    return seq


def _smooth_centered_se3(
    seq: list[dict[str, Any]],
    *,
    window_frames: int,
    min_valid_samples: int,
) -> list[dict[str, Any]]:
    if window_frames % 2 == 0:
        raise ValueError("window_frames must be odd")
    half = window_frames // 2
    smoothed: list[dict[str, Any]] = []
    for i, base in enumerate(seq):
        if not base["valid"]:
            smoothed.append(
                {
                    **base,
                    "ref_valid": False,
                    "ref_t": None,
                    "ref_R": None,
                    "ref_T": None,
                    "window_valid_sample_count": 0,
                }
            )
            continue

        lo = max(0, i - half)
        hi = min(len(seq), i + half + 1)
        translations: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        weights: list[float] = []
        for j in range(lo, hi):
            if not seq[j]["valid"]:
                continue
            dist = abs(j - i)
            w = (half + 1 - dist) / (half + 1)
            translations.append(seq[j]["t"])
            rotations.append(seq[j]["R"])
            weights.append(w)

        sample_count = len(translations)
        if sample_count < min_valid_samples:
            smoothed.append(
                {
                    **base,
                    "ref_valid": False,
                    "ref_t": None,
                    "ref_R": None,
                    "ref_T": None,
                    "window_valid_sample_count": sample_count,
                }
            )
            continue

        ws = np.asarray(weights, dtype=np.float64)
        ws /= ws.sum()
        ref_t = np.sum(ws[:, None] * np.vstack(translations), axis=0)

        q_ref = _rotation_to_quaternion(base["R"])
        quats: list[np.ndarray] = []
        for R in rotations:
            q = _rotation_to_quaternion(R)
            if float(np.dot(q, q_ref)) < 0.0:
                q = -q
            quats.append(q)
        q_avg = np.sum(ws[:, None] * np.vstack(quats), axis=0)
        q_avg /= np.linalg.norm(q_avg)
        ref_R = _quaternion_to_rotation(q_avg)
        if abs(float(np.linalg.det(ref_R)) - 1.0) > ROTATION_DETERMINANT_TOLERANCE:
            raise ValueError("smoothed rotation is not a valid SO(3) element")

        ref_T = np.eye(4, dtype=np.float64)
        ref_T[:3, :3] = ref_R
        ref_T[:3, 3] = ref_t
        smoothed.append(
            {
                **base,
                "ref_valid": True,
                "ref_t": ref_t,
                "ref_R": ref_R,
                "ref_T": ref_T,
                "window_valid_sample_count": sample_count,
            }
        )
    return smoothed


def _reference_quality(source_valid: bool, interpolated: bool, reference_valid: bool) -> str:
    if not reference_valid:
        return "INSUFFICIENT_SUPPORT"
    if interpolated:
        return "INTERPOLATED"
    if source_valid:
        return "VALID"
    return "INSUFFICIENT_SUPPORT"


def build_apriltag_reference_trajectory(
    frames: list[AprilTagPoseFrame],
    config: AprilTagReferenceConfig,
) -> list[AprilTagReferenceRow]:
    if config.method != DEFAULT_METHOD:
        raise ValueError(f"unsupported apriltag reference method: {config.method}")

    interpolated_seq = _interpolate_short_gaps(frames, config.gap_interp_max_frames)
    smoothed_seq = _smooth_centered_se3(
        interpolated_seq,
        window_frames=config.window_frames,
        min_valid_samples=config.min_valid_samples,
    )

    rows: list[AprilTagReferenceRow] = []
    for item in smoothed_seq:
        raw_t = item["t"] if item["source_valid"] and item["T"] is not None else None
        raw_R = item["R"] if item["source_valid"] and item["T"] is not None else None
        raw_q = _rotation_to_quaternion(raw_R) if raw_R is not None else None
        ref_t = item.get("ref_t")
        ref_R = item.get("ref_R")
        ref_q = _rotation_to_quaternion(ref_R) if ref_R is not None else None
        rows.append(
            AprilTagReferenceRow(
                frame_number=int(item["frame_number"]),
                device_timestamp_us=item["device_timestamp_us"],
                source_valid=bool(item["source_valid"]),
                interpolated=bool(item["interpolated"]),
                reference_valid=bool(item.get("ref_valid")),
                reference_quality=_reference_quality(
                    bool(item["source_valid"]),
                    bool(item["interpolated"]),
                    bool(item.get("ref_valid")),
                ),
                raw_tx=None if raw_t is None else float(raw_t[0]),
                raw_ty=None if raw_t is None else float(raw_t[1]),
                raw_tz=None if raw_t is None else float(raw_t[2]),
                raw_qw=None if raw_q is None else float(raw_q[0]),
                raw_qx=None if raw_q is None else float(raw_q[1]),
                raw_qy=None if raw_q is None else float(raw_q[2]),
                raw_qz=None if raw_q is None else float(raw_q[3]),
                ref_tx=None if ref_t is None else float(ref_t[0]),
                ref_ty=None if ref_t is None else float(ref_t[1]),
                ref_tz=None if ref_t is None else float(ref_t[2]),
                ref_qw=None if ref_q is None else float(ref_q[0]),
                ref_qx=None if ref_q is None else float(ref_q[1]),
                ref_qy=None if ref_q is None else float(ref_q[2]),
                ref_qz=None if ref_q is None else float(ref_q[3]),
                smoothing_method=config.method,
                window_frames=config.window_frames,
                gap_interp_max_frames=config.gap_interp_max_frames,
                window_valid_sample_count=int(item.get("window_valid_sample_count", 0)),
                reprojection_error_px=item.get("reprojection_error_px"),
            )
        )
    return rows


REFERENCE_CSV_FIELDS = [
    "frame_number",
    "device_timestamp_us",
    "source_valid",
    "interpolated",
    "reference_valid",
    "reference_quality",
    "raw_tx",
    "raw_ty",
    "raw_tz",
    "raw_qw",
    "raw_qx",
    "raw_qy",
    "raw_qz",
    "ref_tx",
    "ref_ty",
    "ref_tz",
    "ref_qw",
    "ref_qx",
    "ref_qy",
    "ref_qz",
    "smoothing_method",
    "window_frames",
    "gap_interp_max_frames",
    "window_valid_sample_count",
    "reprojection_error_px",
]


def write_apriltag_reference_csv(path: Path, rows: list[AprilTagReferenceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFERENCE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def build_and_write_apriltag_reference(
    session_dir: Path,
    frames: list[AprilTagPoseFrame],
    config: AprilTagReferenceConfig,
) -> dict[str, Any]:
    reference_dir = Path(session_dir) / "derived" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    rows = build_apriltag_reference_trajectory(frames, config)
    csv_path = reference_dir / "apriltag_pose_smoothed.csv"
    write_apriltag_reference_csv(csv_path, rows)

    quality_counts: dict[str, int] = {}
    for row in rows:
        quality_counts[row.reference_quality] = quality_counts.get(row.reference_quality, 0) + 1

    summary = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "method": config.method,
        "window_frames": config.window_frames,
        "window_ms": config.window_frames / 30.0 * 1000.0,
        "gap_interp_max_frames": config.gap_interp_max_frames,
        "min_valid_samples": config.min_valid_samples,
        "row_count": len(rows),
        "source_valid_count": sum(1 for row in rows if row.source_valid),
        "interpolated_count": sum(1 for row in rows if row.interpolated),
        "reference_valid_count": sum(1 for row in rows if row.reference_valid),
        "reference_invalid_count": sum(1 for row in rows if not row.reference_valid),
        "reference_quality_counts": quality_counts,
    }
    write_json(reference_dir / "manifest.json", summary)
    return summary
