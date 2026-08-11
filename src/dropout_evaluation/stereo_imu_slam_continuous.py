"""Phase 4.7-A session-level continuous stereo+IMU SLAM-lite runner."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .rgbd_odometry import transform_magnitude
from .stereo_imu_calibration import (
    StereoImuCalibration,
    calibration_fingerprint,
    load_stereo_imu_calibration,
)
from .stereo_imu_slam_lite import (
    STEREO_IMU_SLAM_LITE_ALGORITHM_ID,
    StereoImuSlamConfig,
    StereoImuSlamTrajectorySample,
    run_stereo_imu_slam_lite,
)
from .stereo_imu_vio_continuous import load_imu_samples, load_stereo_frames
from .stereo_imu_vio_lite import ImuSampleRecord, StereoImuVioConfig, StereoImuVioFrameInput
from .stereo_imu_slam_map import SlamMapConfig


@dataclass(frozen=True)
class StereoImuSlamRuntimeStats:
    total_processing_time_s: float
    real_time_factor: float | None
    frames_per_second: float | None
    mean_frame_ms: float | None
    p90_frame_ms: float | None


@dataclass(frozen=True)
class StereoImuSlamResult:
    samples: list[StereoImuSlamTrajectorySample]
    runtime_stats: StereoImuSlamRuntimeStats
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    pairing_summary: dict[str, Any]


def _longest_invalid_run(samples: Sequence[StereoImuSlamTrajectorySample]) -> int:
    longest = 0
    current = 0
    for sample in samples:
        if sample.valid:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _catastrophic_jumps(samples: Sequence[StereoImuSlamTrajectorySample]) -> list[dict[str, Any]]:
    jumps: list[dict[str, Any]] = []
    prev_pose = None
    prev_frame = None
    for sample in samples:
        if not sample.valid or sample.tx is None:
            prev_pose = None
            continue
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = sample.tx
        T[1, 3] = sample.ty or 0.0
        T[2, 3] = sample.tz or 0.0
        if prev_pose is not None and prev_frame is not None:
            delta = np.linalg.inv(prev_pose) @ T
            trans, rot = transform_magnitude(delta)
            if trans > 0.5 or rot > 45.0:
                jumps.append(
                    {
                        "from_frame": prev_frame,
                        "to_frame": sample.frame_number,
                        "translation_m": trans,
                        "rotation_deg": rot,
                    }
                )
        prev_pose = T
        prev_frame = sample.frame_number
    return jumps


def summarize_slam_trajectory(
    samples: Sequence[StereoImuSlamTrajectorySample],
    *,
    session_duration_s: float | None,
    processing_time_s: float,
    counters: Mapping[str, int],
    final_map_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total = len(samples)
    valid = sum(1 for s in samples if s.valid)
    invalid = total - valid
    frontend_success = sum(1 for s in samples if s.frontend_visual_success)
    map_updates = sum(1 for s in samples if s.map_update_success)
    map_matches = [s.map_match_count for s in samples if s.map_match_count > 0]
    map_inliers = [s.map_inlier_count for s in samples if s.map_inlier_count > 0]

    summary = {
        "total_frames": total,
        "valid_frames": valid,
        "invalid_frames": invalid,
        "availability": float(valid / total) if total else 0.0,
        "longest_invalid_gap": _longest_invalid_run(samples),
        "frontend_visual_success_ratio": float(frontend_success / max(total - 1, 1)),
        "keyframes_created": counters.get("keyframes_created", 0),
        "final_map_points": final_map_points,
        "map_update_attempts": counters.get("map_update_attempts", 0),
        "map_update_successes": counters.get("map_update_successes", 0),
        "map_based_pose_update_count": counters.get("map_based_pose_update_count", 0),
        "relocalization_attempts": counters.get("relocalization_attempts", 0),
        "relocalization_successes": counters.get("relocalization_successes", 0),
        "map_match_count_mean": float(np.mean(map_matches)) if map_matches else 0.0,
        "map_inlier_count_mean": float(np.mean(map_inliers)) if map_inliers else 0.0,
        "map_update_success_frames": map_updates,
        "processing_time_s": processing_time_s,
        "session_duration_s": session_duration_s,
        "real_time_factor": (processing_time_s / session_duration_s) if session_duration_s and session_duration_s > 0 else None,
        "trajectory_finite": all(
            np.isfinite([s.tx, s.ty, s.tz, s.qw, s.qx, s.qy, s.qz]).all()
            for s in samples
            if s.valid and s.tx is not None
        ),
    }

    jumps = _catastrophic_jumps(samples)
    diagnostics = {
        "catastrophic_jump_count": len(jumps),
        "catastrophic_jumps": jumps[:20],
        "state_histogram": {},
        "pose_source_histogram": {},
    }
    for sample in samples:
        diagnostics["state_histogram"][sample.state] = diagnostics["state_histogram"].get(sample.state, 0) + 1
        diagnostics["pose_source_histogram"][sample.pose_source] = (
            diagnostics["pose_source_histogram"].get(sample.pose_source, 0) + 1
        )
    return summary, diagnostics


def run_continuous_stereo_imu_slam(
    frames: Sequence[StereoImuVioFrameInput],
    imu_samples: Sequence[ImuSampleRecord],
    calib: StereoImuCalibration,
    config: StereoImuSlamConfig | None = None,
    *,
    pairing_summary: dict[str, Any] | None = None,
) -> StereoImuSlamResult:
    if not frames:
        raise ValueError("frames must not be empty")
    t0 = time.perf_counter()
    samples, slam_map, counters = run_stereo_imu_slam_lite(frames, imu_samples, calib, config=config)
    processing_time_s = time.perf_counter() - t0
    session_duration_s = None
    if len(frames) >= 2:
        session_duration_s = (frames[-1].device_timestamp_us - frames[0].device_timestamp_us) / 1_000_000.0
    summary, diagnostics = summarize_slam_trajectory(
        samples,
        session_duration_s=session_duration_s,
        processing_time_s=processing_time_s,
        counters=counters,
        final_map_points=slam_map.map_point_count,
    )
    fps = float(len(frames) / processing_time_s) if processing_time_s > 0 else None
    mean_frame_ms = float((processing_time_s / len(frames)) * 1000.0) if frames else None
    runtime_stats = StereoImuSlamRuntimeStats(
        total_processing_time_s=processing_time_s,
        real_time_factor=summary.get("real_time_factor"),
        frames_per_second=fps,
        mean_frame_ms=mean_frame_ms,
        p90_frame_ms=mean_frame_ms,
    )
    return StereoImuSlamResult(
        samples=samples,
        runtime_stats=runtime_stats,
        summary=summary,
        diagnostics=diagnostics,
        pairing_summary=dict(pairing_summary or {}),
    )


def build_slam_provenance(
    *,
    config: StereoImuSlamConfig,
    calib: StereoImuCalibration,
    session_dir: Path,
    frame_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "algorithm_id": STEREO_IMU_SLAM_LITE_ALGORITHM_ID,
        "algorithm_label": "stereo_imu_slam_lite",
        "opencv_version": cv2.__version__,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "session_dir": str(session_dir),
        "frame_range": list(frame_range) if frame_range else None,
        "calibration": calibration_fingerprint(calib),
        "config": {
            "vio": asdict(config.vio),
            "map": asdict(config.map),
        },
        "candidate_uses_apriltag_pose": False,
        "candidate_uses_reference": False,
        "candidate_uses_cup": False,
        "candidate_uses_rgbd_trajectory": False,
        "candidate_uses_dropout_windows": False,
        "stereo_pairing_policy": {
            "method": "device_timestamp_us_exact_then_nearest",
            "tolerance_us": 1000,
        },
        "canonical_frame_source": "rgb_index",
        "imu_units": {
            "accel": "m/s^2",
            "gyro": "rad/s",
            "unit_source": "orbbec_sdk_v2",
        },
    }


def write_slam_outputs(
    output_dir: Path,
    result: StereoImuSlamResult,
    provenance: Mapping[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_number",
        "device_timestamp_us",
        "valid",
        "state",
        "native_left_frame_number",
        "native_right_frame_number",
        "tx",
        "ty",
        "tz",
        "qw",
        "qx",
        "qy",
        "qz",
        "frontend_visual_success",
        "imu_samples_used",
        "keyframe_count",
        "map_point_count",
        "map_match_count",
        "map_inlier_count",
        "map_update_success",
        "relocalization_attempted",
        "relocalization_success",
        "pose_source",
        "failure_reason",
    ]
    with (output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(asdict(sample))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": result.summary,
                "runtime_stats": asdict(result.runtime_stats),
                "pairing_summary": result.pairing_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_slam_config() -> StereoImuSlamConfig:
    return StereoImuSlamConfig(vio=StereoImuVioConfig(), map=SlamMapConfig())
