"""Phase 4.5-A session-level continuous stereo+IMU VIO-lite runner."""

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

from dataset_recorder.reader import DatasetReader

from .canonical_frames import load_canonical_frames_from_rgb_index
from .stereo_pairing import pair_stereo_records, summarize_pairing

from .rgbd_odometry import transform_magnitude
from .stereo_imu_calibration import (
    StereoImuCalibration,
    calibration_fingerprint,
    load_stereo_imu_calibration,
)
from .stereo_imu_vio_lite import (
    STEREO_IMU_VIO_LITE_ALGORITHM_ID,
    ImuSampleRecord,
    StereoImuTrajectorySample,
    StereoImuVioConfig,
    StereoImuVioFrameInput,
    run_stereo_imu_vio_lite,
)


@dataclass(frozen=True)
class StereoImuVioRuntimeStats:
    total_processing_time_s: float
    real_time_factor: float | None
    frames_per_second: float | None


@dataclass(frozen=True)
class StereoImuVioResult:
    samples: list[StereoImuTrajectorySample]
    runtime_stats: StereoImuVioRuntimeStats
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    pairing_summary: dict[str, Any]


def load_imu_samples(reader: DatasetReader) -> list[ImuSampleRecord]:
    accel_rows = list(reader.iterate_accel())
    gyro_rows = list(reader.iterate_gyro())
    if len(accel_rows) != len(gyro_rows):
        raise ValueError("accel/gyro sample count mismatch")
    samples: list[ImuSampleRecord] = []
    for accel, gyro in zip(accel_rows, gyro_rows):
        a_ts = int(accel.row.get("device_timestamp_us") or 0)
        g_ts = int(gyro.row.get("device_timestamp_us") or 0)
        if a_ts != g_ts:
            raise ValueError(f"accel/gyro timestamp mismatch at {a_ts} vs {g_ts}")
        samples.append(
            ImuSampleRecord(
                device_timestamp_us=a_ts,
                accel_m_s2=np.array(
                    [float(accel.row["x"]), float(accel.row["y"]), float(accel.row["z"])],
                    dtype=np.float64,
                ),
                gyro_rad_s=np.array(
                    [float(gyro.row["x"]), float(gyro.row["y"]), float(gyro.row["z"])],
                    dtype=np.float64,
                ),
            )
        )
    return samples


def load_stereo_frames(reader: DatasetReader) -> tuple[list[StereoImuVioFrameInput], dict[str, Any]]:
    left_records = [dict(rec.row) for rec in reader.iterate_left_ir()]
    right_records = [dict(rec.row) for rec in reader.iterate_right_ir()]
    left_paths = {
        int(rec.row.get("device_timestamp_us") or 0): rec.file_path
        for rec in reader.iterate_left_ir()
    }
    right_paths = {
        int(rec.row.get("device_timestamp_us") or 0): rec.file_path
        for rec in reader.iterate_right_ir()
    }
    canonical_frames = load_canonical_frames_from_rgb_index(reader.session_dir)
    canonical_numbers = [frame.frame_number for frame in canonical_frames]
    if len(left_records) != len(canonical_numbers):
        raise ValueError(
            "LEFT_IR row count must match canonical RGB frame count: "
            f"{len(left_records)} vs {len(canonical_numbers)}"
        )

    pairs, unpaired_left = pair_stereo_records(left_records, right_records, canonical_numbers)
    pairing_summary = summarize_pairing(pairs, unpaired_left)

    frames: list[StereoImuVioFrameInput] = []
    for pair in pairs:
        left_path = left_paths.get(pair.left_timestamp_us)
        right_path = right_paths.get(pair.right_timestamp_us)
        if left_path is None or right_path is None or not left_path.is_file() or not right_path.is_file():
            continue
        left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            continue
        frames.append(
            StereoImuVioFrameInput(
                frame_number=pair.canonical_frame_number,
                device_timestamp_us=pair.device_timestamp_us,
                left_gray=left_img,
                right_gray=right_img,
                native_left_frame_number=pair.native_left_frame_number,
                native_right_frame_number=pair.native_right_frame_number,
            )
        )

    pairing_summary["loaded_frames"] = len(frames)
    pairing_summary["canonical_frame_range"] = (
        [frames[0].frame_number, frames[-1].frame_number] if frames else None
    )
    return frames, pairing_summary


def _longest_invalid_run(samples: Sequence[StereoImuTrajectorySample]) -> int:
    longest = 0
    current = 0
    for sample in samples:
        if sample.valid:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _catastrophic_jumps(samples: Sequence[StereoImuTrajectorySample]) -> list[dict[str, Any]]:
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


def summarize_trajectory(
    samples: Sequence[StereoImuTrajectorySample],
    *,
    session_duration_s: float | None,
    processing_time_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total = len(samples)
    valid = sum(1 for s in samples if s.valid)
    propagated_only = sum(1 for s in samples if s.propagated_only)
    invalid = total - valid
    visual_success = sum(1 for s in samples if s.visual_update_success)
    imu_used = sum(s.imu_samples_used for s in samples)
    inliers = [s.visual_inliers for s in samples if s.visual_update_success]
    stereo_pts = [s.stereo_points for s in samples if s.stereo_points > 0]
    init_frame = next((s.frame_number for s in samples if s.state == "init"), None)

    summary = {
        "total_frames": total,
        "valid_frames": valid,
        "invalid_frames": invalid,
        "propagated_only_frames": propagated_only,
        "availability": float(valid / total) if total else 0.0,
        "initialization_frame": init_frame,
        "longest_invalid_gap": _longest_invalid_run(samples),
        "visual_update_success_ratio": float(visual_success / max(total - 1, 1)),
        "visual_inliers_mean": float(np.mean(inliers)) if inliers else 0.0,
        "visual_inliers_median": float(np.median(inliers)) if inliers else 0.0,
        "stereo_points_mean": float(np.mean(stereo_pts)) if stereo_pts else 0.0,
        "stereo_points_median": float(np.median(stereo_pts)) if stereo_pts else 0.0,
        "imu_samples_consumed": imu_used,
        "segment_reset_count": 0,
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
    }
    for sample in samples:
        diagnostics["state_histogram"][sample.state] = diagnostics["state_histogram"].get(sample.state, 0) + 1

    rtf = None
    if session_duration_s and session_duration_s > 0:
        rtf = processing_time_s / session_duration_s
    summary["processing_time_s"] = processing_time_s
    summary["session_duration_s"] = session_duration_s
    summary["real_time_factor"] = rtf
    return summary, diagnostics


def run_continuous_stereo_imu_vio(
    frames: Sequence[StereoImuVioFrameInput],
    imu_samples: Sequence[ImuSampleRecord],
    calib: StereoImuCalibration,
    config: StereoImuVioConfig | None = None,
    *,
    pairing_summary: dict[str, Any] | None = None,
) -> StereoImuVioResult:
    if not frames:
        raise ValueError("frames must not be empty")
    t0 = time.perf_counter()
    samples = run_stereo_imu_vio_lite(frames, imu_samples, calib, config=config)
    processing_time_s = time.perf_counter() - t0
    session_duration_s = None
    if len(frames) >= 2:
        session_duration_s = (frames[-1].device_timestamp_us - frames[0].device_timestamp_us) / 1_000_000.0
    summary, diagnostics = summarize_trajectory(
        samples,
        session_duration_s=session_duration_s,
        processing_time_s=processing_time_s,
    )
    fps = float(len(frames) / processing_time_s) if processing_time_s > 0 else None
    runtime_stats = StereoImuVioRuntimeStats(
        total_processing_time_s=processing_time_s,
        real_time_factor=summary.get("real_time_factor"),
        frames_per_second=fps,
    )
    return StereoImuVioResult(
        samples=samples,
        runtime_stats=runtime_stats,
        summary=summary,
        diagnostics=diagnostics,
        pairing_summary=dict(pairing_summary or {}),
    )


def build_provenance(
    *,
    config: StereoImuVioConfig,
    calib: StereoImuCalibration,
    session_dir: Path,
    frame_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "algorithm_id": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
        "algorithm_label": "stereo_imu_provisional_vio",
        "opencv_version": cv2.__version__,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "session_dir": str(session_dir),
        "frame_range": list(frame_range) if frame_range else None,
        "calibration": calibration_fingerprint(calib),
        "config": asdict(config),
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


def write_vio_outputs(
    output_dir: Path,
    result: StereoImuVioResult,
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
        "visual_inliers",
        "stereo_points",
        "imu_samples_used",
        "visual_update_success",
        "imu_propagated",
        "propagated_only",
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
