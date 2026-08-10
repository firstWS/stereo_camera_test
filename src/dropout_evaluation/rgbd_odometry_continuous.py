"""Phase 4.2-B session-level continuous Open3D RGB-D odometry runner."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .rgbd_odometry import (
    RGBD_ODOMETRY_ALGORITHM_ID,
    PairwiseOdometryResult,
    RgbdOdometryConfig,
    accumulate_odom_pose,
    estimate_rgbd_pair_motion,
    information_matrix_diagnostics,
)
from .rgbd_odometry_motion import (
    MotionPlausibilityConfig,
    PairMotionEvaluation,
    evaluate_pair_motion,
)

TRACKING_STATE_LOCAL = "LOCAL_TRACKING"
TRACKING_STATE_LOST = "TRACKING_LOST"
DEFAULT_MAX_BRIDGE_GAP_FRAMES = 3

PairwiseEstimator = Callable[..., PairwiseOdometryResult]


@dataclass(frozen=True)
class OdometryFrameInput:
    frame_number: int
    device_timestamp_us: int
    rgb: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class ContinuousOdometryConfig:
    odometry: RgbdOdometryConfig = field(default_factory=RgbdOdometryConfig)
    motion: MotionPlausibilityConfig = field(default_factory=MotionPlausibilityConfig)
    max_bridge_gap_frames: int = DEFAULT_MAX_BRIDGE_GAP_FRAMES


@dataclass(frozen=True)
class TrajectorySample:
    frame_number: int
    device_timestamp_us: int
    valid: bool
    tracking_state: str
    segment_id: int
    segment_start: bool
    continuity_from_previous_segment: bool
    tx: float | None
    ty: float | None
    tz: float | None
    qw: float | None
    qx: float | None
    qy: float | None
    qz: float | None
    source_frame: int | None
    pair_gap_frames: int | None
    bridge_recovered: bool


@dataclass(frozen=True)
class PairDiagnosticRecord:
    source_frame: int
    target_frame: int
    source_timestamp_us: int
    target_timestamp_us: int
    dt_sec: float
    open3d_success: bool
    accepted: bool
    reject_reason: str | None
    translation_m: float | None
    rotation_deg: float | None
    linear_speed_m_s: float | None
    angular_speed_deg_s: float | None
    gap_frames: int
    bridge_attempt: bool
    source_valid_depth_ratio: float
    target_valid_depth_ratio: float
    input_prepare_ms: float | None
    odometry_ms: float | None
    total_ms: float | None
    information_finite: bool
    information_symmetric: bool
    information_trace: float | None
    information_condition_number: float | None
    segment_id: int


@dataclass(frozen=True)
class SegmentSummary:
    segment_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    valid_frame_count: int


@dataclass(frozen=True)
class RuntimeStats:
    total_processing_time_s: float
    input_prepare_ms_mean: float
    input_prepare_ms_median: float
    input_prepare_ms_p90: float
    odometry_ms_mean: float
    odometry_ms_median: float
    odometry_ms_p90: float
    total_pair_ms_mean: float
    total_pair_ms_median: float
    total_pair_ms_p90: float
    real_time_factor: float | None


@dataclass(frozen=True)
class ContinuousOdometryResult:
    samples: list[TrajectorySample]
    pair_diagnostics: list[PairDiagnosticRecord]
    segments: list[SegmentSummary]
    runtime_stats: RuntimeStats
    summary: dict[str, Any]


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def pose_to_quaternion_translation(T: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    T = np.asarray(T, dtype=np.float64)
    qw, qx, qy, qz = rotation_matrix_to_quaternion(T[:3, :3])
    return float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), qw, qx, qy, qz


def _depth_valid_ratio(depth_m: np.ndarray) -> float:
    depth = np.asarray(depth_m)
    return float((depth > 0.0).mean())


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def _segment_summaries(samples: Sequence[TrajectorySample]) -> list[SegmentSummary]:
    by_segment: dict[int, list[TrajectorySample]] = {}
    for sample in samples:
        by_segment.setdefault(sample.segment_id, []).append(sample)
    summaries: list[SegmentSummary] = []
    for segment_id in sorted(by_segment):
        rows = by_segment[segment_id]
        summaries.append(
            SegmentSummary(
                segment_id=segment_id,
                start_frame=rows[0].frame_number,
                end_frame=rows[-1].frame_number,
                frame_count=len(rows),
                valid_frame_count=sum(1 for row in rows if row.valid),
            )
        )
    return summaries


def _longest_invalid_run(samples: Sequence[TrajectorySample]) -> int:
    longest = 0
    current = 0
    for sample in samples:
        if sample.valid:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def run_continuous_rgbd_odometry(
    frame_sequence: Sequence[OdometryFrameInput],
    rgb_intrinsics: Any,
    config: ContinuousOdometryConfig | None = None,
    *,
    pairwise_estimator: PairwiseEstimator = estimate_rgbd_pair_motion,
) -> ContinuousOdometryResult:
    if not frame_sequence:
        raise ValueError("frame_sequence must not be empty")
    cfg = config or ContinuousOdometryConfig()
    frames = list(frame_sequence)
    samples: list[TrajectorySample] = []
    pair_diagnostics: list[PairDiagnosticRecord] = []

    segment_id = 0
    last_valid_index = 0
    last_valid_pose = np.eye(4, dtype=np.float64)
    continuity_from_previous_segment = True

    first = frames[0]
    tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(last_valid_pose)
    samples.append(
        TrajectorySample(
            frame_number=first.frame_number,
            device_timestamp_us=first.device_timestamp_us,
            valid=True,
            tracking_state=TRACKING_STATE_LOCAL,
            segment_id=segment_id,
            segment_start=True,
            continuity_from_previous_segment=True,
            tx=tx,
            ty=ty,
            tz=tz,
            qw=qw,
            qx=qx,
            qy=qy,
            qz=qz,
            source_frame=None,
            pair_gap_frames=None,
            bridge_recovered=False,
        )
    )

    input_prepare_times: list[float] = []
    odometry_times: list[float] = []
    total_pair_times: list[float] = []
    run_t0 = time.perf_counter()

    for target_index in range(1, len(frames)):
        target = frames[target_index]
        last_valid = frames[last_valid_index]
        gap_frames = target.frame_number - last_valid.frame_number

        if gap_frames > cfg.max_bridge_gap_frames:
            segment_id += 1
            last_valid_pose = np.eye(4, dtype=np.float64)
            last_valid_index = target_index
            tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(last_valid_pose)
            samples.append(
                TrajectorySample(
                    frame_number=target.frame_number,
                    device_timestamp_us=target.device_timestamp_us,
                    valid=True,
                    tracking_state=TRACKING_STATE_LOCAL,
                    segment_id=segment_id,
                    segment_start=True,
                    continuity_from_previous_segment=False,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    source_frame=None,
                    pair_gap_frames=None,
                    bridge_recovered=False,
                )
            )
            continuity_from_previous_segment = False
            continue

        dt_sec = (target.device_timestamp_us - last_valid.device_timestamp_us) / 1_000_000.0
        bridge_attempt = gap_frames > 1
        pair_result = pairwise_estimator(
            last_valid.rgb,
            last_valid.depth_m,
            target.rgb,
            target.depth_m,
            rgb_intrinsics,
            config=cfg.odometry,
            source_frame=last_valid.frame_number,
            target_frame=target.frame_number,
        )
        info_diag = information_matrix_diagnostics(pair_result.information_matrix)
        evaluation = evaluate_pair_motion(
            pair_result,
            dt_sec=dt_sec,
            motion_config=cfg.motion,
            information_diagnostics=info_diag,
        )
        if evaluation.input_prepare_ms is not None:
            input_prepare_times.append(float(evaluation.input_prepare_ms))
        if evaluation.odometry_ms is not None:
            odometry_times.append(float(evaluation.odometry_ms))
        if evaluation.total_ms is not None:
            total_pair_times.append(float(evaluation.total_ms))

        pair_diagnostics.append(
            PairDiagnosticRecord(
                source_frame=last_valid.frame_number,
                target_frame=target.frame_number,
                source_timestamp_us=last_valid.device_timestamp_us,
                target_timestamp_us=target.device_timestamp_us,
                dt_sec=dt_sec,
                open3d_success=evaluation.open3d_success,
                accepted=evaluation.accepted,
                reject_reason=evaluation.reject_reason,
                translation_m=evaluation.translation_m,
                rotation_deg=evaluation.rotation_deg,
                linear_speed_m_s=evaluation.linear_speed_m_s,
                angular_speed_deg_s=evaluation.angular_speed_deg_s,
                gap_frames=gap_frames,
                bridge_attempt=bridge_attempt,
                source_valid_depth_ratio=_depth_valid_ratio(last_valid.depth_m),
                target_valid_depth_ratio=_depth_valid_ratio(target.depth_m),
                input_prepare_ms=evaluation.input_prepare_ms,
                odometry_ms=evaluation.odometry_ms,
                total_ms=evaluation.total_ms,
                information_finite=bool(info_diag.get("finite")),
                information_symmetric=bool(info_diag.get("symmetric")),
                information_trace=info_diag.get("trace"),
                information_condition_number=info_diag.get("condition_number"),
                segment_id=segment_id,
            )
        )

        if evaluation.accepted and evaluation.transform_target_source is not None:
            last_valid_pose = accumulate_odom_pose(last_valid_pose, evaluation.transform_target_source)
            last_valid_index = target_index
            tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(last_valid_pose)
            samples.append(
                TrajectorySample(
                    frame_number=target.frame_number,
                    device_timestamp_us=target.device_timestamp_us,
                    valid=True,
                    tracking_state=TRACKING_STATE_LOCAL,
                    segment_id=segment_id,
                    segment_start=False,
                    continuity_from_previous_segment=continuity_from_previous_segment,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    source_frame=last_valid.frame_number,
                    pair_gap_frames=gap_frames,
                    bridge_recovered=bridge_attempt,
                )
            )
            continue

        samples.append(
            TrajectorySample(
                frame_number=target.frame_number,
                device_timestamp_us=target.device_timestamp_us,
                valid=False,
                tracking_state=TRACKING_STATE_LOST,
                segment_id=segment_id,
                segment_start=False,
                continuity_from_previous_segment=continuity_from_previous_segment,
                tx=None,
                ty=None,
                tz=None,
                qw=None,
                qx=None,
                qy=None,
                qz=None,
                source_frame=last_valid.frame_number,
                pair_gap_frames=gap_frames,
                bridge_recovered=False,
            )
        )

    total_processing_time_s = time.perf_counter() - run_t0
    segments = _segment_summaries(samples)
    accepted_pairs = sum(1 for row in pair_diagnostics if row.accepted)
    rejected_pairs = sum(1 for row in pair_diagnostics if not row.accepted)
    reject_hist: dict[str, int] = {}
    for row in pair_diagnostics:
        if not row.accepted:
            key = row.reject_reason or "UNKNOWN"
            reject_hist[key] = reject_hist.get(key, 0) + 1
    bridge_attempts = sum(1 for row in pair_diagnostics if row.bridge_attempt)
    bridge_success = sum(1 for row in pair_diagnostics if row.bridge_attempt and row.accepted)
    bridge_failure = bridge_attempts - bridge_success

    session_duration_s = None
    if len(frames) >= 2:
        session_duration_s = (frames[-1].device_timestamp_us - frames[0].device_timestamp_us) / 1_000_000.0
    real_time_factor = None
    if session_duration_s and session_duration_s > 0.0:
        real_time_factor = total_processing_time_s / session_duration_s

    runtime_stats = RuntimeStats(
        total_processing_time_s=total_processing_time_s,
        input_prepare_ms_mean=float(np.mean(input_prepare_times)) if input_prepare_times else 0.0,
        input_prepare_ms_median=float(np.median(input_prepare_times)) if input_prepare_times else 0.0,
        input_prepare_ms_p90=_percentile(input_prepare_times, 0.9),
        odometry_ms_mean=float(np.mean(odometry_times)) if odometry_times else 0.0,
        odometry_ms_median=float(np.median(odometry_times)) if odometry_times else 0.0,
        odometry_ms_p90=_percentile(odometry_times, 0.9),
        total_pair_ms_mean=float(np.mean(total_pair_times)) if total_pair_times else 0.0,
        total_pair_ms_median=float(np.median(total_pair_times)) if total_pair_times else 0.0,
        total_pair_ms_p90=_percentile(total_pair_times, 0.9),
        real_time_factor=real_time_factor,
    )

    summary = {
        "total_frames": len(samples),
        "valid_frames": sum(1 for row in samples if row.valid),
        "invalid_frames": sum(1 for row in samples if not row.valid),
        "valid_ratio": float(sum(1 for row in samples if row.valid) / len(samples)),
        "accepted_pair_count": accepted_pairs,
        "rejected_pair_count": rejected_pairs,
        "reject_reason_histogram": reject_hist,
        "bridge_attempts": bridge_attempts,
        "bridge_success": bridge_success,
        "bridge_failure": bridge_failure,
        "segment_count": len(segments),
        "segment_start_frames": [seg.start_frame for seg in segments],
        "longest_invalid_run": _longest_invalid_run(samples),
    }

    return ContinuousOdometryResult(
        samples=samples,
        pair_diagnostics=pair_diagnostics,
        segments=segments,
        runtime_stats=runtime_stats,
        summary=summary,
    )


def write_trajectory_csv(path: Path, samples: Sequence[TrajectorySample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_number",
        "device_timestamp_us",
        "valid",
        "tracking_state",
        "segment_id",
        "segment_start",
        "continuity_from_previous_segment",
        "tx",
        "ty",
        "tz",
        "qw",
        "qx",
        "qy",
        "qz",
        "source_frame",
        "pair_gap_frames",
        "bridge_recovered",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def write_pair_diagnostics_csv(path: Path, rows: Sequence[PairDiagnosticRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else [
        "source_frame",
        "target_frame",
        "source_timestamp_us",
        "target_timestamp_us",
        "dt_sec",
        "open3d_success",
        "accepted",
        "reject_reason",
        "translation_m",
        "rotation_deg",
        "linear_speed_m_s",
        "angular_speed_deg_s",
        "gap_frames",
        "bridge_attempt",
        "source_valid_depth_ratio",
        "target_valid_depth_ratio",
        "input_prepare_ms",
        "odometry_ms",
        "total_ms",
        "information_finite",
        "information_symmetric",
        "information_trace",
        "information_condition_number",
        "segment_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_provenance(
    *,
    config: ContinuousOdometryConfig,
    calibration_fingerprint: Mapping[str, Any],
    alignment_manifest_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import open3d as o3d

        open3d_version = o3d.__version__
    except ImportError:
        open3d_version = None
    return {
        "algorithm_id": RGBD_ODOMETRY_ALGORITHM_ID,
        "open3d_version": open3d_version,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "jacobian": "Hybrid",
        "depth_min": config.odometry.depth_min_m,
        "depth_max": config.odometry.depth_max_m,
        "depth_diff_max": config.odometry.depth_diff_max_m,
        "depth_scale": config.odometry.depth_scale,
        "depth_trunc": config.odometry.depth_trunc_m,
        "max_linear_speed_m_s": config.motion.max_linear_speed_m_s,
        "max_angular_speed_deg_s": config.motion.max_angular_speed_deg_s,
        "max_bridge_gap_frames": config.max_bridge_gap_frames,
        "calibration_fingerprint": dict(calibration_fingerprint),
        "alignment_manifest_fingerprint": dict(alignment_manifest_fingerprint),
        "tag_texture_visible": True,
        "candidate_uses_apriltag_pose": False,
        "candidate_uses_reference": False,
        "candidate_uses_cup": False,
    }


def write_continuous_outputs(
    output_dir: Path,
    result: ContinuousOdometryResult,
    provenance: Mapping[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_trajectory_csv(output_dir / "trajectory.csv", result.samples)
    write_pair_diagnostics_csv(output_dir / "pair_diagnostics.csv", result.pair_diagnostics)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": result.summary,
                "segments": [asdict(seg) for seg in result.segments],
                "runtime_stats": asdict(result.runtime_stats),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def continuous_config_from_mapping(cfg: Mapping[str, Any] | None) -> ContinuousOdometryConfig:
    raw = dict(cfg or {})
    motion_raw = raw.get("motion", {})
    odometry_raw = raw.get("odometry", {})
    return ContinuousOdometryConfig(
        odometry=RgbdOdometryConfig(
            depth_scale=float(odometry_raw.get("depth_scale", 1.0)),
            depth_trunc_m=float(odometry_raw.get("depth_trunc_m", 4.0)),
            depth_min_m=float(odometry_raw.get("depth_min_m", 0.05)),
            depth_max_m=float(odometry_raw.get("depth_max_m", 4.0)),
            depth_diff_max_m=float(odometry_raw.get("depth_diff_max_m", 0.03)),
            convert_rgb_to_intensity=bool(odometry_raw.get("convert_rgb_to_intensity", True)),
        ),
        motion=MotionPlausibilityConfig(
            max_linear_speed_m_s=float(motion_raw.get("max_linear_speed_m_s", 1.0)),
            max_angular_speed_deg_s=float(motion_raw.get("max_angular_speed_deg_s", 60.0)),
        ),
        max_bridge_gap_frames=int(raw.get("max_bridge_gap_frames", DEFAULT_MAX_BRIDGE_GAP_FRAMES)),
    )
