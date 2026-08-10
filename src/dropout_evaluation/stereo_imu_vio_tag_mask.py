"""Phase 4.5-M2 tag-masked stereo VIO ablation orchestration."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_recorder.session_metadata import write_json

from .dropout_protocol import load_frame_timestamps_from_rgb_index
from .evaluation_io import load_cup_observations_from_csv, load_pose_references_from_csv
from .hold_last_pose_runner import load_dropout_windows_from_manifest
from .ir_tag_mask import (
    MASK_INTERVAL_RECOVERY_FRAME,
    MASK_INTERVAL_START_FRAME,
    FrameTagMaskDiagnostics,
    apply_tag_mask_to_stereo_frames,
    diagnostics_to_jsonable,
    load_rgb_gray_by_canonical_frame,
    load_rgb_ir_mask_calibration,
    summarize_mask_diagnostics,
)
from .open3d_rgbd_evaluation import sanitize_json_value
from .stereo_imu_vio_evaluation import build_metric_snapshot
from .runtime_apriltag import load_runtime_apriltag_poses_from_session
from .stereo_imu_vio_continuous import (
    StereoImuVioResult,
    build_provenance,
    run_continuous_stereo_imu_vio,
)
from .stereo_imu_vio_evaluation import evaluate_stereo_imu_vio_window
from .stereo_imu_vio_lite import (
    STEREO_IMU_VIO_LITE_ALGORITHM_ID,
    StereoImuTrajectorySample,
    StereoImuVioConfig,
    StereoImuVioFrameInput,
)
from .stereo_imu_calibration import StereoImuCalibration

TAG_MASK_WINDOW_ID = "C_pre_cup2__5.0s"
TAG_MASK_ALGORITHM_DIR = "stereo_imu_vio_lite_tag_mask_c5s"
TAG_MASK_ANALYSIS_DIRNAME = "phase4_stereo_imu_vio_lite_tag_mask_c5s"


def build_tag_mask_provenance(
    *,
    base_provenance: Mapping[str, Any],
    mask_summary: Mapping[str, Any],
    margin_px: int,
    dictionary: str,
) -> dict[str, Any]:
    provenance = dict(base_provenance)
    provenance.update(
        {
            "tag_mask_ablation": True,
            "tag_mask_window_id": TAG_MASK_WINDOW_ID,
            "tag_mask_interval": {
                "start_frame": MASK_INTERVAL_START_FRAME,
                "recovery_frame": MASK_INTERVAL_RECOVERY_FRAME,
                "semantics": "half_open [start, recovery)",
            },
            "tag_detector_used_for_mask_generation_only": True,
            "tag_roi_generation_method": mask_summary.get("roi_source_counts"),
            "estimator_uses_tag_pose": False,
            "estimator_receives_tag_roi": False,
            "tag_mask_dictionary": dictionary,
            "tag_mask_margin_px": margin_px,
            "tag_mask_summary": dict(mask_summary),
        }
    )
    return provenance


def write_tag_masked_trajectory_csv(
    output_dir: Path,
    samples: Sequence[StereoImuTrajectorySample],
    mask_diagnostics: Sequence[FrameTagMaskDiagnostics],
) -> None:
    diag_by_frame = {row.frame_number: row for row in mask_diagnostics}
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
        "tag_mask_active",
        "tag_mask_left_area_ratio",
        "tag_mask_right_area_ratio",
    ]
    with (output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            diag = diag_by_frame.get(sample.frame_number)
            row["tag_mask_active"] = bool(diag.tag_mask_active) if diag else False
            row["tag_mask_left_area_ratio"] = diag.left.area_ratio if diag and diag.tag_mask_active else 0.0
            row["tag_mask_right_area_ratio"] = diag.right.area_ratio if diag and diag.tag_mask_active else 0.0
            writer.writerow(row)


def write_tag_masked_vio_outputs(
    output_dir: Path,
    result: StereoImuVioResult,
    provenance: Mapping[str, Any],
    mask_diagnostics: Sequence[FrameTagMaskDiagnostics],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tag_masked_trajectory_csv(output_dir, result.samples, mask_diagnostics)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": result.summary,
                "runtime_stats": {
                    "total_processing_time_s": result.runtime_stats.total_processing_time_s,
                    "real_time_factor": result.runtime_stats.real_time_factor,
                    "frames_per_second": result.runtime_stats.frames_per_second,
                },
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
    write_json(
        output_dir / "tag_mask_diagnostics.json",
        sanitize_json_value(
            {
                "frames": diagnostics_to_jsonable(mask_diagnostics),
                "summary": summarize_mask_diagnostics(mask_diagnostics),
            }
        ),
    )


def run_tag_masked_vio(
    *,
    frames: Sequence[StereoImuVioFrameInput],
    imu_samples: Sequence[Any],
    calib: StereoImuCalibration,
    config: StereoImuVioConfig | None = None,
    pairing_summary: Mapping[str, Any] | None = None,
    margin_px: int,
    dictionary: str,
    rgb_gray_by_frame: Mapping[int, Any] | None = None,
    mask_calib: Any | None = None,
) -> tuple[StereoImuVioResult, list[FrameTagMaskDiagnostics], dict[str, Any]]:
    masked_frames, mask_diagnostics = apply_tag_mask_to_stereo_frames(
        frames,
        rgb_gray_by_frame=rgb_gray_by_frame,
        mask_calib=mask_calib,
        margin_px=margin_px,
        dictionary=dictionary,
    )
    mask_summary = summarize_mask_diagnostics(mask_diagnostics)
    result = run_continuous_stereo_imu_vio(
        masked_frames,
        imu_samples,
        calib,
        config=config,
        pairing_summary=dict(pairing_summary or {}),
    )
    return result, mask_diagnostics, mask_summary


def evaluate_tag_mask_c5s_window(
    *,
    session_dir: Path,
    manifest_path: Path,
    trajectory_csv: Path,
    protocol_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .dropout_protocol import load_dropout_protocol_config

    protocol = load_dropout_protocol_config(protocol_config_path)
    window = next(
        w for w in load_dropout_windows_from_manifest(manifest_path) if w.window_id == TAG_MASK_WINDOW_ID
    )
    runtime_poses = load_runtime_apriltag_poses_from_session(session_dir)
    frame_timestamps = load_frame_timestamps_from_rgb_index(session_dir)
    references = load_pose_references_from_csv(session_dir / protocol.reference_source)
    cup_observations = load_cup_observations_from_csv(session_dir / protocol.cup2_observations_csv)

    generation, evaluation, extras = evaluate_stereo_imu_vio_window(
        window=window,
        local_trajectory_csv=trajectory_csv,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
        references=references,
        cup_observations=cup_observations,
        thresholds=protocol.success_thresholds,
    )
    if evaluation is None:
        raise RuntimeError(f"Tag-mask C5s evaluation failed: {generation.status}")

    eval_dict = sanitize_json_value(evaluation.as_dict())
    row = sanitize_json_value(
        {
            "window_id": TAG_MASK_WINDOW_ID,
            "status": generation.status.value,
            "evaluation": eval_dict,
            "extras": extras,
        }
    )
    return row, build_metric_snapshot(eval_dict)


def load_baseline_c5s_snapshot(baseline_window_results_json: Path) -> dict[str, Any] | None:
    if not baseline_window_results_json.is_file():
        return None
    payload = json.loads(baseline_window_results_json.read_text(encoding="utf-8"))
    for row in payload.get("windows", []):
        if row.get("window_id") == TAG_MASK_WINDOW_ID and row.get("evaluation") is not None:
            return build_metric_snapshot(row["evaluation"])
    return None


def build_tag_mask_validation_summary(
    *,
    session_dir: Path,
    analysis_dir: Path,
    evaluation_dir: Path,
    demo_summary: Mapping[str, Any],
    mask_summary: Mapping[str, Any],
    trajectory_summary: Mapping[str, Any],
    tag_masked_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    local_summary = trajectory_summary.get("summary", trajectory_summary)
    diagnostics = json.loads((analysis_dir / "diagnostics.json").read_text(encoding="utf-8"))
    return sanitize_json_value(
        {
            "dataset": str(session_dir),
            "algorithm_id": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
            "ablation_id": TAG_MASK_ALGORITHM_DIR,
            "window_id": TAG_MASK_WINDOW_ID,
            "masked_frames": {
                "start_frame": MASK_INTERVAL_START_FRAME,
                "recovery_frame": MASK_INTERVAL_RECOVERY_FRAME,
                "last_masked_frame": MASK_INTERVAL_RECOVERY_FRAME - 1,
                "count": mask_summary.get("masked_frame_count"),
            },
            "left_mask_coverage": {
                "median_area_ratio": mask_summary.get("left_area_ratio_median"),
                "p90_area_ratio": mask_summary.get("left_area_ratio_p90"),
                "detection_failures": mask_summary.get("left_detection_failures"),
                "holdover_frames": mask_summary.get("left_holdover_frames"),
            },
            "right_mask_coverage": {
                "median_area_ratio": mask_summary.get("right_area_ratio_median"),
                "p90_area_ratio": mask_summary.get("right_area_ratio_p90"),
                "detection_failures": mask_summary.get("right_detection_failures"),
                "holdover_frames": mask_summary.get("right_holdover_frames"),
            },
            "trajectory_valid_frames": local_summary.get("valid_frames"),
            "trajectory_total_frames": local_summary.get("total_frames"),
            "propagated_only_frames": local_summary.get("propagated_only_frames"),
            "visual_update_success_ratio": local_summary.get("visual_update_success_ratio"),
            "catastrophic_jump_count": diagnostics.get("catastrophic_jump_count"),
            "c5_pose_availability": tag_masked_metrics.get("pose_availability"),
            "c5_translation_median": tag_masked_metrics.get("translation_median"),
            "c5_translation_p90": tag_masked_metrics.get("translation_p90"),
            "c5_rotation_median": tag_masked_metrics.get("rotation_median"),
            "c5_rotation_p90": tag_masked_metrics.get("rotation_p90"),
            "cup2_availability": tag_masked_metrics.get("cup2_availability"),
            "cup2_median": tag_masked_metrics.get("cup2_median"),
            "cup2_p90": tag_masked_metrics.get("cup2_p90"),
            "recovery_latency_frames": tag_masked_metrics.get("recovery_latency_frames"),
            "baseline_unmasked_c5": baseline_metrics,
            "comparison_delta": _metric_delta(tag_masked_metrics, baseline_metrics),
            "tag_detector_used_for_mask_generation_only": True,
            "estimator_uses_tag_pose": False,
            "output_video_path": demo_summary.get("output_path"),
            "analysis_dir": str(analysis_dir),
            "evaluation_dir": str(evaluation_dir),
        }
    )


def _metric_delta(
    tag_masked: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    keys = (
        "pose_availability",
        "translation_median",
        "translation_p90",
        "rotation_median",
        "rotation_p90",
        "cup2_availability",
        "cup2_median",
        "cup2_p90",
    )
    delta: dict[str, Any] = {}
    for key in keys:
        current = tag_masked.get(key)
        base = baseline.get(key)
        if isinstance(current, (int, float)) and isinstance(base, (int, float)):
            delta[key] = float(current) - float(base)
    return delta


def classify_tag_mask_proof_gate(summary: Mapping[str, Any]) -> str:
    valid = int(summary.get("trajectory_valid_frames") or 0)
    total = int(summary.get("trajectory_total_frames") or 0)
    catastrophic = int(summary.get("catastrophic_jump_count") or 0)
    pose_av = summary.get("c5_pose_availability")
    cup2_av = summary.get("cup2_availability")
    masked_count = int((summary.get("masked_frames") or {}).get("count") or 0)

    if masked_count < 140 or valid < total or catastrophic > 0:
        return "TAG_MASKED_VIO_PROOF_FAILED"
    if pose_av is None or cup2_av is None or float(pose_av) < 0.9 or float(cup2_av) < 0.9:
        return "TAG_MASKED_VIO_PROOF_FAILED"

    warnings = False
    if int(summary.get("propagated_only_frames") or 0) > 0:
        warnings = True
    if float(pose_av) < 1.0 or float(cup2_av) < 1.0:
        warnings = True
    left_fail = int((summary.get("left_mask_coverage") or {}).get("detection_failures") or 0)
    right_fail = int((summary.get("right_mask_coverage") or {}).get("detection_failures") or 0)
    if left_fail > 0 or right_fail > 0:
        warnings = True
    delta = summary.get("comparison_delta") or {}
    cup2_delta = delta.get("cup2_median")
    if isinstance(cup2_delta, (int, float)) and abs(float(cup2_delta)) > 0.05:
        warnings = True

    return "TAG_MASKED_VIO_PROOF_READY_WITH_WARNING" if warnings else "TAG_MASKED_VIO_PROOF_READY"
