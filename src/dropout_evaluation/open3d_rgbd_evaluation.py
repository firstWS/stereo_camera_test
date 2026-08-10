"""Phase 4.3 official Open3D RGB-D odometry evaluation runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from dataset_recorder.session_metadata import write_json

from .dropout_protocol import (
    DropoutProtocolConfig,
    DropoutWindow,
    FrameTimestamp,
    SuccessThresholds,
    load_dropout_protocol_config,
    load_frame_timestamps_from_rgb_index,
)
from .evaluation_io import load_cup_observations_from_csv, load_pose_references_from_csv
from .evaluation_metrics import (
    CupObservation,
    EvaluationResult,
    PoseReference,
    evaluate_window,
    rotation_error_deg,
    translation_error,
)
from .hold_last_pose_runner import load_dropout_windows_from_manifest
from .rgbd_odometry import RGBD_ODOMETRY_ALGORITHM_ID
from .rgbd_odometry_adapter import (
    RgbdAdapterStatus,
    RgbdAdapterWindowResult,
    RgbdOdometryAdapterConfig,
    generate_rgbd_odometry_candidates,
    load_local_trajectory_from_csv,
    resolve_adapter_recovery_timing,
)
from .runtime_apriltag import RUNTIME_APRILTAG_SOURCE, RuntimeAprilTagPose, load_runtime_apriltag_poses_from_session


@dataclass(frozen=True)
class Open3dRgbdEvaluationPaths:
    phase3_protocol_config: Path
    dropout_manifest: Path
    session_dir: Path
    continuous_trajectory_csv: Path
    continuous_summary_json: Path
    continuous_provenance_json: Path
    hold_results_dir: Path | None
    output_dir: Path


@dataclass(frozen=True)
class Open3dRgbdEvaluationConfig:
    schema_version: int
    protocol: DropoutProtocolConfig
    paths: Open3dRgbdEvaluationPaths


@dataclass
class Open3dRgbdEvaluationSummary:
    algorithm_id: str
    session_id: str
    windows_total: int
    windows_evaluated: int
    alignment_failures: int
    cup2_applicable_windows: int
    cup2_not_applicable_windows: int
    experiment_design_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "session_id": self.session_id,
            "windows_total": self.windows_total,
            "windows_evaluated": self.windows_evaluated,
            "alignment_failures": self.alignment_failures,
            "cup2_applicable_windows": self.cup2_applicable_windows,
            "cup2_not_applicable_windows": self.cup2_not_applicable_windows,
            "experiment_design_note": self.experiment_design_note,
        }


@dataclass
class Open3dRgbdEvaluationRunResult:
    summary: Open3dRgbdEvaluationSummary
    window_results: list[dict[str, Any]]
    provenance: dict[str, Any]
    hold_comparison: list[dict[str, Any]] | None


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_open3d_rgbd_evaluation_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> Open3dRgbdEvaluationConfig:
    root = repo_root or path.parents[2]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    phase3_config_path = _resolve_path(root, str(payload["phase3_protocol_config"]))
    protocol = load_dropout_protocol_config(phase3_config_path)

    candidate = payload.get("candidate", {})
    comparison = payload.get("comparison", {})
    output = payload.get("output", {})

    session_id = str(output.get("session_id", protocol.session_id))
    session_dir = _resolve_path(root, protocol.session_path)
    manifest_path = _resolve_path(root, protocol.output_root) / session_id / "dropout_windows.json"
    output_dir = (
        _resolve_path(root, str(output.get("root", "out/evaluation/phase4")))
        / session_id
        / str(output.get("algorithm_dir", "open3d_rgbd_odometry"))
    )

    hold_dir_raw = comparison.get("hold_last_pose_results")
    hold_dir = _resolve_path(root, str(hold_dir_raw)) if hold_dir_raw else None

    paths = Open3dRgbdEvaluationPaths(
        phase3_protocol_config=phase3_config_path,
        dropout_manifest=manifest_path,
        session_dir=session_dir,
        continuous_trajectory_csv=_resolve_path(root, str(candidate["continuous_trajectory"])),
        continuous_summary_json=_resolve_path(root, str(candidate["continuous_summary"])),
        continuous_provenance_json=_resolve_path(root, str(candidate["continuous_provenance"])),
        hold_results_dir=hold_dir,
        output_dir=output_dir,
    )
    return Open3dRgbdEvaluationConfig(
        schema_version=int(payload.get("schema_version", 1)),
        protocol=protocol,
        paths=paths,
    )


def sanitize_json_value(value: Any) -> Any:
    """Convert values to JSON-safe form (no NaN/inf)."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    return value


def compute_first_masked_diagnostics(
    *,
    window: DropoutWindow,
    references: Sequence[PoseReference],
    candidates: Sequence,
) -> dict[str, Any]:
    """Analysis-only first-masked and error-growth diagnostics (not official metric schema)."""
    ref_by_frame = {ref.frame_number: ref for ref in references}
    cand_by_frame = {cand.frame_number: cand for cand in candidates}

    masked_frames = sorted(
        cand.frame_number
        for cand in candidates
        if window.start_device_timestamp_us <= cand.device_timestamp_us < window.boundary_timestamp_us
    )

    first_valid_masked_frame: int | None = None
    first_masked_translation_error: float | None = None
    first_masked_rotation_error: float | None = None

    for frame_number in masked_frames:
        reference = ref_by_frame.get(frame_number)
        candidate = cand_by_frame.get(frame_number)
        if reference is None or candidate is None or not candidate.valid:
            continue
        t_err = translation_error(reference, candidate)
        r_err = rotation_error_deg(reference, candidate)
        if t_err is None or r_err is None:
            continue
        first_valid_masked_frame = frame_number
        first_masked_translation_error = t_err
        first_masked_rotation_error = r_err
        break

    end_reference = ref_by_frame.get(window.end_frame)
    end_candidate = cand_by_frame.get(window.end_frame)
    end_translation_error = (
        translation_error(end_reference, end_candidate)
        if end_reference is not None and end_candidate is not None and end_candidate.valid
        else None
    )
    end_rotation_error = (
        rotation_error_deg(end_reference, end_candidate)
        if end_reference is not None and end_candidate is not None and end_candidate.valid
        else None
    )

    translation_error_growth = None
    rotation_error_growth = None
    if end_translation_error is not None and first_masked_translation_error is not None:
        translation_error_growth = end_translation_error - first_masked_translation_error
    if end_rotation_error is not None and first_masked_rotation_error is not None:
        rotation_error_growth = end_rotation_error - first_masked_rotation_error

    return sanitize_json_value(
        {
            "first_valid_masked_frame": first_valid_masked_frame,
            "first_masked_translation_error": first_masked_translation_error,
            "first_masked_rotation_error": first_masked_rotation_error,
            "end_translation_error": end_translation_error,
            "end_rotation_error": end_rotation_error,
            "translation_error_growth": translation_error_growth,
            "rotation_error_growth": rotation_error_growth,
        }
    )


def _b_fairness_metadata(window: DropoutWindow, adapter_result: RgbdAdapterWindowResult) -> dict[str, Any]:
    protocol_anchor = adapter_result.provenance.protocol_runtime_anchor_frame
    alignment_frame = adapter_result.provenance.world_alignment_frame_before_dropout
    applies = window.anchor_id.startswith("B") and (
        protocol_anchor is not None
        and alignment_frame is not None
        and protocol_anchor != alignment_frame
    )
    return sanitize_json_value(
        {
            "applies": applies,
            "protocol_runtime_anchor_frame": protocol_anchor,
            "world_alignment_frame": alignment_frame,
            "world_alignment_age_frames": adapter_result.provenance.alignment_age_frames,
            "world_alignment_age_sec": adapter_result.provenance.alignment_age_sec,
            "alignment_segment_id": adapter_result.provenance.alignment_segment_id,
            "warning": (
                "B absolute error comparison may include initial alignment mismatch between "
                "HOLD runtime anchor and RGB-D retained joint alignment frame."
                if applies
                else None
            ),
            "cleaner_comparison_anchor_match": window.anchor_id.startswith(("C", "D")),
        }
    )


def generate_window_candidates(
    *,
    window: DropoutWindow,
    local_trajectory_csv: Path,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    adapter_config: RgbdOdometryAdapterConfig | None = None,
) -> RgbdAdapterWindowResult:
    """Candidate-only path: continuous trajectory + runtime AprilTag + manifest mask."""
    local_trajectory = load_local_trajectory_from_csv(local_trajectory_csv)
    return generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local_trajectory,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
        config=adapter_config,
    )


def evaluate_open3d_rgbd_window(
    *,
    window: DropoutWindow,
    local_trajectory_csv: Path,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    references: Sequence[PoseReference],
    cup_observations: Sequence[CupObservation],
    adapter_config: RgbdOdometryAdapterConfig | None = None,
    thresholds: SuccessThresholds | None = None,
) -> tuple[RgbdAdapterWindowResult, EvaluationResult | None, dict[str, Any]]:
    """Generate candidates (reference-free) then evaluate with Phase 3 metrics."""
    adapter_cfg = adapter_config or RgbdOdometryAdapterConfig()
    generation = generate_window_candidates(
        window=window,
        local_trajectory_csv=local_trajectory_csv,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
        adapter_config=adapter_cfg,
    )
    candidate_provenance = generation.provenance.as_dict()
    analysis = compute_first_masked_diagnostics(
        window=window,
        references=references,
        candidates=generation.candidates,
    )
    fairness = _b_fairness_metadata(window, generation)

    if generation.status != RgbdAdapterStatus.OK:
        return generation, None, sanitize_json_value(
            {
                "candidate_provenance": candidate_provenance,
                "first_masked_analysis": analysis,
                "fairness": fairness,
            }
        )

    recovery_timing = resolve_adapter_recovery_timing(window=window, result=generation)
    evaluation = evaluate_window(
        window=window,
        references=references,
        candidates=generation.candidates,
        observations=cup_observations,
        algorithm_id=adapter_cfg.algorithm_id,
        thresholds=thresholds,
        cup2_semantic_id="cup2",
        recovery_timing=recovery_timing,
    )
    extras = sanitize_json_value(
        {
            "candidate_provenance": candidate_provenance,
            "first_masked_analysis": analysis,
            "fairness": fairness,
            "segment_reset_during_dropout": generation.provenance.segment_reset_during_dropout,
            "comparison_axes": {
                "accuracy_when_available": "pose and cup2 error distributions",
                "availability_continuity": "pose availability and tracking lost events",
            },
        }
    )
    return generation, evaluation, extras


def load_hold_window_results(hold_results_dir: Path) -> dict[str, dict[str, Any]]:
    path = hold_results_dir / "window_results.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("windows", [])
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        window_id = str(row.get("window_id", ""))
        evaluation = row.get("evaluation")
        if window_id and evaluation is not None:
            by_id[window_id] = evaluation
    return by_id


def _metric_get(distribution: Mapping[str, Any] | None, key: str) -> float | None:
    if distribution is None:
        return None
    value = distribution.get(key)
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def build_hold_comparison_row(
    *,
    window_id: str,
    rgbd_evaluation: EvaluationResult,
    hold_evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rgbd_pose = rgbd_evaluation.pose.as_dict()
    rgbd_recovery = rgbd_evaluation.recovery.as_dict()
    rgbd_cup = rgbd_evaluation.cup2_world.as_dict()

    row: dict[str, Any] = {
        "window_id": window_id,
        "rgbd": sanitize_json_value(
            {
                "pose_availability": (rgbd_pose.get("availability") or {}).get("availability_ratio"),
                "lost_event_count": (rgbd_pose.get("availability") or {}).get("lost_event_count"),
                "translation_median": _metric_get(rgbd_pose.get("translation_error_m"), "median"),
                "translation_p90": _metric_get(rgbd_pose.get("translation_error_m"), "p90"),
                "translation_end": _metric_get(rgbd_pose.get("translation_error_m"), "end_of_dropout_error"),
                "rotation_median": _metric_get(rgbd_pose.get("rotation_error_deg"), "median"),
                "rotation_p90": _metric_get(rgbd_pose.get("rotation_error_deg"), "p90"),
                "rotation_end": _metric_get(rgbd_pose.get("rotation_error_deg"), "end_of_dropout_error"),
                "cup2_median": _metric_get(rgbd_cup.get("position_error"), "median"),
                "cup2_p90": _metric_get(rgbd_cup.get("position_error"), "p90"),
                "cup2_availability": rgbd_cup.get("availability_ratio"),
                "recovery_latency_frames": rgbd_recovery.get("recovery_latency_frames"),
                "recovery_latency_sec": rgbd_recovery.get("recovery_latency_sec"),
            }
        ),
        "hold": None,
    }
    if hold_evaluation is not None:
        hold_pose = hold_evaluation.get("pose", {})
        hold_recovery = hold_evaluation.get("recovery", {})
        hold_cup = hold_evaluation.get("cup2_world", {})
        row["hold"] = sanitize_json_value(
            {
                "pose_availability": (hold_pose.get("availability") or {}).get("availability_ratio"),
                "lost_event_count": (hold_pose.get("availability") or {}).get("lost_event_count"),
                "translation_median": _metric_get(hold_pose.get("translation_error_m"), "median"),
                "translation_p90": _metric_get(hold_pose.get("translation_error_m"), "p90"),
                "translation_end": _metric_get(hold_pose.get("translation_error_m"), "end_of_dropout_error"),
                "rotation_median": _metric_get(hold_pose.get("rotation_error_deg"), "median"),
                "rotation_p90": _metric_get(hold_pose.get("rotation_error_deg"), "p90"),
                "rotation_end": _metric_get(hold_pose.get("rotation_error_deg"), "end_of_dropout_error"),
                "cup2_median": _metric_get(hold_cup.get("position_error"), "median"),
                "cup2_p90": _metric_get(hold_cup.get("position_error"), "p90"),
                "cup2_availability": hold_cup.get("availability_ratio"),
                "recovery_latency_frames": hold_recovery.get("recovery_latency_frames"),
                "recovery_latency_sec": hold_recovery.get("recovery_latency_sec"),
            }
        )
    return sanitize_json_value(row)


def build_evaluation_provenance(
    *,
    config: Open3dRgbdEvaluationConfig,
    continuous_provenance: Mapping[str, Any],
    continuous_summary: Mapping[str, Any],
) -> dict[str, Any]:
    paths = config.paths
    runtime_csv = paths.session_dir / "derived/apriltag/observations.csv"
    reference_csv = paths.session_dir / config.protocol.reference_source
    cup_csv = paths.session_dir / config.protocol.cup2_observations_csv

    runtime_stats = continuous_summary.get("runtime_stats", {})
    return sanitize_json_value(
        {
            "algorithm_id": RGBD_ODOMETRY_ALGORITHM_ID,
            "candidate_source_trajectory": str(paths.continuous_trajectory_csv),
            "candidate_source_trajectory_fingerprint": {
                "trajectory_sha16": _sha16(paths.continuous_trajectory_csv),
                "summary_sha16": _sha16(paths.continuous_summary_json),
            },
            "alignment_cache_fingerprint": continuous_provenance.get("alignment_manifest_fingerprint"),
            "rgb_calibration_fingerprint": continuous_provenance.get("calibration_fingerprint"),
            "runtime_apriltag_source": RUNTIME_APRILTAG_SOURCE,
            "runtime_apriltag_path": str(runtime_csv),
            "runtime_apriltag_fingerprint": {"observations_sha16": _sha16(runtime_csv)} if runtime_csv.is_file() else None,
            "dropout_manifest": str(paths.dropout_manifest),
            "dropout_manifest_fingerprint": {"manifest_sha16": _sha16(paths.dropout_manifest)},
            "reference_source": str(reference_csv),
            "reference_evaluation_only": True,
            "cup_source": str(cup_csv),
            "cup_evaluation_only": True,
            "tag_texture_visible": True,
            "candidate_uses_apriltag_during_dropout": False,
            "candidate_uses_reference": False,
            "candidate_uses_cup": False,
            "algorithm_runtime_source": "phase4_2b_continuous_run",
            "open3d_version": continuous_provenance.get("open3d_version", "0.19.0"),
            "python_version": continuous_provenance.get("python_version"),
            "numpy_version": continuous_provenance.get("numpy_version"),
            "processing_environment": continuous_provenance.get("platform"),
            "total_processing_time_s": runtime_stats.get("total_processing_time_s"),
            "odometry_ms_mean": runtime_stats.get("odometry_ms_mean"),
            "odometry_ms_median": runtime_stats.get("odometry_ms_median"),
            "odometry_ms_p90": runtime_stats.get("odometry_ms_p90"),
            "real_time_factor": runtime_stats.get("real_time_factor"),
            "experiment_design_note": (
                "15 windows are nested durations from the same Scenario A session; "
                "do not treat per-window PASS counts as independent trial statistics."
            ),
        }
    )


def run_open3d_rgbd_evaluation(
    *,
    config: Open3dRgbdEvaluationConfig,
) -> Open3dRgbdEvaluationRunResult:
    """Run official Phase 4 RGB-D evaluation over all manifest windows."""
    paths = config.paths
    windows = load_dropout_windows_from_manifest(paths.dropout_manifest)
    runtime_poses = load_runtime_apriltag_poses_from_session(paths.session_dir)
    frame_timestamps = load_frame_timestamps_from_rgb_index(paths.session_dir)

    references = load_pose_references_from_csv(paths.session_dir / config.protocol.reference_source)
    cup_observations = load_cup_observations_from_csv(
        paths.session_dir / config.protocol.cup2_observations_csv,
    )

    continuous_provenance = json.loads(paths.continuous_provenance_json.read_text(encoding="utf-8"))
    continuous_summary = json.loads(paths.continuous_summary_json.read_text(encoding="utf-8"))

    hold_by_window = (
        load_hold_window_results(paths.hold_results_dir)
        if paths.hold_results_dir is not None
        else {}
    )

    window_results: list[dict[str, Any]] = []
    hold_comparison: list[dict[str, Any]] = []
    alignment_failures = 0
    evaluated = 0
    cup2_applicable = 0
    cup2_not_applicable = 0

    for window in windows:
        generation, evaluation, extras = evaluate_open3d_rgbd_window(
            window=window,
            local_trajectory_csv=paths.continuous_trajectory_csv,
            runtime_poses=runtime_poses,
            frame_timestamps=frame_timestamps,
            references=references,
            cup_observations=cup_observations,
            thresholds=config.protocol.success_thresholds,
        )
        if generation.status != RgbdAdapterStatus.OK or evaluation is None:
            alignment_failures += 1
            window_results.append(
                sanitize_json_value(
                    {
                        "window_id": window.window_id,
                        "status": generation.status.value,
                        "evaluation": None,
                        "extras": extras,
                    }
                )
            )
            continue

        evaluated += 1
        if evaluation.cup2_world.status.value == "NOT_APPLICABLE":
            cup2_not_applicable += 1
        else:
            cup2_applicable += 1

        eval_dict = evaluation.as_dict()
        window_results.append(
            sanitize_json_value(
                {
                    "window_id": window.window_id,
                    "status": "OK",
                    "evaluation": eval_dict,
                    "extras": extras,
                }
            )
        )
        hold_comparison.append(
            build_hold_comparison_row(
                window_id=window.window_id,
                rgbd_evaluation=evaluation,
                hold_evaluation=hold_by_window.get(window.window_id),
            )
        )

    summary = Open3dRgbdEvaluationSummary(
        algorithm_id=RGBD_ODOMETRY_ALGORITHM_ID,
        session_id=config.protocol.session_id,
        windows_total=len(windows),
        windows_evaluated=evaluated,
        alignment_failures=alignment_failures,
        cup2_applicable_windows=cup2_applicable,
        cup2_not_applicable_windows=cup2_not_applicable,
        experiment_design_note=(
            "15 windows are nested durations from the same Scenario A session; "
            "do not treat per-window PASS counts as independent trial statistics."
        ),
    )
    provenance = build_evaluation_provenance(
        config=config,
        continuous_provenance=continuous_provenance,
        continuous_summary=continuous_summary,
    )
    return Open3dRgbdEvaluationRunResult(
        summary=summary,
        window_results=window_results,
        provenance=provenance,
        hold_comparison=hold_comparison if hold_comparison else None,
    )


def write_open3d_rgbd_evaluation_results(
    *,
    output_dir: Path,
    run_result: Open3dRgbdEvaluationRunResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", sanitize_json_value(run_result.summary.as_dict()))
    write_json(output_dir / "window_results.json", sanitize_json_value({"windows": run_result.window_results}))
    write_json(output_dir / "provenance.json", run_result.provenance)
    if run_result.hold_comparison is not None:
        write_json(
            output_dir / "hold_comparison.json",
            sanitize_json_value({"windows": run_result.hold_comparison}),
        )


def run_open3d_rgbd_evaluation_from_paths(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Open3dRgbdEvaluationRunResult:
    config = load_open3d_rgbd_evaluation_config(config_path, repo_root=repo_root)
    result = run_open3d_rgbd_evaluation(config=config)
    target = output_dir or config.paths.output_dir
    write_open3d_rgbd_evaluation_results(output_dir=target, run_result=result)
    return result
