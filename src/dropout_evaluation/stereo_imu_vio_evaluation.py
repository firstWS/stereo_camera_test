"""Phase 4.5-B official stereo+IMU VIO-lite evaluation runner."""

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
from .open3d_rgbd_evaluation import (
    _b_fairness_metadata,
    _metric_get,
    compute_first_masked_diagnostics,
    load_hold_window_results,
    sanitize_json_value,
)
from .rgbd_odometry_adapter import (
    RgbdAdapterStatus,
    RgbdAdapterWindowResult,
    resolve_adapter_recovery_timing,
)
from .runtime_apriltag import RUNTIME_APRILTAG_SOURCE, RuntimeAprilTagPose, load_runtime_apriltag_poses_from_session
from .stereo_imu_vio_adapter import (
    StereoImuVioAdapterConfig,
    generate_stereo_imu_vio_candidates,
    load_vio_trajectory_from_csv,
    vio_trajectory_to_local_trajectory,
)
from .stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID


@dataclass(frozen=True)
class StereoImuVioEvaluationPaths:
    phase3_protocol_config: Path
    dropout_manifest: Path
    session_dir: Path
    continuous_trajectory_csv: Path
    continuous_summary_json: Path
    continuous_provenance_json: Path
    hold_results_dir: Path | None
    rgbd_results_dir: Path | None
    output_dir: Path


@dataclass(frozen=True)
class StereoImuVioEvaluationConfig:
    schema_version: int
    protocol: DropoutProtocolConfig
    paths: StereoImuVioEvaluationPaths


@dataclass
class StereoImuVioEvaluationSummary:
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
class StereoImuVioEvaluationRunResult:
    summary: StereoImuVioEvaluationSummary
    window_results: list[dict[str, Any]]
    provenance: dict[str, Any]
    comparison: list[dict[str, Any]] | None


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_stereo_imu_vio_evaluation_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> StereoImuVioEvaluationConfig:
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
        / str(output.get("algorithm_dir", "stereo_imu_vio_lite"))
    )

    hold_dir_raw = comparison.get("hold_last_pose_results")
    rgbd_dir_raw = comparison.get("rgbd_results")
    hold_dir = _resolve_path(root, str(hold_dir_raw)) if hold_dir_raw else None
    rgbd_dir = _resolve_path(root, str(rgbd_dir_raw)) if rgbd_dir_raw else None

    paths = StereoImuVioEvaluationPaths(
        phase3_protocol_config=phase3_config_path,
        dropout_manifest=manifest_path,
        session_dir=session_dir,
        continuous_trajectory_csv=_resolve_path(root, str(candidate["continuous_trajectory"])),
        continuous_summary_json=_resolve_path(root, str(candidate["continuous_summary"])),
        continuous_provenance_json=_resolve_path(root, str(candidate["continuous_provenance"])),
        hold_results_dir=hold_dir,
        rgbd_results_dir=rgbd_dir,
        output_dir=output_dir,
    )
    return StereoImuVioEvaluationConfig(
        schema_version=int(payload.get("schema_version", 1)),
        protocol=protocol,
        paths=paths,
    )


def load_baseline_window_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    path = results_dir / "window_results.json"
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


def build_metric_snapshot(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    pose = evaluation.get("pose", {})
    recovery = evaluation.get("recovery", {})
    cup = evaluation.get("cup2_world", {})
    return sanitize_json_value(
        {
            "pose_availability": (pose.get("availability") or {}).get("availability_ratio"),
            "lost_event_count": (pose.get("availability") or {}).get("lost_event_count"),
            "translation_median": _metric_get(pose.get("translation_error_m"), "median"),
            "translation_p90": _metric_get(pose.get("translation_error_m"), "p90"),
            "translation_end": _metric_get(pose.get("translation_error_m"), "end_of_dropout_error"),
            "rotation_median": _metric_get(pose.get("rotation_error_deg"), "median"),
            "rotation_p90": _metric_get(pose.get("rotation_error_deg"), "p90"),
            "rotation_end": _metric_get(pose.get("rotation_error_deg"), "end_of_dropout_error"),
            "cup2_median": _metric_get(cup.get("position_error"), "median"),
            "cup2_p90": _metric_get(cup.get("position_error"), "p90"),
            "cup2_availability": cup.get("availability_ratio"),
            "cup2_status": cup.get("status"),
            "recovery_latency_frames": recovery.get("recovery_latency_frames"),
            "recovery_latency_sec": recovery.get("recovery_latency_sec"),
        }
    )


def build_three_way_comparison_row(
    *,
    window_id: str,
    vio_evaluation: EvaluationResult,
    hold_evaluation: Mapping[str, Any] | None,
    rgbd_evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return sanitize_json_value(
        {
            "window_id": window_id,
            "vio": build_metric_snapshot(vio_evaluation.as_dict()),
            "hold": build_metric_snapshot(hold_evaluation) if hold_evaluation is not None else None,
            "rgbd": build_metric_snapshot(rgbd_evaluation) if rgbd_evaluation is not None else None,
        }
    )


def generate_window_candidates(
    *,
    window: DropoutWindow,
    local_trajectory_csv: Path,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    adapter_config: StereoImuVioAdapterConfig | None = None,
) -> RgbdAdapterWindowResult:
    local_trajectory = vio_trajectory_to_local_trajectory(load_vio_trajectory_from_csv(local_trajectory_csv))
    return generate_stereo_imu_vio_candidates(
        window=window,
        local_trajectory=local_trajectory,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
        config=adapter_config,
    )


def evaluate_stereo_imu_vio_window(
    *,
    window: DropoutWindow,
    local_trajectory_csv: Path,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    references: Sequence[PoseReference],
    cup_observations: Sequence[CupObservation],
    adapter_config: StereoImuVioAdapterConfig | None = None,
    thresholds: SuccessThresholds | None = None,
) -> tuple[RgbdAdapterWindowResult, EvaluationResult | None, dict[str, Any]]:
    adapter_cfg = adapter_config or StereoImuVioAdapterConfig()
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


def build_evaluation_provenance(
    *,
    config: StereoImuVioEvaluationConfig,
    continuous_provenance: Mapping[str, Any],
    continuous_summary: Mapping[str, Any],
) -> dict[str, Any]:
    paths = config.paths
    runtime_csv = paths.session_dir / "derived/apriltag/observations.csv"
    reference_csv = paths.session_dir / config.protocol.reference_source
    cup_csv = paths.session_dir / config.protocol.cup2_observations_csv
    runtime_stats = continuous_summary.get("runtime_stats", {})
    summary = continuous_summary.get("summary", continuous_summary)

    return sanitize_json_value(
        {
            "algorithm_id": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
            "algorithm_label": "stereo_imu_provisional_vio",
            "candidate_source_trajectory": str(paths.continuous_trajectory_csv),
            "candidate_source_trajectory_fingerprint": {
                "trajectory_sha16": _sha16(paths.continuous_trajectory_csv),
                "summary_sha16": _sha16(paths.continuous_summary_json),
            },
            "calibration": continuous_provenance.get("calibration"),
            "stereo_pairing_policy": continuous_provenance.get("stereo_pairing_policy"),
            "canonical_frame_source": continuous_provenance.get("canonical_frame_source", "rgb_index"),
            "runtime_apriltag_source": RUNTIME_APRILTAG_SOURCE,
            "runtime_apriltag_path": str(runtime_csv),
            "runtime_apriltag_fingerprint": {"observations_sha16": _sha16(runtime_csv)} if runtime_csv.is_file() else None,
            "dropout_manifest": str(paths.dropout_manifest),
            "dropout_manifest_fingerprint": {"manifest_sha16": _sha16(paths.dropout_manifest)},
            "reference_source": str(reference_csv),
            "reference_evaluation_only": True,
            "cup_source": str(cup_csv),
            "cup_evaluation_only": True,
            "candidate_uses_apriltag_during_dropout": False,
            "candidate_uses_reference": False,
            "candidate_uses_cup": False,
            "algorithm_runtime_source": "phase4_5a_frozen_continuous_run",
            "opencv_version": continuous_provenance.get("opencv_version"),
            "python_version": continuous_provenance.get("python_version"),
            "numpy_version": continuous_provenance.get("numpy_version"),
            "processing_environment": continuous_provenance.get("platform"),
            "total_processing_time_s": runtime_stats.get("total_processing_time_s"),
            "real_time_factor": runtime_stats.get("real_time_factor"),
            "session_local_availability": summary.get("availability"),
            "propagated_only_frames": summary.get("propagated_only_frames"),
            "experiment_design_note": (
                "15 windows are nested durations from the same Scenario A session; "
                "do not treat per-window PASS counts as independent trial statistics. "
                "Stereo+IMU VIO-lite pipeline feasibility is evaluated as a whole; "
                "propagated-only=0 in the frozen run so IMU-only continuity was not exercised."
            ),
        }
    )


def run_stereo_imu_vio_evaluation(
    *,
    config: StereoImuVioEvaluationConfig,
) -> StereoImuVioEvaluationRunResult:
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
    rgbd_by_window = (
        load_baseline_window_results(paths.rgbd_results_dir)
        if paths.rgbd_results_dir is not None
        else {}
    )

    window_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    alignment_failures = 0
    evaluated = 0
    cup2_applicable = 0
    cup2_not_applicable = 0

    for window in windows:
        generation, evaluation, extras = evaluate_stereo_imu_vio_window(
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
        comparison_rows.append(
            build_three_way_comparison_row(
                window_id=window.window_id,
                vio_evaluation=evaluation,
                hold_evaluation=hold_by_window.get(window.window_id),
                rgbd_evaluation=rgbd_by_window.get(window.window_id),
            )
        )

    summary = StereoImuVioEvaluationSummary(
        algorithm_id=STEREO_IMU_VIO_LITE_ALGORITHM_ID,
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
    return StereoImuVioEvaluationRunResult(
        summary=summary,
        window_results=window_results,
        provenance=provenance,
        comparison=comparison_rows if comparison_rows else None,
    )


def write_stereo_imu_vio_evaluation_results(
    *,
    output_dir: Path,
    run_result: StereoImuVioEvaluationRunResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", sanitize_json_value(run_result.summary.as_dict()))
    write_json(output_dir / "window_results.json", sanitize_json_value({"windows": run_result.window_results}))
    write_json(output_dir / "provenance.json", run_result.provenance)
    if run_result.comparison is not None:
        write_json(output_dir / "comparison.json", sanitize_json_value({"windows": run_result.comparison}))


def run_stereo_imu_vio_evaluation_from_paths(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> StereoImuVioEvaluationRunResult:
    config = load_stereo_imu_vio_evaluation_config(config_path, repo_root=repo_root)
    result = run_stereo_imu_vio_evaluation(config=config)
    target = output_dir or config.paths.output_dir
    write_stereo_imu_vio_evaluation_results(output_dir=target, run_result=result)
    return result
