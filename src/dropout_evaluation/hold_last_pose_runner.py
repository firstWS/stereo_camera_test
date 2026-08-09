"""Runner for HOLD_LAST_POSE baseline evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
from .evaluation_metrics import CupObservation, EvaluationResult, PoseReference, RecoveryTiming, evaluate_window
from .hold_last_pose import (
    HOLD_LAST_POSE_ALGORITHM_ID,
    HoldLastPoseConfig,
    HoldLastPoseStatus,
    HoldLastPoseWindowResult,
    generate_hold_last_pose_candidates,
)
from .runtime_apriltag import RuntimeAprilTagPose, load_runtime_apriltag_poses_from_session


@dataclass
class HoldLastPoseRunSummary:
    algorithm_id: str
    session_id: str
    protocol_manifest: str
    windows_total: int
    windows_evaluated: int
    anchor_failures: int
    recovery_failures: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "session_id": self.session_id,
            "protocol_manifest": self.protocol_manifest,
            "windows_total": self.windows_total,
            "windows_evaluated": self.windows_evaluated,
            "anchor_failures": self.anchor_failures,
            "recovery_failures": self.recovery_failures,
        }


@dataclass
class HoldLastPoseRunResult:
    summary: HoldLastPoseRunSummary
    window_results: list[dict[str, Any]]
    provenance: list[dict[str, Any]]


def load_dropout_windows_from_manifest(manifest_path: Path) -> list[DropoutWindow]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows: list[DropoutWindow] = []
    for raw in payload["windows"]:
        windows.append(
            DropoutWindow(
                window_id=str(raw["window_id"]),
                anchor_id=str(raw["anchor_id"]),
                session_id=str(raw["session_id"]),
                start_frame=int(raw["start_frame"]),
                start_device_timestamp_us=int(raw["start_device_timestamp_us"]),
                boundary_timestamp_us=int(raw["boundary_timestamp_us"]),
                end_frame=int(raw["end_frame"]),
                end_device_timestamp_us=int(raw["end_device_timestamp_us"]),
                recovery_frame=raw.get("recovery_frame"),
                recovery_device_timestamp_us=raw.get("recovery_device_timestamp_us"),
                target_duration_sec=float(raw["target_duration_sec"]),
                masked_sample_span_sec=float(raw["masked_sample_span_sec"]),
                frame_count=int(raw["frame_count"]),
                motion_class=str(raw["motion_class"]),
                includes_cup2_appearance=bool(raw["includes_cup2_appearance"]),
                anchor_convention=str(raw["anchor_convention"]),
            )
        )
    return windows


def evaluate_hold_last_pose_window(
    *,
    window: DropoutWindow,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    references: Sequence[PoseReference],
    cup_observations: Sequence[CupObservation],
    thresholds: SuccessThresholds,
    config: HoldLastPoseConfig | None = None,
) -> tuple[HoldLastPoseWindowResult, EvaluationResult | None]:
    generation = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
        config=config,
    )
    if generation.status != HoldLastPoseStatus.OK:
        return generation, None
    actual_ts: int | None = None
    if generation.provenance.recovery_actual_frame is not None:
        for candidate in generation.candidates:
            if candidate.frame_number == generation.provenance.recovery_actual_frame:
                actual_ts = candidate.device_timestamp_us
                break
    recovery_timing = RecoveryTiming(
        recovery_requested_frame=generation.provenance.recovery_requested_frame,
        recovery_requested_device_timestamp_us=window.recovery_device_timestamp_us,
        recovery_actual_frame=generation.provenance.recovery_actual_frame,
        recovery_actual_device_timestamp_us=actual_ts,
    )
    evaluation = evaluate_window(
        window=window,
        references=references,
        candidates=generation.candidates,
        observations=cup_observations,
        algorithm_id=HOLD_LAST_POSE_ALGORITHM_ID,
        thresholds=thresholds,
        cup2_semantic_id="cup2",
        recovery_timing=recovery_timing,
    )
    return generation, evaluation


def run_hold_last_pose_evaluation(
    *,
    config: DropoutProtocolConfig,
    manifest_path: Path,
    session_dir: Path,
    hold_config: HoldLastPoseConfig | None = None,
) -> HoldLastPoseRunResult:
    """Evaluate HOLD_LAST_POSE across manifest windows (candidate generation is reference-free)."""
    hold_cfg = hold_config or HoldLastPoseConfig()
    windows = load_dropout_windows_from_manifest(manifest_path)
    runtime_poses = load_runtime_apriltag_poses_from_session(session_dir)
    frame_timestamps = load_frame_timestamps_from_rgb_index(session_dir)

    references = load_pose_references_from_csv(session_dir / config.reference_source)
    cup_observations = load_cup_observations_from_csv(
        session_dir / config.cup2_observations_csv,
    )

    window_results: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    anchor_failures = 0
    recovery_failures = 0
    evaluated = 0

    for window in windows:
        generation, evaluation = evaluate_hold_last_pose_window(
            window=window,
            runtime_poses=runtime_poses,
            frame_timestamps=frame_timestamps,
            references=references,
            cup_observations=cup_observations,
            thresholds=config.success_thresholds,
            config=hold_cfg,
        )
        provenance_rows.append(generation.provenance.as_dict())
        if generation.status != HoldLastPoseStatus.OK:
            anchor_failures += 1
            window_results.append(
                {
                    "window_id": window.window_id,
                    "status": generation.status.value,
                    "evaluation": None,
                }
            )
            continue
        if generation.provenance.recovery_actual_frame is None:
            recovery_failures += 1
        evaluated += 1
        assert evaluation is not None
        window_results.append(
            {
                "window_id": window.window_id,
                "status": generation.status.value,
                "evaluation": evaluation.as_dict(),
            }
        )

    summary = HoldLastPoseRunSummary(
        algorithm_id=hold_cfg.algorithm_id,
        session_id=config.session_id,
        protocol_manifest=str(manifest_path),
        windows_total=len(windows),
        windows_evaluated=evaluated,
        anchor_failures=anchor_failures,
        recovery_failures=recovery_failures,
    )
    return HoldLastPoseRunResult(
        summary=summary,
        window_results=window_results,
        provenance=provenance_rows,
    )


def write_hold_last_pose_results(
    *,
    output_dir: Path,
    run_result: HoldLastPoseRunResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", run_result.summary.as_dict())
    write_json(output_dir / "window_results.json", {"windows": run_result.window_results})
    write_json(output_dir / "provenance.json", {"windows": run_result.provenance})


def run_hold_last_pose_from_paths(
    *,
    config_path: Path,
    manifest_path: Path,
    session_dir: Path | None = None,
    output_dir: Path | None = None,
    hold_config: HoldLastPoseConfig | None = None,
) -> HoldLastPoseRunResult:
    config = load_dropout_protocol_config(config_path)
    resolved_session = session_dir or (config_path.parents[2] / config.session_path)
    result = run_hold_last_pose_evaluation(
        config=config,
        manifest_path=manifest_path,
        session_dir=resolved_session,
        hold_config=hold_config,
    )
    if output_dir is not None:
        write_hold_last_pose_results(output_dir=output_dir, run_result=result)
    return result
