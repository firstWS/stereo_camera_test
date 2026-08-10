"""Runner for Phase 4.2-C RGB-D odometry Phase 3 harness adapter smoke."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_recorder.session_metadata import write_json

from .dropout_protocol import DropoutProtocolConfig, load_dropout_protocol_config, load_frame_timestamps_from_rgb_index
from .hold_last_pose_runner import load_dropout_windows_from_manifest
from .rgbd_odometry_adapter import (
    RgbdAdapterStatus,
    RgbdAdapterWindowResult,
    RgbdOdometryAdapterConfig,
    generate_rgbd_odometry_candidates,
    load_local_trajectory_from_csv,
)
from .runtime_apriltag import load_runtime_apriltag_poses_from_session


@dataclass
class RgbdAdapterRunSummary:
    algorithm_id: str
    session_id: str
    protocol_manifest: str
    continuous_trajectory: str
    windows_total: int
    windows_generated: int
    alignment_failures: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "session_id": self.session_id,
            "protocol_manifest": self.protocol_manifest,
            "continuous_trajectory": self.continuous_trajectory,
            "windows_total": self.windows_total,
            "windows_generated": self.windows_generated,
            "alignment_failures": self.alignment_failures,
        }


@dataclass
class RgbdAdapterRunResult:
    summary: RgbdAdapterRunSummary
    window_diagnostics: list[dict[str, Any]]
    provenance: list[dict[str, Any]]


def run_rgbd_adapter_smoke(
    *,
    config: DropoutProtocolConfig,
    manifest_path: Path,
    session_dir: Path,
    continuous_trajectory_csv: Path,
    adapter_config: RgbdOdometryAdapterConfig | None = None,
) -> RgbdAdapterRunResult:
    """Generate candidate trajectories for all dropout windows (no metric evaluation)."""
    adapter_cfg = adapter_config or RgbdOdometryAdapterConfig()
    windows = load_dropout_windows_from_manifest(manifest_path)
    local_trajectory = load_local_trajectory_from_csv(continuous_trajectory_csv)
    runtime_poses = load_runtime_apriltag_poses_from_session(session_dir)
    frame_timestamps = load_frame_timestamps_from_rgb_index(session_dir)

    diagnostics: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    generated = 0
    alignment_failures = 0

    for window in windows:
        result = generate_rgbd_odometry_candidates(
            window=window,
            local_trajectory=local_trajectory,
            runtime_poses=runtime_poses,
            frame_timestamps=frame_timestamps,
            config=adapter_cfg,
        )
        provenance_rows.append(result.provenance.as_dict())
        diagnostics.append(result.diagnostic.as_dict())
        if result.status == RgbdAdapterStatus.OK:
            generated += 1
        else:
            alignment_failures += 1

    summary = RgbdAdapterRunSummary(
        algorithm_id=adapter_cfg.algorithm_id,
        session_id=config.session_id,
        protocol_manifest=str(manifest_path),
        continuous_trajectory=str(continuous_trajectory_csv),
        windows_total=len(windows),
        windows_generated=generated,
        alignment_failures=alignment_failures,
    )
    return RgbdAdapterRunResult(
        summary=summary,
        window_diagnostics=diagnostics,
        provenance=provenance_rows,
    )


def write_rgbd_adapter_smoke_outputs(
    *,
    output_dir: Path,
    run_result: RgbdAdapterRunResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", run_result.summary.as_dict())
    write_json(output_dir / "window_diagnostics.json", {"windows": run_result.window_diagnostics})
    write_json(output_dir / "provenance.json", {"windows": run_result.provenance})
    (output_dir / "README.txt").write_text(
        "Phase 4.2-C adapter smoke output (not an official Phase 4 benchmark artifact).\n",
        encoding="utf-8",
    )


def run_rgbd_adapter_smoke_from_paths(
    *,
    config_path: Path,
    manifest_path: Path,
    session_dir: Path | None = None,
    continuous_trajectory_csv: Path,
    output_dir: Path | None = None,
    adapter_config: RgbdOdometryAdapterConfig | None = None,
) -> RgbdAdapterRunResult:
    config = load_dropout_protocol_config(config_path)
    resolved_session = session_dir or (config_path.parents[2] / config.session_path)
    result = run_rgbd_adapter_smoke(
        config=config,
        manifest_path=manifest_path,
        session_dir=resolved_session,
        continuous_trajectory_csv=continuous_trajectory_csv,
        adapter_config=adapter_config,
    )
    if output_dir is not None:
        write_rgbd_adapter_smoke_outputs(output_dir=output_dir, run_result=result)
    return result
