"""Phase 4.7-A Scenario A stereo+IMU SLAM-lite analysis runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.reader import DatasetReader  # noqa: E402
from dropout_evaluation.stereo_imu_calibration import load_stereo_imu_calibration  # noqa: E402
from dropout_evaluation.stereo_imu_slam_continuous import (  # noqa: E402
    build_slam_provenance,
    default_slam_config,
    load_imu_samples,
    load_stereo_frames,
    run_continuous_stereo_imu_slam,
    write_slam_outputs,
)

DEFAULT_SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
DEFAULT_OUTPUT = ROOT / "out/analysis/phase47_stereo_imu_slam_lite"
CHECKPOINT_FRAMES = (81, 202, 248)


def _checkpoint_pose_availability(trajectory_path: Path) -> dict[str, bool]:
    if not trajectory_path.is_file():
        return {f"frame_{n}": False for n in CHECKPOINT_FRAMES}
    with trajectory_path.open(encoding="utf-8", newline="") as handle:
        rows = {int(row["frame_number"]): row for row in csv.DictReader(handle)}
    out: dict[str, bool] = {}
    for frame_number in CHECKPOINT_FRAMES:
        row = rows.get(frame_number)
        out[f"frame_{frame_number}"] = bool(
            row
            and row.get("valid") in (True, "True", "true", "1", 1)
            and row.get("tx") not in (None, "", "None")
        )
    return out


def evaluate_gate(summary: dict, diagnostics: dict, checkpoints: dict[str, bool]) -> str:
    if summary.get("total_frames", 0) < 400:
        return "SLAM_LITE_CORE_NOT_READY"
    if summary.get("invalid_frames", 0) > summary.get("total_frames", 1):
        return "SLAM_LITE_CORE_NOT_READY"
    if summary.get("keyframes_created", 0) <= 1:
        return "SLAM_LITE_CORE_NOT_READY"
    if summary.get("final_map_points", 0) <= 0:
        return "SLAM_LITE_CORE_NOT_READY"
    if summary.get("map_update_successes", 0) <= 0:
        return "SLAM_LITE_CORE_NOT_READY"
    if summary.get("map_based_pose_update_count", 0) <= 0:
        return "SLAM_LITE_CORE_NOT_READY"
    if diagnostics.get("catastrophic_jump_count", 0) > 0:
        return "SLAM_LITE_CORE_NOT_READY"
    if not all(checkpoints.values()):
        return "SLAM_LITE_CORE_NOT_READY"
    if summary.get("valid_frames", 0) < summary.get("total_frames", 0) * 0.5:
        return "SLAM_LITE_CORE_READY_WITH_WARNING"
    return "SLAM_LITE_CORE_READY"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.7-A stereo+IMU SLAM-lite")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run first 60 canonical frames")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and (args.output / "trajectory.csv").is_file() and not args.force:
        print(f"REFUSING: output already exists at {args.output}")
        return 2

    reader = DatasetReader(args.session)
    calib = load_stereo_imu_calibration(
        reader.calibration_intrinsics(),
        reader.calibration_extrinsics(),
    )
    frames, pairing_summary = load_stereo_frames(reader)
    imu_samples = load_imu_samples(reader)

    if args.smoke:
        frames = frames[:60]
        pairing_summary = {**pairing_summary, "smoke_subset": True, "smoke_frame_count": len(frames)}
    else:
        start_frame = args.start_frame
        end_frame = args.end_frame
        if start_frame is not None or end_frame is not None:
            lo = start_frame if start_frame is not None else frames[0].frame_number
            hi = end_frame if end_frame is not None else frames[-1].frame_number
            frames = [f for f in frames if lo <= f.frame_number <= hi]

    if not frames:
        print("No stereo frames to process")
        return 1

    config = default_slam_config()
    result = run_continuous_stereo_imu_slam(
        frames,
        imu_samples,
        calib,
        config=config,
        pairing_summary=pairing_summary,
    )
    frame_range = (frames[0].frame_number, frames[-1].frame_number)
    provenance = build_slam_provenance(
        config=config,
        calib=calib,
        session_dir=args.session,
        frame_range=frame_range,
    )
    write_slam_outputs(args.output, result, provenance)

    checkpoints = _checkpoint_pose_availability(args.output / "trajectory.csv")
    gate = evaluate_gate(result.summary, result.diagnostics, checkpoints)
    gate_payload = {
        "gate": gate,
        "checkpoints": checkpoints,
        "summary": result.summary,
        "diagnostics": {
            "catastrophic_jump_count": result.diagnostics.get("catastrophic_jump_count"),
            "pose_source_histogram": result.diagnostics.get("pose_source_histogram"),
        },
        "runtime_stats": result.runtime_stats,
    }
    (args.output / "gate.json").write_text(json.dumps(gate_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(gate_payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
