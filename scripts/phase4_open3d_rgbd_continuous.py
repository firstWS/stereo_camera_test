"""Scenario A full continuous Open3D RGB-D odometry analysis run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.reader import DatasetReader  # noqa: E402
from dataset_recorder.rgb_depth_geometry import load_rgb_depth_calibration  # noqa: E402
from dropout_evaluation.rgbd_odometry_continuous import (  # noqa: E402
    ContinuousOdometryConfig,
    OdometryFrameInput,
    build_provenance,
    run_continuous_rgbd_odometry,
    write_continuous_outputs,
)
from dropout_evaluation.rgbd_odometry import pinhole_intrinsic_from_rgb_calibration  # noqa: E402

SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
ALIGNMENT_ROOT = ROOT / "out/evaluation/phase4/20260807_161354_scenario_a/rgbd_alignment"
OUTPUT_DIR = ROOT / "out/analysis/phase4_open3d_rgbd_continuous"


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_frames() -> list[OdometryFrameInput]:
    reader = DatasetReader(SESSION)
    frames: list[OdometryFrameInput] = []
    for record in reader.iterate_rgb():
        frame_number = int(record.row.get("frame_number") or 0)
        if record.file_path is None or not record.file_path.is_file():
            raise RuntimeError(f"Missing RGB file for frame {frame_number}")
        bgr = cv2.imread(str(record.file_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read RGB frame {frame_number}")
        depth_path = ALIGNMENT_ROOT / "aligned_depth" / f"frame_{frame_number:06d}.npy"
        depth = np.load(depth_path)
        frames.append(
            OdometryFrameInput(
                frame_number=frame_number,
                device_timestamp_us=int(record.row.get("device_timestamp_us") or 0),
                rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                depth_m=depth,
            )
        )
    return frames


def main() -> int:
    if OUTPUT_DIR.exists() and (OUTPUT_DIR / "trajectory.csv").is_file():
        print(f"REFUSING: output already exists at {OUTPUT_DIR}")
        return 2

    nondeterminism = ROOT / "out/analysis/phase4_open3d_rgbd_continuous/nondeterminism_report.json"
    if nondeterminism.is_file():
        report = json.loads(nondeterminism.read_text(encoding="utf-8"))
        if not report.get("gate_ready_for_full_run", False):
            print("Nondeterminism gate not ready; refusing full run")
            return 3

    reader = DatasetReader(SESSION)
    calib = load_rgb_depth_calibration(
        reader.calibration_intrinsics(),
        reader.calibration_extrinsics(),
    )
    intrinsic = pinhole_intrinsic_from_rgb_calibration(calib)
    frames = _load_frames()
    config = ContinuousOdometryConfig()
    result = run_continuous_rgbd_odometry(frames, intrinsic, config=config)
    provenance = build_provenance(
        config=config,
        calibration_fingerprint={
            "intrinsics_sha16": _sha16(SESSION / "calibration/intrinsics.json"),
            "extrinsics_sha16": _sha16(SESSION / "calibration/extrinsics.json"),
        },
        alignment_manifest_fingerprint={
            "manifest_sha16": _sha16(ALIGNMENT_ROOT / "manifest.json"),
            "frame_index_sha16": _sha16(ALIGNMENT_ROOT / "frame_index.json"),
        },
    )
    write_continuous_outputs(OUTPUT_DIR, result, provenance)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print("OUTPUT_DIR", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
