"""Phase 4.5-A Scenario A stereo+IMU provisional VIO analysis runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.reader import DatasetReader  # noqa: E402
from dropout_evaluation.evaluation_io import load_pose_references_from_csv  # noqa: E402
from dropout_evaluation.rgbd_odometry import relative_transform_target_source, transform_magnitude  # noqa: E402
from dropout_evaluation.stereo_imu_calibration import load_stereo_imu_calibration  # noqa: E402
from dropout_evaluation.stereo_imu_vio_continuous import (  # noqa: E402
    build_provenance,
    load_imu_samples,
    load_stereo_frames,
    run_continuous_stereo_imu_vio,
    write_vio_outputs,
)
from dropout_evaluation.stereo_imu_vio_lite import StereoImuVioConfig  # noqa: E402

DEFAULT_SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
DEFAULT_OUTPUT = ROOT / "out/analysis/phase4_stereo_imu_vio_lite"


def _pose_from_row(row: dict) -> tuple[int, object] | None:
    if row.get("valid") not in (True, "True", "true", "1", 1):
        return None
    if not row.get("qw"):
        return None
    import numpy as np

    qw, qx, qy, qz = (float(row[k]) for k in ("qw", "qx", "qy", "qz"))
    tx, ty, tz = (float(row[k]) for k in ("tx", "ty", "tz"))
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return int(row["frame_number"]), T


def run_posthoc_reference(session_dir: Path, output_dir: Path) -> dict:
    import csv

    trajectory_path = output_dir / "trajectory.csv"
    ref_path = session_dir / "derived/reference/apriltag_pose_smoothed.csv"
    if not trajectory_path.is_file() or not ref_path.is_file():
        return {"available": False, "reason": "missing_trajectory_or_reference"}

    with trajectory_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    refs = {
        ref.frame_number: ref.T_world_camera
        for ref in load_pose_references_from_csv(ref_path)
        if ref.valid and ref.T_world_camera is not None
    }
    valid_rows = []
    for row in rows:
        parsed = _pose_from_row(row)
        if parsed is None:
            continue
        frame_number, T = parsed
        if frame_number in refs:
            valid_rows.append((frame_number, T))
    if len(valid_rows) < 2:
        return {"available": False, "reason": "insufficient_overlap_with_reference"}

    import numpy as np

    start_frame, T_odom_start = valid_rows[0]
    T_ref_start = refs[start_frame]
    T_align = T_ref_start @ np.linalg.inv(T_odom_start)
    trans_errors = []
    rot_errors = []
    for frame_number, T_odom in valid_rows[1:]:
        T_ref = refs[frame_number]
        T_candidate_world = T_align @ T_odom
        delta = relative_transform_target_source(T_ref, T_candidate_world)
        trans, rot = transform_magnitude(delta)
        trans_errors.append(trans)
        rot_errors.append(rot)

    return {
        "available": True,
        "aligned_frame_count": len(valid_rows),
        "translation_error_m_median": float(np.median(trans_errors)) if trans_errors else None,
        "translation_error_m_p90": float(np.quantile(trans_errors, 0.9)) if trans_errors else None,
        "rotation_error_deg_median": float(np.median(rot_errors)) if rot_errors else None,
        "rotation_error_deg_p90": float(np.quantile(rot_errors, 0.9)) if rot_errors else None,
        "rotation_error_deg_max": float(np.max(rot_errors)) if rot_errors else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.5-A stereo+IMU provisional VIO")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run frames 1-120 subset smoke")
    parser.add_argument("--posthoc-reference", action="store_true")
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
        frames = frames[:120]
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

    config = StereoImuVioConfig()
    result = run_continuous_stereo_imu_vio(
        frames,
        imu_samples,
        calib,
        config=config,
        pairing_summary=pairing_summary,
    )
    frame_range = (frames[0].frame_number, frames[-1].frame_number)
    provenance = build_provenance(
        config=config,
        calib=calib,
        session_dir=args.session,
        frame_range=frame_range,
    )
    write_vio_outputs(args.output, result, provenance)

    if args.posthoc_reference:
        posthoc = run_posthoc_reference(args.session, args.output)
        diag_path = args.output / "diagnostics.json"
        diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
        diagnostics["posthoc_reference"] = posthoc
        diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"summary": result.summary, "pairing_summary": result.pairing_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
