"""Phase 4.5-M2 tag-masked stereo VIO C5s proof runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.reader import DatasetReader  # noqa: E402
from dataset_recorder.session_metadata import write_json  # noqa: E402
from dropout_evaluation.ir_tag_mask import (  # noqa: E402
    DEFAULT_MASK_MARGIN_PX,
    DEFAULT_TAG_DICTIONARY,
    load_rgb_gray_by_canonical_frame,
    load_rgb_ir_mask_calibration,
)
from dropout_evaluation.phase45_vio_tag_mask_demo import generate_tag_mask_demo_video  # noqa: E402
from dropout_evaluation.stereo_imu_calibration import load_stereo_imu_calibration  # noqa: E402
from dropout_evaluation.stereo_imu_vio_continuous import (  # noqa: E402
    build_provenance,
    load_imu_samples,
    load_stereo_frames,
)
from dropout_evaluation.stereo_imu_vio_lite import StereoImuVioConfig  # noqa: E402
from dropout_evaluation.stereo_imu_vio_tag_mask import (  # noqa: E402
    TAG_MASK_ALGORITHM_DIR,
    TAG_MASK_ANALYSIS_DIRNAME,
    build_tag_mask_provenance,
    build_tag_mask_validation_summary,
    classify_tag_mask_proof_gate,
    evaluate_tag_mask_c5s_window,
    load_baseline_c5s_snapshot,
    run_tag_masked_vio,
    write_tag_masked_vio_outputs,
)

DEFAULT_SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
DEFAULT_ANALYSIS = ROOT / "out/analysis" / TAG_MASK_ANALYSIS_DIRNAME
DEFAULT_EVAL = ROOT / "out/evaluation/phase4/20260807_161354_scenario_a" / TAG_MASK_ALGORITHM_DIR
DEFAULT_DEMO = ROOT / "out/demo/phase45_vio_tag_mask/scenario_a_vio_tag_mask_c5s_demo.mp4"
DEFAULT_MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"
DEFAULT_PROTOCOL = ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml"
BASELINE_EVAL = ROOT / "out/evaluation/phase4/20260807_161354_scenario_a/stereo_imu_vio_lite/window_results.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.5-M2 tag-masked VIO C5s proof")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--demo-output", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--margin-px", type=int, default=DEFAULT_MASK_MARGIN_PX)
    parser.add_argument("--dictionary", type=str, default=DEFAULT_TAG_DICTIONARY)
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.analysis_output.exists() and (args.analysis_output / "trajectory.csv").is_file() and not args.force:
        print(f"REFUSING: analysis output already exists at {args.analysis_output}")
        return 2

    reader = DatasetReader(args.session)
    calib = load_stereo_imu_calibration(
        reader.calibration_intrinsics(),
        reader.calibration_extrinsics(),
    )
    frames, pairing_summary = load_stereo_frames(reader)
    imu_samples = load_imu_samples(reader)
    rgb_gray_by_frame = load_rgb_gray_by_canonical_frame(args.session)
    mask_calib = load_rgb_ir_mask_calibration(
        reader.calibration_intrinsics(),
        reader.calibration_extrinsics(),
    )
    if len(frames) != 436:
        print(f"Expected 436 frames, got {len(frames)}")
        return 1

    config = StereoImuVioConfig()
    result, mask_diagnostics, mask_summary = run_tag_masked_vio(
        frames=frames,
        imu_samples=imu_samples,
        calib=calib,
        config=config,
        pairing_summary=pairing_summary,
        margin_px=args.margin_px,
        dictionary=args.dictionary,
        rgb_gray_by_frame=rgb_gray_by_frame,
        mask_calib=mask_calib,
    )
    provenance = build_tag_mask_provenance(
        base_provenance=build_provenance(
            config=config,
            calib=calib,
            session_dir=args.session,
            frame_range=(frames[0].frame_number, frames[-1].frame_number),
        ),
        mask_summary=mask_summary,
        margin_px=args.margin_px,
        dictionary=args.dictionary,
    )
    write_tag_masked_vio_outputs(args.analysis_output, result, provenance, mask_diagnostics)

    eval_row, tag_masked_metrics = evaluate_tag_mask_c5s_window(
        session_dir=args.session,
        manifest_path=args.manifest,
        trajectory_csv=args.analysis_output / "trajectory.csv",
        protocol_config_path=args.protocol_config,
    )
    baseline_metrics = load_baseline_c5s_snapshot(BASELINE_EVAL)
    args.evaluation_output.mkdir(parents=True, exist_ok=True)
    write_json(args.evaluation_output / "window_results.json", {"windows": [eval_row]})
    write_json(
        args.evaluation_output / "summary.json",
        {
            "algorithm_id": provenance["algorithm_id"],
            "session_id": args.session.name,
            "windows_total": 1,
            "windows_evaluated": 1,
            "window_id": eval_row["window_id"],
            "metrics": tag_masked_metrics,
            "baseline_unmasked_c5": baseline_metrics,
        },
    )
    write_json(
        args.evaluation_output / "provenance.json",
        {
            **provenance,
            "evaluation_scope": "single_window_c5s_only",
            "baseline_window_results": str(BASELINE_EVAL),
        },
    )

    demo_summary: dict = {"skipped": True}
    if not args.skip_demo:
        if args.demo_output.is_file() and not args.force:
            print(f"REFUSING: demo already exists at {args.demo_output}")
            return 2
        demo_summary = generate_tag_mask_demo_video(
            session_dir=args.session,
            trajectory_csv=args.analysis_output / "trajectory.csv",
            manifest_path=args.manifest,
            mask_diagnostics_json=args.analysis_output / "tag_mask_diagnostics.json",
            output_mp4=args.demo_output,
        )
        write_json(args.demo_output.parent / "demo_summary.json", demo_summary)

    trajectory_summary = json.loads((args.analysis_output / "summary.json").read_text(encoding="utf-8"))
    validation_summary = build_tag_mask_validation_summary(
        session_dir=args.session,
        analysis_dir=args.analysis_output,
        evaluation_dir=args.evaluation_output,
        demo_summary=demo_summary,
        mask_summary=mask_summary,
        trajectory_summary=trajectory_summary,
        tag_masked_metrics=tag_masked_metrics,
        baseline_metrics=baseline_metrics,
    )
    gate = classify_tag_mask_proof_gate(validation_summary)
    validation_summary["gate"] = gate
    write_json(args.demo_output.parent / "tag_mask_validation_summary.json", validation_summary)

    print(json.dumps({"gate": gate, "validation_summary": validation_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
