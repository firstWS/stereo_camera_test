"""Generate Phase 4.5 Application World final demo (frozen M2 artifacts only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.phase45_vio_evidence_demo import generate_evidence_demo_video  # noqa: E402

DEFAULT_SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
DEFAULT_TRAJECTORY = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/trajectory.csv"
DEFAULT_MASK_DIAG = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/tag_mask_diagnostics.json"
DEFAULT_EVAL = ROOT / "out/evaluation/phase4/20260807_161354_scenario_a/stereo_imu_vio_lite_tag_mask_c5s"
DEFAULT_VALIDATION = ROOT / "out/demo/phase45_vio_tag_mask/tag_mask_validation_summary.json"
DEFAULT_MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"
DEFAULT_OUTPUT_DIR = ROOT / "out/demo/phase45_vio_application_world_final"
DEFAULT_OUTPUT_MP4 = DEFAULT_OUTPUT_DIR / "scenario_a_vio_application_world_final.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.5 Application World final demo")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--mask-diagnostics", type=Path, default=DEFAULT_MASK_DIAG)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MP4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.is_file() and not args.force:
        print(f"REFUSING: output already exists at {args.output}")
        return 2

    summary = generate_evidence_demo_video(
        session_dir=args.session,
        trajectory_csv=args.trajectory,
        manifest_path=args.manifest,
        mask_diagnostics_json=args.mask_diagnostics,
        evaluation_dir=args.evaluation_dir,
        validation_summary_json=args.validation_summary,
        output_mp4=args.output,
        presentation_version="application_world_final",
        show_world_top_view=False,
        highlight_tag_dropout_red=True,
        use_application_world=True,
        fps=30.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
