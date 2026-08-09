"""Run HOLD_LAST_POSE baseline evaluation for Phase 3 dropout windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dropout_evaluation.hold_last_pose_runner import run_hold_last_pose_from_paths  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HOLD_LAST_POSE baseline on Phase 3 dropout windows.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml",
        help="Phase 3 dropout protocol config path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Dropout windows manifest path.",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Dataset session directory (defaults to config session path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = args.config
    manifest_path = args.manifest
    if manifest_path is None:
        from dropout_evaluation.dropout_protocol import load_dropout_protocol_config

        config = load_dropout_protocol_config(config_path)
        manifest_path = (
            ROOT
            / config.output_root
            / config.session_id
            / "dropout_windows.json"
        )

    output_dir = args.output
    if output_dir is None:
        from dropout_evaluation.dropout_protocol import load_dropout_protocol_config

        config = load_dropout_protocol_config(config_path)
        output_dir = (
            ROOT
            / config.output_root
            / config.session_id
            / "hold_last_pose"
        )

    result = run_hold_last_pose_from_paths(
        config_path=config_path,
        manifest_path=manifest_path,
        session_dir=args.session,
        output_dir=output_dir,
    )
    print(
        f"HOLD_LAST_POSE evaluated {result.summary.windows_evaluated}/"
        f"{result.summary.windows_total} windows -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
