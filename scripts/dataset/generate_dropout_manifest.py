"""Generate Phase 3 dropout window manifest (read-only on official session)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    generate_dropout_windows,
    load_dropout_protocol_config,
    load_frame_timestamps_from_reference_csv,
    resolve_cup2_first_appearance_for_config,
    write_dropout_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Phase 3 dropout window manifest from anchor timestamps.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml",
        help="Phase 3 dropout protocol config path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest output path. Defaults to <output.root>/<session_id>/dropout_windows.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_dropout_protocol_config(args.config)
    session_dir = ROOT / config.session_path
    reference_csv = session_dir / config.reference_source
    frames = load_frame_timestamps_from_reference_csv(reference_csv)
    cup2_first_appearance = resolve_cup2_first_appearance_for_config(config, session_dir)
    windows = generate_dropout_windows(
        config,
        frames,
        cup2_first_appearance=cup2_first_appearance,
    )

    if args.output is not None:
        output_path = args.output
    else:
        output_path = (
            ROOT
            / config.output_root
            / config.session_id
            / "dropout_windows.json"
        )

    write_dropout_manifest(
        config=config,
        windows=windows,
        output_path=output_path,
        cup2_first_appearance=cup2_first_appearance,
    )
    print(f"Wrote {len(windows)} dropout windows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
