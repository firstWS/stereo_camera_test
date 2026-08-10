"""Run official Phase 4 Open3D RGB-D odometry evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dropout_evaluation.open3d_rgbd_evaluation import (  # noqa: E402
    load_open3d_rgbd_evaluation_config,
    run_open3d_rgbd_evaluation_from_paths,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run official Phase 4 Open3D RGB-D odometry evaluation on Phase 3 dropout windows. "
            "Uses frozen continuous trajectory (no Open3D re-run)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/evaluation/phase4_open3d_rgbd_scenario_a.yaml",
        help="Phase 4 official evaluation config path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for official evaluation artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_open3d_rgbd_evaluation_config(args.config, repo_root=ROOT)
    output_dir = args.output or config.paths.output_dir

    result = run_open3d_rgbd_evaluation_from_paths(
        config_path=args.config,
        output_dir=output_dir,
        repo_root=ROOT,
    )
    print(
        f"open3d_hybrid_rgbd evaluated {result.summary.windows_evaluated}/"
        f"{result.summary.windows_total} windows -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
