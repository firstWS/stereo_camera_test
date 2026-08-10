"""Official Phase 4.5-B stereo+IMU VIO-lite Scenario A evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.stereo_imu_vio_evaluation import (  # noqa: E402
    load_stereo_imu_vio_evaluation_config,
    run_stereo_imu_vio_evaluation_from_paths,
)

DEFAULT_CONFIG = ROOT / "configs/evaluation/phase45_stereo_imu_vio_scenario_a.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.5-B official VIO-lite evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_stereo_imu_vio_evaluation_config(args.config, repo_root=ROOT)
    output_dir = args.output or config.paths.output_dir
    if output_dir.exists() and (output_dir / "summary.json").is_file() and not args.force:
        print(f"REFUSING: output already exists at {output_dir}")
        return 2

    result = run_stereo_imu_vio_evaluation_from_paths(
        config_path=args.config,
        output_dir=output_dir,
        repo_root=ROOT,
    )
    print(json.dumps(result.summary.as_dict(), indent=2))
    return 0 if result.summary.alignment_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
