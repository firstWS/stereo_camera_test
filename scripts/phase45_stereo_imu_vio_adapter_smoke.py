"""Phase 4.5-B VIO adapter smoke: 15-window candidate generation (no official benchmark)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.stereo_imu_vio_adapter_runner import run_stereo_imu_vio_adapter_smoke_from_paths  # noqa: E402

CONFIG = ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml"
MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"
SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
CONTINUOUS = ROOT / "out/analysis/phase4_stereo_imu_vio_lite/trajectory.csv"
OUTPUT = ROOT / "out/analysis/phase4_stereo_imu_vio_adapter"


def main() -> int:
    if not CONTINUOUS.is_file():
        print(f"Continuous trajectory not found: {CONTINUOUS}")
        return 1
    if not MANIFEST.is_file():
        print(f"Dropout manifest not found: {MANIFEST}")
        return 1

    result = run_stereo_imu_vio_adapter_smoke_from_paths(
        config_path=CONFIG,
        manifest_path=MANIFEST,
        session_dir=SESSION,
        continuous_trajectory_csv=CONTINUOUS,
        output_dir=OUTPUT,
    )
    print(json.dumps(result.summary.as_dict(), indent=2))
    return 0 if result.summary.windows_generated == result.summary.windows_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
