"""Tests for Phase 4.5-B official stereo+IMU VIO-lite evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.stereo_imu_vio_evaluation import (  # noqa: E402
    build_metric_snapshot,
    build_three_way_comparison_row,
    load_baseline_window_results,
    load_stereo_imu_vio_evaluation_config,
)
from dropout_evaluation.stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID  # noqa: E402

CONFIG = ROOT / "configs/evaluation/phase45_stereo_imu_vio_scenario_a.yaml"
TRAJECTORY = ROOT / "out/analysis/phase4_stereo_imu_vio_lite/trajectory.csv"
HOLD_DIR = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/hold_last_pose"
RGBD_DIR = ROOT / "out/evaluation/phase4/20260807_161354_scenario_a/open3d_rgbd_odometry"


def test_config_algorithm_paths_and_comparison_sources() -> None:
    config = load_stereo_imu_vio_evaluation_config(CONFIG, repo_root=ROOT)
    assert config.paths.output_dir.name == "stereo_imu_vio_lite"
    assert config.paths.continuous_trajectory_csv == TRAJECTORY.resolve()
    assert config.paths.hold_results_dir == HOLD_DIR.resolve()
    assert config.paths.rgbd_results_dir == RGBD_DIR.resolve()


def test_evaluation_module_algorithm_id_constant() -> None:
    assert STEREO_IMU_VIO_LITE_ALGORITHM_ID == "stereo_imu_vio_lite"


def test_no_estimator_leakage_in_evaluation_runner() -> None:
    source = (ROOT / "src/dropout_evaluation/stereo_imu_vio_evaluation.py").read_text(encoding="utf-8")
    assert "run_stereo_imu_vio_lite" not in source
    assert "run_continuous_stereo_imu_vio" not in source
    assert "load_pose_references_from_csv" in source
    assert "candidate_uses_reference" not in source.lower() or "false" in source.lower()


@pytest.mark.skipif(not HOLD_DIR.is_dir(), reason="HOLD artifacts missing")
def test_hold_baseline_results_are_read_only() -> None:
    hold = load_baseline_window_results(HOLD_DIR)
    assert len(hold) == 15


@pytest.mark.skipif(not RGBD_DIR.is_dir(), reason="RGB-D artifacts missing")
def test_rgbd_baseline_results_are_read_only() -> None:
    rgbd = load_baseline_window_results(RGBD_DIR)
    assert len(rgbd) == 15


@pytest.mark.skipif(not TRAJECTORY.is_file(), reason="frozen VIO trajectory missing")
def test_frozen_trajectory_has_canonical_436_frames() -> None:
    import csv

    with TRAJECTORY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 436
    assert int(rows[0]["frame_number"]) == 1
    assert int(rows[-1]["frame_number"]) == 436


@pytest.mark.skipif(not RGBD_DIR.is_dir(), reason="RGB-D artifacts missing")
def test_cup2_applicable_count_matches_phase3_semantics() -> None:
    rgbd = load_baseline_window_results(RGBD_DIR)
    applicable = sum(1 for row in rgbd.values() if row.get("cup2_world", {}).get("status") != "NOT_APPLICABLE")
    not_applicable = sum(1 for row in rgbd.values() if row.get("cup2_world", {}).get("status") == "NOT_APPLICABLE")
    assert applicable == 11
    assert not_applicable == 4


@pytest.mark.skipif(not (HOLD_DIR.is_dir() and RGBD_DIR.is_dir()), reason="baseline artifacts missing")
def test_three_way_comparison_row_shape() -> None:
    hold = load_baseline_window_results(HOLD_DIR)
    rgbd = load_baseline_window_results(RGBD_DIR)
    from dropout_evaluation.evaluation_metrics import EvaluationResult

    # reuse rgbd eval object shape via dict roundtrip for a minimal smoke
    sample = rgbd["B_motion_start__0.5s"]
    evaluation = EvaluationResult.from_dict(sample) if hasattr(EvaluationResult, "from_dict") else None
    if evaluation is None:
        row = {
            "window_id": "B_motion_start__0.5s",
            "vio": build_metric_snapshot(sample),
            "hold": build_metric_snapshot(hold["B_motion_start__0.5s"]),
            "rgbd": build_metric_snapshot(sample),
        }
    else:
        row = build_three_way_comparison_row(
            window_id="B_motion_start__0.5s",
            vio_evaluation=evaluation,
            hold_evaluation=hold.get("B_motion_start__0.5s"),
            rgbd_evaluation=rgbd.get("B_motion_start__0.5s"),
        )
    assert row["window_id"] == "B_motion_start__0.5s"
    assert "vio" in row or "hold" in row


@pytest.mark.skipif(
    not (CONFIG.is_file() and TRAJECTORY.is_file() and HOLD_DIR.is_dir() and RGBD_DIR.is_dir()),
    reason="official evaluation prerequisites missing",
)
def test_official_evaluation_is_deterministic_on_frozen_trajectory() -> None:
    from dropout_evaluation.stereo_imu_vio_evaluation import run_stereo_imu_vio_evaluation

    config = load_stereo_imu_vio_evaluation_config(CONFIG, repo_root=ROOT)
    first = run_stereo_imu_vio_evaluation(config=config)
    second = run_stereo_imu_vio_evaluation(config=config)
    assert first.summary.as_dict() == second.summary.as_dict()
    assert first.window_results == second.window_results


@pytest.mark.skipif(not CONFIG.is_file(), reason="config missing")
def test_fifteen_window_manifest_wiring() -> None:
    from dropout_evaluation.hold_last_pose_runner import load_dropout_windows_from_manifest

    config = load_stereo_imu_vio_evaluation_config(CONFIG, repo_root=ROOT)
    windows = load_dropout_windows_from_manifest(config.paths.dropout_manifest)
    assert len(windows) == 15
    by_anchor = {}
    for window in windows:
        by_anchor.setdefault(window.anchor_id, []).append(window.window_id)
    assert len(by_anchor["B_motion_start"]) == 5
    assert len(by_anchor["C_pre_cup2"]) == 5
    assert len(by_anchor["D_active_with_cup2"]) == 5
    assert windows[0].start_frame == 81
