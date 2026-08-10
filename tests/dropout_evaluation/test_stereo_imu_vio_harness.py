"""Tests for stereo pairing, canonical frames, and VIO harness adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.canonical_frames import load_canonical_frames_from_rgb_index  # noqa: E402
from dropout_evaluation.stereo_pairing import (  # noqa: E402
    DEFAULT_STEREO_PAIR_TOLERANCE_US,
    pair_stereo_records,
    summarize_pairing,
)
from dropout_evaluation.stereo_imu_vio_adapter import (  # noqa: E402
    load_vio_trajectory_from_csv,
    vio_trajectory_to_local_trajectory,
)

SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"


def _load_index_rows(stream: str) -> list[dict]:
    import csv

    path = SESSION / "streams" / stream / "index.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.skipif(not SESSION.is_dir(), reason="Scenario A dataset not available")
def test_exact_and_nearest_stereo_pairing_recovers_436() -> None:
    left_rows = _load_index_rows("left_ir")
    right_rows = _load_index_rows("right_ir")
    canonical = [frame.frame_number for frame in load_canonical_frames_from_rgb_index(SESSION)]
    pairs, unpaired = pair_stereo_records(left_rows, right_rows, canonical)
    summary = summarize_pairing(pairs, unpaired)
    assert summary["paired_frames"] == 436
    assert summary["unpaired_left_frames"] == 0
    assert summary["exact_pairs"] == 433
    assert summary["nearest_pairs"] == 3
    assert summary["max_timestamp_skew_us"] == 1


def test_tolerance_rejects_large_skew() -> None:
    left = [{"frame_number": 1, "device_timestamp_us": 1000}]
    right = [{"frame_number": 1, "device_timestamp_us": 5000}]
    pairs, unpaired = pair_stereo_records(left, right, [1], tolerance_us=1000)
    assert pairs == []
    assert unpaired == [0]


@pytest.mark.skipif(not SESSION.is_dir(), reason="Scenario A dataset not available")
def test_canonical_frame_mapping_1_to_436() -> None:
    canonical = load_canonical_frames_from_rgb_index(SESSION)
    assert len(canonical) == 436
    assert canonical[0].frame_number == 1
    assert canonical[-1].frame_number == 436


@pytest.mark.skipif(not SESSION.is_dir(), reason="Scenario A dataset not available")
def test_native_frame_numbers_are_preserved_in_pairs() -> None:
    left_rows = _load_index_rows("left_ir")
    right_rows = _load_index_rows("right_ir")
    canonical = [frame.frame_number for frame in load_canonical_frames_from_rgb_index(SESSION)]
    pairs, _ = pair_stereo_records(left_rows, right_rows, canonical)
    assert pairs[0].canonical_frame_number == 1
    assert pairs[0].native_left_frame_number == 12
    assert pairs[-1].canonical_frame_number == 436
    assert pairs[-1].native_left_frame_number == 447


@pytest.mark.skipif(not SESSION.is_dir(), reason="Scenario A dataset not available")
def test_b_c_d_canonical_anchors() -> None:
    canonical = load_canonical_frames_from_rgb_index(SESSION)
    by_frame = {frame.frame_number: frame for frame in canonical}
    assert by_frame[81].frame_number == 81
    assert by_frame[202].frame_number == 202
    assert by_frame[248].frame_number == 248


def test_no_reference_dependency_in_estimator() -> None:
    import dropout_evaluation.stereo_imu_vio_lite as mod

    source = Path(mod.__file__).read_text(encoding="utf-8").lower()
    assert "apriltag" not in source
    assert "load_pose_references" not in source
    assert "cup" not in source


@pytest.mark.skipif(
    not (ROOT / "out/analysis/phase4_stereo_imu_vio_lite/trajectory.csv").is_file(),
    reason="VIO trajectory not generated",
)
def test_vio_trajectory_csv_has_canonical_and_native_frames() -> None:
    samples = load_vio_trajectory_from_csv(ROOT / "out/analysis/phase4_stereo_imu_vio_lite/trajectory.csv")
    assert len(samples) == 436
    assert samples[0].frame_number == 1
    assert samples[-1].frame_number == 436
    assert samples[0].native_left_frame_number == 12
    local = vio_trajectory_to_local_trajectory(samples)
    assert all(item.segment_id == 0 for item in local)
