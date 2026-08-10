"""Tests for Phase 4.5 VIO demo video generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.phase45_vio_demo import (  # noqa: E402
    DEMO_DURATION_SEC,
    DEMO_START_FRAME,
    DEMO_WINDOW_ID,
    build_demo_replay_states,
    dropout_overlay_lines,
    load_cup_bboxes_by_frame,
    load_demo_window,
    load_rgb_frame_paths,
    load_vio_trajectory_from_csv,
)
from dropout_evaluation.stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID

SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
TRAJECTORY = ROOT / "out/analysis/phase4_stereo_imu_vio_lite/trajectory.csv"
MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="dropout manifest missing")
def test_demo_window_c5s_start_frame_202() -> None:
    window = load_demo_window(MANIFEST)
    assert window.window_id == DEMO_WINDOW_ID
    assert window.start_frame == DEMO_START_FRAME
    assert window.target_duration_sec == DEMO_DURATION_SEC
    assert window.recovery_frame == 352


@pytest.mark.skipif(not SESSION.is_dir(), reason="Scenario A missing")
def test_rgb_frame_mapping_canonical_1_to_436() -> None:
    paths = load_rgb_frame_paths(SESSION)
    assert len(paths) == 436
    assert 1 in paths and 436 in paths


@pytest.mark.skipif(not TRAJECTORY.is_file(), reason="frozen trajectory missing")
def test_frozen_trajectory_read_canonical() -> None:
    samples = load_vio_trajectory_from_csv(TRAJECTORY)
    assert len(samples) == 436
    assert samples[0].frame_number == 1


@pytest.mark.skipif(not (SESSION / "derived/cups/observations.csv").is_file(), reason="cup obs missing")
def test_cup2_overlay_mapping_frame_203() -> None:
    cups = load_cup_bboxes_by_frame(SESSION / "derived/cups/observations.csv")
    frame_cups = cups[203]
    ids = {cup.semantic_id for cup in frame_cups}
    assert "cup1" in ids
    assert "cup2" in ids


@pytest.mark.skipif(
    not (TRAJECTORY.is_file() and MANIFEST.is_file() and SESSION.is_dir()),
    reason="demo prerequisites missing",
)
def test_demo_replay_states_cover_canonical_frames() -> None:
    states = build_demo_replay_states(
        session_dir=SESSION,
        trajectory_csv=TRAJECTORY,
        manifest_path=MANIFEST,
    )
    assert 1 in states
    assert 436 in states
    assert states[250].masked is True
    assert states[201].masked is False


def test_dropout_overlay_semantics() -> None:
    class _Window:
        start_frame = 202
        recovery_frame = 352

        @staticmethod
        def boundary_timestamp_us() -> int:
            return 0

    window = load_demo_window(MANIFEST) if MANIFEST.is_file() else None
    if window is None:
        pytest.skip("manifest missing")
    pre = dropout_overlay_lines(
        frame_number=201,
        replay=None,
        window=window,
        device_timestamp_us=window.start_device_timestamp_us - 1,
    )
    assert "VISIBLE" in pre[0]
    mid = dropout_overlay_lines(
        frame_number=250,
        replay=None,
        window=window,
        device_timestamp_us=window.start_device_timestamp_us + 1_000_000,
    )
    assert "SOFTWARE DROPOUT" in mid[0]
    assert "STEREO + IMU VIO" in mid[2]
    post = dropout_overlay_lines(
        frame_number=352,
        replay=None,
        window=window,
        device_timestamp_us=window.recovery_device_timestamp_us or window.boundary_timestamp_us,
    )
    assert "RE-ANCHORED" in post[0]


@pytest.mark.skipif(
    not (ROOT / "out/demo/phase45_vio/scenario_a_vio_c5s_demo.mp4").is_file(),
    reason="demo mp4 not generated",
)
def test_demo_video_smoke_openable() -> None:
    import cv2

    mp4 = ROOT / "out/demo/phase45_vio/scenario_a_vio_c5s_demo.mp4"
    cap = cv2.VideoCapture(str(mp4))
    assert cap.isOpened()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 400 <= count <= 436
    cap.release()


def test_algorithm_id_constant() -> None:
    assert STEREO_IMU_VIO_LITE_ALGORITHM_ID == "stereo_imu_vio_lite"
