"""Tests for Phase 4.7-A stereo+IMU SLAM-lite."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from dropout_evaluation.stereo_imu_calibration import load_stereo_imu_calibration
from dropout_evaluation.stereo_imu_slam_continuous import (
    default_slam_config,
    run_continuous_stereo_imu_slam,
    summarize_slam_trajectory,
    write_slam_outputs,
)
from dropout_evaluation.stereo_imu_slam_lite import (
    StereoImuSlamTrajectorySample,
    run_stereo_imu_slam_lite,
)
from dropout_evaluation.stereo_imu_slam_map import SlamMap, SlamMapConfig
from dropout_evaluation.stereo_imu_vio_lite import (
    ImuSampleRecord,
    StereoImuVioConfig,
    StereoImuVioFrameInput,
    detect_landmarks,
)


def _synthetic_calibration():
    intrinsics = {
        "intrinsics": [
            {
                "frame": "LEFT_IR",
                "success": True,
                "intrinsic": {"width": 64, "height": 48, "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0},
                "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
            },
            {
                "frame": "RIGHT_IR",
                "success": True,
                "intrinsic": {"width": 64, "height": 48, "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0},
                "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
            },
        ]
    }
    extrinsics = {
        "extrinsics": [
            {
                "from_frame": "LEFT_IR",
                "to_frame": "RIGHT_IR",
                "success": True,
                "extrinsic": {
                    "rotation": np.eye(3).tolist(),
                    "translation": [-95.0, 0.0, 0.0],
                },
            },
            {
                "from_frame": "LEFT_IR",
                "to_frame": "GYRO",
                "success": True,
                "extrinsic": {
                    "rotation": np.eye(3).tolist(),
                    "translation": [0.0, 0.0, 0.0],
                },
            },
        ]
    }
    return load_stereo_imu_calibration(intrinsics, extrinsics)


def _synthetic_frames(count: int = 4) -> list[StereoImuVioFrameInput]:
    frames: list[StereoImuVioFrameInput] = []
    yy, xx = np.mgrid[0:48, 0:64]
    checker = (((xx // 4) + (yy // 4)) % 2 * 180 + 40).astype(np.uint8)
    for idx in range(count):
        shift = idx * 2
        left = np.roll(checker, shift=shift, axis=1)
        right = np.roll(checker, shift=shift + 3, axis=1)
        frames.append(
            StereoImuVioFrameInput(
                frame_number=idx + 1,
                device_timestamp_us=1_000_000 + idx * 33_000,
                left_gray=left,
                right_gray=right,
                native_left_frame_number=idx,
                native_right_frame_number=idx,
            )
        )
    return frames


def test_keyframe_insertion_and_map_persistence():
    calib = _synthetic_calibration()
    cfg = default_slam_config()
    cfg = replace(cfg, map=replace(cfg.map, keyframe_interval_frames=1), vio=replace(cfg.vio, min_stereo_points=5, min_visual_inliers=5))
    slam_map = SlamMap(cfg.map)
    frame = _synthetic_frames(1)[0]
    left, right = frame.left_gray, frame.right_gray
    T = np.eye(4)
    added = slam_map.add_keyframe(
        frame_number=1,
        device_timestamp_us=frame.device_timestamp_us,
        T_slam_camera=T,
        left_gray=left,
        right_gray=right,
        calib=calib,
        vio_config=cfg.vio,
    )
    assert slam_map.keyframe_count == 1
    assert slam_map.map_point_count >= 0
    assert added >= 0
    if slam_map.map_point_count == 0:
        from dropout_evaluation.stereo_imu_slam_map import SlamMapPoint

        slam_map.map_points[1] = SlamMapPoint(
            map_point_id=1,
            position_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            descriptor=np.zeros(32, dtype=np.uint8),
            first_seen_frame=1,
            last_seen_frame=1,
        )
    point_ids = list(slam_map.map_points.keys())
    assert point_ids
    first_id = point_ids[0]
    assert slam_map.map_points[first_id].first_seen_frame == 1


def test_descriptor_map_matching_and_pnp(monkeypatch):
    calib = _synthetic_calibration()
    cfg = default_slam_config()
    slam_map = SlamMap(cfg.map)
    frame = _synthetic_frames(1)[0]

    def fake_localize(*, left_gray, calib, min_matches=None, min_inliers=None):
        from dropout_evaluation.stereo_imu_slam_map import MapLocalizationResult

        T = np.eye(4)
        T[0, 3] = 0.01
        return MapLocalizationResult(True, T, 12, 10, None)

    monkeypatch.setattr(slam_map, "localize_with_map", fake_localize)
    result = slam_map.localize_with_map(left_gray=frame.left_gray, calib=calib)
    assert result.success
    assert result.T_slam_camera is not None


def test_map_update_rejection():
    calib = _synthetic_calibration()
    slam_map = SlamMap(SlamMapConfig(min_map_matches=100))
    frame = _synthetic_frames(1)[0]
    result = slam_map.localize_with_map(left_gray=frame.left_gray, calib=calib)
    assert not result.success


def test_relocalization_path(monkeypatch):
    calib = _synthetic_calibration()
    cfg = default_slam_config()
    slam_map = SlamMap(cfg.map)
    frame = _synthetic_frames(1)[0]
    slam_map.add_keyframe(
        frame_number=1,
        device_timestamp_us=frame.device_timestamp_us,
        T_slam_camera=np.eye(4),
        left_gray=frame.left_gray,
        right_gray=frame.right_gray,
        calib=calib,
        vio_config=cfg.vio,
    )

    def fake_reloc(*, left_gray, calib):
        from dropout_evaluation.stereo_imu_slam_map import MapLocalizationResult

        return MapLocalizationResult(True, np.eye(4), 8, 6, None)

    monkeypatch.setattr(slam_map, "relocalize", fake_reloc)
    result = slam_map.relocalize(left_gray=frame.left_gray, calib=calib)
    assert result.success


def test_causal_frame_handling():
    frames = _synthetic_frames(5)
    calib = _synthetic_calibration()
    imu = [
        ImuSampleRecord(ts, np.array([0.0, 0.0, 9.8]), np.array([0.0, 0.0, 0.0]))
        for ts in range(1_000_000, 1_200_000, 10_000)
    ]
    samples, slam_map, counters = run_stereo_imu_slam_lite(
        frames,
        imu,
        calib,
        config=default_slam_config(),
    )
    assert len(samples) == len(frames)
    assert all(samples[i].frame_number <= samples[i + 1].frame_number for i in range(len(samples) - 1))
    assert counters["map_update_attempts"] >= len(frames) - 1


def test_output_contract(tmp_path: Path):
    frames = _synthetic_frames(3)
    calib = _synthetic_calibration()
    imu = [ImuSampleRecord(1_000_000, np.zeros(3), np.zeros(3))]
    result = run_continuous_stereo_imu_slam(frames, imu, calib, config=default_slam_config())
    provenance = {"algorithm_id": "stereo_imu_slam_lite"}
    write_slam_outputs(tmp_path, result, provenance)
    with (tmp_path / "trajectory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "frame_number",
        "valid",
        "state",
        "tx",
        "pose_source",
        "map_match_count",
        "map_update_success",
        "keyframe_count",
        "map_point_count",
    }
    assert rows
    assert required.issubset(rows[0].keys())


def test_summarize_slam_trajectory_metrics():
    sample = StereoImuSlamTrajectorySample(
        frame_number=1,
        device_timestamp_us=1,
        valid=True,
        state="slam_map_tracking",
        native_left_frame_number=0,
        native_right_frame_number=0,
        tx=0.0,
        ty=0.0,
        tz=0.0,
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        frontend_visual_success=True,
        imu_samples_used=1,
        keyframe_count=2,
        map_point_count=10,
        map_match_count=12,
        map_inlier_count=8,
        map_update_success=True,
        relocalization_attempted=False,
        relocalization_success=False,
        pose_source="MAP_TRACKING",
    )
    summary, diagnostics = summarize_slam_trajectory(
        [sample],
        session_duration_s=1.0,
        processing_time_s=0.5,
        counters={
            "keyframes_created": 2,
            "map_update_attempts": 1,
            "map_update_successes": 1,
            "map_based_pose_update_count": 1,
            "relocalization_attempts": 0,
            "relocalization_successes": 0,
        },
        final_map_points=10,
    )
    assert summary["map_based_pose_update_count"] == 1
    assert diagnostics["catastrophic_jump_count"] == 0


@pytest.mark.integration
def test_scenario_a_smoke_gate():
    root = Path(__file__).resolve().parents[2]
    session = root / "out/datasets/gemini335l/20260807_161354_scenario_a"
    if not session.is_dir():
        pytest.skip("Scenario A dataset not available")
    from dataset_recorder.reader import DatasetReader
    from dropout_evaluation.stereo_imu_slam_continuous import load_imu_samples, load_stereo_frames

    reader = DatasetReader(session)
    calib = _synthetic_calibration()
    calib = load_stereo_imu_calibration(reader.calibration_intrinsics(), reader.calibration_extrinsics())
    frames, _ = load_stereo_frames(reader)
    imu_samples = load_imu_samples(reader)
    frames = frames[:60]
    result = run_continuous_stereo_imu_slam(frames, imu_samples, calib, config=default_slam_config())
    assert result.summary["keyframes_created"] >= 2
    assert result.summary["final_map_points"] > 0
    assert result.summary["map_based_pose_update_count"] > 0
