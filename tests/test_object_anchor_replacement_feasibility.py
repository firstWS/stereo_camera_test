from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect import UltralyticsYOLODetector
from object_anchor_detector import UltralyticsPoseDetector
from object_anchor_world import ObjectAnchorWorldSettings, ObjectAnchorWorldTracker


MODULE_PATH = (
    ROOT
    / "experiments"
    / "object_anchor_replacement_feasibility.py"
)
SPEC = importlib.util.spec_from_file_location("object_anchor_replacement_feasibility", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _transform(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    result[0, 3] = x
    return result


def test_reprojection_buckets_use_requested_boundaries() -> None:
    assert MODULE.reprojection_bucket(None) == "no_raw_pnp"
    assert MODULE.reprojection_bucket(0.0) == "0_to_3"
    assert MODULE.reprojection_bucket(3.0) == "0_to_3"
    assert MODULE.reprojection_bucket(5.0) == "3_to_5"
    assert MODULE.reprojection_bucket(7.5) == "5_to_7_5"
    assert MODULE.reprojection_bucket(10.0) == "7_5_to_10"
    assert MODULE.reprojection_bucket(10.1) == "over_10"


def test_transform_comparison_and_shared_point() -> None:
    tag = _transform(0.0)
    anchor = _transform(0.03, yaw_deg=3.0)
    translation_m, rotation_deg = MODULE.compare_transforms(tag, anchor)
    assert translation_m is not None and np.isclose(translation_m, 0.03)
    assert rotation_deg is not None and np.isclose(rotation_deg, 3.0)

    point = np.array([0.1, 0.2, 1.0])
    tag_point = MODULE.transform_point(tag, point)
    anchor_point = MODULE.transform_point(anchor, point)
    assert np.allclose(tag_point, point)
    assert not np.allclose(anchor_point, point)


def test_distribution_reports_requested_statistics() -> None:
    result = MODULE.distribution([1.0, 2.0, 3.0, 10.0])
    assert result["count"] == 4
    assert result["median"] == 2.5
    assert result["max"] == 10.0
    assert np.isclose(result["p90"], 7.9)


def test_apriltag_candidate_solver_records_raw_pose_for_same_corners() -> None:
    K = np.array(
        [[610.0, 0.0, 640.0], [0.0, 611.0, 400.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([-0.03, 0.04, 0.0002, -0.0004, -0.01, 0.0, 0.0, 0.0])
    object_points = MODULE._tag_object_points(0.135)
    rvec = np.array([0.1, -0.2, 0.03], dtype=np.float64)
    tvec = np.array([0.02, -0.01, 1.2], dtype=np.float64)
    corners, _ = MODULE.cv2.projectPoints(object_points, rvec, tvec, K, dist)
    corners = corners.reshape(4, 2)

    zero = MODULE.solve_apriltag_candidate(
        corners, 0.135, K, np.zeros((8, 1), dtype=np.float64)
    )
    corrected = MODULE.solve_apriltag_candidate(corners, 0.135, K, dist)

    assert zero is not None and corrected is not None
    assert zero["rvec"].shape == (3,)
    assert corrected["tvec"].shape == (3,)
    assert corrected["T_camera_tag"].shape == (4, 4)
    assert corrected["reprojection_error_px"] <= zero["reprojection_error_px"]


def test_tag_pixel_geometry_reports_size_center_and_boundary() -> None:
    centered = np.array(
        [[600.0, 360.0], [680.0, 360.0], [680.0, 440.0], [600.0, 440.0]]
    )
    result = MODULE.tag_pixel_geometry(
        centered,
        (800, 1280),
        boundary_margin_px=10.0,
        occlusion_edge_ratio=0.4,
    )
    assert np.isclose(result["polygon_area_px2"], 6400.0)
    assert np.isclose(result["mean_width_px"], 80.0)
    assert np.isclose(result["mean_height_px"], 80.0)
    assert np.isclose(result["center_distance_px"], 0.0)
    assert result["near_image_boundary"] is False
    assert result["partial_occlusion_suspected"] is False

    boundary = centered.copy()
    boundary[:, 0] -= 598.0
    boundary_result = MODULE.tag_pixel_geometry(
        boundary,
        (800, 1280),
        boundary_margin_px=10.0,
        occlusion_edge_ratio=0.4,
    )
    assert boundary_result["near_image_boundary"] is True
    assert boundary_result["partial_occlusion_suspected"] is True


def test_pose_cluster_summary_separates_large_pose_groups() -> None:
    transforms = [
        _transform(0.00),
        _transform(0.01, 1.0),
        _transform(-0.01, -1.0),
        _transform(2.00, 56.0),
        _transform(2.01, 57.0),
        _transform(1.99, 55.0),
    ]
    summary = MODULE.pose_cluster_summary(
        transforms,
        translation_threshold_m=0.25,
        rotation_threshold_deg=20.0,
        min_samples=2,
    )
    assert summary["cluster_count"] == 2
    assert summary["noise_count"] == 0
    assert summary["cluster_sizes"] == [3, 3]
    assert np.isclose(summary["largest_cluster_ratio"], 0.5)


def test_diagnostic_classifies_shared_small_tag_jumps_as_corner_localization() -> None:
    method = {
        "pose_clusters": {
            "cluster_count": 2,
            "largest_cluster_ratio": 0.7,
        }
    }
    rows = [
        {
            "frame_idx": 1,
            "zero_rotation_jump": True,
            "zero_translation_jump": False,
            "distortion_rotation_jump": True,
            "distortion_translation_jump": False,
        }
    ]
    size = {
        "zero": {
            "jump_tag_area_median_lt_75pct_normal": True,
            "all_jump_or_transition_areas_below_normal_p25": True,
        },
        "distortion": {
            "jump_tag_area_median_lt_75pct_normal": True,
            "all_jump_or_transition_areas_below_normal_p25": True,
        },
    }
    result = MODULE.classify_apriltag_diagnostic(method, method, rows, size)
    assert result["classification"] == "small_tag_corner_localization_primary"
    assert result["jump_frame_overlap_ratio"] == 1.0
    assert result["registration_config_preparation_allowed"] is False


def test_default_orbbec_config_uses_full99_without_cup_regression() -> None:
    root = MODULE_PATH.parents[1]
    config = yaml.safe_load(
        (root / "configs" / "orbbec_gemini.yaml").read_text(encoding="utf-8")
    )
    assert config["object_anchor"]["enabled"] is True
    assert (
        config["object_anchor"]["model_path"]
        == "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt"
    )
    assert config["object_anchor"]["detector_conf"] == 0.25
    assert config["detector"] == {
        "kind": "yolo",
        "model_path": "yolo11s.pt",
        "conf": 0.4,
        "iou": 0.45,
        "imgsz": 640,
        "device": None,
        "class_ids": [41],
    }
    assert config["apriltag_world"]["enabled"] is True


def test_cup_and_object_anchor_use_distinct_detector_classes() -> None:
    assert UltralyticsYOLODetector is not UltralyticsPoseDetector
    assert UltralyticsYOLODetector.__module__ == "detect"
    assert UltralyticsPoseDetector.__module__ == "object_anchor_detector"


def test_missing_anchor_registration_never_injects_identity(tmp_path: Path) -> None:
    missing = tmp_path / "missing_world_pose.yaml"
    tracker = ObjectAnchorWorldTracker(
        object_id="tissue_box_01",
        settings=ObjectAnchorWorldSettings(),
        registration_file=missing,
        session_dir=tmp_path / "session",
        start_registration=False,
    )
    try:
        assert tracker.registered_world_pose is None
        assert not missing.exists()
        assert tracker.registration is None
    finally:
        tracker.close()
