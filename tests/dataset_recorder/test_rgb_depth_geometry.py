from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.rgb_depth_geometry import (  # noqa: E402
    METHOD_BASELINE_MEDIAN,
    METHOD_ROBUST_NEAR_QUANTILE,
    CupDepthEstimatorConfig,
    RgbDepthCalibration,
    build_depth_timestamp_index,
    cup_depth_estimator_config_from_mapping,
    depth_meters_from_raw,
    estimate_bbox_camera_xyz,
    estimate_cup_xyz_from_rgb_bbox,
    extrinsic_to_homogeneous_4x4,
    load_rgb_depth_calibration,
    match_nearest_depth_timestamp,
    normalize_depth_image,
)
from stereo_types import BBox  # noqa: E402


def _identity_calib() -> RgbDepthCalibration:
    k = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 24.0], [0.0, 0.0, 1.0]])
    return RgbDepthCalibration(
        k_rgb=k,
        k_depth=k.copy(),
        t_rgb_to_depth=np.eye(4),
        rgb_width=64,
        rgb_height=48,
        depth_width=64,
        depth_height=48,
    )


def test_normalize_depth_image_hw_uint16_unchanged() -> None:
    depth = np.array([[0, 1500], [65535, 944]], dtype=np.uint16)
    normalized = normalize_depth_image(depth)
    assert normalized.shape == (2, 2)
    assert normalized.dtype == np.uint16
    assert np.array_equal(normalized, depth)


def test_normalize_depth_image_hw1_to_hw_bit_identical() -> None:
    depth = np.array([[[0], [1500]], [[65535], [944]]], dtype=np.uint16)
    normalized = normalize_depth_image(depth)
    assert normalized.shape == (2, 2)
    assert normalized.dtype == np.uint16
    assert np.array_equal(normalized, depth[:, :, 0])
    assert np.shares_memory(depth, normalized)


def test_normalize_depth_image_rejects_hw3() -> None:
    depth = np.zeros((4, 4, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match=r"\(H, W\) or \(H, W, 1\)"):
        normalize_depth_image(depth)


def test_normalize_depth_image_rejects_unexpected_ndim() -> None:
    depth = np.zeros((4,), dtype=np.uint16)
    with pytest.raises(ValueError, match=r"\(H, W\) or \(H, W, 1\)"):
        normalize_depth_image(depth)


def test_depth_meters_from_raw_hw1_matches_hw() -> None:
    depth_hw = np.full((48, 64), 1500, dtype=np.uint16)
    depth_hw1 = depth_hw[:, :, np.newaxis]
    meters_hw = depth_meters_from_raw(depth_hw, depth_scale=1.0, depth_is_millimeters=True)
    meters_hw1 = depth_meters_from_raw(depth_hw1, depth_scale=1.0, depth_is_millimeters=True)
    assert meters_hw.shape == (48, 64)
    assert meters_hw1.shape == (48, 64)
    assert np.array_equal(meters_hw, meters_hw1)


def test_estimate_cup_xyz_from_hw1_raw_depth_path() -> None:
    calib = _identity_calib()
    depth_raw = np.full((48, 64, 1), 1500, dtype=np.uint16)
    depth_m = depth_meters_from_raw(depth_raw, depth_scale=1.0, depth_is_millimeters=True)
    bbox = BBox(xyxy=(16.0, 12.0, 48.0, 36.0), confidence=1.0, class_id=41, label="cup")
    estimate = estimate_cup_xyz_from_rgb_bbox(
        depth_m,
        bbox,
        calib,
        min_valid_ratio=0.03,
    )
    assert estimate.valid is True
    assert estimate.Z == pytest.approx(1.5, rel=1e-3)


def test_depth_meters_from_raw_orbbec_style() -> None:
    raw = np.array([[1500, 0], [0, 1500]], dtype=np.uint16)
    depth = depth_meters_from_raw(raw, depth_scale=1.0, depth_is_millimeters=True)
    assert depth[0, 0] == pytest.approx(1.5)


def test_nearest_timestamp_prefers_closest_depth() -> None:
    rows = [
        ({"frame_number": 0, "device_timestamp_us": 100}, "a"),
        ({"frame_number": 1, "device_timestamp_us": 200}, "b"),
        ({"frame_number": 2, "device_timestamp_us": 300}, "c"),
    ]
    timestamps, depth_rows, depth_paths = build_depth_timestamp_index(rows)
    match = match_nearest_depth_timestamp(
        205,
        timestamps,
        depth_rows,
        depth_paths,
        max_delta_us=20,
    )
    assert match is not None
    assert match.depth_frame_number == 1
    assert match.rgb_depth_delta_us == 5


def test_nearest_timestamp_rejects_large_skew() -> None:
    rows = [({"frame_number": 0, "device_timestamp_us": 100}, "a")]
    timestamps, depth_rows, depth_paths = build_depth_timestamp_index(rows)
    assert (
        match_nearest_depth_timestamp(
            10_000,
            timestamps,
            depth_rows,
            depth_paths,
            max_delta_us=100,
        )
        is None
    )


def test_identity_extrinsic_projects_center_bbox() -> None:
    calib = _identity_calib()
    depth_m = np.full((48, 64), 1.5, dtype=np.float64)
    bbox = BBox(xyxy=(16.0, 12.0, 48.0, 36.0), confidence=1.0, class_id=41, label="cup")
    estimate = estimate_cup_xyz_from_rgb_bbox(
        depth_m,
        bbox,
        calib,
        min_valid_ratio=0.03,
    )
    assert estimate.valid is True
    assert estimate.Z == pytest.approx(1.5, rel=1e-3)
    assert estimate.X == pytest.approx(0.0, abs=0.05)
    assert estimate.Y == pytest.approx(0.0, abs=0.05)


def test_translated_extrinsic_changes_rgb_coordinates() -> None:
    depth_m = np.full((48, 64), 1.0, dtype=np.float64)
    bbox = BBox(xyxy=(20.0, 16.0, 44.0, 32.0), confidence=1.0, class_id=41, label="cup")
    identity = estimate_cup_xyz_from_rgb_bbox(
        depth_m,
        bbox,
        _identity_calib(),
        min_valid_ratio=0.01,
    )
    t_rgb_to_depth = np.eye(4)
    t_rgb_to_depth[0, 3] = 0.05
    translated = RgbDepthCalibration(
        k_rgb=_identity_calib().k_rgb,
        k_depth=_identity_calib().k_depth,
        t_rgb_to_depth=t_rgb_to_depth,
        rgb_width=64,
        rgb_height=48,
        depth_width=64,
        depth_height=48,
    )
    shifted = estimate_cup_xyz_from_rgb_bbox(depth_m, bbox, translated, min_valid_ratio=0.01)
    assert identity.valid is True
    assert shifted.valid is True
    assert shifted.X != pytest.approx(identity.X)


def test_extrinsic_direction_uses_rgb_to_depth_inverse() -> None:
    extrinsic = {
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation": [50.0, 0.0, 0.0],
    }
    transform = extrinsic_to_homogeneous_4x4(extrinsic)
    assert transform[0, 3] == pytest.approx(0.05)


def test_load_rgb_depth_calibration_from_snapshot() -> None:
    intrinsics = {
        "intrinsics": [
            {
                "frame": "RGB",
                "success": True,
                "intrinsic": {"width": 1280, "height": 800, "fx": 610.0, "fy": 611.0, "cx": 646.0, "cy": 399.0},
            },
            {
                "frame": "DEPTH",
                "success": True,
                "intrinsic": {"width": 848, "height": 480, "fx": 412.0, "fy": 412.0, "cx": 421.0, "cy": 240.0},
            },
        ]
    }
    extrinsics = {
        "extrinsics": [
            {
                "from_frame": "RGB",
                "to_frame": "DEPTH",
                "success": True,
                "extrinsic": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [23.694, 0.156, -0.127],
                },
            }
        ]
    }
    calib = load_rgb_depth_calibration(intrinsics, extrinsics)
    assert calib.depth_width == 848
    assert calib.t_rgb_to_depth[0, 3] == pytest.approx(0.023694, rel=1e-3)


def test_empty_projection_is_safe() -> None:
    calib = _identity_calib()
    depth_m = np.zeros((48, 64), dtype=np.float64)
    bbox = BBox(xyxy=(10.0, 10.0, 20.0, 20.0), confidence=1.0, class_id=41, label="cup")
    estimate = estimate_cup_xyz_from_rgb_bbox(depth_m, bbox, calib, min_valid_ratio=0.03)
    assert estimate.valid is False


def _baseline_cfg() -> CupDepthEstimatorConfig:
    return CupDepthEstimatorConfig(method=METHOD_BASELINE_MEDIAN)


def _q25_cfg() -> CupDepthEstimatorConfig:
    return CupDepthEstimatorConfig(method=METHOD_ROBUST_NEAR_QUANTILE, near_quantile=0.25)


def test_baseline_median_matches_legacy_estimate() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    y = np.zeros(4)
    z = np.array([0.6, 0.65, 2.0, 2.1], dtype=np.float64)
    estimate = estimate_bbox_camera_xyz(x, y, z, config=_baseline_cfg(), valid_pixel_ratio=0.1)
    assert estimate.valid is True
    assert estimate.notes == "rgb_bbox_projected_median"
    assert estimate.X == pytest.approx(0.25)
    assert estimate.Z == pytest.approx(1.325)


def test_robust_near_quantile_selects_near_subset() -> None:
    x = np.array([0.0] * 80 + [1.0] * 20, dtype=np.float64)
    y = np.zeros(100)
    z = np.array([0.63] * 80 + [2.0] * 20, dtype=np.float64)
    estimate = estimate_bbox_camera_xyz(x, y, z, config=_q25_cfg(), valid_pixel_ratio=0.1)
    assert estimate.valid is True
    assert estimate.notes == "rgb_bbox_robust_near_quantile_median"
    assert estimate.Z == pytest.approx(0.63, abs=1e-9)
    assert estimate.X == pytest.approx(0.0, abs=1e-9)


def test_study_like_frame_b_far_support_does_not_move_q25() -> None:
    cfg = _q25_cfg()
    x = np.array([0.0] * 30 + [1.0] * 70, dtype=np.float64)
    y = np.zeros(100)
    z = np.array([0.65] * 30 + [2.0] * 70, dtype=np.float64)
    baseline = estimate_bbox_camera_xyz(x, y, z, config=_baseline_cfg(), valid_pixel_ratio=0.1)
    q25 = estimate_bbox_camera_xyz(x, y, z, config=cfg, valid_pixel_ratio=0.1)
    assert baseline.valid is True
    assert q25.valid is True
    assert baseline.Z > 1.0
    assert q25.Z == pytest.approx(0.65, abs=1e-9)


def test_single_minimum_pixel_does_not_dominate_q25() -> None:
    z = np.concatenate(
        [
            np.array([0.01], dtype=np.float64),
            np.full(19, 0.63, dtype=np.float64),
            np.full(20, 2.0, dtype=np.float64),
        ]
    )
    x = np.arange(z.size, dtype=np.float64)
    y = np.zeros(z.size)
    estimate = estimate_bbox_camera_xyz(x, y, z, config=_q25_cfg(), valid_pixel_ratio=0.1)
    assert estimate.valid is True
    assert estimate.Z == pytest.approx(np.median(z[z <= np.quantile(z, 0.25, method="linear")]), abs=1e-9)
    assert estimate.Z != pytest.approx(0.01)
    assert estimate.Z == pytest.approx(0.63, abs=0.02)


def test_non_finite_and_non_positive_z_are_excluded() -> None:
    x = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    y = np.zeros(3)
    z = np.array([np.nan, 0.0, 0.7], dtype=np.float64)
    estimate = estimate_bbox_camera_xyz(x, y, z, config=_q25_cfg(), valid_pixel_ratio=0.1)
    assert estimate.valid is True
    assert estimate.Z == pytest.approx(0.7)


def test_empty_points_are_invalid() -> None:
    estimate = estimate_bbox_camera_xyz(
        np.array([]),
        np.array([]),
        np.array([]),
        config=_q25_cfg(),
        valid_pixel_ratio=0.0,
    )
    assert estimate.valid is False
    assert estimate.notes == "no_finite_positive_z"


def test_insufficient_near_support_is_invalid() -> None:
    cfg = CupDepthEstimatorConfig(
        method=METHOD_ROBUST_NEAR_QUANTILE,
        near_quantile=0.25,
        min_near_points=100,
    )
    x = np.ones(10)
    y = np.zeros(10)
    z = np.linspace(0.6, 0.7, 10)
    estimate = estimate_bbox_camera_xyz(x, y, z, config=cfg, valid_pixel_ratio=0.1)
    assert estimate.valid is False
    assert estimate.notes == "insufficient_near_quantile_points"


def test_cup_depth_config_validation_rejects_bad_quantile() -> None:
    with pytest.raises(ValueError, match="near_quantile"):
        cup_depth_estimator_config_from_mapping({"near_quantile": 0.0})
    with pytest.raises(ValueError, match="near_quantile"):
        cup_depth_estimator_config_from_mapping({"near_quantile": 1.0})
    with pytest.raises(ValueError, match="near_quantile"):
        cup_depth_estimator_config_from_mapping({"near_quantile": -0.1})


def test_cup_depth_config_validation_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unsupported cup_depth.method"):
        cup_depth_estimator_config_from_mapping({"method": "magic_depth"})


def test_robust_near_quantile_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    z = np.concatenate([rng.normal(0.65, 0.01, size=150), rng.normal(2.0, 0.05, size=50)])
    a = estimate_bbox_camera_xyz(x, y, z, config=_q25_cfg(), valid_pixel_ratio=0.1)
    b = estimate_bbox_camera_xyz(x, y, z, config=_q25_cfg(), valid_pixel_ratio=0.1)
    assert a.valid is True and b.valid is True
    assert (a.X, a.Y, a.Z) == (b.X, b.Y, b.Z)


def test_unimodal_distribution_q25_close_to_baseline() -> None:
    z = np.linspace(0.60, 0.70, 50)
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    baseline = estimate_bbox_camera_xyz(x, y, z, config=_baseline_cfg(), valid_pixel_ratio=0.1)
    q25 = estimate_bbox_camera_xyz(x, y, z, config=_q25_cfg(), valid_pixel_ratio=0.1)
    assert baseline.valid and q25.valid
    assert q25.Z == pytest.approx(baseline.Z, abs=0.05)
