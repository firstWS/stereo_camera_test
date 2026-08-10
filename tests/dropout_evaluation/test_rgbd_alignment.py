from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.rgb_depth_geometry import (  # noqa: E402
    RgbDepthCalibration,
    estimate_cup_xyz_from_rgb_bbox,
    extrinsic_to_homogeneous_4x4,
    project_depth_pixels_to_rgb,
)
from dropout_evaluation.rgbd_alignment import (  # noqa: E402
    INVALID_DEPTH_VALUE,
    align_depth_meters_to_rgb,
    align_depth_to_rgb,
    build_alignment_manifest_provenance,
    compute_coverage_metrics,
    compute_spatial_coverage,
)
from stereo_types import BBox  # noqa: E402


def _identity_calib(*, rgb_width: int = 64, rgb_height: int = 48) -> RgbDepthCalibration:
    k = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 24.0], [0.0, 0.0, 1.0]])
    return RgbDepthCalibration(
        k_rgb=k,
        k_depth=k.copy(),
        t_rgb_to_depth=np.eye(4),
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        depth_width=64,
        depth_height=48,
    )


def test_identity_extrinsic_same_intrinsics_preserves_pixel_location() -> None:
    calib = _identity_calib()
    depth_m = np.zeros((48, 64), dtype=np.float64)
    depth_m[24, 32] = 1.5
    result = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    assert result.aligned_depth.shape == (48, 64)
    assert result.aligned_depth[24, 32] == pytest.approx(1.5, rel=1e-5)
    assert result.aligned_depth[0, 0] == INVALID_DEPTH_VALUE


def test_known_translation_extrinsic_shifts_projected_pixel() -> None:
    depth_m = np.zeros((48, 64), dtype=np.float64)
    depth_m[24, 32] = 1.0
    identity = align_depth_meters_to_rgb(depth_m, _identity_calib(), z_min_m=0.05, z_max_m=40.0)
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
    shifted = align_depth_meters_to_rgb(depth_m, translated, z_min_m=0.05, z_max_m=40.0)
    identity_uv = np.argwhere(identity.aligned_depth > 0.0)[0]
    shifted_uv = np.argwhere(shifted.aligned_depth > 0.0)[0]
    assert shifted_uv[1] != identity_uv[1]


def test_positive_z_only_recorded() -> None:
    calib = _identity_calib()
    depth_m = np.zeros((48, 64), dtype=np.float64)
    depth_m[24, 32] = 1.0
    result = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    valid = result.aligned_depth > 0.0
    assert np.all(result.aligned_depth[valid] > 0.0)


def test_behind_camera_points_excluded() -> None:
    calib = _identity_calib()
    depth_m = np.zeros((48, 64), dtype=np.float64)
    depth_m[24, 32] = 1.0
    t_rgb_to_depth = np.eye(4)
    t_rgb_to_depth[2, 3] = 2.0
    behind_calib = RgbDepthCalibration(
        k_rgb=calib.k_rgb,
        k_depth=calib.k_depth,
        t_rgb_to_depth=t_rgb_to_depth,
        rgb_width=64,
        rgb_height=48,
        depth_width=64,
        depth_height=48,
    )
    result = align_depth_meters_to_rgb(depth_m, behind_calib, z_min_m=0.05, z_max_m=40.0)
    assert result.diagnostics.behind_rgb_camera > 0
    assert result.coverage.valid_pixel_count == 0


def test_out_of_bounds_points_excluded() -> None:
    calib = RgbDepthCalibration(
        k_rgb=np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 24.0], [0.0, 0.0, 1.0]]),
        k_depth=np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 24.0], [0.0, 0.0, 1.0]]),
        t_rgb_to_depth=np.eye(4),
        rgb_width=20,
        rgb_height=20,
        depth_width=64,
        depth_height=48,
    )
    depth_m = np.zeros((48, 64), dtype=np.float64)
    depth_m[24, 32] = 1.0
    result = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    assert result.diagnostics.projected_out_of_bounds > 0
    assert result.coverage.valid_pixel_count == 0


def test_z_buffer_nearest_z_wins() -> None:
    from dropout_evaluation.rgbd_alignment import _z_buffer_projected_points

    z_rgb = np.array([2.0, 1.0], dtype=np.float64)
    u_i = np.array([10, 10], dtype=np.int64)
    v_i = np.array([10, 10], dtype=np.int64)
    aligned, collision_count, unique_count = _z_buffer_projected_points(
        z_rgb,
        u_i,
        v_i,
        rgb_width=64,
        rgb_height=48,
    )
    assert aligned[10, 10] == pytest.approx(1.0, rel=1e-5)
    assert collision_count == 1
    assert unique_count == 1


def test_invalid_source_depth_excluded() -> None:
    calib = _identity_calib()
    depth_m = np.zeros((48, 64), dtype=np.float64)
    result = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    assert result.diagnostics.invalid_source_depth == depth_m.size
    assert result.coverage.valid_pixel_count == 0


def test_output_shape_matches_rgb_hw() -> None:
    calib = RgbDepthCalibration(
        k_rgb=_identity_calib().k_rgb,
        k_depth=_identity_calib().k_depth,
        t_rgb_to_depth=np.eye(4),
        rgb_width=1280,
        rgb_height=800,
        depth_width=848,
        depth_height=480,
    )
    depth_m = np.full((480, 848), 1.5, dtype=np.float64)
    result = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    assert result.aligned_depth.shape == (800, 1280)
    assert result.aligned_depth.dtype == np.float32


def test_dtype_and_unit_meters() -> None:
    calib = _identity_calib()
    depth_raw = np.full((48, 64), 1500, dtype=np.uint16)
    result = align_depth_to_rgb(
        depth_raw,
        calib.k_depth,
        calib.k_rgb,
        calib.t_rgb_to_depth,
        depth_scale=1.0,
        rgb_width=64,
        rgb_height=48,
        depth_is_millimeters=True,
    )
    valid = result.aligned_depth > 0.0
    assert result.aligned_depth.dtype == np.float32
    assert float(np.median(result.aligned_depth[valid])) == pytest.approx(1.5, rel=1e-3)


def test_holes_remain_invalid() -> None:
    calib = _identity_calib()
    depth_m = np.zeros((48, 64), dtype=np.float64)
    depth_m[24, 32] = 1.0
    result = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    holes = result.aligned_depth == INVALID_DEPTH_VALUE
    assert int(holes.sum()) == depth_m.size - 1


def test_deterministic_output() -> None:
    calib = _identity_calib()
    depth_m = np.full((48, 64), 1.2, dtype=np.float64)
    depth_m[0:10, 0:10] = 0.0
    a = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    b = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    assert np.array_equal(a.aligned_depth, b.aligned_depth)


def test_transform_direction_regression_matches_phase2_inverse() -> None:
    extrinsic = {
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation": [50.0, 0.0, 0.0],
    }
    t_rgb_to_depth = extrinsic_to_homogeneous_4x4(extrinsic)
    calib = RgbDepthCalibration(
        k_rgb=_identity_calib().k_rgb,
        k_depth=_identity_calib().k_depth,
        t_rgb_to_depth=t_rgb_to_depth,
        rgb_width=64,
        rgb_height=48,
        depth_width=64,
        depth_height=48,
    )
    depth_m = np.full((48, 64), 1.0, dtype=np.float64)
    bbox = BBox(xyxy=(20.0, 16.0, 44.0, 32.0), confidence=1.0, class_id=41, label="cup")
    cup = estimate_cup_xyz_from_rgb_bbox(depth_m, bbox, calib, min_valid_ratio=0.01)
    projection = project_depth_pixels_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    aligned = align_depth_meters_to_rgb(depth_m, calib, z_min_m=0.05, z_max_m=40.0)
    assert cup.valid is True
    assert aligned.coverage.valid_pixel_count > 0
    assert projection.z_rgb.size > 0


def test_coverage_and_spatial_helpers() -> None:
    aligned = np.zeros((9, 9), dtype=np.float32)
    aligned[3:6, 3:6] = 1.0
    coverage = compute_coverage_metrics(aligned, odometry_z_min_m=0.05, odometry_z_max_m=4.0)
    spatial = compute_spatial_coverage(aligned)
    assert coverage.valid_pixel_count == 9
    assert coverage.valid_pixel_ratio == pytest.approx(9 / 81)
    assert spatial.center_valid_ratio == pytest.approx(1.0)
    assert spatial.left_valid_ratio < spatial.center_valid_ratio


def test_manifest_provenance_fields() -> None:
    manifest = build_alignment_manifest_provenance(
        session_id="test_session",
        source_rgb_resolution=(1280, 800),
        source_depth_resolution=(848, 480),
        output_resolution=(1280, 800),
    )
    assert manifest["schema_version"] == 1
    assert manifest["depth_unit"] == "meters"
    assert manifest["invalid_value"] == 0.0
    assert manifest["z_buffer_policy"] == "nearest_positive_z_rgb"
    assert manifest["hole_fill_policy"] == "none"
    assert manifest["timestamp_pairing_policy"] == "nearest_device_timestamp"
