from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

open3d = pytest.importorskip("open3d")

from dataset_recorder.rgb_depth_geometry import RgbDepthCalibration  # noqa: E402
from dropout_evaluation.rgbd_odometry import (  # noqa: E402
    DEFAULT_DEPTH_SCALE,
    DEFAULT_DEPTH_TRUNC_M,
    RgbdOdometryConfig,
    accumulate_odom_pose,
    assert_rgb_depth_same_resolution,
    build_rgbd_image_from_arrays,
    depth_input_stats,
    estimate_rgbd_pair_motion,
    invert_rigid_transform,
    make_odometry_option,
    pinhole_intrinsic_from_rgb_calibration,
    relative_transform_target_source,
    transform_is_finite,
    transform_magnitude,
)


def _calib() -> RgbDepthCalibration:
    k = np.array([[610.0, 0.0, 640.0], [0.0, 610.0, 400.0], [0.0, 0.0, 1.0]])
    return RgbDepthCalibration(
        k_rgb=k,
        k_depth=k.copy(),
        t_rgb_to_depth=np.eye(4),
        rgb_width=1280,
        rgb_height=800,
        depth_width=1280,
        depth_height=800,
    )


def test_open3d_version_import() -> None:
    assert open3d.__version__ == "0.19.0"


def test_meter_depth_uses_depth_scale_one() -> None:
    color = np.zeros((800, 1280, 3), dtype=np.uint8)
    depth = np.zeros((800, 1280), dtype=np.float32)
    depth[400, 640] = 1.5
    result = build_rgbd_image_from_arrays(color, depth, config=RgbdOdometryConfig(depth_scale=1.0))
    depth_after = np.asarray(result.rgbd_image.depth)
    assert depth_after[400, 640] == pytest.approx(1.5, rel=1e-5)


def test_depth_trunc_four_meters() -> None:
    color = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.zeros((32, 32), dtype=np.float32)
    depth[16, 16] = 5.0
    result = build_rgbd_image_from_arrays(
        color,
        depth,
        config=RgbdOdometryConfig(depth_scale=1.0, depth_trunc_m=4.0),
    )
    depth_after = np.asarray(result.rgbd_image.depth)
    assert depth_after[16, 16] == 0.0


def test_rgb_depth_same_resolution_required() -> None:
    color = np.zeros((800, 1280, 3), dtype=np.uint8)
    depth = np.zeros((480, 848), dtype=np.float32)
    with pytest.raises(ValueError, match="resolution mismatch"):
        assert_rgb_depth_same_resolution(color, depth)


def test_pinhole_intrinsic_uses_rgb_calibration() -> None:
    calib = _calib()
    intrinsic = pinhole_intrinsic_from_rgb_calibration(calib)
    assert intrinsic.width == 1280
    assert intrinsic.height == 800
    assert intrinsic.intrinsic_matrix[0, 0] == pytest.approx(610.0)
    assert intrinsic.intrinsic_matrix[1, 1] == pytest.approx(610.0)
    assert intrinsic.intrinsic_matrix[0, 2] == pytest.approx(640.0)
    assert intrinsic.intrinsic_matrix[1, 2] == pytest.approx(400.0)


def test_hybrid_jacobian_and_option_values() -> None:
    option = make_odometry_option(RgbdOdometryConfig())
    assert option.depth_min == pytest.approx(0.05)
    assert option.depth_max == pytest.approx(4.0)
    assert option.depth_diff_max == pytest.approx(0.03)
    jacobian = open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    assert jacobian is not None


def test_failure_returns_invalid_result_not_identity() -> None:
    color = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.float32)
    intrinsic = pinhole_intrinsic_from_rgb_calibration(
        RgbDepthCalibration(
            k_rgb=np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]]),
            k_depth=np.eye(3),
            t_rgb_to_depth=np.eye(4),
            rgb_width=64,
            rgb_height=64,
            depth_width=64,
            depth_height=64,
        )
    )
    with patch(
        "open3d.pipelines.odometry.compute_rgbd_odometry",
        return_value=(False, np.eye(4), np.eye(6)),
    ):
        result = estimate_rgbd_pair_motion(
            color,
            depth,
            color,
            depth,
            intrinsic,
            source_frame=1,
            target_frame=2,
        )
    assert result.success is False
    assert result.transform_target_source is None
    assert result.information_matrix is None
    assert result.failure_reason == "open3d_compute_rgbd_odometry_failed"


def test_accumulate_odom_pose_algebra() -> None:
    T_odom_source = np.eye(4)
    T_odom_source[0, 3] = 1.0
    M = np.eye(4)
    M[0, 3] = 0.1
    T_odom_target = accumulate_odom_pose(T_odom_source, M)
    expected = T_odom_source @ invert_rigid_transform(M)
    assert np.allclose(T_odom_target, expected)


def test_relative_transform_target_source_regression() -> None:
    T_world_s = np.eye(4)
    T_world_s[0, 3] = 1.0
    T_world_t = np.eye(4)
    T_world_t[0, 3] = 1.1
    M = relative_transform_target_source(T_world_s, T_world_t)
    P_s = np.array([0.0, 0.0, 0.0, 1.0])
    P_t = M @ P_s
    P_world_s = T_world_s @ P_s
    P_world_t = T_world_t @ P_t
    assert P_world_s[:3] == pytest.approx(P_world_t[:3], abs=1e-9)


def test_candidate_module_has_no_reference_dependency() -> None:
    source = (ROOT / "src/dropout_evaluation/rgbd_odometry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.names[0].name
        for node in tree.body
        if isinstance(node, ast.Import)
        for node in [node]
    }
    import_from = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "evaluation_io" not in import_from
    assert "apriltag_reference" not in import_from
    assert "reference" not in " ".join(import_from)


def test_depth_input_stats_meter_scaling() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = 0.8
    depth[2, 2] = 5.0
    stats = depth_input_stats(depth, odometry_z_min_m=0.05, odometry_z_max_m=4.0)
    assert stats.nonzero_count == 2
    assert stats.z_median_m == pytest.approx(2.9, rel=1e-3)
    assert stats.odometry_range_count == 1


def test_processing_result_serialization_finite() -> None:
    transform = np.eye(4)
    info = np.eye(6)
    assert transform_is_finite(transform)
    assert transform_is_finite(None) is False
    trans, rot = transform_magnitude(transform)
    assert trans == 0.0
    assert rot == 0.0


def test_width_height_not_swapped() -> None:
    color = np.zeros((800, 1280, 3), dtype=np.uint8)
    depth = np.zeros((800, 1280), dtype=np.float32)
    result = build_rgbd_image_from_arrays(color, depth)
    depth_after = np.asarray(result.rgbd_image.depth)
    assert depth_after.shape == (800, 1280)
    assert DEFAULT_DEPTH_SCALE == 1.0
    assert DEFAULT_DEPTH_TRUNC_M == 4.0
