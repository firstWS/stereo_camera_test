"""Phase 4.2 Open3D RGB-D pairwise odometry candidate (no Phase 3 harness coupling)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from dataset_recorder.rgb_depth_geometry import RgbDepthCalibration

RGBD_ODOMETRY_ALGORITHM_ID = "open3d_hybrid_rgbd"
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_DEPTH_TRUNC_M = 4.0
DEFAULT_DEPTH_MIN_M = 0.05
DEFAULT_DEPTH_MAX_M = 4.0
DEFAULT_DEPTH_DIFF_MAX_M = 0.03


@dataclass(frozen=True)
class RgbdOdometryConfig:
    depth_scale: float = DEFAULT_DEPTH_SCALE
    depth_trunc_m: float = DEFAULT_DEPTH_TRUNC_M
    depth_min_m: float = DEFAULT_DEPTH_MIN_M
    depth_max_m: float = DEFAULT_DEPTH_MAX_M
    depth_diff_max_m: float = DEFAULT_DEPTH_DIFF_MAX_M
    convert_rgb_to_intensity: bool = True


@dataclass(frozen=True)
class DepthInputStats:
    shape: tuple[int, int]
    dtype: str
    nonzero_count: int
    z_min_m: float | None
    z_median_m: float | None
    z_p90_m: float | None
    z_max_m: float | None
    odometry_range_count: int


@dataclass(frozen=True)
class RgbdImageBuildResult:
    rgbd_image: Any
    depth_stats_before: DepthInputStats
    depth_stats_after: DepthInputStats
    input_prepare_ms: float


@dataclass(frozen=True)
class PairwiseOdometryResult:
    success: bool
    source_frame: int | None
    target_frame: int | None
    transform_target_source: np.ndarray | None
    information_matrix: np.ndarray | None
    input_prepare_ms: float
    odometry_ms: float
    total_ms: float
    translation_magnitude_m: float | None
    rotation_magnitude_deg: float | None
    failure_reason: str | None = None

    def to_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.transform_target_source is not None:
            payload["transform_target_source"] = self.transform_target_source.tolist()
        else:
            payload["transform_target_source"] = None
        if self.information_matrix is not None:
            payload["information_matrix"] = self.information_matrix.tolist()
        else:
            payload["information_matrix"] = None
        return payload


def _require_open3d():
    import open3d as o3d

    return o3d


def depth_input_stats(depth_m: np.ndarray, *, odometry_z_min_m: float, odometry_z_max_m: float) -> DepthInputStats:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W), got shape {depth.shape}")
    valid = depth > 0.0
    z = depth[valid]
    odometry_range = valid & (depth >= odometry_z_min_m) & (depth <= odometry_z_max_m)
    if z.size == 0:
        return DepthInputStats(
            shape=(int(depth.shape[0]), int(depth.shape[1])),
            dtype=str(depth.dtype),
            nonzero_count=0,
            z_min_m=None,
            z_median_m=None,
            z_p90_m=None,
            z_max_m=None,
            odometry_range_count=0,
        )
    return DepthInputStats(
        shape=(int(depth.shape[0]), int(depth.shape[1])),
        dtype=str(depth.dtype),
        nonzero_count=int(valid.sum()),
        z_min_m=float(np.min(z)),
        z_median_m=float(np.median(z)),
        z_p90_m=float(np.quantile(z, 0.9)),
        z_max_m=float(np.max(z)),
        odometry_range_count=int(odometry_range.sum()),
    )


def assert_rgb_depth_same_resolution(color_rgb: np.ndarray, depth_m: np.ndarray) -> None:
    if color_rgb.ndim != 3 or color_rgb.shape[2] != 3:
        raise ValueError(f"color must be HxWx3, got shape {color_rgb.shape}")
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth_m.shape}")
    if color_rgb.shape[0] != depth_m.shape[0] or color_rgb.shape[1] != depth_m.shape[1]:
        raise ValueError(
            "RGB/depth resolution mismatch: "
            f"color={color_rgb.shape[:2]} depth={depth_m.shape[:2]}"
        )


def pinhole_intrinsic_from_rgb_calibration(calib: RgbDepthCalibration):
    o3d = _require_open3d()
    fx = float(calib.k_rgb[0, 0])
    fy = float(calib.k_rgb[1, 1])
    cx = float(calib.k_rgb[0, 2])
    cy = float(calib.k_rgb[1, 2])
    return o3d.camera.PinholeCameraIntrinsic(
        int(calib.rgb_width),
        int(calib.rgb_height),
        fx,
        fy,
        cx,
        cy,
    )


def build_rgbd_image_from_arrays(
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    config: RgbdOdometryConfig | None = None,
) -> RgbdImageBuildResult:
    """Convert aligned RGB + metric depth arrays into an Open3D RGBDImage."""
    o3d = _require_open3d()
    cfg = config or RgbdOdometryConfig()
    color_rgb = np.asarray(color_rgb)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    assert_rgb_depth_same_resolution(color_rgb, depth_m)
    stats_before = depth_input_stats(
        depth_m,
        odometry_z_min_m=cfg.depth_min_m,
        odometry_z_max_m=cfg.depth_max_m,
    )
    t0 = time.perf_counter()
    if color_rgb.dtype != np.uint8:
        color_u8 = np.clip(color_rgb, 0, 255).astype(np.uint8)
    else:
        color_u8 = color_rgb
    color_o3d = o3d.geometry.Image(color_u8)
    depth_o3d = o3d.geometry.Image(depth_m)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=float(cfg.depth_scale),
        depth_trunc=float(cfg.depth_trunc_m),
        convert_rgb_to_intensity=bool(cfg.convert_rgb_to_intensity),
    )
    depth_after = np.asarray(rgbd.depth)
    stats_after = depth_input_stats(
        depth_after,
        odometry_z_min_m=cfg.depth_min_m,
        odometry_z_max_m=cfg.depth_max_m,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return RgbdImageBuildResult(
        rgbd_image=rgbd,
        depth_stats_before=stats_before,
        depth_stats_after=stats_after,
        input_prepare_ms=elapsed_ms,
    )


def make_odometry_option(config: RgbdOdometryConfig | None = None):
    o3d = _require_open3d()
    cfg = config or RgbdOdometryConfig()
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_min = float(cfg.depth_min_m)
    option.depth_max = float(cfg.depth_max_m)
    option.depth_diff_max = float(cfg.depth_diff_max_m)
    return option


def transform_magnitude(transform_4x4: np.ndarray) -> tuple[float, float]:
    T = np.asarray(transform_4x4, dtype=np.float64)
    translation = float(np.linalg.norm(T[:3, 3]))
    R = T[:3, :3]
    trace = float(np.trace(R))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cos_theta)))
    return translation, rotation_deg


def relative_transform_target_source(
    T_world_source: np.ndarray,
    T_world_target: np.ndarray,
) -> np.ndarray:
    """Return T_target_source such that P_target = T_target_source @ P_source."""
    T_s = np.asarray(T_world_source, dtype=np.float64)
    T_t = np.asarray(T_world_target, dtype=np.float64)
    return np.linalg.inv(T_t) @ T_s


def accumulate_odom_pose(
    T_odom_camera_source: np.ndarray,
    transform_target_source: np.ndarray,
) -> np.ndarray:
    """Accumulate local odometry pose when Open3D returns P_target = M @ P_source."""
    T_src = np.asarray(T_odom_camera_source, dtype=np.float64)
    M = np.asarray(transform_target_source, dtype=np.float64)
    return T_src @ np.linalg.inv(M)


def invert_rigid_transform(transform_4x4: np.ndarray) -> np.ndarray:
    T = np.asarray(transform_4x4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def transform_is_finite(transform_4x4: np.ndarray | None) -> bool:
    if transform_4x4 is None:
        return False
    return bool(np.isfinite(np.asarray(transform_4x4)).all())


def information_matrix_diagnostics(information_matrix: np.ndarray | None) -> dict[str, Any]:
    if information_matrix is None:
        return {
            "shape": None,
            "finite": False,
            "symmetric": False,
            "trace": None,
            "condition_number": None,
        }
    info = np.asarray(information_matrix, dtype=np.float64)
    finite = bool(np.isfinite(info).all())
    symmetric = bool(np.allclose(info, info.T, atol=1e-6, rtol=1e-6)) if finite else False
    trace = float(np.trace(info)) if finite else None
    condition_number = None
    if finite and info.shape == (6, 6):
        try:
            condition_number = float(np.linalg.cond(info))
        except np.linalg.LinAlgError:
            condition_number = None
    return {
        "shape": list(info.shape),
        "finite": finite,
        "symmetric": symmetric,
        "trace": trace,
        "condition_number": condition_number,
    }


def estimate_rgbd_pair_motion(
    source_rgb: np.ndarray,
    source_depth_m: np.ndarray,
    target_rgb: np.ndarray,
    target_depth_m: np.ndarray,
    rgb_intrinsics: Any,
    *,
    config: RgbdOdometryConfig | None = None,
    source_frame: int | None = None,
    target_frame: int | None = None,
) -> PairwiseOdometryResult:
    """Estimate relative motion from source RGB-D to target RGB-D using Open3D Hybrid odometry."""
    o3d = _require_open3d()
    cfg = config or RgbdOdometryConfig()
    total_t0 = time.perf_counter()

    source_build = build_rgbd_image_from_arrays(source_rgb, source_depth_m, config=cfg)
    target_build = build_rgbd_image_from_arrays(target_rgb, target_depth_m, config=cfg)
    input_prepare_ms = source_build.input_prepare_ms + target_build.input_prepare_ms

    option = make_odometry_option(cfg)
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    odometry_t0 = time.perf_counter()
    success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_build.rgbd_image,
        target_build.rgbd_image,
        rgb_intrinsics,
        np.eye(4, dtype=np.float64),
        jacobian,
        option,
    )
    odometry_ms = (time.perf_counter() - odometry_t0) * 1000.0
    total_ms = (time.perf_counter() - total_t0) * 1000.0

    if not success:
        return PairwiseOdometryResult(
            success=False,
            source_frame=source_frame,
            target_frame=target_frame,
            transform_target_source=None,
            information_matrix=None,
            input_prepare_ms=input_prepare_ms,
            odometry_ms=odometry_ms,
            total_ms=total_ms,
            translation_magnitude_m=None,
            rotation_magnitude_deg=None,
            failure_reason="open3d_compute_rgbd_odometry_failed",
        )

    transform_np = np.asarray(transform, dtype=np.float64)
    info_np = np.asarray(information, dtype=np.float64)
    if not transform_is_finite(transform_np) or not np.isfinite(info_np).all():
        return PairwiseOdometryResult(
            success=False,
            source_frame=source_frame,
            target_frame=target_frame,
            transform_target_source=None,
            information_matrix=None,
            input_prepare_ms=input_prepare_ms,
            odometry_ms=odometry_ms,
            total_ms=total_ms,
            translation_magnitude_m=None,
            rotation_magnitude_deg=None,
            failure_reason="non_finite_transform_or_information",
        )

    translation_m, rotation_deg = transform_magnitude(transform_np)
    return PairwiseOdometryResult(
        success=True,
        source_frame=source_frame,
        target_frame=target_frame,
        transform_target_source=transform_np,
        information_matrix=info_np,
        input_prepare_ms=input_prepare_ms,
        odometry_ms=odometry_ms,
        total_ms=total_ms,
        translation_magnitude_m=translation_m,
        rotation_magnitude_deg=rotation_deg,
        failure_reason=None,
    )


def rgbd_odometry_config_from_mapping(cfg: Mapping[str, Any] | None) -> RgbdOdometryConfig:
    raw = dict(cfg or {})
    return RgbdOdometryConfig(
        depth_scale=float(raw.get("depth_scale", DEFAULT_DEPTH_SCALE)),
        depth_trunc_m=float(raw.get("depth_trunc_m", DEFAULT_DEPTH_TRUNC_M)),
        depth_min_m=float(raw.get("depth_min_m", DEFAULT_DEPTH_MIN_M)),
        depth_max_m=float(raw.get("depth_max_m", DEFAULT_DEPTH_MAX_M)),
        depth_diff_max_m=float(raw.get("depth_diff_max_m", DEFAULT_DEPTH_DIFF_MAX_M)),
        convert_rgb_to_intensity=bool(raw.get("convert_rgb_to_intensity", True)),
    )
