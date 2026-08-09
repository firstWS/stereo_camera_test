"""Offline RGB–Depth geometry for Phase 2 derived cup depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from stereo_types import BBox, DepthEstimate


@dataclass(frozen=True)
class RgbDepthCalibration:
    k_rgb: np.ndarray
    k_depth: np.ndarray
    t_rgb_to_depth: np.ndarray  # 4x4 homogeneous; maps P_rgb -> P_depth (meters)
    rgb_width: int
    rgb_height: int
    depth_width: int
    depth_height: int


@dataclass(frozen=True)
class DepthPairingConfig:
    max_rgb_depth_delta_us: int = 33_333
    depth_is_millimeters: bool = True
    z_min_m: float = 0.05
    z_max_m: float = 40.0


METHOD_BASELINE_MEDIAN = "baseline_median"
METHOD_ROBUST_NEAR_QUANTILE = "robust_near_quantile"
DEFAULT_NEAR_QUANTILE = 0.25
DEFAULT_MIN_NEAR_POINTS = 1
QUANTILE_METHOD = "linear"


@dataclass(frozen=True)
class CupDepthEstimatorConfig:
    method: str = METHOD_ROBUST_NEAR_QUANTILE
    near_quantile: float = DEFAULT_NEAR_QUANTILE
    min_near_points: int = DEFAULT_MIN_NEAR_POINTS


def cup_depth_estimator_config_from_mapping(cfg: Mapping[str, Any] | None) -> CupDepthEstimatorConfig:
    raw = dict(cfg or {})
    method = str(raw.get("method", METHOD_ROBUST_NEAR_QUANTILE))
    if method not in (METHOD_BASELINE_MEDIAN, METHOD_ROBUST_NEAR_QUANTILE):
        raise ValueError(f"unsupported cup_depth.method: {method}")
    near_quantile = float(raw.get("near_quantile", DEFAULT_NEAR_QUANTILE))
    if not (0.0 < near_quantile < 1.0) or not np.isfinite(near_quantile):
        raise ValueError("cup_depth.near_quantile must satisfy 0 < q < 1")
    min_near_points = int(raw.get("min_near_points", DEFAULT_MIN_NEAR_POINTS))
    if min_near_points < 1:
        raise ValueError("cup_depth.min_near_points must be >= 1")
    return CupDepthEstimatorConfig(
        method=method,
        near_quantile=near_quantile,
        min_near_points=min_near_points,
    )


@dataclass(frozen=True)
class DepthTimestampMatch:
    depth_frame_number: int
    depth_device_timestamp_us: int
    depth_file_path: Any
    depth_row: dict[str, Any]
    rgb_depth_delta_us: int


def _intrinsic_matrix(entry: Mapping[str, Any]) -> np.ndarray:
    intrinsic = entry.get("intrinsic") or {}
    fx = float(intrinsic.get("fx", 0.0))
    fy = float(intrinsic.get("fy", 0.0))
    cx = float(intrinsic.get("cx", 0.0))
    cy = float(intrinsic.get("cy", 0.0))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _find_intrinsic(calibration: Mapping[str, Any], frame: str) -> dict[str, Any] | None:
    for entry in calibration.get("intrinsics", []):
        if entry.get("frame") == frame and entry.get("success"):
            return entry
    return None


def _find_extrinsic(calibration: Mapping[str, Any], from_frame: str, to_frame: str) -> dict[str, Any] | None:
    for entry in calibration.get("extrinsics", []):
        if (
            entry.get("from_frame") == from_frame
            and entry.get("to_frame") == to_frame
            and entry.get("success")
        ):
            return entry
    return None


def extrinsic_to_homogeneous_4x4(extrinsic: Mapping[str, Any]) -> np.ndarray:
    """Build 4x4 transform from SDK-exported extrinsic (translation in millimeters)."""
    rotation = np.asarray(extrinsic["rotation"], dtype=np.float64).reshape(3, 3)
    translation_mm = np.asarray(extrinsic["translation"], dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_mm / 1000.0
    return transform


def load_rgb_depth_calibration(
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
) -> RgbDepthCalibration:
    rgb_entry = _find_intrinsic(intrinsics, "RGB")
    depth_entry = _find_intrinsic(intrinsics, "DEPTH")
    if rgb_entry is None or depth_entry is None:
        raise ValueError("RGB and DEPTH intrinsics are required for RGB–Depth geometry")
    rgb_intr = rgb_entry.get("intrinsic") or {}
    depth_intr = depth_entry.get("intrinsic") or {}
    extrinsic_entry = _find_extrinsic(extrinsics, "RGB", "DEPTH")
    if extrinsic_entry is None or extrinsic_entry.get("extrinsic") is None:
        raise ValueError("RGB->DEPTH extrinsic is required for RGB–Depth geometry")
    return RgbDepthCalibration(
        k_rgb=_intrinsic_matrix(rgb_entry),
        k_depth=_intrinsic_matrix(depth_entry),
        t_rgb_to_depth=extrinsic_to_homogeneous_4x4(extrinsic_entry["extrinsic"]),
        rgb_width=int(rgb_intr.get("width", 0)),
        rgb_height=int(rgb_intr.get("height", 0)),
        depth_width=int(depth_intr.get("width", 0)),
        depth_height=int(depth_intr.get("height", 0)),
    )


def normalize_depth_image(depth: np.ndarray) -> np.ndarray:
    """Return a 2D depth image; only shape normalization, no value conversion."""
    if depth.ndim == 2:
        return depth
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[:, :, 0]
    raise ValueError(
        f"depth image must be (H, W) or (H, W, 1), got shape {tuple(depth.shape)}"
    )


def depth_pairing_config_from_mapping(cfg: Mapping[str, Any] | None) -> DepthPairingConfig:
    raw = dict(cfg or {})
    return DepthPairingConfig(
        max_rgb_depth_delta_us=int(raw.get("max_rgb_depth_delta_us", 33_333)),
        depth_is_millimeters=bool(raw.get("depth_is_millimeters", True)),
        z_min_m=float(raw.get("z_min_m", 0.05)),
        z_max_m=float(raw.get("z_max_m", 40.0)),
    )


def depth_meters_from_raw(
    depth_raw: np.ndarray,
    *,
    depth_scale: float | None,
    depth_is_millimeters: bool,
) -> np.ndarray:
    depth_raw = normalize_depth_image(depth_raw)
    scale = 1.0 if depth_scale in (None, "") else float(depth_scale)
    depth = depth_raw.astype(np.float64) * scale
    if depth_is_millimeters:
        depth /= 1000.0
    return depth


def build_depth_timestamp_index(
    depth_rows: Sequence[tuple[dict[str, Any], Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[Any]]:
    """Return sorted timestamps and parallel depth metadata."""
    if not depth_rows:
        return np.array([], dtype=np.int64), [], []
    order = sorted(
        range(len(depth_rows)),
        key=lambda index: int(depth_rows[index][0].get("device_timestamp_us") or 0),
    )
    timestamps = np.array(
        [int(depth_rows[index][0].get("device_timestamp_us") or 0) for index in order],
        dtype=np.int64,
    )
    rows = [depth_rows[index][0] for index in order]
    paths = [depth_rows[index][1] for index in order]
    return timestamps, rows, paths


def match_nearest_depth_timestamp(
    rgb_timestamp_us: int,
    depth_timestamps_us: np.ndarray,
    depth_rows: Sequence[dict[str, Any]],
    depth_paths: Sequence[Any],
    *,
    max_delta_us: int,
) -> DepthTimestampMatch | None:
    if depth_timestamps_us.size == 0:
        return None
    index = int(np.searchsorted(depth_timestamps_us, rgb_timestamp_us))
    candidates: list[int] = []
    if 0 <= index < depth_timestamps_us.size:
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    best_index = min(
        candidates,
        key=lambda candidate: abs(int(depth_timestamps_us[candidate]) - rgb_timestamp_us),
    )
    depth_ts = int(depth_timestamps_us[best_index])
    delta = abs(depth_ts - rgb_timestamp_us)
    if delta > max_delta_us:
        return None
    row = depth_rows[best_index]
    return DepthTimestampMatch(
        depth_frame_number=int(row.get("frame_number") or 0),
        depth_device_timestamp_us=depth_ts,
        depth_file_path=depth_paths[best_index],
        depth_row=dict(row),
        rgb_depth_delta_us=delta,
    )


def _depth_pixels_to_rgb_points(
    depth_m: np.ndarray,
    calib: RgbDepthCalibration,
    *,
    z_min_m: float,
    z_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth_m.shape[:2]
    v_d, u_d = np.indices((height, width), dtype=np.float64)
    z_d = depth_m.astype(np.float64)
    valid = np.isfinite(z_d) & (z_d >= z_min_m) & (z_d <= z_max_m)
    if not np.any(valid):
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    fx_d = calib.k_depth[0, 0]
    fy_d = calib.k_depth[1, 1]
    cx_d = calib.k_depth[0, 2]
    cy_d = calib.k_depth[1, 2]

    z = z_d[valid]
    u = u_d[valid]
    v = v_d[valid]
    x_d = (u - cx_d) * z / fx_d
    y_d = (v - cy_d) * z / fy_d

    points_depth = np.stack([x_d, y_d, z, np.ones_like(z)], axis=0)
    t_depth_to_rgb = np.linalg.inv(calib.t_rgb_to_depth)
    points_rgb = t_depth_to_rgb @ points_depth
    x_rgb = points_rgb[0]
    y_rgb = points_rgb[1]
    z_rgb = points_rgb[2]
    front = z_rgb > z_min_m
    return x_rgb[front], y_rgb[front], z_rgb[front]


def _invalid_depth_estimate(*, valid_pixel_ratio: float, notes: str) -> DepthEstimate:
    return DepthEstimate(
        track="A_rgbd",
        X=0.0,
        Y=0.0,
        Z=0.0,
        disparity=None,
        valid=False,
        valid_pixel_ratio=valid_pixel_ratio,
        notes=notes,
    )


def estimate_bbox_camera_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    config: CupDepthEstimatorConfig,
    valid_pixel_ratio: float,
) -> DepthEstimate:
    """Estimate representative RGB-camera XYZ from bbox-projected valid points."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0.0)
    x = x[finite]
    y = y[finite]
    z = z[finite]
    if x.size == 0:
        return _invalid_depth_estimate(valid_pixel_ratio=valid_pixel_ratio, notes="no_finite_positive_z")

    if config.method == METHOD_BASELINE_MEDIAN:
        return DepthEstimate(
            track="A_rgbd",
            X=float(np.median(x)),
            Y=float(np.median(y)),
            Z=float(np.median(z)),
            disparity=None,
            valid=True,
            valid_pixel_ratio=valid_pixel_ratio,
            notes="rgb_bbox_projected_median",
        )

    qz = float(np.quantile(z, config.near_quantile, method=QUANTILE_METHOD))
    near = z <= qz
    near_count = int(near.sum())
    if near_count < config.min_near_points:
        return _invalid_depth_estimate(
            valid_pixel_ratio=valid_pixel_ratio,
            notes="insufficient_near_quantile_points",
        )
    return DepthEstimate(
        track="A_rgbd",
        X=float(np.median(x[near])),
        Y=float(np.median(y[near])),
        Z=float(np.median(z[near])),
        disparity=None,
        valid=True,
        valid_pixel_ratio=valid_pixel_ratio,
        notes="rgb_bbox_robust_near_quantile_median",
    )


def estimate_cup_xyz_from_rgb_bbox(
    depth_m: np.ndarray,
    bbox: BBox,
    calib: RgbDepthCalibration,
    *,
    min_valid_ratio: float,
    z_min_m: float = 0.05,
    z_max_m: float = 40.0,
    cup_depth_config: CupDepthEstimatorConfig | None = None,
) -> DepthEstimate:
    """Project native depth pixels into the RGB image and sample inside ``bbox``."""
    depth_cfg = cup_depth_config or CupDepthEstimatorConfig(method=METHOD_BASELINE_MEDIAN)
    x_rgb, y_rgb, z_rgb = _depth_pixels_to_rgb_points(
        depth_m,
        calib,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
    )
    if x_rgb.size == 0:
        return _invalid_depth_estimate(valid_pixel_ratio=0.0, notes="no_projected_depth")

    fx_r = calib.k_rgb[0, 0]
    fy_r = calib.k_rgb[1, 1]
    cx_r = calib.k_rgb[0, 2]
    cy_r = calib.k_rgb[1, 2]
    u_rgb = fx_r * x_rgb / z_rgb + cx_r
    v_rgb = fy_r * y_rgb / z_rgb + cy_r

    x1, y1, x2, y2 = bbox.xyxy
    inside = (u_rgb >= x1) & (u_rgb <= x2) & (v_rgb >= y1) & (v_rgb <= y2)
    if not np.any(inside):
        return _invalid_depth_estimate(valid_pixel_ratio=0.0, notes="no_depth_in_rgb_bbox")

    x_sel = x_rgb[inside]
    y_sel = y_rgb[inside]
    z_sel = z_rgb[inside]
    bbox_area = max(1.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
    ratio = float(x_sel.size) / bbox_area
    if ratio < min_valid_ratio:
        return _invalid_depth_estimate(valid_pixel_ratio=ratio, notes="insufficient_projected_depth")

    return estimate_bbox_camera_xyz(
        x_sel,
        y_sel,
        z_sel,
        config=depth_cfg,
        valid_pixel_ratio=ratio,
    )
