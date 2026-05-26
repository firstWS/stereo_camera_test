"""RGB-D: depth map ROIs and pinhole back-projection (meters)."""

from __future__ import annotations

import numpy as np

from stereo_types import BBox, DepthEstimate


def backproject_xyz_from_uvz(u: float, v: float, z_m: float, K: np.ndarray) -> tuple[float, float, float]:
    """Camera frame XYZ in meters; z_m is depth along optical axis (positive forward)."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    X = (float(u) - cx) * z_m / fx
    Y = (float(v) - cy) * z_m / fy
    return X, Y, z_m


def median_depth_in_bbox(
    depth_m: np.ndarray,
    bbox: BBox,
    *,
    z_min_m: float = 0.05,
    z_max_m: float = 40.0,
) -> tuple[float | None, float]:
    """
    Median of finite depths inside ``bbox`` (inclusive integer bounds).

    Returns ``(z_median_m, valid_pixel_ratio)`` where ratio is w.r.t. ROI area.
    """
    x1, y1, x2, y2 = map(int, bbox.xyxy)
    h, w = depth_m.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    roi = depth_m[y1 : y2 + 1, x1 : x2 + 1].astype(np.float64)
    valid = np.isfinite(roi) & (roi >= z_min_m) & (roi <= z_max_m)
    area = float(roi.size)
    if area <= 0 or not np.any(valid):
        return None, 0.0
    ratio = float(np.sum(valid)) / area
    z_med = float(np.median(roi[valid]))
    return z_med, ratio


def depth_estimate_rgbd_bbox(
    depth_m: np.ndarray,
    bbox: BBox,
    K: np.ndarray,
    *,
    min_valid_ratio: float = 0.03,
    z_min_m: float = 0.05,
    z_max_m: float = 40.0,
) -> DepthEstimate:
    """Track A-style estimate: ROI median depth (like dense stereo) with ``(u,v)`` at box center."""
    x1, y1, x2, y2 = bbox.xyxy
    u = (float(x1) + float(x2)) * 0.5
    v = (float(y1) + float(y2)) * 0.5
    z_med, ratio = median_depth_in_bbox(depth_m, bbox, z_min_m=z_min_m, z_max_m=z_max_m)
    if z_med is None or ratio < min_valid_ratio:
        return DepthEstimate(
            track="A_rgbd",
            X=0.0,
            Y=0.0,
            Z=0.0,
            disparity=None,
            valid=False,
            valid_pixel_ratio=ratio,
            notes="no_valid_depth",
        )
    X, Y, Z = backproject_xyz_from_uvz(u, v, z_med, K)
    return DepthEstimate(
        track="A_rgbd",
        X=X,
        Y=Y,
        Z=Z,
        disparity=None,
        valid=True,
        valid_pixel_ratio=ratio,
        notes="rgbd_median_bbox",
    )


def orbbec_sparse_stub() -> DepthEstimate:
    """Track B is not used for Orbbec RGB-D path."""
    return DepthEstimate(
        track="B_sparse",
        X=0.0,
        Y=0.0,
        Z=0.0,
        disparity=None,
        valid=False,
        valid_pixel_ratio=None,
        notes="orbbec_no_sparse",
    )
