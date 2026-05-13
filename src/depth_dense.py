"""Track A: dense SGBM disparity + median inside detection box."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from stereo_types import BBox, DepthEstimate
from triangulate import xyz_from_disparity_pixel


@dataclass
class SGBMConfig:
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


def make_sgbm(cfg: SGBMConfig) -> cv2.StereoSGBM:
    return cv2.StereoSGBM.create(
        minDisparity=cfg.min_disparity,
        numDisparities=cfg.num_disparities,
        blockSize=cfg.block_size,
        P1=8 * 3 * cfg.block_size**2,
        P2=32 * 3 * cfg.block_size**2,
        disp12MaxDiff=cfg.disp12_max_diff,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window_size,
        speckleRange=cfg.speckle_range,
        preFilterCap=cfg.pre_filter_cap,
        mode=cfg.mode,
    )


def compute_disparity_map(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    sgbm: cv2.StereoSGBM,
    scale_down: int = 1,
) -> np.ndarray:
    """
    Returns float disparity in pixels (same scale as input if scale_down=1).
    SGBM output is fixed-point * 16; convert to float.
    """
    gl, gr = gray_left, gray_right
    if scale_down > 1:
        gl = cv2.resize(gl, None, fx=1 / scale_down, fy=1 / scale_down, interpolation=cv2.INTER_AREA)
        gr = cv2.resize(gr, None, fx=1 / scale_down, fy=1 / scale_down, interpolation=cv2.INTER_AREA)
    raw = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    if scale_down > 1:
        raw = cv2.resize(raw, (gray_left.shape[1], gray_left.shape[0]), interpolation=cv2.INTER_LINEAR)
        raw *= float(scale_down)
    return raw


def depth_dense_track_a(
    disparity: np.ndarray,
    bbox: BBox,
    Q: np.ndarray,
    min_disp: float = 1.0,
) -> DepthEstimate:
    x1, y1, x2, y2 = map(int, bbox.xyxy)
    h, w = disparity.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return DepthEstimate(
            track="A_dense", X=0, Y=0, Z=0, disparity=None, valid=False, notes="bad_roi"
        )
    roi = disparity[y1 : y2 + 1, x1 : x2 + 1]
    valid = roi > min_disp
    if not np.any(valid):
        return DepthEstimate(
            track="A_dense",
            X=0,
            Y=0,
            Z=0,
            disparity=None,
            valid=False,
            valid_pixel_ratio=0.0,
            notes="no_valid_disp",
        )
    d_med = float(np.median(roi[valid]))
    ratio = float(np.mean(valid))
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    try:
        X, Y, Z = xyz_from_disparity_pixel(u, v, d_med, Q)
    except ValueError as e:
        return DepthEstimate(
            track="A_dense",
            X=0,
            Y=0,
            Z=0,
            disparity=d_med,
            valid=False,
            valid_pixel_ratio=ratio,
            notes=str(e),
        )
    return DepthEstimate(
        track="A_dense",
        X=X,
        Y=Y,
        Z=Z,
        disparity=d_med,
        valid=True,
        valid_pixel_ratio=ratio,
        notes="",
    )
