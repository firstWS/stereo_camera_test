"""Track B: sparse NCC matching along epipolar line (rectified images)."""

from __future__ import annotations

import cv2
import numpy as np

from stereo_types import DepthEstimate
from triangulate import xyz_from_disparity_pixel


def depth_sparse_track_b(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    u: float,
    v: float,
    Q: np.ndarray,
    template_radius: int = 7,
    max_disparity: int = 128,
    min_disparity: int = 1,
) -> DepthEstimate:
    """
    Search for best match on same row in right image (rectified stereo).
    """
    h, w = gray_left.shape
    ui, vi = int(round(u)), int(round(v))
    if vi < template_radius or vi >= h - template_radius or ui < template_radius or ui >= w - template_radius:
        return DepthEstimate(
            track="B_sparse", X=0, Y=0, Z=0, disparity=None, valid=False, notes="point_margin"
        )
    tpl = gray_left[
        vi - template_radius : vi + template_radius + 1,
        ui - template_radius : ui + template_radius + 1,
    ].astype(np.float32)
    tpl -= float(np.mean(tpl))
    denom = float(np.linalg.norm(tpl))
    if denom < 1e-6:
        return DepthEstimate(
            track="B_sparse", X=0, Y=0, Z=0, disparity=None, valid=False, notes="flat_template"
        )
    tpl /= denom

    best_ncc = -1.0
    best_d = 0
    ur_start = max(template_radius, ui - max_disparity)
    ur_end = ui - min_disparity
    if ur_end < ur_start:
        return DepthEstimate(
            track="B_sparse", X=0, Y=0, Z=0, disparity=None, valid=False, notes="invalid_search_range"
        )

    for ur in range(ur_start, ur_end + 1):
        patch = gray_right[
            vi - template_radius : vi + template_radius + 1,
            ur - template_radius : ur + template_radius + 1,
        ].astype(np.float32)
        patch -= float(np.mean(patch))
        pn = float(np.linalg.norm(patch))
        if pn < 1e-6:
            continue
        ncc = float(np.sum(tpl * (patch / pn)))
        if ncc > best_ncc:
            best_ncc = ncc
            best_d = ui - ur

    if best_d <= 0 or best_ncc < 0.3:
        return DepthEstimate(
            track="B_sparse",
            X=0,
            Y=0,
            Z=0,
            disparity=float(best_d) if best_d > 0 else None,
            valid=False,
            notes=f"low_ncc={best_ncc:.3f}",
        )
    try:
        X, Y, Z = xyz_from_disparity_pixel(u, v, float(best_d), Q)
    except ValueError as e:
        return DepthEstimate(
            track="B_sparse",
            X=0,
            Y=0,
            Z=0,
            disparity=float(best_d),
            valid=False,
            notes=str(e),
        )
    return DepthEstimate(
        track="B_sparse",
        X=X,
        Y=Y,
        Z=Z,
        disparity=float(best_d),
        valid=True,
        notes=f"ncc={best_ncc:.3f}",
    )
