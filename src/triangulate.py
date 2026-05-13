"""Triangle: disparity -> 3D using Q matrix (OpenCV convention)."""

from __future__ import annotations

import numpy as np


def xyz_from_disparity_pixel(
    u: float,
    v: float,
    disp: float,
    Q: np.ndarray,
) -> tuple[float, float, float]:
    """
    Rectified stereo: disparity d = x_left - x_right (pixels), OpenCV SGBM convention.
    """
    if disp <= 1e-6:
        raise ValueError("disparity must be positive")
    hom = Q @ np.array([u, v, disp, 1.0], dtype=np.float64)
    if abs(hom[3]) < 1e-9:
        raise ValueError("invalid homogeneous w")
    X, Y, Z, W = hom
    return float(X / W), float(Y / W), float(Z / W)
