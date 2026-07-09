from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BBox:
    xyxy: tuple[float, float, float, float]
    confidence: float = 0.0
    class_id: int = -1
    label: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, y2)


@dataclass
class DetectionResult:
    """Fixed schema from DetectorAdapter."""

    boxes: list[BBox]
    image_shape_hw: tuple[int, int]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferencePoint:
    """Point in rectified left image used for depth (pixel coords, subpixel ok)."""

    u: float
    v: float
    source_bbox_index: int = 0


@dataclass
class DepthEstimate:
    track: str
    X: float
    Y: float
    Z: float
    disparity: float | None
    valid: bool
    valid_pixel_ratio: float | None = None
    notes: str = ""


@dataclass
class StereoFrame:
    left_bgr: np.ndarray
    right_bgr: np.ndarray

    def gray_left(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.left_bgr, cv2.COLOR_BGR2GRAY)

    def gray_right(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.right_bgr, cv2.COLOR_BGR2GRAY)
