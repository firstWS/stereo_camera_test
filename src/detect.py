"""DetectorAdapter: pluggable 2D detection with fixed output schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from stereo_types import BBox, DetectionResult, ReferencePoint


class DetectorAdapter(ABC):
    @abstractmethod
    def predict(self, bgr: np.ndarray) -> DetectionResult:
        raise NotImplementedError


class UltralyticsYOLODetector(DetectorAdapter):
    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int | None = 640,
        device: str | None = None,
        class_ids: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.class_ids = class_ids

    def predict(self, bgr: np.ndarray) -> DetectionResult:
        h, w = bgr.shape[:2]
        kwargs: dict[str, Any] = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            kwargs["device"] = self.device
        if self.class_ids:
            kwargs["classes"] = self.class_ids
        results = self._model.predict(bgr, **kwargs)
        boxes: list[BBox] = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return DetectionResult(boxes=[], image_shape_hw=(h, w))
        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = getattr(r, "names", None) or getattr(self._model, "names", {})
        for i in range(len(xyxy)):
            label = str(names.get(cls[i], cls[i])) if isinstance(names, dict) else None
            x1, y1, x2, y2 = map(float, xyxy[i])
            boxes.append(
                BBox(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(conf[i]),
                    class_id=int(cls[i]),
                    label=label,
                )
            )
        return DetectionResult(boxes=boxes, image_shape_hw=(h, w))


class DummyCenterDetector(DetectorAdapter):
    """Returns a synthetic box at image center for pipeline tests without weights."""

    def __init__(self, frac: float = 0.2) -> None:
        self.frac = frac

    def predict(self, bgr: np.ndarray) -> DetectionResult:
        h, w = bgr.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        dx, dy = w * self.frac * 0.5, h * self.frac * 0.5
        box = BBox(
            xyxy=(cx - dx, cy - dy, cx + dx, cy + dy),
            confidence=1.0,
            class_id=0,
            label="dummy",
        )
        return DetectionResult(boxes=[box], image_shape_hw=(h, w))


def pick_primary_box(dets: DetectionResult) -> BBox | None:
    if not dets.boxes:
        return None
    return max(dets.boxes, key=lambda b: b.confidence)


def box_to_reference(dets: DetectionResult, box_index: int = 0) -> ReferencePoint | None:
    if box_index >= len(dets.boxes):
        return None
    b = dets.boxes[box_index]
    u, v = b.center
    return ReferencePoint(u=u, v=v, source_bbox_index=box_index)
