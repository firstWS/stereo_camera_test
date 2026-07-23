"""Detector interface for Object Anchor keypoint models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ObjectAnchorDetection:
    keypoints_xy: np.ndarray
    keypoint_confidences: np.ndarray
    bbox_xyxy: tuple[float, float, float, float] | None = None
    score: float = 1.0
    class_id: int = 0
    label: str = "object_anchor"
    keypoint_visibility: np.ndarray | None = None


class ObjectAnchorDetector(ABC):
    @abstractmethod
    def predict(self, bgr: np.ndarray) -> list[ObjectAnchorDetection]:
        raise NotImplementedError


class UltralyticsPoseDetector(ObjectAnchorDetector):
    """YOLO-Pose adapter. A custom tissue-box pose weight file is required."""

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int | None = 640,
        device: str | None = None,
        max_detections: int = 1,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = imgsz
        self.device = device
        self.max_detections = int(max_detections)

    def predict(self, bgr: np.ndarray) -> list[ObjectAnchorDetection]:
        kwargs: dict[str, Any] = {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            kwargs["device"] = self.device

        results = self._model.predict(bgr, **kwargs)
        if not results:
            return []
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return []

        xy = result.keypoints.xy.cpu().numpy()
        conf_tensor = result.keypoints.conf
        if conf_tensor is None:
            keypoint_conf = np.ones(xy.shape[:2], dtype=np.float64)
        else:
            keypoint_conf = conf_tensor.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", None) or getattr(self._model, "names", {})

        count = min(len(xy), len(boxes))
        detections: list[ObjectAnchorDetection] = []
        for index in range(count):
            class_id = int(classes[index])
            label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id)
            detections.append(
                ObjectAnchorDetection(
                    keypoints_xy=np.asarray(xy[index], dtype=np.float64),
                    keypoint_confidences=np.asarray(keypoint_conf[index], dtype=np.float64),
                    bbox_xyxy=tuple(float(value) for value in boxes[index]),
                    score=float(scores[index]),
                    class_id=class_id,
                    label=label,
                )
            )
        return detections
