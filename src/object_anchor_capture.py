"""Timed raw RGB capture for Object Anchor training data."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAPTURE_MANIFEST_FIELDS = (
    "filename",
    "capture_type",
    "captured_at",
    "frame_width",
    "frame_height",
    "capture_interval_seconds",
    "target_count",
    "camera_serial",
    "object_anchor_detected",
    "object_anchor_confidence",
    "apriltag_detected",
    "loaded_model_path",
)


@dataclass(frozen=True)
class ObjectAnchorCaptureSettings:
    capture_type: str
    target_count: int = 100
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.capture_type not in {"positive", "negative"}:
            raise ValueError("capture_type must be 'positive' or 'negative'")
        if self.target_count <= 0:
            raise ValueError("target_count must be greater than zero")
        if self.interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be greater than zero")


class ObjectAnchorCaptureSession:
    """Append-only capture session; only successful image writes advance the count."""

    def __init__(
        self,
        root_dir: Path,
        settings: ObjectAnchorCaptureSettings,
        *,
        camera_serial: str,
        loaded_model_path: str,
        session_time: datetime | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.settings = settings
        self.camera_serial = str(camera_serial)
        self.loaded_model_path = str(loaded_model_path)
        self.session_time = session_time or datetime.now().astimezone()
        self.session_stamp = self.session_time.strftime("%Y%m%d_%H%M%S")
        self.image_dir = self.root_dir / settings.capture_type / "images"
        self.label_dir = self.root_dir / "negative" / "labels"
        self.manifest_path = self.root_dir / "capture_manifest.csv"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if settings.capture_type == "negative":
            self.label_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0
        self.last_saved_monotonic: float | None = None
        self.last_filename = "-"

    @property
    def complete(self) -> bool:
        return self.saved_count >= self.settings.target_count

    def due(self, now_monotonic: float | None = None) -> bool:
        if self.complete:
            return False
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        return (
            self.last_saved_monotonic is None
            or now - self.last_saved_monotonic >= self.settings.interval_seconds
        )

    def _next_paths(self) -> tuple[str, Path, Path | None]:
        sequence = self.saved_count + 1
        while True:
            stem = f"tissue_{self.settings.capture_type}_{self.session_stamp}_{sequence:03d}"
            filename = f"{stem}.jpg"
            image_path = self.image_dir / filename
            label_path = self.label_dir / f"{stem}.txt" if self.settings.capture_type == "negative" else None
            if not image_path.exists() and (label_path is None or not label_path.exists()):
                return filename, image_path, label_path
            sequence += 1

    def _append_manifest(self, row: dict[str, Any]) -> None:
        write_header = not self.manifest_path.exists() or self.manifest_path.stat().st_size == 0
        with self.manifest_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CAPTURE_MANIFEST_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())

    def save(
        self,
        bgr: np.ndarray,
        *,
        now_monotonic: float | None = None,
        captured_at: datetime | None = None,
        object_anchor_detected: bool = False,
        object_anchor_confidence: float | None = None,
        apriltag_detected: bool = False,
    ) -> Path | None:
        if self.complete:
            return None
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not self.due(now):
            return None
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("bgr must be an HxWx3 OpenCV image")

        filename, image_path, label_path = self._next_paths()
        if not cv2.imwrite(str(image_path), bgr):
            return None

        try:
            if label_path is not None:
                label_path.touch(exist_ok=False)
            h, w = bgr.shape[:2]
            timestamp = (captured_at or datetime.now().astimezone()).astimezone().isoformat()
            self._append_manifest(
                {
                    "filename": filename,
                    "capture_type": self.settings.capture_type,
                    "captured_at": timestamp,
                    "frame_width": w,
                    "frame_height": h,
                    "capture_interval_seconds": self.settings.interval_seconds,
                    "target_count": self.settings.target_count,
                    "camera_serial": self.camera_serial,
                    "object_anchor_detected": bool(object_anchor_detected),
                    "object_anchor_confidence": (
                        "" if object_anchor_confidence is None else float(object_anchor_confidence)
                    ),
                    "apriltag_detected": bool(apriltag_detected),
                    "loaded_model_path": self.loaded_model_path,
                }
            )
        except Exception:
            image_path.unlink(missing_ok=True)
            if label_path is not None:
                label_path.unlink(missing_ok=True)
            raise

        self.saved_count += 1
        self.last_saved_monotonic = now
        self.last_filename = filename
        return image_path

