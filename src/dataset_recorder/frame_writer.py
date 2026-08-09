"""Per-stream frame file writer for Phase 2 datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameSnapshot:
    stream: str
    sequence: int
    width: int
    height: int
    format_name: str
    data: bytes
    depth_scale: float | None


def decode_snapshot(snapshot: FrameSnapshot) -> np.ndarray | None:
    import cv2

    format_name = snapshot.format_name.upper()
    image: np.ndarray | None = None
    if "MJPG" in format_name or "MJPEG" in format_name:
        image = cv2.imdecode(np.frombuffer(snapshot.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    elif snapshot.stream == "DEPTH" or any(
        token in format_name for token in ("Y16", "Z16", "Y10", "Y12")
    ):
        expected = snapshot.width * snapshot.height
        values = np.frombuffer(snapshot.data, dtype=np.uint16)
        if values.size >= expected:
            image = values[:expected].reshape(snapshot.height, snapshot.width)
    elif any(token in format_name for token in ("RGB", "BGR")):
        expected = snapshot.width * snapshot.height * 3
        values = np.frombuffer(snapshot.data, dtype=np.uint8)
        if values.size >= expected:
            image = values[:expected].reshape(snapshot.height, snapshot.width, 3)
            if "RGB" in format_name and "BGR" not in format_name:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif "Y8" in format_name:
        expected = snapshot.width * snapshot.height
        values = np.frombuffer(snapshot.data, dtype=np.uint8)
        if values.size >= expected:
            image = values[:expected].reshape(snapshot.height, snapshot.width)
    return image


def save_frame_file(
    snapshot: FrameSnapshot,
    frames_dir: Path,
    *,
    frame_number: int | None,
) -> dict[str, Any]:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    index = frame_number if frame_number is not None else snapshot.sequence
    stem = f"frame_{index:06d}"
    image = decode_snapshot(snapshot)
    sidecar: dict[str, Any] = {
        "stream": snapshot.stream,
        "received_sequence": snapshot.sequence,
        "width": snapshot.width,
        "height": snapshot.height,
        "format": snapshot.format_name,
        "depth_scale": snapshot.depth_scale,
        "depth_storage": (
            "raw_uint16_png_no_resize"
            if snapshot.stream == "DEPTH" and image is not None
            else None
        ),
    }
    if image is not None:
        path = frames_dir / f"{stem}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"cv2.imwrite failed: {path}")
    else:
        path = frames_dir / f"{stem}.bin"
        path.write_bytes(snapshot.data)
        sidecar["fallback_reason"] = "format_not_losslessly_decoded"
    sidecar_path = frames_dir / f"{stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "file_name": path.name,
        "data_path": str(path),
        "sidecar_path": str(sidecar_path),
    }
