"""Non-blocking RGB preview for Phase 2 live recording."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .frame_writer import FrameSnapshot, decode_snapshot

PREVIEW_PHASE_LABELS: dict[str, str] = {
    "hold": "HOLD",
    "prepare": "PREPARE",
    "yaw_start": "PAN RIGHT",
    "final_hold": "FINAL HOLD",
}

PREVIEW_PHASE_HINTS: dict[str, str] = {
    "yaw_start": "Yaw target: 20-30 deg | Natural small translation allowed",
}


@dataclass
class _PreviewSnapshot:
    width: int
    height: int
    format_name: str
    data: bytes
    stream: str = "RGB"


class RgbPreview:
    """Display-only RGB preview; never mutates frames written to disk."""

    def __init__(self, *, enabled: bool, max_width: int = 960) -> None:
        self.enabled = bool(enabled)
        self._max_width = max(1, int(max_width))
        self._lock = threading.Lock()
        self._latest: _PreviewSnapshot | None = None
        self._window = "Gemini335L RGB Preview"
        self._opened = False

    def update_from_snapshot(self, snapshot: FrameSnapshot) -> None:
        if not self.enabled or snapshot.stream != "RGB":
            return
        try:
            payload = _PreviewSnapshot(
                width=int(snapshot.width),
                height=int(snapshot.height),
                format_name=str(snapshot.format_name),
                data=bytes(snapshot.data),
            )
            with self._lock:
                self._latest = payload
        except Exception:
            return

    def _decode_latest(self) -> np.ndarray | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        frame = FrameSnapshot(
            stream=latest.stream,
            sequence=0,
            width=latest.width,
            height=latest.height,
            format_name=latest.format_name,
            data=latest.data,
            depth_scale=None,
        )
        return decode_snapshot(frame)

    def render(
        self,
        *,
        scenario_slug: str | None,
        elapsed_seconds: float,
        phase_cue_id: str | None,
    ) -> None:
        if not self.enabled:
            return
        try:
            import cv2
        except Exception:
            return
        try:
            bgr = self._decode_latest()
            if bgr is None:
                bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            display = bgr
            height, width = display.shape[:2]
            if width > self._max_width:
                scale = self._max_width / float(width)
                display = cv2.resize(
                    display,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            overlay = display.copy()
            phase = PREVIEW_PHASE_LABELS.get(phase_cue_id or "", "RECORDING")
            lines = [
                f"Scenario: {scenario_slug or 'unknown'}",
                f"Elapsed: {elapsed_seconds:0.1f}s",
                f"Phase: {phase}",
            ]
            hint = PREVIEW_PHASE_HINTS.get(phase_cue_id or "")
            if hint:
                lines.append(hint)
            y = 24
            for line in lines:
                cv2.putText(
                    overlay,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y += 24
            cv2.imshow(self._window, overlay)
            cv2.waitKey(1)
            self._opened = True
        except Exception:
            return

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            import cv2

            if self._opened:
                cv2.destroyWindow(self._window)
            else:
                cv2.destroyAllWindows()
        except Exception:
            return
