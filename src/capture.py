"""UVC capture and side-by-side split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from stereo_types import StereoFrame


@dataclass
class SBSSplitConfig:
    """If full frame is W x H with left=W_left, the default is equal halves."""

    left_width: int | None = None
    """If None, uses frame.shape[1] // 2."""

    swap_eyes: bool = False


class CaptureAdapter:
    """USB UVC SBS stereo capture."""

    def __init__(
        self,
        device_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        backend: int | None = None,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        api = self.backend if self.backend is not None else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.device_index, api)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.device_index}")
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_raw(self) -> tuple[bool, np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Capture not open")
        return self._cap.read()

    def read_stereo(self, split: SBSSplitConfig) -> tuple[bool, StereoFrame]:
        ok, frame = self.read_raw()
        if not ok or frame is None:
            return False, StereoFrame(np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8))
        h, w = frame.shape[:2]
        lw = split.left_width if split.left_width is not None else w // 2
        rw = w - lw
        if lw <= 0 or rw <= 0:
            raise ValueError(f"Invalid SBS widths: total={w}, left_width={split.left_width}")
        left = frame[:, :lw].copy()
        right = frame[:, lw:].copy()
        if split.swap_eyes:
            left, right = right, left
        return True, StereoFrame(left_bgr=left, right_bgr=right)

    def __enter__(self) -> CaptureAdapter:
        self.open()
        return self

    def __exit__(self, *args):  # noqa: ANN001
        self.release()


def split_sbs_frame(frame_bgr: np.ndarray, split: SBSSplitConfig) -> StereoFrame:
    h, w = frame_bgr.shape[:2]
    lw = split.left_width if split.left_width is not None else w // 2
    left = frame_bgr[:, :lw].copy()
    right = frame_bgr[:, lw:].copy()
    if split.swap_eyes:
        left, right = right, left
    return StereoFrame(left_bgr=left, right_bgr=right)


def load_image_pair_paths(
    left_dir: Path, pattern_left: str = "*.png"
) -> list[tuple[Path, Path]]:
    left_dir = Path(left_dir)
    lefts = sorted(left_dir.glob(pattern_left))
    pairs: list[tuple[Path, Path]] = []
    for lp in lefts:
        rp = lp.parent / lp.name.replace("left", "right")
        if not rp.exists():
            stem = lp.stem
            if stem.startswith("left_"):
                rp = lp.with_name("right_" + stem[5:] + lp.suffix)
        if rp.exists():
            pairs.append((lp, rp))
    return pairs


def iter_stereo_images_from_dirs(
    left_dir: Path, right_dir: Path | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    left_dir = Path(left_dir)
    if right_dir is None:
        for lp, rp in load_image_pair_paths(left_dir):
            yield cv2.imread(str(lp)), cv2.imread(str(rp))
    else:
        right_dir = Path(right_dir)
        for lp in sorted(left_dir.glob("*.png")):
            rp = right_dir / lp.name
            if rp.exists():
                yield cv2.imread(str(lp)), cv2.imread(str(rp))
