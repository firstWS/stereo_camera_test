"""UVC capture and side-by-side split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from stereo_types import StereoFrame


def enumerate_aligned_stereo_pairs(
    left_dir: Path,
    right_dir: Path,
    patterns: Sequence[str],
) -> list[tuple[Path, Path]]:
    """
    Pairs images that share the same filename in left_dir and right_dir.

    Patterns are globs relative to left_dir (e.g. '*.png').
    Sorted by filename case-insensitive.
    """
    left_root = Path(left_dir)
    right_root = Path(right_dir)
    by_name: dict[str, Path] = {}
    for pat in patterns:
        for p in sorted(left_root.glob(pat)):
            if p.is_file() and p.name not in by_name:
                by_name[p.name] = p
    out: list[tuple[Path, Path]] = []
    skipped_left: list[str] = []
    for name in sorted(by_name.keys(), key=str.lower):
        lp = by_name[name]
        rp = right_root / name
        if rp.is_file():
            out.append((lp, rp))
        else:
            skipped_left.append(name)
    if skipped_left:
        print(
            f"[StereoImageFolder] Skipped {len(skipped_left)} left file(s) with no matching name in '{right_root}': "
            + ", ".join(skipped_left[:5])
            + (" ..." if len(skipped_left) > 5 else "")
        )
    if not out:
        raise RuntimeError(
            f"No usable left/right pairs under {left_root} / {right_root}. "
            "Use identical filenames in both folders."
        )
    return out


class StereoImageFolderReader:
    """Read stereo sequences from two folders with matching filenames."""

    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        self._pairs = pairs
        self._idx = 0

    @classmethod
    def from_dirs(cls, left_dir: Path, right_dir: Path, patterns: Sequence[str]) -> StereoImageFolderReader:
        pairs = enumerate_aligned_stereo_pairs(left_dir, right_dir, list(patterns))
        return cls(pairs)

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    def release(self) -> None:
        self._idx = 0

    def read_stereo_pair(self) -> tuple[bool, StereoFrame]:
        bad = StereoFrame(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
        )
        while self._idx < len(self._pairs):
            lp, rp = self._pairs[self._idx]
            self._idx += 1
            L = cv2.imread(str(lp))
            R = cv2.imread(str(rp))
            if L is not None and R is not None:
                return True, StereoFrame(left_bgr=L, right_bgr=R)
            print(f"[StereoImageFolder] Skipping unreadable pair: {lp.name} / {rp.name}")
        return False, bad


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
