"""Canonical Scenario A frame table for Phase 3-compatible evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalFrame:
    frame_number: int
    device_timestamp_us: int


def load_canonical_frames_from_rgb_index(session_dir: Path) -> list[CanonicalFrame]:
    """Load canonical frame numbers (1..N) from the RGB stream index."""
    index_path = Path(session_dir) / "streams" / "rgb" / "index.csv"
    frames: list[CanonicalFrame] = []
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frames.append(
                CanonicalFrame(
                    frame_number=int(row["frame_number"]),
                    device_timestamp_us=int(row["device_timestamp_us"]),
                )
            )
    frames.sort(key=lambda item: item.frame_number)
    return frames


def canonical_frame_numbers(session_dir: Path) -> list[int]:
    return [frame.frame_number for frame in load_canonical_frames_from_rgb_index(session_dir)]
