"""Offline replay reader for Phase 2 datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .session_metadata import load_json
from .types import IMU_STREAMS, STREAM_DIR_NAMES, VIDEO_STREAMS


@dataclass(frozen=True)
class FrameRecord:
    stream: str
    row: dict[str, Any]
    file_path: Path | None


@dataclass(frozen=True)
class ImuSample:
    stream: str
    row: dict[str, Any]


class DatasetReader:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)

    def session_meta(self) -> dict[str, Any]:
        return load_json(self.session_dir / "session.json")

    def scenario_meta(self) -> dict[str, Any]:
        return load_json(self.session_dir / "scenario.json")

    def recording_state(self) -> dict[str, Any]:
        return load_json(self.session_dir / "recording_state.json")

    def calibration_intrinsics(self) -> dict[str, Any]:
        return load_json(self.session_dir / "calibration" / "intrinsics.json")

    def calibration_extrinsics(self) -> dict[str, Any]:
        return load_json(self.session_dir / "calibration" / "extrinsics.json")

    def calibration_camera_imu(self) -> dict[str, Any]:
        return load_json(self.session_dir / "calibration" / "camera_imu.json")

    def _iterate_index(self, stream: str) -> Iterator[FrameRecord]:
        folder = STREAM_DIR_NAMES[stream]
        index_path = self.session_dir / "streams" / folder / "index.csv"
        frames_dir = self.session_dir / "streams" / folder / "frames"
        with index_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                file_name = row.get("file_name") or ""
                file_path = frames_dir / file_name if file_name else None
                if file_path is not None and not file_path.is_file():
                    file_path = None
                yield FrameRecord(stream=stream, row=dict(row), file_path=file_path)

    def _iterate_imu(self, stream: str) -> Iterator[ImuSample]:
        path = self.session_dir / "streams" / f"{stream.lower()}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield ImuSample(stream=stream, row=dict(row))

    def iterate_rgb(self) -> Iterator[FrameRecord]:
        yield from self._iterate_index("RGB")

    def iterate_depth(self) -> Iterator[FrameRecord]:
        yield from self._iterate_index("DEPTH")

    def iterate_left_ir(self) -> Iterator[FrameRecord]:
        yield from self._iterate_index("LEFT_IR")

    def iterate_right_ir(self) -> Iterator[FrameRecord]:
        yield from self._iterate_index("RIGHT_IR")

    def iterate_accel(self) -> Iterator[ImuSample]:
        yield from self._iterate_imu("ACCEL")

    def iterate_gyro(self) -> Iterator[ImuSample]:
        yield from self._iterate_imu("GYRO")

    def stream_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stream in VIDEO_STREAMS:
            folder = STREAM_DIR_NAMES[stream]
            index_path = self.session_dir / "streams" / folder / "index.csv"
            if not index_path.is_file():
                counts[stream] = 0
                continue
            with index_path.open(encoding="utf-8", newline="") as handle:
                counts[stream] = sum(1 for _ in csv.DictReader(handle))
        for stream in IMU_STREAMS:
            path = self.session_dir / "streams" / f"{stream.lower()}.csv"
            if not path.is_file():
                counts[stream] = 0
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                counts[stream] = sum(1 for _ in csv.DictReader(handle))
        return counts

    def derived_manifest(self) -> dict[str, Any] | None:
        path = self.session_dir / "derived" / "manifest.json"
        return load_json(path) if path.is_file() else None
