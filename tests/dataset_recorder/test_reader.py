from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.reader import DatasetReader  # noqa: E402


def test_reader_iterates_all_streams(synthetic_session: Path) -> None:
    reader = DatasetReader(synthetic_session)
    assert len(list(reader.iterate_rgb())) == 5
    assert len(list(reader.iterate_depth())) == 5
    assert len(list(reader.iterate_left_ir())) == 5
    assert len(list(reader.iterate_right_ir())) == 5
    assert len(list(reader.iterate_accel())) == 20
    assert len(list(reader.iterate_gyro())) == 20


def test_reader_returns_existing_frame_paths(synthetic_session: Path) -> None:
    reader = DatasetReader(synthetic_session)
    rgb = next(reader.iterate_rgb())
    assert rgb.file_path is not None
    assert rgb.file_path.is_file()
    assert rgb.row["frame_number"] == "0"


def test_stream_counts_match_iteration(synthetic_session: Path) -> None:
    reader = DatasetReader(synthetic_session)
    counts = reader.stream_counts()
    assert counts["RGB"] == 5
    assert counts["ACCEL"] == 20
    assert reader.derived_manifest() is None
