from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.timestamp_analysis import (  # noqa: E402
    analyze_timestamp_series,
    pair_nearest_timestamps,
    read_csv_rows,
    summarize_stream_rows,
)


def test_regular_30fps_timestamp_series() -> None:
    period_us = 1_000_000.0 / 30.0
    timestamps = [index * period_us for index in range(301)]
    result = analyze_timestamp_series(
        timestamps,
        configured_rate_hz=30.0,
        frame_numbers=list(range(301)),
    )
    assert result["monotonic_increasing"] is True
    assert result["reverse_count"] == 0
    assert result["duplicate_count"] == 0
    assert result["gap_count"] == 0
    assert result["frame_number"]["missing_count"] == 0
    assert result["measured_rate_hz"] == pytest.approx(30.0)


def test_reverse_duplicate_gap_and_missing_frame_number() -> None:
    result = analyze_timestamp_series(
        [0, 33_333, 33_333, 20_000, 133_333],
        configured_rate_hz=30.0,
        frame_numbers=[0, 1, 1, 0, 4],
    )
    assert result["reverse_count"] == 1
    assert result["duplicate_count"] == 1
    assert result["monotonic_increasing"] is False
    assert result["gap_count"] == 1
    assert result["estimated_missing_from_timestamp"] == 2
    assert result["frame_number"]["duplicate_count"] == 1
    assert result["frame_number"]["reverse_count"] == 1
    assert result["frame_number"]["missing_count"] == 3


def test_empty_and_single_sample_are_safe() -> None:
    empty = analyze_timestamp_series([], configured_rate_hz=30.0)
    single = analyze_timestamp_series([123.0], configured_rate_hz=30.0, frame_numbers=[7])
    assert empty["monotonic_increasing"] is None
    assert empty["measured_rate_hz"] is None
    assert single["monotonic_increasing"] is True
    assert single["duration_us"] == 0.0
    assert single["frame_number"]["missing_count"] == 0


def test_nearest_pairing_with_tolerance() -> None:
    result = pair_nearest_timestamps(
        [0, 10_000, 20_000, 30_000],
        [500, 10_500, 50_000],
        tolerance_us=1_000,
    )
    assert result["paired_count"] == 2
    assert result["failed_count"] == 2
    assert result["failure_rate"] == pytest.approx(0.5)
    assert result["median_abs_offset_us"] == pytest.approx(500.0)


def test_irregular_imu_rate_and_row_wrapper() -> None:
    rows = [
        {"device_timestamp_us": "0", "frame_number": "0"},
        {"device_timestamp_us": "5000", "frame_number": "1"},
        {"device_timestamp_us": "17000", "frame_number": "3"},
    ]
    result = summarize_stream_rows(
        rows,
        timestamp_field="device_timestamp_us",
        configured_rate_hz=200.0,
    )
    assert result["gap_count"] == 1
    assert result["frame_number"]["missing_count"] == 1


def test_partially_damaged_csv_returns_rows_and_warning(tmp_path: Path) -> None:
    path = tmp_path / "partial.csv"
    path.write_text("a,b\n1,2\n3,4,extra\n", encoding="utf-8")
    rows, warnings = read_csv_rows(path)
    assert len(rows) == 2
    assert warnings == ["extra_columns_at_line:3"]


def test_missing_csv_is_reported(tmp_path: Path) -> None:
    rows, warnings = read_csv_rows(tmp_path / "missing.csv")
    assert rows == []
    assert warnings and warnings[0].startswith("missing_file:")
