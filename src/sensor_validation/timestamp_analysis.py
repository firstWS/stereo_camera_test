"""Pure timestamp, frame-drop, and cross-stream pairing analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    parsed = _finite_float(value)
    return int(parsed) if parsed is not None else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _interval_statistics(intervals_us: np.ndarray) -> dict[str, float | None]:
    positive = intervals_us[intervals_us > 0.0]
    if not positive.size:
        return {
            "count": 0,
            "mean_us": None,
            "median_us": None,
            "std_us": None,
            "p90_us": None,
            "p95_us": None,
            "p99_us": None,
            "min_us": None,
            "max_us": None,
        }
    return {
        "count": int(positive.size),
        "mean_us": float(np.mean(positive)),
        "median_us": float(np.median(positive)),
        "std_us": float(np.std(positive)),
        "p90_us": _percentile(positive, 90.0),
        "p95_us": _percentile(positive, 95.0),
        "p99_us": _percentile(positive, 99.0),
        "min_us": float(np.min(positive)),
        "max_us": float(np.max(positive)),
    }


def analyze_timestamp_series(
    timestamps_us: Sequence[float | int | None],
    *,
    configured_rate_hz: float | None = None,
    frame_numbers: Sequence[int | None] | None = None,
    gap_factor: float = 1.5,
) -> dict[str, Any]:
    """Analyze one clock domain without fabricating unavailable values."""

    values = [_finite_float(value) for value in timestamps_us]
    valid_values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    summary: dict[str, Any] = {
        "sample_count": len(timestamps_us),
        "valid_timestamp_count": int(valid_values.size),
        "invalid_timestamp_count": len(timestamps_us) - int(valid_values.size),
        "start_timestamp_us": float(valid_values[0]) if valid_values.size else None,
        "end_timestamp_us": float(valid_values[-1]) if valid_values.size else None,
        "duration_us": (
            float(valid_values[-1] - valid_values[0]) if valid_values.size >= 2 else 0.0 if valid_values.size == 1 else None
        ),
        "monotonic_increasing": None,
        "reverse_count": 0,
        "duplicate_count": 0,
        "interval": _interval_statistics(np.asarray([], dtype=np.float64)),
        "configured_rate_hz": configured_rate_hz,
        "measured_rate_hz": None,
        "configured_rate_ratio": None,
        "gap_factor": float(gap_factor),
        "gap_count": 0,
        "estimated_missing_from_timestamp": 0,
        "timestamp_missing_is_estimate": True,
        "frame_number": {
            "available": False,
            "valid_count": 0,
            "missing_count": None,
            "duplicate_count": None,
            "reverse_count": None,
        },
    }
    if valid_values.size >= 2:
        deltas = np.diff(valid_values)
        reverse_count = int(np.count_nonzero(deltas < 0.0))
        duplicate_count = int(np.count_nonzero(deltas == 0.0))
        summary["reverse_count"] = reverse_count
        summary["duplicate_count"] = duplicate_count
        summary["monotonic_increasing"] = reverse_count == 0 and duplicate_count == 0
        summary["interval"] = _interval_statistics(deltas)
        duration_s = float(valid_values[-1] - valid_values[0]) / 1_000_000.0
        if duration_s > 0.0:
            summary["measured_rate_hz"] = float((valid_values.size - 1) / duration_s)
    elif valid_values.size == 1:
        summary["monotonic_increasing"] = True

    expected_interval_us: float | None = None
    if configured_rate_hz is not None and configured_rate_hz > 0.0:
        expected_interval_us = 1_000_000.0 / float(configured_rate_hz)
        measured = summary["measured_rate_hz"]
        if measured is not None:
            summary["configured_rate_ratio"] = float(measured / configured_rate_hz)
        if valid_values.size >= 2:
            positive_deltas = np.diff(valid_values)
            positive_deltas = positive_deltas[positive_deltas > 0.0]
            gaps = positive_deltas[positive_deltas > expected_interval_us * gap_factor]
            summary["gap_count"] = int(gaps.size)
            summary["estimated_missing_from_timestamp"] = int(
                sum(max(int(round(delta / expected_interval_us)) - 1, 0) for delta in gaps)
            )
    summary["expected_interval_us"] = expected_interval_us

    if frame_numbers is not None:
        frame_values = [_finite_int(value) for value in frame_numbers]
        valid_frames = np.asarray([value for value in frame_values if value is not None], dtype=np.int64)
        frame_summary = summary["frame_number"]
        frame_summary["available"] = bool(valid_frames.size)
        frame_summary["valid_count"] = int(valid_frames.size)
        if valid_frames.size >= 2:
            frame_deltas = np.diff(valid_frames)
            frame_summary["missing_count"] = int(
                np.sum(np.maximum(frame_deltas[frame_deltas > 1] - 1, 0))
            )
            frame_summary["duplicate_count"] = int(np.count_nonzero(frame_deltas == 0))
            frame_summary["reverse_count"] = int(np.count_nonzero(frame_deltas < 0))
        elif valid_frames.size == 1:
            frame_summary["missing_count"] = 0
            frame_summary["duplicate_count"] = 0
            frame_summary["reverse_count"] = 0
    return summary


def pair_nearest_timestamps(
    reference_timestamps_us: Sequence[float | int | None],
    target_timestamps_us: Sequence[float | int | None],
    *,
    tolerance_us: float | None,
) -> dict[str, Any]:
    """Pair each reference timestamp to the closest target in the same clock domain."""

    references = np.asarray(
        sorted(value for raw in reference_timestamps_us if (value := _finite_float(raw)) is not None),
        dtype=np.float64,
    )
    targets = np.asarray(
        sorted(value for raw in target_timestamps_us if (value := _finite_float(raw)) is not None),
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "reference_count": int(references.size),
        "target_count": int(targets.size),
        "paired_count": 0,
        "failed_count": int(references.size),
        "failure_rate": 1.0 if references.size else None,
        "tolerance_us": tolerance_us,
        "median_abs_offset_us": None,
        "p90_abs_offset_us": None,
        "p99_abs_offset_us": None,
        "max_abs_offset_us": None,
        "median_signed_offset_us": None,
    }
    if not references.size or not targets.size:
        return result

    insertion = np.searchsorted(targets, references, side="left")
    signed_offsets: list[float] = []
    for reference, position in zip(references, insertion, strict=True):
        candidates: list[float] = []
        if position < targets.size:
            candidates.append(float(targets[position] - reference))
        if position > 0:
            candidates.append(float(targets[position - 1] - reference))
        nearest = min(candidates, key=lambda offset: abs(offset))
        if tolerance_us is None or abs(nearest) <= tolerance_us:
            signed_offsets.append(nearest)

    offsets = np.asarray(signed_offsets, dtype=np.float64)
    absolute = np.abs(offsets)
    paired_count = int(offsets.size)
    result.update(
        {
            "paired_count": paired_count,
            "failed_count": int(references.size) - paired_count,
            "failure_rate": float((references.size - paired_count) / references.size),
            "median_abs_offset_us": float(np.median(absolute)) if absolute.size else None,
            "p90_abs_offset_us": _percentile(absolute, 90.0),
            "p99_abs_offset_us": _percentile(absolute, 99.0),
            "max_abs_offset_us": float(np.max(absolute)) if absolute.size else None,
            "median_signed_offset_us": float(np.median(offsets)) if offsets.size else None,
        }
    )
    return result


def read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a partially damaged CSV, returning usable rows and parse warnings."""

    csv_path = Path(path)
    if not csv_path.is_file():
        return [], [f"missing_file:{csv_path}"]
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return [], ["missing_header"]
            for index, row in enumerate(reader, start=2):
                if None in row:
                    warnings.append(f"extra_columns_at_line:{index}")
                rows.append({str(key): value for key, value in row.items() if key is not None})
    except (OSError, csv.Error, UnicodeError) as error:
        warnings.append(f"csv_error:{type(error).__name__}:{error}")
    return rows, warnings


def summarize_stream_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    timestamp_field: str,
    frame_number_field: str = "frame_number",
    configured_rate_hz: float | None = None,
    gap_factor: float = 1.5,
) -> dict[str, Any]:
    materialized = list(rows)
    return analyze_timestamp_series(
        [row.get(timestamp_field) for row in materialized],
        configured_rate_hz=configured_rate_hz,
        frame_numbers=[row.get(frame_number_field) for row in materialized],
        gap_factor=gap_factor,
    )
