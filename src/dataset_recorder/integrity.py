"""Offline dataset integrity validation for Phase 2 sessions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from sensor_validation.timestamp_analysis import analyze_timestamp_series, read_csv_rows

from .session_metadata import load_json, validate_scenario_payload
from .types import IMU_STREAMS, STREAM_DIR_NAMES, VIDEO_STREAMS


def _read_index_rows(session_dir: Path, stream: str) -> list[dict[str, Any]]:
    folder = STREAM_DIR_NAMES[stream]
    path = session_dir / "streams" / folder / "index.csv"
    if not path.is_file():
        return []
    rows, _warnings = read_csv_rows(path)
    return rows


def _read_imu_rows(session_dir: Path, stream: str) -> list[dict[str, Any]]:
    path = session_dir / "streams" / f"{stream.lower()}.csv"
    if not path.is_file():
        return []
    rows, _warnings = read_csv_rows(path)
    return rows


def _frame_file_count(session_dir: Path, stream: str) -> int:
    folder = STREAM_DIR_NAMES[stream]
    frames_dir = session_dir / "streams" / folder / "frames"
    if not frames_dir.is_dir():
        return 0
    return sum(1 for path in frames_dir.iterdir() if path.suffix in {".png", ".bin"})


def _analyze_stream_rows(
    rows: list[dict[str, Any]],
    *,
    stream: str,
    configured_rate_hz: float | None,
    gap_factor: float,
) -> dict[str, Any]:
    timestamps = [row.get("device_timestamp_us") for row in rows]
    frame_numbers = [row.get("frame_number") for row in rows]
    analysis = analyze_timestamp_series(
        timestamps,
        configured_rate_hz=configured_rate_hz,
        frame_numbers=frame_numbers,
        gap_factor=gap_factor,
    )
    status = "PASS"
    reasons: list[str] = []
    if analysis["invalid_timestamp_count"]:
        status = "INVALID"
        reasons.append("device_timestamp_zero")
    if analysis["reverse_count"] or analysis["duplicate_count"]:
        status = "INVALID"
        reasons.append("timestamp_reverse_or_duplicate")
    if analysis.get("frame_number", {}).get("missing_count", 0):
        status = "WARNING" if status == "PASS" else status
        reasons.append("frame_number_missing")
    return {
        "stream": stream,
        "status": status,
        "reasons": reasons,
        "sample_count": analysis["sample_count"],
        "device_clock": analysis,
    }


def validate_dataset_session(
    session_dir: Path,
    *,
    gap_factor: float = 1.5,
    video_rate_hz: float = 30.0,
    imu_rate_hz: float = 200.0,
) -> dict[str, Any]:
    session_dir = Path(session_dir)
    blockers: list[str] = []
    warnings: list[str] = []
    streams: dict[str, Any] = {}

    required_files = [
        "session.json",
        "scenario.json",
        "recording_state.json",
        "events.csv",
        "calibration/intrinsics.json",
        "calibration/extrinsics.json",
        "calibration/camera_imu.json",
    ]
    for relative in required_files:
        if not (session_dir / relative).is_file():
            blockers.append(f"missing:{relative}")

    if (session_dir / "scenario.json").is_file():
        scenario_errors = validate_scenario_payload(load_json(session_dir / "scenario.json"))
        for error in scenario_errors:
            blockers.append(error)

    for stream in VIDEO_STREAMS:
        rows = _read_index_rows(session_dir, stream)
        if not rows:
            blockers.append(f"empty_stream:{stream}")
            continue
        result = _analyze_stream_rows(
            rows, stream=stream, configured_rate_hz=video_rate_hz, gap_factor=gap_factor
        )
        file_count = _frame_file_count(session_dir, stream)
        if file_count != len(rows):
            blockers.append(f"frame_file_mismatch:{stream}:{len(rows)}!={file_count}")
            result["status"] = "INVALID"
        streams[stream] = result

    for stream in IMU_STREAMS:
        rows = _read_imu_rows(session_dir, stream)
        if not rows:
            blockers.append(f"empty_stream:{stream}")
            continue
        result = _analyze_stream_rows(
            rows, stream=stream, configured_rate_hz=imu_rate_hz, gap_factor=gap_factor
        )
        streams[stream] = result

    recording_state_path = session_dir / "recording_state.json"
    if recording_state_path.is_file():
        state = load_json(recording_state_path)
        if state.get("overall_status") != "COMPLETE":
            warnings.append("recording_incomplete")
        if state.get("queue_overflow_counts"):
            blockers.append("queue_overflow")
        if state.get("writer_errors") or state.get("callback_errors") or state.get("stop_errors"):
            blockers.append("recorder_errors")

    camera_imu_path = session_dir / "calibration" / "camera_imu.json"
    if camera_imu_path.is_file():
        camera_imu = load_json(camera_imu_path)
        if camera_imu.get("camera_imu_extrinsic_status") != "AVAILABLE":
            warnings.append("camera_imu_extrinsic_not_available")

    if blockers:
        overall = "INVALID"
    elif warnings:
        overall = "WARNING"
    else:
        overall = "VALID"

    return {
        "session": str(session_dir),
        "overall_status": overall,
        "blockers": blockers,
        "warnings": warnings,
        "streams": streams,
    }
