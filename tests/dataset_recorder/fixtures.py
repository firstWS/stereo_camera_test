from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.session_metadata import (  # noqa: E402
    build_session_metadata,
    write_calibration_snapshot,
    write_json,
    write_scenario_file,
)
from dataset_recorder.types import (  # noqa: E402
    EVENT_FIELDS,
    IMU_CSV_FIELDS,
    STREAM_INDEX_FIELDS,
)

SCENARIO_A = {
    "schema_version": 1,
    "scenario_name": "Scenario A - Rightward Yaw Pan",
    "scenario_slug": "scenario_a",
    "camera_motion": "rightward_yaw",
    "anchor_visibility": "visible",
    "cup1_visibility": "visible",
    "cup2": {"initial_state": "initially_hidden", "notes": "metadata only"},
    "planned_translation_m": None,
    "planned_yaw_deg": 25.0,
    "planned_duration_sec": 15,
    "planned_motion_windows": {
        "initial_hold_sec": [0.0, 3.0],
        "yaw_pan_sec": [3.0, 5.0],
        "final_hold_sec": [5.0, 15.0],
    },
    "operator": "manual",
    "notes": "test fixture",
}


def _write_video_stream(
    session_dir: Path,
    stream: str,
    *,
    frame_count: int,
    width: int,
    height: int,
    period_us: float,
) -> None:
    folder = {
        "RGB": "rgb",
        "DEPTH": "depth",
        "LEFT_IR": "left_ir",
        "RIGHT_IR": "right_ir",
    }[stream]
    stream_dir = session_dir / "streams" / folder
    frames_dir = stream_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    with (stream_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STREAM_INDEX_FIELDS)
        writer.writeheader()
        for index in range(frame_count):
            ts = int(index * period_us) + 1
            stem = f"frame_{index:06d}"
            if stream == "DEPTH":
                image = np.full((height, width), 1500 + index, dtype=np.uint16)
            else:
                image = np.full((height, width, 3), 40 + index, dtype=np.uint8)
            path = frames_dir / f"{stem}.png"
            cv2.imwrite(str(path), image)
            writer.writerow(
                {
                    "received_sequence": index + 1,
                    "callback_sequence": index + 1,
                    "frame_number": index,
                    "device_timestamp_us": ts,
                    "system_timestamp_us": ts + 1000,
                    "global_timestamp_us": ts + 2000,
                    "host_monotonic_ns": 1_000_000 + index,
                    "host_wall_time_ns": 2_000_000 + index,
                    "width": width,
                    "height": height,
                    "format": "Y16" if stream == "DEPTH" else "RGB",
                    "data_size_bytes": width * height * (2 if stream == "DEPTH" else 3),
                    "depth_scale": 1.0 if stream == "DEPTH" else "",
                    "metadata_json": "{}",
                    "file_name": path.name,
                }
            )


def _write_imu_stream(
    session_dir: Path,
    stream: str,
    *,
    sample_count: int,
    period_us: float,
) -> None:
    path = session_dir / "streams" / f"{stream.lower()}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMU_CSV_FIELDS)
        writer.writeheader()
        for index in range(sample_count):
            ts = int(index * period_us) + 1
            writer.writerow(
                {
                    "received_sequence": index + 1,
                    "callback_sequence": index + 1,
                    "frame_number": index,
                    "device_timestamp_us": ts,
                    "system_timestamp_us": ts + 1000,
                    "global_timestamp_us": ts + 2000,
                    "host_monotonic_ns": 1_000_000 + index,
                    "host_wall_time_ns": 2_000_000 + index,
                    "sample_rate": 200,
                    "full_scale_range": "4g" if stream == "ACCEL" else "1000dps",
                    "temperature": 25.0,
                    "x": 0.01 * index,
                    "y": 0.02 * index,
                    "z": 9.8,
                    "metadata_json": "{}",
                }
            )


def build_synthetic_session(
    tmp_path: Path,
    *,
    frame_count: int = 5,
    imu_count: int = 20,
    scenario: dict[str, Any] | None = None,
    complete: bool = True,
    camera_imu_available: bool = False,
) -> Path:
    session_dir = tmp_path / "20260807_120000_scenario_a"
    session_dir.mkdir(parents=True)
    scenario_payload = dict(scenario or SCENARIO_A)
    write_scenario_file(session_dir, scenario_payload)

    calibration = {
        "source": "synthetic",
        "intrinsics": [
            {
                "frame": "RGB",
                "success": True,
                "intrinsic": {
                    "width": 64,
                    "height": 48,
                    "fx": 600.0,
                    "fy": 600.0,
                    "cx": 32.0,
                    "cy": 24.0,
                },
            },
            {
                "frame": "DEPTH",
                "success": True,
                "intrinsic": {
                    "width": 64,
                    "height": 48,
                    "fx": 600.0,
                    "fy": 600.0,
                    "cx": 32.0,
                    "cy": 24.0,
                },
            },
        ],
        "extrinsics": [
            {
                "from_frame": "RGB",
                "to_frame": "DEPTH",
                "success": True,
                "extrinsic": {
                    "rotation": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "translation": [0.0, 0.0, 0.0],
                    "translation_unit": "millimeter_from_installed_sdk_example",
                },
            }
        ],
        "identity_substitution_used": True,
        "camera_imu_extrinsic_status": "AVAILABLE" if camera_imu_available else "UNAVAILABLE",
        "camera_imu_attempts": 1,
        "camera_imu_successes": 1 if camera_imu_available else 0,
    }
    write_calibration_snapshot(session_dir, calibration)
    write_json(
        session_dir / "device_info.json",
        {"connected_device": {"name": "Gemini 335L"}, "sdk": {"runtime_version": "2.8.6"}},
    )
    write_json(
        session_dir / "selected_profiles.json",
        {"selection": [], "selected_profile_ids": {"RGB": "rgb"}},
    )
    write_json(
        session_dir / "recording_state.json",
        {
            "overall_status": "COMPLETE" if complete else "INCOMPLETE",
            "requested_duration_seconds": 15.0,
            "elapsed_seconds": 15.0,
            "queue_overflow_counts": {},
            "writer_errors": [],
            "callback_errors": [],
            "stop_errors": [],
        },
    )

    video_period_us = 1_000_000.0 / 30.0
    imu_period_us = 1_000_000.0 / 200.0
    for stream, size in (
        ("RGB", (64, 48)),
        ("DEPTH", (64, 48)),
        ("LEFT_IR", (64, 48)),
        ("RIGHT_IR", (64, 48)),
    ):
        _write_video_stream(
            session_dir,
            stream,
            frame_count=frame_count,
            width=size[0],
            height=size[1],
            period_us=video_period_us,
        )
    _write_imu_stream(session_dir, "ACCEL", sample_count=imu_count, period_us=imu_period_us)
    _write_imu_stream(session_dir, "GYRO", sample_count=imu_count, period_us=imu_period_us)

    with (session_dir / "events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "event_time_host_monotonic_ns": 1,
                "event_time_device_us": 1,
                "event_type": "recording_start",
                "message": "synthetic",
            }
        )

    session_meta = build_session_metadata(
        session_id=session_dir.name,
        dataset_root=session_dir.parent,
        scenario=scenario_payload,
        device_info=json.loads((session_dir / "device_info.json").read_text(encoding="utf-8")),
        selected_profiles=json.loads(
            (session_dir / "selected_profiles.json").read_text(encoding="utf-8")
        ),
        recording={"duration_requested_sec": 15.0, "save_policy": "all_frames"},
        status="COMPLETE" if complete else "INCOMPLETE",
        integrity_status="VALID",
    )
    write_json(session_dir / "session.json", session_meta)
    return session_dir
