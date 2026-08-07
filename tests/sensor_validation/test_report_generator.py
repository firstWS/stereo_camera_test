from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.report_generator import analyze_session  # noqa: E402
from sensor_validation.sensor_recorder import IMU_FIELDS, VIDEO_FIELDS  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_session(path: Path, *, camera_imu_status: str) -> dict[str, object]:
    selection = []
    for stream in ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR"):
        selection.append(
            {
                "sensor": stream,
                "selected": {"kind": "video", "fps": 30.0},
            }
        )
    for stream in ("ACCEL", "GYRO"):
        selection.append(
            {
                "sensor": stream,
                "selected": {"kind": stream.lower(), "sample_rate": "200_HZ"},
            }
        )
    _write_json(path / "selected_profiles.json", {"selection": selection})
    _write_json(
        path / "calibration.json",
        {
            "camera_imu_extrinsic_status": camera_imu_status,
            "identity_substitution_used": False,
        },
    )
    _write_json(
        path / "session_state.json",
        {"overall_status": "COMPLETE", "mode": "static"},
    )

    with (path / "video_frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_FIELDS)
        writer.writeheader()
        for stream_offset, stream in enumerate(("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR")):
            for index in range(31):
                device_us = index * (1_000_000 / 30) + stream_offset * 100
                row = dict.fromkeys(VIDEO_FIELDS, "")
                row.update(
                    {
                        "stream": stream,
                        "received_sequence": index + 1,
                        "frame_number": index,
                        "device_timestamp_us": device_us,
                        "host_monotonic_ns": int(device_us * 1000),
                    }
                )
                writer.writerow(row)
    with (path / "imu_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMU_FIELDS)
        writer.writeheader()
        for stream in ("ACCEL", "GYRO"):
            for index in range(201):
                device_us = index * 5000
                row = dict.fromkeys(IMU_FIELDS, "")
                row.update(
                    {
                        "stream": stream,
                        "received_sequence": index + 1,
                        "frame_number": index,
                        "device_timestamp_us": device_us,
                        "host_monotonic_ns": int(device_us * 1000),
                        "x": 0.0 if stream == "ACCEL" else 0.001,
                        "y": 0.0,
                        "z": 9.80665 if stream == "ACCEL" else 0.0,
                    }
                )
                writer.writerow(row)
    return yaml.safe_load(
        (ROOT / "configs" / "sensor_validation" / "gemini335l_phase1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_complete_synthetic_session_passes_and_writes_reports(tmp_path: Path) -> None:
    config = _make_session(tmp_path, camera_imu_status="AVAILABLE")
    summary = analyze_session(tmp_path, config)
    assert summary["overall_status"] == "PASS"
    assert summary["phase2_readiness"] == "READY"
    assert all(
        summary["streams"][stream]["status"] == "PASS"
        for stream in ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR", "ACCEL", "GYRO")
    )
    assert (tmp_path / "validation_summary.json").is_file()
    assert "Gemini 335L" in (tmp_path / "phase1_report.md").read_text(encoding="utf-8")


def test_missing_camera_imu_extrinsic_blocks_phase2(tmp_path: Path) -> None:
    config = _make_session(tmp_path, camera_imu_status="NOT_EXPOSED")
    summary = analyze_session(tmp_path, config)
    assert summary["overall_status"] == "BLOCKED"
    assert summary["phase2_readiness"] == "BLOCKED"
    assert summary["calibration"]["identity_substitution_used"] is False
    assert "CALIBRATION:camera_imu_extrinsic_not_available" in summary["blockers"]
