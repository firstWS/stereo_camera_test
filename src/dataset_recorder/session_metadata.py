"""Session and scenario metadata helpers for Phase 2 datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .types import (
    DATASET_SCHEMA_VERSION,
    RECORD_TOOL_NAME,
    RECORD_TOOL_VERSION,
    SCENARIO_SCHEMA_VERSION,
    SUPPORTED_SCENARIOS,
)

REQUIRED_SCENARIO_KEYS = (
    "scenario_name",
    "scenario_slug",
    "camera_motion",
    "anchor_visibility",
    "cup1_visibility",
    "cup2",
    "planned_translation_m",
    "planned_yaw_deg",
    "planned_duration_sec",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_scenario_payload(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_SCENARIO_KEYS:
        if key not in payload:
            errors.append(f"missing scenario field: {key}")
    slug = str(payload.get("scenario_slug", ""))
    if slug and slug not in SUPPORTED_SCENARIOS:
        errors.append(f"unsupported scenario_slug: {slug}")
    # Key must exist; null means intentional translation is not planned.
    if "planned_translation_m" in payload:
        translation = payload.get("planned_translation_m")
        if translation is not None:
            try:
                float(translation)
            except (TypeError, ValueError):
                errors.append("planned_translation_m must be a number or null")
    if "planned_yaw_deg" in payload and payload.get("planned_yaw_deg") is None:
        errors.append("planned_yaw_deg must be a number (null not allowed)")
    return errors


def load_scenario_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario root must be an object: {path}")
    payload.setdefault("schema_version", SCENARIO_SCHEMA_VERSION)
    errors = validate_scenario_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def write_calibration_snapshot(session_dir: Path, calibration: Mapping[str, Any]) -> None:
    cal_dir = session_dir / "calibration"
    write_json(
        cal_dir / "intrinsics.json",
        {
            "source": calibration.get("source"),
            "intrinsics": calibration.get("intrinsics", []),
        },
    )
    write_json(
        cal_dir / "extrinsics.json",
        {
            "source": calibration.get("source"),
            "extrinsics": calibration.get("extrinsics", []),
            "identity_substitution_used": calibration.get("identity_substitution_used", False),
        },
    )
    write_json(
        cal_dir / "camera_imu.json",
        {
            "camera_imu_extrinsic_status": calibration.get("camera_imu_extrinsic_status"),
            "camera_imu_attempts": calibration.get("camera_imu_attempts"),
            "camera_imu_successes": calibration.get("camera_imu_successes"),
            "identity_substitution_used": calibration.get("identity_substitution_used", False),
        },
    )


def build_session_metadata(
    *,
    session_id: str,
    dataset_root: Path,
    scenario: Mapping[str, Any],
    device_info: Mapping[str, Any] | None,
    selected_profiles: Mapping[str, Any] | None,
    recording: Mapping[str, Any],
    status: str,
    integrity_status: str | None = None,
) -> dict[str, Any]:
    device: dict[str, Any] = {}
    sdk_block: dict[str, Any] = {}
    if device_info:
        connected = device_info.get("connected_device") or {}
        if isinstance(connected, dict):
            device = {
                "model": connected.get("name") or connected.get("product_name"),
                "firmware": connected.get("firmware_version"),
                "hardware": connected.get("hardware_version"),
                "usb": connected.get("connection_type") or connected.get("usb_type"),
            }
        sdk = device_info.get("sdk") or {}
        if isinstance(sdk, dict):
            sdk_block = {
                "package": sdk.get("distribution_package"),
                "package_version": sdk.get("distribution_version"),
                "runtime_version": sdk.get("runtime_version"),
            }

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "session_id": session_id,
        "created_at_utc": utc_now_iso(),
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "phase": "phase2_common_dataset",
        "recorder": {
            "tool": RECORD_TOOL_NAME,
            "version": RECORD_TOOL_VERSION,
        },
        "device": device,
        "sdk": sdk_block,
        "profiles": selected_profiles or {},
        "recording": dict(recording),
        "scenario": {
            "scenario_name": scenario.get("scenario_name"),
            "scenario_slug": scenario.get("scenario_slug"),
            "camera_motion": scenario.get("camera_motion"),
        },
        "paths": {
            "streams": "streams/",
            "calibration": "calibration/",
            "events": "events.csv",
            "scenario": "scenario.json",
            "derived": "derived/",
        },
        "status": status,
        "integrity_status": integrity_status,
    }


def write_scenario_file(session_dir: Path, scenario: Mapping[str, Any]) -> None:
    payload = dict(scenario)
    payload.setdefault("schema_version", SCENARIO_SCHEMA_VERSION)
    payload.setdefault("operator", "manual")
    payload.setdefault("notes", "")
    write_json(session_dir / "scenario.json", payload)
