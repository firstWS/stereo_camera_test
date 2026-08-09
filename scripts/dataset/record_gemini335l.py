#!/usr/bin/env python
"""Gemini 335L Phase-2 common dataset record, validate, and derive CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_recorder.derive_observations import derive_observations
from dataset_recorder.integrity import validate_dataset_session
from dataset_recorder.recorder import DatasetRecorder
from dataset_recorder.session_metadata import (
    build_session_metadata,
    load_scenario_file,
    write_calibration_snapshot,
    write_json,
    write_scenario_file,
)
from sensor_validation.calibration_exporter import export_profile_calibration
from sensor_validation.device_inspector import inspect_connected_device
from sensor_validation.profile_selector import (
    ProfileDecision,
    resolve_profile_handles,
    select_imu_profile,
    select_video_profile,
)
from sensor_validation.sdk_adapter import OrbbecSdkAdapter

DEFAULT_CONFIG = ROOT / "configs" / "dataset" / "gemini335l_phase2.yaml"
SCENARIO_DIR = ROOT / "configs" / "dataset" / "scenarios"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _session_dir(config: Mapping[str, Any], output: str | None, scenario_slug: str) -> Path:
    configured = config.get("output", {}).get("root", "out/datasets/gemini335l")
    root = Path(output or configured)
    if not root.is_absolute():
        root = ROOT / root
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    session = root / f"{timestamp}_{scenario_slug}"
    suffix = 1
    while session.exists():
        session = root / f"{timestamp}_{scenario_slug}_{suffix}"
        suffix += 1
    session.mkdir(parents=True)
    return session


def _select_profiles(
    profiles: list[Any],
    config: Mapping[str, Any],
    *,
    allow_fallback_override: bool,
) -> tuple[list[ProfileDecision], dict[str, Any]]:
    profile_config = dict(config.get("profiles", {}))
    allow_fallback = bool(
        allow_fallback_override or profile_config.get("allow_fallback", False)
    )
    decisions: list[ProfileDecision] = []
    for sensor in ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR"):
        decisions.append(
            select_video_profile(
                profiles,
                sensor=sensor,
                request=dict(profile_config.get(sensor, {})),
                allow_fallback=allow_fallback,
            )
        )
    for sensor in ("ACCEL", "GYRO"):
        decisions.append(
            select_imu_profile(
                profiles,
                sensor=sensor,
                request=dict(profile_config.get(sensor, {})),
                allow_fallback=allow_fallback,
            )
        )
    selected = resolve_profile_handles(profiles, decisions)
    return decisions, selected


def _device_choice(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[str | None, int | None]:
    device = dict(config.get("device", {}))
    serial = args.serial if args.serial is not None else device.get("serial_number")
    index = (
        args.device_index
        if args.device_index is not None
        else device.get("device_index")
    )
    return (str(serial) if serial else None), (int(index) if index is not None else None)


def _resolve_scenario(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario_file:
        return load_scenario_file(Path(args.scenario_file))
    if args.scenario:
        path = SCENARIO_DIR / f"{args.scenario}.yaml"
        return load_scenario_file(path)
    raise SystemExit("record requires --scenario or --scenario-file")


def cmd_record(args: argparse.Namespace) -> int:
    if not args.i_confirm_device_access:
        print(
            "Refusing to open the camera without --i-confirm-device-access.",
            file=sys.stderr,
        )
        return 2
    config = _load_config(Path(args.config))
    scenario = _resolve_scenario(args)
    slug = str(scenario["scenario_slug"])
    session_dir = _session_dir(config, args.output, slug)
    write_scenario_file(session_dir, scenario)

    adapter = OrbbecSdkAdapter()
    serial, index = _device_choice(args, config)
    inspection = inspect_connected_device(
        adapter,
        serial_number=serial,
        device_index=index,
    )
    decisions, selected = _select_profiles(
        inspection.profiles,
        config,
        allow_fallback_override=bool(args.allow_fallback),
    )
    calibration = export_profile_calibration(selected)
    write_calibration_snapshot(session_dir, calibration)
    write_json(session_dir / "device_info.json", inspection.device_info)
    write_json(
        session_dir / "selected_profiles.json",
        {
            "selection": [decision.as_dict() for decision in decisions],
            "selected_profile_ids": {
                name: profile.profile_id for name, profile in selected.items()
            },
        },
    )

    duration = float(args.duration or scenario.get("planned_duration_sec") or 15.0)
    preview_enabled = bool(getattr(args, "preview", False))
    recorder = DatasetRecorder(
        adapter,
        device=inspection.opened.device,
        selected_profiles=selected,
        session_dir=session_dir,
        queue_size=int(config.get("record", {}).get("queue_size", 8192)),
        preview_enabled=preview_enabled,
    )
    recording_state = recorder.record(
        duration_seconds=duration,
        scenario_slug=slug,
    )
    recording_state["scenario_slug"] = slug
    recording_state["command"] = "record"
    recording_state["preview_enabled"] = preview_enabled
    write_json(session_dir / "recording_state.json", recording_state)

    session_meta = build_session_metadata(
        session_id=session_dir.name,
        dataset_root=session_dir.parent,
        scenario=scenario,
        device_info=inspection.device_info,
        selected_profiles=load_json_payload(session_dir / "selected_profiles.json"),
        recording={
            "duration_requested_sec": duration,
            "duration_elapsed_sec": recording_state.get("elapsed_seconds"),
            "scenario_slug": slug,
            "save_policy": "all_frames",
            "preview_enabled": preview_enabled,
        },
        status=recording_state.get("overall_status", "INCOMPLETE"),
        integrity_status=None,
    )
    write_json(session_dir / "session.json", session_meta)

    integrity = validate_dataset_session(session_dir)
    write_json(session_dir / "integrity.json", integrity)
    session_meta["integrity_status"] = integrity.get("overall_status")
    write_json(session_dir / "session.json", session_meta)
    print(f"기록 결과: {session_dir}")
    print(f"무결성 판정: {integrity.get('overall_status')}")
    return 0 if integrity.get("overall_status") != "INVALID" else 1


def load_json_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_validate(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    integrity_cfg = dict(config.get("integrity", {}))
    result = validate_dataset_session(
        Path(args.session),
        gap_factor=float(integrity_cfg.get("gap_factor", 1.5)),
        video_rate_hz=float(integrity_cfg.get("video_rate_hz", 30.0)),
        imu_rate_hz=float(integrity_cfg.get("imu_rate_hz", 200.0)),
    )
    write_json(Path(args.session) / "integrity.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("overall_status") != "INVALID" else 1


def cmd_derive(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    derive_cfg = dict(config.get("derive", {}))
    apriltag_cfg = dict(derive_cfg.get("apriltag_world") or {})
    min_valid = float(derive_cfg.get("min_valid_depth_ratio", 0.03))
    apriltag_enabled = bool(apriltag_cfg.get("enabled", False))
    if not apriltag_enabled:
        print(
            "WARNING: derive.apriltag_world.enabled is false; "
            "AprilTag observations will not be generated.",
            file=sys.stderr,
        )
    manifest = derive_observations(
        Path(args.session),
        apriltag_config=apriltag_cfg,
        min_valid_depth_ratio=min_valid,
        cup_mot_config=dict(derive_cfg.get("cup_mot") or {}),
        depth_pairing_config=derive_cfg,
        cup_depth_config=dict(derive_cfg.get("cup_depth") or {}),
        apriltag_reference_config=dict(derive_cfg.get("apriltag_reference") or {}),
    )
    for warning in manifest.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini 335L Phase-2 common dataset recorder"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="record a dataset session (live device)")
    record.add_argument("--i-confirm-device-access", action="store_true")
    record.add_argument("--serial")
    record.add_argument("--device-index", type=int)
    record.add_argument("--allow-fallback", action="store_true")
    record.add_argument("--output")
    record.add_argument("--scenario", choices=["scenario_a", "scenario_b"])
    record.add_argument("--scenario-file")
    record.add_argument("--duration", type=float)
    preview_group = record.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview",
        action="store_true",
        help="show non-blocking RGB preview during recording",
    )
    preview_group.add_argument(
        "--no-preview",
        action="store_true",
        help="disable RGB preview (default)",
    )

    validate = sub.add_parser("validate", help="validate an existing dataset session")
    validate.add_argument("--session", required=True)

    derive = sub.add_parser("derive", help="offline AprilTag/cup derived observations")
    derive.add_argument("--session", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "record":
        return cmd_record(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "derive":
        return cmd_derive(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
