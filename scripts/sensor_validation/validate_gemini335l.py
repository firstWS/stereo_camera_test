#!/usr/bin/env python
"""Gemini 335L Phase-1 inspect, record, and offline analysis CLI."""

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

from sensor_validation.calibration_exporter import export_profile_calibration
from sensor_validation.device_inspector import inspect_connected_device
from sensor_validation.profile_selector import (
    ProfileDecision,
    resolve_profile_handles,
    select_imu_profile,
    select_video_profile,
)
from sensor_validation.report_generator import analyze_session
from sensor_validation.sdk_adapter import OrbbecSdkAdapter
from sensor_validation.sensor_recorder import SensorRecorder, probe_profile_combinations

DEFAULT_CONFIG = ROOT / "configs" / "sensor_validation" / "gemini335l_phase1.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _session_dir(config: Mapping[str, Any], output: str | None, command: str) -> Path:
    configured = config.get("output", {}).get("root", "out/sensor_validation")
    root = Path(output or configured)
    if not root.is_absolute():
        root = ROOT / root
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    session = root / f"{timestamp}_{command}"
    suffix = 1
    while session.exists():
        session = root / f"{timestamp}_{command}_{suffix}"
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


def _inspect_summary(
    *,
    selected: Mapping[str, Any],
    decisions: list[ProfileDecision],
    calibration: Mapping[str, Any],
    probes: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    for decision in decisions:
        if decision.selected is None:
            blockers.append(f"PROFILE:{decision.sensor}:{decision.reason}")
    if calibration.get("camera_imu_extrinsic_status") != "AVAILABLE":
        blockers.append("CALIBRATION:camera_imu_extrinsic_not_available")
    if any(item.get("status") != "success" for item in probes):
        blockers.append("STREAM_PROBE:one_or_more_combinations_failed")
    return {
        "schema_version": 1,
        "overall_status": "NOT_TESTED",
        "phase2_readiness": "BLOCKED",
        "command": "inspect",
        "selected_streams": sorted(selected),
        "camera_imu_extrinsic_status": calibration.get(
            "camera_imu_extrinsic_status", "NOT_TESTED"
        ),
        "blockers": blockers or ["RECORDING:not_performed"],
        "warnings": [],
    }


def _require_device_confirmation(args: argparse.Namespace) -> None:
    if not args.i_confirm_device_access:
        raise PermissionError(
            "Live device access was not confirmed. Add --i-confirm-device-access."
        )


def _run_inspect(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    _require_device_confirmation(args)
    session = _session_dir(config, args.output, "inspect")
    _write_json(
        session / "session_state.json",
        {"overall_status": "INCOMPLETE", "command": "inspect"},
    )
    try:
        adapter = OrbbecSdkAdapter()
        serial, index = _device_choice(args, config)
        inspection = inspect_connected_device(
            adapter, serial_number=serial, device_index=index
        )
        decisions, selected = _select_profiles(
            inspection.profiles,
            config,
            allow_fallback_override=args.allow_fallback,
        )
        calibration = export_profile_calibration(selected)
        probes = probe_profile_combinations(
            adapter,
            device=inspection.opened.device,
            selected_profiles=selected,
            duration_seconds=float(
                args.probe_duration
                if args.probe_duration is not None
                else config.get("probe", {}).get("duration_seconds", 1.0)
            ),
        )
        _write_json(session / "device_info.json", inspection.device_info)
        _write_json(session / "stream_profiles.json", inspection.stream_profiles)
        _write_json(
            session / "selected_profiles.json",
            {
                "selection": [decision.as_dict() for decision in decisions],
                "selected_profile_ids": {
                    name: profile.profile_id for name, profile in selected.items()
                },
            },
        )
        _write_json(session / "calibration.json", calibration)
        _write_json(session / "probe_results.json", {"attempts": probes})
        summary = _inspect_summary(
            selected=selected,
            decisions=decisions,
            calibration=calibration,
            probes=probes,
        )
        _write_json(session / "validation_summary.json", summary)
        _write_json(
            session / "session_state.json",
            {
                "overall_status": "COMPLETE",
                "command": "inspect",
                "output": str(session),
            },
        )
    except Exception as error:
        _write_json(
            session / "session_state.json",
            {
                "overall_status": "INCOMPLETE",
                "command": "inspect",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        print(f"검사 실패: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"부분 결과: {session}", file=sys.stderr)
        return 2
    print(f"검사 결과: {session}")
    return 0


def _run_record(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    _require_device_confirmation(args)
    session = _session_dir(config, args.output, f"record_{args.mode}")
    duration = float(
        args.duration
        if args.duration is not None
        else config.get("record", {})
        .get("default_duration_seconds", {})
        .get(args.mode, 60)
    )
    initial_state = {
        "overall_status": "INCOMPLETE",
        "command": "record",
        "mode": args.mode,
        "requested_duration_seconds": duration,
    }
    _write_json(session / "session_state.json", initial_state)
    try:
        adapter = OrbbecSdkAdapter()
        serial, index = _device_choice(args, config)
        inspection = inspect_connected_device(
            adapter, serial_number=serial, device_index=index
        )
        decisions, selected = _select_profiles(
            inspection.profiles,
            config,
            allow_fallback_override=args.allow_fallback,
        )
        if not selected:
            raise RuntimeError("No requested stream profile was selected.")
        calibration = export_profile_calibration(selected)
        _write_json(session / "device_info.json", inspection.device_info)
        _write_json(session / "stream_profiles.json", inspection.stream_profiles)
        _write_json(
            session / "selected_profiles.json",
            {
                "selection": [decision.as_dict() for decision in decisions],
                "selected_profile_ids": {
                    name: profile.profile_id for name, profile in selected.items()
                },
            },
        )
        _write_json(session / "calibration.json", calibration)
        record_config = dict(config.get("record", {}))
        recorder = SensorRecorder(
            adapter,
            device=inspection.opened.device,
            selected_profiles=selected,
            output_dir=session,
            queue_size=int(record_config.get("queue_size", 8192)),
            sample_fps=float(
                args.sample_fps
                if args.sample_fps is not None
                else record_config.get("sample_fps", 1.0)
            ),
            save_all_frames=args.save_all_frames,
            no_preview=args.no_preview,
        )
        state = recorder.record(duration_seconds=duration, mode=args.mode)
        state["command"] = "record"
        _write_json(session / "session_state.json", state)
        summary = analyze_session(session, config)
    except Exception as error:
        _write_json(
            session / "session_state.json",
            {
                **initial_state,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        print(f"기록 실패: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"부분 결과: {session}", file=sys.stderr)
        return 2
    print(f"기록 결과: {session}")
    print(f"Phase 1 판정: {summary['overall_status']}")
    return 0 if state["overall_status"] == "COMPLETE" else 2


def _run_analyze(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    session = Path(args.session)
    if not session.is_absolute():
        session = ROOT / session
    if not session.is_dir():
        print(f"Session directory not found: {session}", file=sys.stderr)
        return 2
    summary = analyze_session(session, config)
    print(f"분석 결과: {session / 'validation_summary.json'}")
    print(f"Phase 1 판정: {summary['overall_status']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orbbec Gemini 335L Phase-1 sensor validation"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("inspect", "record"):
        live = subparsers.add_parser(name)
        live.add_argument("--i-confirm-device-access", action="store_true")
        live.add_argument("--serial")
        live.add_argument("--device-index", type=int)
        live.add_argument("--allow-fallback", action="store_true")
        live.add_argument("--output", help="output root; a timestamped session is created")
        if name == "inspect":
            live.add_argument("--probe-duration", type=float)
        else:
            live.add_argument(
                "--mode",
                choices=("static", "translation", "translation-yaw"),
                required=True,
            )
            live.add_argument("--duration", type=float)
            live.add_argument("--sample-fps", type=float)
            live.add_argument("--save-all-frames", action="store_true")
            live.add_argument("--no-preview", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--session", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(Path(args.config))
        if args.command == "inspect":
            return _run_inspect(args, config)
        if args.command == "record":
            return _run_record(args, config)
        return _run_analyze(args, config)
    except (OSError, ValueError, PermissionError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
