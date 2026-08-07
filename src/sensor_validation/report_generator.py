"""Offline session analysis, Phase-1 verdict, and Korean Markdown report."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .imu_analysis import analyze_accelerometer, analyze_gyroscope
from .profile_selector import sample_rate_hz
from .timestamp_analysis import (
    pair_nearest_timestamps,
    read_csv_rows,
    summarize_stream_rows,
)

REQUIRED_STREAMS = ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR", "ACCEL", "GYRO")
VIDEO_STREAMS = ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _selected_profiles(session_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(session_dir / "selected_profiles.json")
    result: dict[str, dict[str, Any]] = {}
    for entry in payload.get("selection", []):
        if isinstance(entry, dict) and isinstance(entry.get("selected"), dict):
            result[str(entry.get("sensor"))] = dict(entry["selected"])
    return result


def _configured_rate(profile: Mapping[str, Any]) -> float | None:
    if profile.get("kind") == "video":
        return _float(profile.get("fps"))
    return sample_rate_hz(str(profile.get("sample_rate") or ""))


def _host_us(rows: list[dict[str, Any]]) -> list[float | None]:
    result: list[float | None] = []
    for row in rows:
        value = _float(row.get("host_monotonic_ns"))
        result.append(value / 1000.0 if value is not None else None)
    return result


def _missing_ratio(result: Mapping[str, Any]) -> tuple[float | None, str]:
    frame = result.get("frame_number", {})
    missing = frame.get("missing_count") if isinstance(frame, dict) else None
    valid_count = frame.get("valid_count") if isinstance(frame, dict) else None
    if isinstance(missing, int) and isinstance(valid_count, int) and valid_count:
        return missing / max(valid_count + missing, 1), "frame_number_authoritative"
    estimated = result.get("estimated_missing_from_timestamp")
    valid_timestamp_count = result.get("valid_timestamp_count")
    if isinstance(estimated, int) and isinstance(valid_timestamp_count, int) and valid_timestamp_count:
        return estimated / max(valid_timestamp_count + estimated, 1), "timestamp_estimate"
    return None, "unavailable"


def _stream_verdict(
    device_analysis: Mapping[str, Any],
    *,
    minimum_rate_ratio: float,
    maximum_drop_ratio: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not device_analysis.get("valid_timestamp_count"):
        return "NOT_TESTED", ["no_valid_device_timestamps"]
    if device_analysis.get("reverse_count", 0) > 0:
        reasons.append("device_timestamp_reversal")
    if device_analysis.get("duplicate_count", 0) > 0:
        reasons.append("device_timestamp_duplicate")
    rate_ratio = device_analysis.get("configured_rate_ratio")
    if rate_ratio is not None and rate_ratio < minimum_rate_ratio:
        reasons.append("measured_rate_below_threshold")
    drop_ratio, source = _missing_ratio(device_analysis)
    if drop_ratio is not None and drop_ratio > maximum_drop_ratio:
        reasons.append(f"drop_ratio_above_threshold:{source}")
    return ("BLOCKED" if reasons else "PASS"), reasons


def analyze_session(
    session_dir: str | Path,
    config: Mapping[str, Any],
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    session_path = Path(session_dir)
    video_rows, video_warnings = read_csv_rows(session_path / "video_frames.csv")
    imu_rows, imu_warnings = read_csv_rows(session_path / "imu_samples.csv")
    rows_by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in video_rows + imu_rows:
        rows_by_stream[str(row.get("stream") or "UNKNOWN")].append(row)

    selected = _selected_profiles(session_path)
    analysis_config = dict(config.get("analysis", {}))
    gap_factor = float(analysis_config.get("gap_factor", 1.5))
    minimum_rate_ratio = float(analysis_config.get("minimum_rate_ratio", 0.95))
    maximum_drop_ratio = float(analysis_config.get("maximum_drop_ratio", 0.01))
    pairing_config = dict(analysis_config.get("pairing_tolerance_us", {}))

    streams: dict[str, Any] = {}
    blockers: list[str] = []
    warnings = video_warnings + imu_warnings
    for stream in REQUIRED_STREAMS:
        rows = rows_by_stream.get(stream, [])
        rate = _configured_rate(selected.get(stream, {}))
        device = summarize_stream_rows(
            rows,
            timestamp_field="device_timestamp_us",
            configured_rate_hz=rate,
            gap_factor=gap_factor,
        )
        host_rows = [dict(row, host_timestamp_us=value) for row, value in zip(rows, _host_us(rows), strict=True)]
        host = summarize_stream_rows(
            host_rows,
            timestamp_field="host_timestamp_us",
            configured_rate_hz=rate,
            gap_factor=gap_factor,
        )
        verdict, reasons = _stream_verdict(
            device,
            minimum_rate_ratio=minimum_rate_ratio,
            maximum_drop_ratio=maximum_drop_ratio,
        )
        if stream in selected and not rows:
            verdict = "BLOCKED"
            reasons.append("selected_stream_has_no_samples")
        if stream not in selected:
            verdict = "BLOCKED"
            reasons.append("stream_not_selected_or_unsupported")
        drop_ratio, drop_source = _missing_ratio(device)
        streams[stream] = {
            "status": verdict,
            "reasons": reasons,
            "configured_rate_hz": rate,
            "device_clock": device,
            "host_monotonic_clock": host,
            "drop_ratio": drop_ratio,
            "drop_source": drop_source,
        }
        if verdict == "BLOCKED":
            blockers.extend(f"{stream}:{reason}" for reason in reasons)

    pairing: dict[str, Any] = {}
    if rows_by_stream.get("RGB"):
        rgb_device = [row.get("device_timestamp_us") for row in rows_by_stream["RGB"]]
        for target in ("DEPTH", "LEFT_IR", "RIGHT_IR"):
            if rows_by_stream.get(target):
                pairing[f"RGB_{target}_device"] = pair_nearest_timestamps(
                    rgb_device,
                    [row.get("device_timestamp_us") for row in rows_by_stream[target]],
                    tolerance_us=float(pairing_config.get("video", 20_000)),
                )
    if rows_by_stream.get("ACCEL") and rows_by_stream.get("GYRO"):
        pairing["ACCEL_GYRO_device"] = pair_nearest_timestamps(
            [row.get("device_timestamp_us") for row in rows_by_stream["ACCEL"]],
            [row.get("device_timestamp_us") for row in rows_by_stream["GYRO"]],
            tolerance_us=float(pairing_config.get("imu", 10_000)),
        )
    pairing["VIDEO_IMU_device"] = {
        "status": "UNAVAILABLE",
        "reason": "separate_pipeline_device_clock_compatibility_not_verified",
    }
    for video_stream in VIDEO_STREAMS:
        if not rows_by_stream.get(video_stream):
            continue
        for imu_stream in ("ACCEL", "GYRO"):
            if not rows_by_stream.get(imu_stream):
                continue
            pairing[f"{video_stream}_{imu_stream}_host_monotonic"] = (
                pair_nearest_timestamps(
                    _host_us(rows_by_stream[video_stream]),
                    _host_us(rows_by_stream[imu_stream]),
                    tolerance_us=float(
                        pairing_config.get("video_imu_host", 20_000)
                    ),
                )
            )

    session_state = _load_json(session_path / "session_state.json")
    mode = str(session_state.get("mode") or "unknown")
    imu_config = dict(analysis_config.get("imu", {}))
    if mode == "static":
        gyro_rows = [
            {
                "gyro_x": row.get("x"),
                "gyro_y": row.get("y"),
                "gyro_z": row.get("z"),
            }
            for row in rows_by_stream.get("GYRO", [])
        ]
        accel_rows = [
            {
                "accel_x": row.get("x"),
                "accel_y": row.get("y"),
                "accel_z": row.get("z"),
            }
            for row in rows_by_stream.get("ACCEL", [])
        ]
        gyro = analyze_gyroscope(
            gyro_rows,
            mad_multiplier=float(imu_config.get("mad_multiplier", 6.0)),
            physical_spike_threshold_rad_s=_float(
                imu_config.get("gyro_physical_spike_threshold_rad_s")
            ),
        )
        accel = analyze_accelerometer(
            accel_rows,
            mad_multiplier=float(imu_config.get("mad_multiplier", 6.0)),
            physical_spike_threshold_mps2=_float(
                imu_config.get("accel_physical_spike_threshold_mps2")
            ),
        )
    else:
        gyro = {"status": "NOT_TESTED", "reason": "mode_is_not_static"}
        accel = {"status": "NOT_TESTED", "reason": "mode_is_not_static"}

    imu_status = "PASS"
    if gyro.get("status") != "PASS" or accel.get("status") != "PASS":
        imu_status = "NOT_TESTED"
    else:
        bias = gyro["bias_proxy_rad_s"]
        bias_norm = float(
            np.linalg.norm([bias["x"], bias["y"], bias["z"]])
        )
        gravity_error = float(accel["norm"]["mean_abs_error_from_gravity_mps2"])
        gyro_spike_ratio = gyro["spikes"]["count"] / max(gyro["sample_count"], 1)
        accel_spike_ratio = accel["spikes"]["count"] / max(accel["sample_count"], 1)
        if bias_norm > float(imu_config.get("maximum_gyro_bias_norm_rad_s", 0.05)):
            blockers.append("IMU:gyro_bias_proxy_above_threshold")
            imu_status = "BLOCKED"
        if gravity_error > float(
            imu_config.get("maximum_accel_gravity_error_mps2", 0.5)
        ):
            blockers.append("IMU:accel_gravity_error_above_threshold")
            imu_status = "BLOCKED"
        maximum_spike_ratio = float(imu_config.get("maximum_spike_ratio", 0.01))
        if max(gyro_spike_ratio, accel_spike_ratio) > maximum_spike_ratio:
            blockers.append("IMU:spike_ratio_above_threshold")
            imu_status = "BLOCKED"

    calibration = _load_json(session_path / "calibration.json")
    camera_imu_status = calibration.get(
        "camera_imu_extrinsic_status", "NOT_TESTED"
    )
    if camera_imu_status != "AVAILABLE":
        blockers.append("CALIBRATION:camera_imu_extrinsic_not_available")
    if session_state.get("overall_status") not in {None, "COMPLETE"}:
        blockers.append("SESSION:recording_incomplete")
    overflow_counts = session_state.get("queue_overflow_counts", {})
    received_counts = session_state.get("received_counts", {})
    if isinstance(overflow_counts, dict) and any(
        int(value or 0) > 0 for value in overflow_counts.values()
    ):
        total_overflow = sum(int(value or 0) for value in overflow_counts.values())
        total_received = (
            sum(int(value or 0) for value in received_counts.values())
            if isinstance(received_counts, dict)
            else 0
        )
        overflow_ratio = total_overflow / max(total_received, 1)
        warnings.append(f"callback_queue_overflow:{total_overflow}")
        if overflow_ratio > maximum_drop_ratio:
            blockers.append("SESSION:callback_queue_overflow_ratio_above_threshold")
    for key in ("writer_errors", "callback_errors", "stop_errors"):
        errors = session_state.get(key, [])
        if errors:
            blockers.append(f"SESSION:{key}")
    if session_state.get("snapshot_errors"):
        warnings.append("representative_snapshot_errors")

    if not selected and not video_rows and not imu_rows:
        overall_status = "NOT_TESTED"
    elif blockers:
        overall_status = "BLOCKED"
    elif warnings:
        overall_status = "PASS_WITH_WARNINGS"
    else:
        overall_status = "PASS"
    summary = {
        "schema_version": 1,
        "session": str(session_path.resolve()),
        "overall_status": overall_status,
        "phase2_readiness": "READY" if overall_status.startswith("PASS") else "BLOCKED",
        "mode": mode,
        "streams": streams,
        "pairing": pairing,
        "imu_static": {
            "status": imu_status,
            "gyroscope": gyro,
            "accelerometer": accel,
        },
        "calibration": {
            "camera_imu_extrinsic_status": camera_imu_status,
            "identity_substitution_used": calibration.get(
                "identity_substitution_used", False
            ),
        },
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
    }
    if write_outputs:
        _write_json(session_path / "validation_summary.json", summary)
        (session_path / "phase1_report.md").write_text(
            render_markdown_report(summary), encoding="utf-8"
        )
    return summary


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gemini 335L Phase 1 센서 검증 보고서",
        "",
        f"- 전체 판정: `{summary.get('overall_status')}`",
        f"- Phase 2 준비 상태: `{summary.get('phase2_readiness')}`",
        f"- 기록 모드: `{summary.get('mode')}`",
        "",
        "## 스트림 판정",
        "",
    ]
    streams = summary.get("streams", {})
    for name in REQUIRED_STREAMS:
        item = streams.get(name, {}) if isinstance(streams, dict) else {}
        lines.append(
            f"- {name}: `{item.get('status', 'NOT_TESTED')}` "
            f"(samples={item.get('device_clock', {}).get('sample_count', 0)})"
        )
    lines.extend(["", "## IMU 및 Calibration", ""])
    imu = summary.get("imu_static", {})
    calibration = summary.get("calibration", {})
    lines.append(f"- 정지 IMU 판정: `{imu.get('status', 'NOT_TESTED')}`")
    lines.append(
        "- Camera–IMU extrinsic: "
        f"`{calibration.get('camera_imu_extrinsic_status', 'NOT_TESTED')}`"
    )
    lines.extend(["", "## 차단 사유", ""])
    blockers = summary.get("blockers", [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 해석 주의사항",
            "",
            "- Device timestamp와 host monotonic timestamp는 서로 다른 clock domain으로 분석했습니다.",
            "- 별도 pipeline의 video–IMU device clock 호환성은 검증 전까지 unavailable입니다.",
            "- SDK가 노출하지 않은 calibration 값은 identity로 대체하지 않았습니다.",
            "- Accel 축 평균은 중력 투영을 포함하므로 bias로 해석하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)
