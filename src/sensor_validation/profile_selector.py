"""Deterministic, fully logged profile selection without silent downgrade."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .sdk_adapter import ProfileHandle


@dataclass(frozen=True)
class ProfileDecision:
    sensor: str
    requested: dict[str, Any]
    selected_profile_id: str | None
    selected: dict[str, Any] | None
    exact_match: bool
    fallback_used: bool
    reason: str
    candidates_considered: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "requested": self.requested,
            "selected_profile_id": self.selected_profile_id,
            "selected": self.selected,
            "exact_match": self.exact_match,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "candidates_considered": self.candidates_considered,
        }


def _normalized_name(value: Any) -> str | None:
    if value is None or value == "":
        return None
    normalized = str(value).upper()
    for prefix in ("OBFORMAT.", "OBSENSORTYPE.", "OB_FORMAT_", "OB_SENSOR_TYPE_"):
        normalized = normalized.replace(prefix, "")
    if normalized.endswith("_SENSOR"):
        normalized = normalized[: -len("_SENSOR")]
    return normalized


def _profile_matches_sensor(profile: ProfileHandle, sensor: str) -> bool:
    expected = _normalized_name(sensor) or ""
    actual = _normalized_name(profile.sensor_type) or ""
    stream = _normalized_name(profile.stream_type) or ""
    aliases = {
        "RGB": ("COLOR",),
        "COLOR": ("COLOR",),
        "DEPTH": ("DEPTH",),
        "LEFT_IR": ("LEFT_IR",),
        "RIGHT_IR": ("RIGHT_IR",),
        "ACCEL": ("ACCEL",),
        "GYRO": ("GYRO",),
    }
    tokens = aliases.get(expected, (expected,))
    return any(token in actual or token in stream for token in tokens)


def _video_exact(profile: ProfileHandle, request: Mapping[str, Any]) -> bool:
    checks = (
        request.get("width") in {None, profile.width},
        request.get("height") in {None, profile.height},
        request.get("fps") in {None, profile.fps},
        _normalized_name(request.get("format"))
        in {None, _normalized_name(profile.format)},
    )
    return all(checks)


def _video_score(profile: ProfileHandle, request: Mapping[str, Any]) -> tuple[float, ...]:
    requested_fps = float(request.get("fps") or 0.0)
    requested_width = int(request.get("width") or 0)
    requested_height = int(request.get("height") or 0)
    requested_format = _normalized_name(request.get("format"))
    return (
        0.0 if requested_format in {None, _normalized_name(profile.format)} else 1.0,
        abs(float(profile.fps or 0.0) - requested_fps) if requested_fps else 0.0,
        abs(int(profile.width or 0) - requested_width) if requested_width else 0.0,
        abs(int(profile.height or 0) - requested_height) if requested_height else 0.0,
        -(float(profile.fps or 0.0)),
        -(int(profile.width or 0) * int(profile.height or 0)),
    )


def select_video_profile(
    profiles: Iterable[ProfileHandle],
    *,
    sensor: str,
    request: Mapping[str, Any],
    allow_fallback: bool,
) -> ProfileDecision:
    candidates = [
        profile
        for profile in profiles
        if profile.kind == "video" and _profile_matches_sensor(profile, sensor)
    ]
    exact = [profile for profile in candidates if _video_exact(profile, request)]
    selected: ProfileHandle | None = None
    reason: str
    fallback_used = False
    if exact:
        selected = sorted(
            exact, key=lambda item: (not item.is_default, item.profile_id)
        )[0]
        reason = (
            "exact_requested_default_profile"
            if selected.is_default
            else "exact_requested_profile"
        )
    elif allow_fallback and candidates:
        selected = min(candidates, key=lambda item: _video_score(item, request))
        fallback_used = True
        reason = "explicit_fallback_ranked_by_format_fps_resolution"
    elif candidates:
        reason = "requested_profile_not_available_and_fallback_disabled"
    else:
        reason = "sensor_or_profile_unavailable"
    return ProfileDecision(
        sensor=sensor,
        requested=dict(request),
        selected_profile_id=selected.profile_id if selected else None,
        selected=selected.as_dict() if selected else None,
        exact_match=bool(exact),
        fallback_used=fallback_used,
        reason=reason,
        candidates_considered=len(candidates),
    )


def sample_rate_hz(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.upper().split(".")[-1].replace("SAMPLE_RATE_", "")
    multiplier = 1000.0 if normalized.endswith("_KHZ") else 1.0
    normalized = re.sub(r"_(K?HZ)$", "", normalized)
    pieces = re.findall(r"\d+", normalized)
    if not pieces:
        return None
    number = pieces[0] if len(pieces) == 1 else f"{pieces[0]}.{''.join(pieces[1:])}"
    return float(number) * multiplier


def _imu_exact(profile: ProfileHandle, request: Mapping[str, Any]) -> bool:
    requested_rate = request.get("sample_rate_hz")
    requested_range = _normalized_name(request.get("full_scale_range"))
    profile_rate = sample_rate_hz(profile.sample_rate)
    return (
        (requested_rate is None or profile_rate == float(requested_rate))
        and requested_range in {None, _normalized_name(profile.full_scale_range)}
    )


def select_imu_profile(
    profiles: Iterable[ProfileHandle],
    *,
    sensor: str,
    request: Mapping[str, Any],
    allow_fallback: bool,
) -> ProfileDecision:
    expected_kind = "accel" if _normalized_name(sensor) == "ACCEL" else "gyro"
    candidates = [
        profile
        for profile in profiles
        if profile.kind == expected_kind and _profile_matches_sensor(profile, sensor)
    ]
    exact = [profile for profile in candidates if _imu_exact(profile, request)]
    selected: ProfileHandle | None = None
    reason: str
    fallback_used = False
    if exact:
        selected = sorted(exact, key=lambda item: item.profile_id)[0]
        reason = "exact_requested_profile"
    elif allow_fallback and candidates:
        requested_rate = float(request.get("sample_rate_hz") or 0.0)
        selected = min(
            candidates,
            key=lambda item: (
                abs((sample_rate_hz(item.sample_rate) or 0.0) - requested_rate)
                if requested_rate
                else -(sample_rate_hz(item.sample_rate) or 0.0),
                item.profile_id,
            ),
        )
        fallback_used = True
        reason = "explicit_fallback_ranked_by_sample_rate"
    elif candidates:
        reason = "requested_profile_not_available_and_fallback_disabled"
    else:
        reason = "sensor_or_profile_unavailable"
    return ProfileDecision(
        sensor=sensor,
        requested=dict(request),
        selected_profile_id=selected.profile_id if selected else None,
        selected=selected.as_dict() if selected else None,
        exact_match=bool(exact),
        fallback_used=fallback_used,
        reason=reason,
        candidates_considered=len(candidates),
    )


def resolve_profile_handles(
    profiles: Iterable[ProfileHandle],
    decisions: Iterable[ProfileDecision],
) -> dict[str, ProfileHandle]:
    by_id = {profile.profile_id: profile for profile in profiles}
    return {
        decision.sensor: by_id[decision.selected_profile_id]
        for decision in decisions
        if decision.selected_profile_id in by_id
    }


def build_video_probe_matrix(selected: Mapping[str, ProfileHandle]) -> list[dict[str, Any]]:
    """Build ordered, bounded combinations while preserving RGB+Depth first."""

    baseline = [sensor for sensor in ("RGB", "DEPTH") if sensor in selected]
    attempts: list[tuple[str, list[str]]] = [("rgb_depth", baseline)]
    if "LEFT_IR" in selected:
        attempts.append(("rgb_depth_left_ir", baseline + ["LEFT_IR"]))
    if "RIGHT_IR" in selected:
        attempts.append(("rgb_depth_right_ir", baseline + ["RIGHT_IR"]))
    if "LEFT_IR" in selected and "RIGHT_IR" in selected:
        attempts.append(
            ("rgb_depth_dual_ir", baseline + ["LEFT_IR", "RIGHT_IR"])
        )
    return [
        {
            "name": name,
            "sensors": sensors,
            "profile_ids": [selected[sensor].profile_id for sensor in sensors],
        }
        for name, sensors in attempts
        if sensors
    ]
