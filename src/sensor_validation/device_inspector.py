"""Device, sensor, profile, and installed SDK capability inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sdk_adapter import OpenedDevice, OrbbecSdkAdapter, ProfileHandle, safe_call


@dataclass
class InspectionResult:
    opened: OpenedDevice
    profiles: list[ProfileHandle]
    device_info: dict[str, Any]
    stream_profiles: dict[str, Any]


def _api_capabilities(sdk: Any) -> dict[str, bool]:
    frame_type = getattr(sdk, "Frame", None)
    profile_type = getattr(sdk, "StreamProfile", None)
    pipeline_type = getattr(sdk, "Pipeline", None)
    return {
        "frame_get_timestamp_us": hasattr(frame_type, "get_timestamp_us"),
        "frame_get_system_timestamp_us": hasattr(
            frame_type, "get_system_timestamp_us"
        ),
        "frame_get_global_timestamp_us": hasattr(
            frame_type, "get_global_timestamp_us"
        ),
        "frame_get_index": hasattr(frame_type, "get_index"),
        "frame_metadata": hasattr(frame_type, "has_metadata")
        and hasattr(frame_type, "get_metadata_value"),
        "profile_get_extrinsic_to": hasattr(profile_type, "get_extrinsic_to"),
        "pipeline_camera_param": hasattr(pipeline_type, "get_camera_param"),
        "pipeline_frame_sync": hasattr(pipeline_type, "enable_frame_sync"),
        "native_record_device": hasattr(sdk, "RecordDevice"),
        "native_recording_lifecycle_validated": False,
    }


def inspect_connected_device(
    adapter: OrbbecSdkAdapter,
    *,
    serial_number: str | None,
    device_index: int | None,
) -> InspectionResult:
    opened, device_list_info = adapter.open_device(
        serial_number=serial_number,
        device_index=device_index,
    )
    profiles, profile_errors = adapter.enumerate_profiles(opened.device)
    device_info = {
        **adapter.environment_info(),
        "sdk_runtime_version": safe_call(adapter.sdk, "get_version"),
        "selected_device_index": opened.index,
        "connected_devices": device_list_info,
        "selected_device": adapter.device_info(opened.device),
        "api_capabilities": _api_capabilities(adapter.sdk),
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        grouped.setdefault(profile.sensor_type, []).append(profile.as_dict())
    stream_profiles = {
        "profiles": [profile.as_dict() for profile in profiles],
        "profiles_by_sensor": grouped,
        "enumeration_errors": profile_errors,
        "profile_count": len(profiles),
    }
    return InspectionResult(
        opened=opened,
        profiles=profiles,
        device_info=device_info,
        stream_profiles=stream_profiles,
    )
