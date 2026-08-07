"""Installed pyorbbecsdk2 2.1.1 adapter with no import-time hardware access."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable


class SensorValidationSdkError(RuntimeError):
    """Base error for capability-safe SDK operations."""


class NoDeviceError(SensorValidationSdkError):
    pass


class MultipleDevicesError(SensorValidationSdkError):
    pass


def enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def safe_call(obj: Any, method: str, *, default: Any = None) -> Any:
    function = getattr(obj, method, None)
    if not callable(function):
        return default
    try:
        return function()
    except Exception:
        return default


@dataclass
class ProfileHandle:
    profile_id: str
    sensor_type: str
    stream_type: str | None
    kind: str
    format: str | None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: str | None = None
    full_scale_range: str | None = None
    is_default: bool = False
    raw: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "sensor_type": self.sensor_type,
            "stream_type": self.stream_type,
            "kind": self.kind,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "full_scale_range": self.full_scale_range,
            "is_default": self.is_default,
        }


@dataclass
class OpenedDevice:
    context: Any
    device_list: Any
    device: Any
    index: int


class OrbbecSdkAdapter:
    """Thin adapter around only APIs present in the installed SDK stubs/examples."""

    distribution_name = "pyorbbecsdk2"
    module_name = "pyorbbecsdk"

    def __init__(self, sdk_module: Any | None = None) -> None:
        self.sdk = sdk_module if sdk_module is not None else importlib.import_module(self.module_name)

    @classmethod
    def environment_info(cls) -> dict[str, Any]:
        try:
            distribution_version = importlib.metadata.version(cls.distribution_name)
        except importlib.metadata.PackageNotFoundError:
            distribution_version = None
        return {
            "sdk_distribution": cls.distribution_name,
            "sdk_distribution_version": distribution_version,
            "sdk_import_module": cls.module_name,
            "python_version": sys.version,
            "os": platform.platform(),
        }

    def create_context(self) -> Any:
        return self.sdk.Context()

    def list_devices(self, context: Any) -> tuple[Any, list[dict[str, Any]]]:
        device_list = context.query_devices()
        count = int(device_list.get_count())
        devices: list[dict[str, Any]] = []
        for index in range(count):
            info: dict[str, Any] = {"index": index}
            for output_name, method_name in (
                ("name", "get_device_name_by_index"),
                ("serial_number", "get_device_serial_number_by_index"),
                ("pid", "get_device_pid_by_index"),
                ("vid", "get_device_vid_by_index"),
                ("uid", "get_device_uid_by_index"),
                ("connection_type", "get_device_connection_type_by_index"),
            ):
                method = getattr(device_list, method_name, None)
                try:
                    info[output_name] = method(index) if callable(method) else None
                except Exception as error:
                    info[output_name] = None
                    info[f"{output_name}_error"] = f"{type(error).__name__}: {error}"
            devices.append(info)
        return device_list, devices

    def open_device(
        self,
        *,
        serial_number: str | None,
        device_index: int | None,
    ) -> tuple[OpenedDevice, list[dict[str, Any]]]:
        context = self.create_context()
        device_list, devices = self.list_devices(context)
        if not devices:
            raise NoDeviceError("No Orbbec device was returned by Context.query_devices().")
        serial = (serial_number or "").strip()
        if not serial and device_index is None and len(devices) > 1:
            raise MultipleDevicesError(
                "Multiple Orbbec devices are connected; select serial_number or device_index explicitly."
            )
        if serial:
            device = device_list.get_device_by_serial_number(serial)
            selected_index = next(
                (
                    int(item["index"])
                    for item in devices
                    if str(item.get("serial_number") or "") == serial
                ),
                -1,
            )
        else:
            selected_index = int(device_index or 0)
            if selected_index < 0 or selected_index >= len(devices):
                raise SensorValidationSdkError(
                    f"device_index {selected_index} is outside 0..{len(devices) - 1}"
                )
            device = device_list.get_device_by_index(selected_index)
        return OpenedDevice(context, device_list, device, selected_index), devices

    @staticmethod
    def device_info(device: Any) -> dict[str, Any]:
        info = device.get_device_info()
        result: dict[str, Any] = {}
        for output_name, method_name in (
            ("product_name", "get_name"),
            ("serial_number", "get_serial_number"),
            ("uid", "get_uid"),
            ("firmware_version", "get_firmware_version"),
            ("hardware_version", "get_hardware_version"),
            ("supported_min_sdk_version", "get_supported_min_sdk_version"),
            ("pid", "get_pid"),
            ("vid", "get_vid"),
            ("connection_type", "get_connection_type"),
        ):
            method = getattr(info, method_name, None)
            try:
                result[output_name] = method() if callable(method) else None
            except Exception as error:
                result[output_name] = None
                result[f"{output_name}_error"] = f"{type(error).__name__}: {error}"
        return result

    @staticmethod
    def _stream_profile(
        profile: Any,
        sensor_type: str,
        index: int,
        *,
        is_default: bool = False,
    ) -> ProfileHandle:
        is_video = bool(safe_call(profile, "is_video_stream_profile", default=False))
        is_accel = bool(safe_call(profile, "is_accel_stream_profile", default=False))
        is_gyro = bool(safe_call(profile, "is_gyro_stream_profile", default=False))
        kind = "unknown"
        if is_video:
            kind = "video"
            profile = profile.as_video_stream_profile()
        elif is_accel:
            kind = "accel"
            profile = profile.as_accel_stream_profile()
        elif is_gyro:
            kind = "gyro"
            profile = profile.as_gyro_stream_profile()
        return ProfileHandle(
            profile_id=f"{sensor_type}:{index}",
            sensor_type=sensor_type,
            stream_type=enum_name(safe_call(profile, "get_type")),
            kind=kind,
            format=enum_name(safe_call(profile, "get_format")),
            width=_safe_int(safe_call(profile, "get_width")) if kind == "video" else None,
            height=_safe_int(safe_call(profile, "get_height")) if kind == "video" else None,
            fps=_safe_float(safe_call(profile, "get_fps")) if kind == "video" else None,
            sample_rate=(
                enum_name(safe_call(profile, "get_sample_rate"))
                if kind in {"accel", "gyro"}
                else None
            ),
            full_scale_range=(
                enum_name(safe_call(profile, "get_full_scale_range"))
                if kind in {"accel", "gyro"}
                else None
            ),
            is_default=is_default,
            raw=profile,
        )

    def enumerate_profiles(self, device: Any) -> tuple[list[ProfileHandle], list[dict[str, Any]]]:
        sensor_list = device.get_sensor_list()
        profiles: list[ProfileHandle] = []
        errors: list[dict[str, Any]] = []
        count = int(sensor_list.get_count())
        for sensor_index in range(count):
            sensor_type_value = sensor_list.get_type_by_index(sensor_index)
            sensor_type = enum_name(sensor_type_value) or f"sensor_{sensor_index}"
            try:
                sensor = sensor_list.get_sensor_by_index(sensor_index)
                profile_list = sensor.get_stream_profile_list()
                profile_count = int(profile_list.get_count())
                default_video = safe_call(
                    profile_list, "get_default_video_stream_profile"
                )
            except Exception as error:
                errors.append(
                    {
                        "sensor_type": sensor_type,
                        "stage": "get_stream_profile_list",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            for profile_index in range(profile_count):
                try:
                    raw_profile = profile_list.get_stream_profile_by_index(profile_index)
                    handle = self._stream_profile(
                        raw_profile, sensor_type, profile_index
                    )
                    if default_video is not None and handle.kind == "video":
                        default_handle = self._stream_profile(
                            default_video, sensor_type, -1
                        )
                        handle.is_default = _same_profile(handle, default_handle)
                    profiles.append(handle)
                except Exception as error:
                    errors.append(
                        {
                            "sensor_type": sensor_type,
                            "profile_index": profile_index,
                            "stage": "read_profile",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        return profiles, errors

    def create_pipeline(self, device: Any) -> Any:
        return self.sdk.Pipeline(device)

    def create_config(self) -> Any:
        return self.sdk.Config()

    @staticmethod
    def enable_profiles(config: Any, profiles: Iterable[ProfileHandle]) -> None:
        for profile in profiles:
            config.enable_stream(profile.raw)

    @staticmethod
    def stop_pipeline(pipeline: Any | None) -> str | None:
        if pipeline is None:
            return None
        try:
            pipeline.stop()
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _same_profile(left: ProfileHandle, right: ProfileHandle) -> bool:
    return all(
        getattr(left, field_name) == getattr(right, field_name)
        for field_name in (
            "sensor_type",
            "stream_type",
            "kind",
            "format",
            "width",
            "height",
            "fps",
            "sample_rate",
            "full_scale_range",
        )
    )
