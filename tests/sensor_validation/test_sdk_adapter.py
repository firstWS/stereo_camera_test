from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.device_inspector import inspect_connected_device  # noqa: E402
from sensor_validation.sdk_adapter import (  # noqa: E402
    MultipleDevicesError,
    NoDeviceError,
    OrbbecSdkAdapter,
)


class FakeVideoProfile:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps

    def is_video_stream_profile(self) -> bool:
        return True

    def is_accel_stream_profile(self) -> bool:
        return False

    def is_gyro_stream_profile(self) -> bool:
        return False

    def as_video_stream_profile(self) -> "FakeVideoProfile":
        return self

    def get_type(self) -> SimpleNamespace:
        return SimpleNamespace(name="COLOR")

    def get_format(self) -> SimpleNamespace:
        return SimpleNamespace(name="RGB")

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_fps(self) -> int:
        return self.fps


class FakeProfileList:
    def __init__(self) -> None:
        self.profiles = [
            FakeVideoProfile(640, 480, 30),
            FakeVideoProfile(1280, 800, 30),
        ]

    def get_count(self) -> int:
        return len(self.profiles)

    def get_stream_profile_by_index(self, index: int) -> FakeVideoProfile:
        return self.profiles[index]

    def get_default_video_stream_profile(self) -> FakeVideoProfile:
        return self.profiles[1]


class FakeSensor:
    def get_stream_profile_list(self) -> FakeProfileList:
        return FakeProfileList()


class FakeSensorList:
    def get_count(self) -> int:
        return 1

    def get_type_by_index(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(name="COLOR_SENSOR")

    def get_sensor_by_index(self, _index: int) -> FakeSensor:
        return FakeSensor()


class FakeDeviceInfo:
    def get_name(self) -> str:
        return "Gemini 335L"

    def get_serial_number(self) -> str:
        return "SERIAL"

    def get_firmware_version(self) -> str:
        return "1.0"


class FakeDevice:
    def get_sensor_list(self) -> FakeSensorList:
        return FakeSensorList()

    def get_device_info(self) -> FakeDeviceInfo:
        return FakeDeviceInfo()


class FakeDeviceList:
    def __init__(self, count: int) -> None:
        self.devices = [FakeDevice() for _ in range(count)]

    def get_count(self) -> int:
        return len(self.devices)

    def get_device_by_index(self, index: int) -> FakeDevice:
        return self.devices[index]

    def get_device_by_serial_number(self, _serial: str) -> FakeDevice:
        return self.devices[0]

    def get_device_name_by_index(self, _index: int) -> str:
        return "Gemini 335L"

    def get_device_serial_number_by_index(self, index: int) -> str:
        return f"SERIAL-{index}"


def _sdk(device_count: int) -> SimpleNamespace:
    device_list = FakeDeviceList(device_count)
    return SimpleNamespace(
        Context=lambda: SimpleNamespace(query_devices=lambda: device_list)
    )


def test_inspector_enumerates_profiles_and_marks_default() -> None:
    result = inspect_connected_device(
        OrbbecSdkAdapter(_sdk(1)),
        serial_number=None,
        device_index=0,
    )
    assert result.device_info["selected_device"]["product_name"] == "Gemini 335L"
    assert len(result.profiles) == 2
    assert result.profiles[0].is_default is False
    assert result.profiles[1].is_default is True
    assert result.stream_profiles["enumeration_errors"] == []


def test_device_selection_refuses_ambiguous_or_missing_device() -> None:
    with pytest.raises(MultipleDevicesError):
        OrbbecSdkAdapter(_sdk(2)).open_device(
            serial_number=None,
            device_index=None,
        )
    with pytest.raises(NoDeviceError):
        OrbbecSdkAdapter(_sdk(0)).open_device(
            serial_number=None,
            device_index=0,
        )
