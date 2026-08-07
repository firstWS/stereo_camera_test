from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.calibration_exporter import export_profile_calibration  # noqa: E402
from sensor_validation.profile_selector import (  # noqa: E402
    build_video_probe_matrix,
    resolve_profile_handles,
    sample_rate_hz,
    select_imu_profile,
    select_video_profile,
)
from sensor_validation.sdk_adapter import ProfileHandle  # noqa: E402


class FakeProfile:
    def __init__(self, *, fail_extrinsic: bool = False) -> None:
        self.fail_extrinsic = fail_extrinsic

    def get_intrinsic(self) -> SimpleNamespace:
        return SimpleNamespace(
            width=1280,
            height=800,
            fx=600.0,
            fy=601.0,
            cx=640.0,
            cy=400.0,
            bias=[0.0, 0.0, 0.0],
            gravity=[0.0, 0.0, 9.80665],
            scale_misalignment=[1.0] * 9,
            temp_slope=[0.0, 0.0, 0.0],
            noise_density=[0.1, 0.1, 0.1],
            random_walk=[0.01, 0.01, 0.01],
            reference_temp=25.0,
        )

    def get_distortion(self) -> SimpleNamespace:
        return SimpleNamespace(
            k1=0.1,
            k2=0.01,
            k3=0.0,
            k4=0.0,
            k5=0.0,
            k6=0.0,
            p1=0.0,
            p2=0.0,
            model=SimpleNamespace(name="BROWN_CONRADY"),
        )

    def get_extrinsic_to(self, _other: object) -> SimpleNamespace:
        if self.fail_extrinsic:
            raise RuntimeError("not exposed")
        return SimpleNamespace(
            rot=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            transform=[0.0, 0.0, 0.0],
        )


def _video(
    profile_id: str,
    sensor: str,
    width: int,
    height: int,
    fps: float,
) -> ProfileHandle:
    return ProfileHandle(
        profile_id=profile_id,
        sensor_type=sensor,
        stream_type=sensor,
        kind="video",
        format="OB_FORMAT_RGB",
        width=width,
        height=height,
        fps=fps,
        raw=FakeProfile(),
    )


def test_video_profile_exact_and_explicit_fallback() -> None:
    profiles = [
        _video("rgb:0", "COLOR", 640, 480, 30.0),
        _video("rgb:1", "COLOR", 1280, 800, 30.0),
    ]
    exact = select_video_profile(
        profiles,
        sensor="RGB",
        request={"width": 1280, "height": 800, "fps": 30.0, "format": "OB_FORMAT_RGB"},
        allow_fallback=False,
    )
    fallback = select_video_profile(
        profiles,
        sensor="RGB",
        request={"width": 1920, "height": 1080, "fps": 30.0, "format": "OB_FORMAT_RGB"},
        allow_fallback=True,
    )
    assert exact.selected_profile_id == "rgb:1"
    assert exact.exact_match is True
    assert fallback.selected_profile_id == "rgb:1"
    assert fallback.fallback_used is True


def test_fallback_disabled_does_not_select_profile() -> None:
    decision = select_video_profile(
        [_video("rgb:0", "COLOR", 640, 480, 30.0)],
        sensor="RGB",
        request={"width": 1280, "height": 800, "fps": 30.0},
        allow_fallback=False,
    )
    assert decision.selected is None
    assert "fallback_disabled" in decision.reason


def test_imu_selection_and_probe_order() -> None:
    accel = ProfileHandle(
        profile_id="accel:0",
        sensor_type="ACCEL",
        stream_type="ACCEL",
        kind="accel",
        format="ACCEL",
        sample_rate="OB_SAMPLE_RATE_200_HZ",
        full_scale_range="OB_ACCEL_FULL_SCALE_RANGE_4G",
        raw=FakeProfile(),
    )
    decision = select_imu_profile(
        [accel],
        sensor="ACCEL",
        request={
            "sample_rate_hz": 200,
            "full_scale_range": "OB_ACCEL_FULL_SCALE_RANGE_4G",
        },
        allow_fallback=False,
    )
    assert decision.selected_profile_id == "accel:0"

    videos = {
        "RGB": _video("rgb", "COLOR", 1280, 800, 30),
        "DEPTH": _video("depth", "DEPTH", 1280, 800, 30),
        "LEFT_IR": _video("lir", "LEFT_IR", 1280, 800, 30),
        "RIGHT_IR": _video("rir", "RIGHT_IR", 1280, 800, 30),
    }
    probes = build_video_probe_matrix(videos)
    assert [attempt["name"] for attempt in probes] == [
        "rgb_depth",
        "rgb_depth_left_ir",
        "rgb_depth_right_ir",
        "rgb_depth_dual_ir",
    ]
    assert sample_rate_hz("SAMPLE_RATE_12_5_HZ") == 12.5
    assert sample_rate_hz("SAMPLE_RATE_1_5625_HZ") == 1.5625
    assert sample_rate_hz("SAMPLE_RATE_16_KHZ") == 16_000


def test_calibration_export_preserves_sdk_identity_without_substitution() -> None:
    video = _video("rgb", "COLOR", 1280, 800, 30)
    accel = ProfileHandle(
        profile_id="accel",
        sensor_type="ACCEL",
        stream_type="ACCEL",
        kind="accel",
        format="ACCEL",
        sample_rate="OB_SAMPLE_RATE_200_HZ",
        raw=FakeProfile(),
    )
    result = export_profile_calibration({"RGB": video, "ACCEL": accel})
    assert result["camera_imu_extrinsic_status"] == "AVAILABLE"
    assert result["identity_substitution_used"] is False
    assert result["extrinsics"][0]["extrinsic"]["identity_returned_by_sdk"] is True


def test_missing_camera_imu_extrinsic_is_not_fabricated() -> None:
    video = _video("rgb", "COLOR", 1280, 800, 30)
    video.raw = FakeProfile(fail_extrinsic=True)
    accel = ProfileHandle(
        profile_id="accel",
        sensor_type="ACCEL",
        stream_type="ACCEL",
        kind="accel",
        format="ACCEL",
        raw=FakeProfile(fail_extrinsic=True),
    )
    result = export_profile_calibration({"RGB": video, "ACCEL": accel})
    assert result["camera_imu_extrinsic_status"] == "NOT_EXPOSED"
    assert all(item["extrinsic"] is None for item in result["extrinsics"])


def test_resolve_handles_ignores_unselected_decisions() -> None:
    profiles = [_video("rgb", "COLOR", 1280, 800, 30)]
    selected = select_video_profile(
        profiles,
        sensor="RGB",
        request={"width": 1280, "height": 800, "fps": 30.0},
        allow_fallback=False,
    )
    missing = select_video_profile(
        profiles,
        sensor="DEPTH",
        request={"width": 1280},
        allow_fallback=False,
    )
    assert resolve_profile_handles(profiles, [selected, missing]) == {
        "RGB": profiles[0]
    }
