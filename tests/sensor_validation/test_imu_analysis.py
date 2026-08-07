from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.imu_analysis import (  # noqa: E402
    STANDARD_GRAVITY_MPS2,
    analyze_accelerometer,
    analyze_gyroscope,
)


def _gyro_rows(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"gyro_x": float(row[0]), "gyro_y": float(row[1]), "gyro_z": float(row[2])}
        for row in values
    ]


def _accel_rows(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"accel_x": float(row[0]), "accel_y": float(row[1]), "accel_z": float(row[2])}
        for row in values
    ]


def test_stationary_gyro_has_small_bias_proxy() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 0.002, size=(1000, 3))
    result = analyze_gyroscope(_gyro_rows(values))
    assert result["status"] == "PASS"
    assert result["sample_count"] == 1000
    assert abs(result["bias_proxy_rad_s"]["x"]) < 0.001
    assert result["norm"]["std"] < 0.003


def test_constant_gyro_bias_is_reported_as_proxy() -> None:
    values = np.tile(np.array([0.02, -0.01, 0.005]), (100, 1))
    result = analyze_gyroscope(_gyro_rows(values))
    assert result["bias_proxy_rad_s"]["x"] == pytest.approx(0.02)
    assert result["bias_proxy_rad_s"]["y"] == pytest.approx(-0.01)
    assert "not a calibrated bias" in result["bias_proxy_rad_s"]["definition"]


def test_gyro_spike_is_detected() -> None:
    values = np.zeros((100, 3), dtype=np.float64)
    values[-1, 0] = 5.0
    result = analyze_gyroscope(
        _gyro_rows(values),
        physical_spike_threshold_rad_s=1.0,
    )
    assert result["spikes"]["count"] == 1
    assert result["spikes"]["physical_count"] == 1


def test_gravity_only_accelerometer_is_stable() -> None:
    values = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2]), (200, 1))
    result = analyze_accelerometer(_accel_rows(values))
    assert result["norm"]["mean"] == pytest.approx(STANDARD_GRAVITY_MPS2)
    assert result["norm"]["std"] == pytest.approx(0.0)
    assert "not reported as accelerometer bias" in result["axis_bias_interpretation"]


def test_noisy_accelerometer_has_larger_norm_std() -> None:
    rng = np.random.default_rng(11)
    quiet = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2]), (500, 1))
    noisy = quiet + rng.normal(0.0, 0.8, size=quiet.shape)
    quiet_result = analyze_accelerometer(_accel_rows(quiet))
    noisy_result = analyze_accelerometer(_accel_rows(noisy))
    assert noisy_result["norm"]["std"] > quiet_result["norm"]["std"]


def test_invalid_and_empty_samples_are_safe() -> None:
    result = analyze_accelerometer(
        [{"accel_x": "", "accel_y": None, "accel_z": "bad"}]
    )
    assert result["status"] == "NOT_TESTED"
    assert result["sample_count"] == 0
    assert result["invalid_sample_count"] == 1
