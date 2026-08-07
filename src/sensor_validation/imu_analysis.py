"""Pure static IMU statistics for Gemini 335L Phase-1 validation."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

STANDARD_GRAVITY_MPS2 = 9.80665


def _finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _vectors(
    samples: Iterable[Mapping[str, Any]],
    fields: tuple[str, str, str],
) -> tuple[np.ndarray, int]:
    rows = list(samples)
    vectors: list[tuple[float, float, float]] = []
    for row in rows:
        values = tuple(_finite_float(row.get(field)) for field in fields)
        if all(value is not None for value in values):
            vectors.append((float(values[0]), float(values[1]), float(values[2])))
    return (
        np.asarray(vectors, dtype=np.float64).reshape((-1, 3)),
        len(rows) - len(vectors),
    )


def _axis_stats(values: np.ndarray, *, include_p95_abs: bool) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, axis in enumerate(("x", "y", "z")):
        column = values[:, index]
        stats = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "median": float(np.median(column)),
        }
        if include_p95_abs:
            stats["p95_abs"] = float(np.percentile(np.abs(column), 95.0))
        result[axis] = stats
    return result


def _spike_summary(
    norms: np.ndarray,
    *,
    mad_multiplier: float,
    physical_threshold: float | None,
) -> dict[str, Any]:
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    robust_sigma = 1.4826 * mad
    robust_threshold = median + mad_multiplier * robust_sigma
    robust_mask = norms > robust_threshold if robust_sigma > 0.0 else np.zeros(norms.shape, dtype=bool)
    physical_mask = (
        norms > float(physical_threshold)
        if physical_threshold is not None
        else np.zeros(norms.shape, dtype=bool)
    )
    return {
        "count": int(np.count_nonzero(robust_mask | physical_mask)),
        "robust_count": int(np.count_nonzero(robust_mask)),
        "physical_count": int(np.count_nonzero(physical_mask)),
        "median_norm": median,
        "mad_norm": mad,
        "mad_multiplier": float(mad_multiplier),
        "robust_threshold": robust_threshold,
        "physical_threshold": physical_threshold,
    }


def analyze_gyroscope(
    samples: Iterable[Mapping[str, Any]],
    *,
    fields: tuple[str, str, str] = ("gyro_x", "gyro_y", "gyro_z"),
    mad_multiplier: float = 6.0,
    physical_spike_threshold_rad_s: float | None = None,
) -> dict[str, Any]:
    values, invalid_count = _vectors(samples, fields)
    if not values.size:
        return {
            "status": "NOT_TESTED",
            "sample_count": 0,
            "invalid_sample_count": invalid_count,
            "axis": None,
            "norm": None,
            "spikes": None,
            "bias_proxy_rad_s": None,
        }
    norms = np.linalg.norm(values, axis=1)
    means = np.mean(values, axis=0)
    return {
        "status": "PASS",
        "sample_count": int(values.shape[0]),
        "invalid_sample_count": invalid_count,
        "axis": _axis_stats(values, include_p95_abs=True),
        "norm": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
        },
        "spikes": _spike_summary(
            norms,
            mad_multiplier=mad_multiplier,
            physical_threshold=physical_spike_threshold_rad_s,
        ),
        "bias_proxy_rad_s": {
            "x": float(means[0]),
            "y": float(means[1]),
            "z": float(means[2]),
            "definition": "stationary axis mean; not a calibrated bias",
        },
    }


def analyze_accelerometer(
    samples: Iterable[Mapping[str, Any]],
    *,
    fields: tuple[str, str, str] = ("accel_x", "accel_y", "accel_z"),
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
    mad_multiplier: float = 6.0,
    physical_spike_threshold_mps2: float | None = None,
) -> dict[str, Any]:
    values, invalid_count = _vectors(samples, fields)
    if not values.size:
        return {
            "status": "NOT_TESTED",
            "sample_count": 0,
            "invalid_sample_count": invalid_count,
            "axis": None,
            "norm": None,
            "spikes": None,
            "axis_bias_interpretation": "unavailable",
        }
    norms = np.linalg.norm(values, axis=1)
    norm_mean = float(np.mean(norms))
    return {
        "status": "PASS",
        "sample_count": int(values.shape[0]),
        "invalid_sample_count": invalid_count,
        "axis": _axis_stats(values, include_p95_abs=False),
        "norm": {
            "mean": norm_mean,
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
            "gravity_reference_mps2": float(gravity_mps2),
            "mean_minus_gravity_mps2": norm_mean - float(gravity_mps2),
            "mean_abs_error_from_gravity_mps2": abs(norm_mean - float(gravity_mps2)),
        },
        "spikes": _spike_summary(
            norms,
            mad_multiplier=mad_multiplier,
            physical_threshold=physical_spike_threshold_mps2,
        ),
        "axis_bias_interpretation": (
            "axis means include gravity projection because camera orientation is unknown; "
            "they are not reported as accelerometer bias"
        ),
    }
