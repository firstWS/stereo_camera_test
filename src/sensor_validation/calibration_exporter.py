"""Factory calibration export using only installed SDK profile APIs."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .sdk_adapter import ProfileHandle, enum_name


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "name"):
        return enum_name(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    return parsed if math.isfinite(parsed) else None


def _fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: _json_value(getattr(value, name))
        for name in names
        if hasattr(value, name)
    }


def _camera_intrinsic(value: Any) -> dict[str, Any]:
    return _fields(value, ("width", "height", "fx", "fy", "cx", "cy"))


def _distortion(value: Any) -> dict[str, Any]:
    result = _fields(value, ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"))
    model = getattr(value, "model", None)
    result["model"] = enum_name(model) if model is not None else None
    return result


def _imu_intrinsic(value: Any) -> dict[str, Any]:
    return _fields(
        value,
        (
            "bias",
            "gravity",
            "scale_misalignment",
            "temp_slope",
            "noise_density",
            "random_walk",
            "reference_temp",
        ),
    )


def _extrinsic(value: Any) -> dict[str, Any]:
    rotation = np.asarray(getattr(value, "rot"), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(
        getattr(value, "transform"), dtype=np.float64
    ).reshape(3)
    finite = bool(np.isfinite(rotation).all() and np.isfinite(translation).all())
    orthogonality_error = (
        float(np.linalg.norm(rotation.T @ rotation - np.eye(3))) if finite else None
    )
    determinant = float(np.linalg.det(rotation)) if finite else None
    return {
        "rotation": rotation.tolist() if finite else _json_value(rotation),
        "translation": translation.tolist() if finite else _json_value(translation),
        "rotation_representation": "row_major_3x3",
        "translation_unit": "millimeter_from_installed_sdk_example",
        "finite": finite,
        "rotation_orthogonality_error": orthogonality_error,
        "rotation_determinant": determinant,
        "identity_returned_by_sdk": bool(
            finite
            and np.allclose(rotation, np.eye(3), atol=1e-8)
            and np.allclose(translation, np.zeros(3), atol=1e-8)
        ),
    }


def _query_profile_calibration(name: str, profile: ProfileHandle) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "frame": name,
        "profile": profile.as_dict(),
        "source": "factory_calibration_via_stream_profile",
        "success": False,
        "sdk_api": None,
        "intrinsic": None,
        "distortion": None,
        "error": None,
    }
    try:
        if profile.kind == "video":
            entry["sdk_api"] = "VideoStreamProfile.get_intrinsic/get_distortion"
            entry["intrinsic"] = _camera_intrinsic(profile.raw.get_intrinsic())
            try:
                entry["distortion"] = _distortion(profile.raw.get_distortion())
            except Exception as error:
                entry["distortion_error"] = f"{type(error).__name__}: {error}"
        elif profile.kind in {"accel", "gyro"}:
            entry["sdk_api"] = (
                f"{type(profile.raw).__name__}.get_intrinsic"
            )
            entry["intrinsic"] = _imu_intrinsic(profile.raw.get_intrinsic())
        else:
            entry["error"] = "unsupported_profile_kind"
            return entry
        entry["success"] = True
    except Exception as error:
        entry["error"] = f"{type(error).__name__}: {error}"
    return entry


def export_profile_calibration(
    selected_profiles: Mapping[str, ProfileHandle],
) -> dict[str, Any]:
    """Export per-profile intrinsic and ordered pairwise extrinsic records."""

    intrinsics = [
        _query_profile_calibration(name, profile)
        for name, profile in selected_profiles.items()
    ]
    extrinsics: list[dict[str, Any]] = []
    camera_imu_attempts = 0
    camera_imu_successes = 0
    for from_name, from_profile in selected_profiles.items():
        for to_name, to_profile in selected_profiles.items():
            if from_name == to_name:
                continue
            is_camera_imu = (
                from_profile.kind in {"accel", "gyro"}
                and to_profile.kind == "video"
            ) or (
                from_profile.kind == "video"
                and to_profile.kind in {"accel", "gyro"}
            )
            if is_camera_imu:
                camera_imu_attempts += 1
            entry: dict[str, Any] = {
                "source": "factory_calibration_via_StreamProfile.get_extrinsic_to",
                "from_frame": from_name,
                "to_frame": to_name,
                "sdk_api": "StreamProfile.get_extrinsic_to",
                "success": False,
                "extrinsic": None,
                "error": None,
            }
            try:
                entry["extrinsic"] = _extrinsic(
                    from_profile.raw.get_extrinsic_to(to_profile.raw)
                )
                entry["success"] = True
                if is_camera_imu:
                    camera_imu_successes += 1
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
            extrinsics.append(entry)
    if camera_imu_successes:
        camera_imu_status = "AVAILABLE"
    elif camera_imu_attempts:
        camera_imu_status = "NOT_EXPOSED"
    else:
        camera_imu_status = "NOT_TESTED"
    return {
        "source": "installed_pyorbbecsdk2_profile_api",
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "camera_imu_extrinsic_status": camera_imu_status,
        "camera_imu_attempts": camera_imu_attempts,
        "camera_imu_successes": camera_imu_successes,
        "identity_substitution_used": False,
    }


def export_active_rgb_depth_camera_param(pipeline: Any) -> dict[str, Any]:
    """Optional active RGB/Depth pair query; errors remain explicit."""

    result: dict[str, Any] = {
        "source": "Pipeline.get_camera_param",
        "success": False,
        "depth_intrinsic": None,
        "depth_distortion": None,
        "rgb_intrinsic": None,
        "rgb_distortion": None,
        "depth_to_rgb": None,
        "error": None,
    }
    try:
        parameter = pipeline.get_camera_param()
        result.update(
            {
                "success": True,
                "depth_intrinsic": _camera_intrinsic(parameter.depth_intrinsic),
                "depth_distortion": _distortion(parameter.depth_distortion),
                "rgb_intrinsic": _camera_intrinsic(parameter.rgb_intrinsic),
                "rgb_distortion": _distortion(parameter.rgb_distortion),
                "depth_to_rgb": _extrinsic(parameter.transform),
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result
