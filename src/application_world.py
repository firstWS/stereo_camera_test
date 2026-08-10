"""Tag0-centric evaluation world <-> 1st MVP Application World compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from apriltag_world import AprilTagWorldConfig, build_apriltag_world_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPLICATION_WORLD_CONFIG_PATH = REPO_ROOT / "configs/orbbec_gemini.yaml"
DEFAULT_TAG0_WORLD_CONFIG_PATH = REPO_ROOT / "configs/dataset/gemini335l_phase2.yaml"
DEFAULT_APPLICATION_TAG_ID = 0
EXPECTED_TRANSLATION_DELTA_M = np.array([1.0, 2.0, 0.0], dtype=np.float64)
TRANSFORM_CONVENTION = "T_A_B maps coordinates in frame B to frame A (P_A = T_A_B @ P_B)"


@dataclass(frozen=True)
class ApplicationWorldContract:
    """Authoritative Application World contract loaded from config."""

    config_path: str
    tag_id: int
    T_application_tag0: np.ndarray
    tag0_position_application_world_m: tuple[float, float, float]
    front_normal: str
    top_direction: str
    transform_convention: str = TRANSFORM_CONVENTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "tag_id": self.tag_id,
            "T_application_tag0": self.T_application_tag0.tolist(),
            "tag0_position_application_world_m": list(self.tag0_position_application_world_m),
            "front_normal": self.front_normal,
            "top_direction": self.top_direction,
            "transform_convention": self.transform_convention,
            "axis_definitions": application_world_axis_definitions(self.front_normal, self.top_direction),
        }


def _as_transform(matrix: np.ndarray) -> np.ndarray:
    T = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(T).all():
        raise ValueError("transform must be finite")
    return T


def _as_point(point_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(point_xyz, dtype=np.float64).reshape(3)


def load_apriltag_world_section(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("apriltag_world")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain apriltag_world config")
    return section


def load_apriltag_world_config(config_path: str | Path) -> AprilTagWorldConfig:
    return build_apriltag_world_config(load_apriltag_world_section(config_path))


def application_world_axis_definitions(
    front_normal: str = "+Z",
    top_direction: str = "+Y",
) -> dict[str, Any]:
    return {
        "application_world": {
            "+X": "wall right (cross(top, front))",
            "+Y": f"vertical/up ({top_direction})",
            "+Z": f"tag front/out-of-wall ({front_normal})",
        },
        "tag0_world": {
            "note": "Same axis orientation as Application World; Tag0 origin at (0,0,0)",
        },
        "object_anchor_local": {
            "note": "Separate Z-up object frame; not modified by this layer",
        },
    }


def load_application_world_contract(
    *,
    config_path: str | Path | None = None,
    tag_id: int = DEFAULT_APPLICATION_TAG_ID,
) -> ApplicationWorldContract:
    path = Path(config_path or DEFAULT_APPLICATION_WORLD_CONFIG_PATH)
    section = load_apriltag_world_section(path)
    config = build_apriltag_world_config(section)
    if tag_id not in config.tags:
        raise ValueError(f"tag_id {tag_id} not found in {path}")
    T_application_tag0 = config.tags[tag_id].T_world_tag.copy()
    tag_cfg = section.get("tags", {}).get(tag_id) or section.get("tags", {}).get(str(tag_id), {})
    position = np.asarray(tag_cfg.get("position", T_application_tag0[:3, 3]), dtype=np.float64).reshape(3)
    return ApplicationWorldContract(
        config_path=str(path),
        tag_id=tag_id,
        T_application_tag0=T_application_tag0,
        tag0_position_application_world_m=(float(position[0]), float(position[1]), float(position[2])),
        front_normal=str(section.get("front_normal", "+Z")),
        top_direction=str(section.get("top_direction", "+Y")),
    )


def load_T_application_tag0(
    *,
    config_path: str | Path | None = None,
    tag_id: int = DEFAULT_APPLICATION_TAG_ID,
) -> np.ndarray:
    """Return T_application_tag0 from authoritative Application World config."""
    return load_application_world_contract(config_path=config_path, tag_id=tag_id).T_application_tag0.copy()


def transform_point(
    transform: np.ndarray,
    point_xyz: np.ndarray,
) -> np.ndarray:
    T = _as_transform(transform)
    P = _as_point(point_xyz)
    homogeneous = np.array([P[0], P[1], P[2], 1.0], dtype=np.float64)
    return (T @ homogeneous)[:3]


def compose_transforms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _as_transform(left) @ _as_transform(right)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_as_transform(transform))


def tag0_world_point_to_application_world(
    point_tag0_world: np.ndarray,
    *,
    T_application_tag0: np.ndarray | None = None,
    config_path: str | Path | None = None,
    tag_id: int = DEFAULT_APPLICATION_TAG_ID,
) -> np.ndarray:
    T = T_application_tag0 if T_application_tag0 is not None else load_T_application_tag0(
        config_path=config_path,
        tag_id=tag_id,
    )
    return transform_point(T, point_tag0_world)


def application_world_point_to_tag0_world(
    point_application_world: np.ndarray,
    *,
    T_application_tag0: np.ndarray | None = None,
    config_path: str | Path | None = None,
    tag_id: int = DEFAULT_APPLICATION_TAG_ID,
) -> np.ndarray:
    T = T_application_tag0 if T_application_tag0 is not None else load_T_application_tag0(
        config_path=config_path,
        tag_id=tag_id,
    )
    return transform_point(invert_transform(T), point_application_world)


def tag0_world_pose_to_application_world(
    T_tag0_target: np.ndarray,
    *,
    T_application_tag0: np.ndarray | None = None,
    config_path: str | Path | None = None,
    tag_id: int = DEFAULT_APPLICATION_TAG_ID,
) -> np.ndarray:
    T = T_application_tag0 if T_application_tag0 is not None else load_T_application_tag0(
        config_path=config_path,
        tag_id=tag_id,
    )
    return compose_transforms(T, T_tag0_target)


def application_world_pose_to_tag0_world(
    T_application_target: np.ndarray,
    *,
    T_application_tag0: np.ndarray | None = None,
    config_path: str | Path | None = None,
    tag_id: int = DEFAULT_APPLICATION_TAG_ID,
) -> np.ndarray:
    T = T_application_tag0 if T_application_tag0 is not None else load_T_application_tag0(
        config_path=config_path,
        tag_id=tag_id,
    )
    return compose_transforms(invert_transform(T), T_application_target)


def rotation_unchanged(
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
    *,
    atol: float = 1e-9,
) -> bool:
    return bool(np.allclose(left_rotation, right_rotation, atol=atol))


def translation_delta(
    point_application_world: np.ndarray,
    point_tag0_world: np.ndarray,
) -> np.ndarray:
    return _as_point(point_application_world) - _as_point(point_tag0_world)
