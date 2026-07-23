"""Object Anchor configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ObjectAnchorKeypoint:
    keypoint_id: int
    name: str
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ObjectAnchorPoseSettings:
    confidence_threshold: float = 0.5
    min_visibility: int = 1
    min_correspondences: int = 4
    min_inliers: int = 4
    ransac_reprojection_error_px: float = 8.0
    max_mean_reprojection_error_px: float = 5.0
    ransac_confidence: float = 0.99
    ransac_iterations: int = 100
    refine_lm: bool = True


@dataclass(frozen=True)
class ObjectAnchorConfig:
    object_id: str
    object_type: str
    anchor_mode: str
    unit: str
    size: dict[str, float]
    coordinate_system: dict[str, str]
    keypoints: tuple[ObjectAnchorKeypoint, ...]
    skeleton: tuple[tuple[int, int], ...]
    pose_settings: ObjectAnchorPoseSettings
    world_pose: dict[str, Any] | None = None

    @property
    def object_points(self) -> np.ndarray:
        return np.asarray([point.xyz for point in self.keypoints], dtype=np.float64)

    @property
    def keypoint_names(self) -> tuple[str, ...]:
        return tuple(point.name for point in self.keypoints)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def load_object_anchor_config(path: str | Path) -> ObjectAnchorConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}

    cfg = _require_mapping(root.get("object_anchor"), "object_anchor")
    object_id = str(cfg.get("id") or "").strip()
    if not object_id:
        raise ValueError("object_anchor.id is required")
    unit = str(cfg.get("unit", "meter")).strip().lower()
    if unit not in {"meter", "metre", "m"}:
        raise ValueError("object_anchor.unit must be meter for this pipeline")
    anchor_mode = str(cfg.get("mode", "cuboid_8point")).strip().lower()
    if anchor_mode not in {"cuboid_8point", "front_only"}:
        raise ValueError("object_anchor.mode must be cuboid_8point or front_only")

    size_raw = _require_mapping(cfg.get("size"), "object_anchor.size")
    size = {axis: float(size_raw[axis]) for axis in ("width", "depth", "height")}
    if any(value <= 0.0 for value in size.values()):
        raise ValueError("object_anchor size values must be positive")

    coord_raw = _require_mapping(cfg.get("coordinate_system"), "object_anchor.coordinate_system")
    coordinate_system = {str(key): str(value) for key, value in coord_raw.items()}

    points_raw = cfg.get("keypoints_3d")
    if not isinstance(points_raw, list) or len(points_raw) < 4:
        raise ValueError("object_anchor.keypoints_3d must contain at least four points")
    points: list[ObjectAnchorKeypoint] = []
    for expected_id, raw_point in enumerate(points_raw):
        point = _require_mapping(raw_point, f"object_anchor.keypoints_3d[{expected_id}]")
        point_id = int(point.get("id", -1))
        if point_id != expected_id:
            raise ValueError(
                "object_anchor keypoint IDs must be contiguous and ordered; "
                f"expected {expected_id}, got {point_id}"
            )
        xyz = np.asarray(point.get("xyz"), dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f"keypoint {point_id} xyz must contain three finite values")
        points.append(
            ObjectAnchorKeypoint(
                keypoint_id=point_id,
                name=str(point.get("name") or f"keypoint_{point_id}"),
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )
        )
    if anchor_mode == "front_only":
        expected_names = (
            "front_top_left",
            "front_top_right",
            "front_bottom_right",
            "front_bottom_left",
        )
        if tuple(point.name for point in points) != expected_names:
            raise ValueError(
                "front_only keypoints must be ordered: " + ", ".join(expected_names)
            )
        if len(points) != 4:
            raise ValueError("front_only mode requires exactly four keypoints")

    skeleton_raw = cfg.get("skeleton") or []
    skeleton: list[tuple[int, int]] = []
    for edge_index, raw_edge in enumerate(skeleton_raw):
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ValueError(f"object_anchor.skeleton[{edge_index}] must be [start, end]")
        edge = (int(raw_edge[0]), int(raw_edge[1]))
        if min(edge) < 0 or max(edge) >= len(points):
            raise ValueError(f"object_anchor.skeleton[{edge_index}] references an unknown keypoint")
        skeleton.append(edge)

    pose_raw = _require_mapping(cfg.get("pose_estimation") or {}, "object_anchor.pose_estimation")
    pose_settings = ObjectAnchorPoseSettings(
        confidence_threshold=float(pose_raw.get("confidence_threshold", 0.5)),
        min_visibility=int(pose_raw.get("min_visibility", 1)),
        min_correspondences=int(pose_raw.get("min_correspondences", 4)),
        min_inliers=int(pose_raw.get("min_inliers", 4)),
        ransac_reprojection_error_px=float(pose_raw.get("ransac_reprojection_error_px", 8.0)),
        max_mean_reprojection_error_px=float(
            pose_raw.get("max_mean_reprojection_error_px", 5.0)
        ),
        ransac_confidence=float(pose_raw.get("ransac_confidence", 0.99)),
        ransac_iterations=int(pose_raw.get("ransac_iterations", 100)),
        refine_lm=bool(pose_raw.get("refine_lm", True)),
    )
    if pose_settings.min_correspondences < 4 or pose_settings.min_inliers < 4:
        raise ValueError("PnP requires at least four correspondences and four inliers")

    world_pose = cfg.get("world_pose")
    if world_pose is not None:
        world_pose = _require_mapping(world_pose, "object_anchor.world_pose")

    return ObjectAnchorConfig(
        object_id=object_id,
        object_type=str(cfg.get("type", "cuboid")),
        anchor_mode=anchor_mode,
        unit="meter",
        size=size,
        coordinate_system=coordinate_system,
        keypoints=tuple(points),
        skeleton=tuple(skeleton),
        pose_settings=pose_settings,
        world_pose=world_pose,
    )
