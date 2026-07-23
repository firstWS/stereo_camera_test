"""Validation and sanitation primitives for Ultralytics YOLO Pose labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PoseLabelValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SanitizedPoseLabel:
    class_id: int
    bbox_cxcywh: tuple[float, float, float, float]
    keypoints_xyv: np.ndarray
    hidden_keypoints_rewritten: int

    @property
    def visible_ids(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in np.flatnonzero(self.keypoints_xyv[:, 2] > 0.0)
        )

    def to_yolo_line(self) -> str:
        values = [str(self.class_id)]
        values.extend(f"{value:.6f}" for value in self.bbox_cxcywh)
        for x, y, visibility in self.keypoints_xyv:
            values.extend((f"{x:.6f}", f"{y:.6f}", str(int(visibility))))
        return " ".join(values)


def sanitize_yolo_pose_line(
    line: str,
    *,
    expected_keypoints: int = 8,
    source: str = "label",
) -> SanitizedPoseLabel:
    tokens = line.strip().split()
    expected_values = 1 + 4 + expected_keypoints * 3
    if len(tokens) != expected_values:
        raise PoseLabelValidationError(
            f"{source}: expected {expected_values} values, got {len(tokens)}"
        )
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise PoseLabelValidationError(f"{source}: contains a non-numeric value") from exc
    if not np.all(np.isfinite(values)):
        raise PoseLabelValidationError(f"{source}: contains NaN or infinity")

    class_value = float(values[0])
    class_id = int(round(class_value))
    if class_id < 0 or abs(class_value - class_id) > 1e-9:
        raise PoseLabelValidationError(f"{source}: class ID must be a non-negative integer")

    bbox = values[1:5]
    bbox_names = ("cx", "cy", "width", "height")
    for name, value in zip(bbox_names, bbox):
        if not 0.0 <= float(value) <= 1.0:
            raise PoseLabelValidationError(
                f"{source}: bbox {name}={value:.6f} is outside [0,1]"
            )
    if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
        raise PoseLabelValidationError(f"{source}: bbox width and height must be positive")

    keypoints = values[5:].reshape(expected_keypoints, 3).copy()
    rewritten = 0
    for index, point in enumerate(keypoints):
        x, y, visibility_value = map(float, point)
        visibility = int(round(visibility_value))
        if visibility not in (0, 1, 2) or abs(visibility_value - visibility) > 1e-9:
            raise PoseLabelValidationError(
                f"{source}: keypoint {index} visibility must be 0, 1, or 2"
            )
        if visibility == 0:
            if x != 0.0 or y != 0.0:
                rewritten += 1
            keypoints[index] = [0.0, 0.0, 0.0]
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise PoseLabelValidationError(
                f"{source}: visible keypoint {index} ({x:.6f},{y:.6f},v={visibility}) "
                "is outside [0,1]"
            )
        keypoints[index, 2] = float(visibility)

    return SanitizedPoseLabel(
        class_id=class_id,
        bbox_cxcywh=tuple(float(value) for value in bbox),
        keypoints_xyv=keypoints,
        hidden_keypoints_rewritten=rewritten,
    )


def cuboid_face_keypoint_ids(object_points: np.ndarray) -> dict[str, tuple[int, ...]]:
    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object_points must have shape (N,3)")
    axes = {
        "left": (0, float(np.min(points[:, 0]))),
        "right": (0, float(np.max(points[:, 0]))),
        "front": (1, float(np.min(points[:, 1]))),
        "back": (1, float(np.max(points[:, 1]))),
        "bottom": (2, float(np.min(points[:, 2]))),
        "top": (2, float(np.max(points[:, 2]))),
    }
    return {
        name: tuple(int(index) for index in np.flatnonzero(np.isclose(points[:, axis], value)))
        for name, (axis, value) in axes.items()
    }


def infer_named_view(stem: str) -> str | None:
    normalized = stem.lower().replace("-", "_")
    for view in ("front", "back", "left", "right", "top", "bottom"):
        if normalized == view or normalized.endswith(f"_{view}") or f"_{view}_" in normalized:
            return view
    return None


def find_skeleton_crossings(
    keypoints_xyv: np.ndarray,
    skeleton: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return non-adjacent visible skeleton edges that properly cross in 2D."""
    points = np.asarray(keypoints_xyv, dtype=np.float64)

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def properly_cross(edge_a: tuple[int, int], edge_b: tuple[int, int]) -> bool:
        if set(edge_a) & set(edge_b):
            return False
        a, b = points[edge_a[0], :2], points[edge_a[1], :2]
        c, d = points[edge_b[0], :2], points[edge_b[1], :2]
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        epsilon = 1e-12
        return o1 * o2 < -epsilon and o3 * o4 < -epsilon

    visible_edges = [
        edge
        for edge in skeleton
        if points[edge[0], 2] > 0.0 and points[edge[1], 2] > 0.0
    ]
    crossings: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for first_index, first in enumerate(visible_edges):
        for second in visible_edges[first_index + 1 :]:
            if properly_cross(first, second):
                crossings.append((first, second))
    return crossings
