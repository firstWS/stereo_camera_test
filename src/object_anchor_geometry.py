"""Small 2D geometry checks shared by Object Anchor runtime and validation."""

from __future__ import annotations

import numpy as np


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross_strict(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 * o2 < -epsilon) and (o3 * o4 < -epsilon)


def find_skeleton_crossings(
    keypoints_xy: np.ndarray,
    skeleton: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return pairs of non-adjacent skeleton edge indices that intersect."""
    points = np.asarray(keypoints_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints_xy must have shape (N, 2)")
    valid = np.all(np.isfinite(points), axis=1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(mask) != len(points):
            raise ValueError("valid_mask must match keypoint count")
        valid &= mask

    crossings: list[tuple[int, int]] = []
    edges = list(skeleton)
    for first_index, (a_index, b_index) in enumerate(edges):
        if not (valid[a_index] and valid[b_index]):
            continue
        for second_index in range(first_index + 1, len(edges)):
            c_index, d_index = edges[second_index]
            if len({a_index, b_index, c_index, d_index}) < 4:
                continue
            if not (valid[c_index] and valid[d_index]):
                continue
            if _segments_cross_strict(
                points[a_index],
                points[b_index],
                points[c_index],
                points[d_index],
            ):
                crossings.append((first_index, second_index))
    return tuple(crossings)
