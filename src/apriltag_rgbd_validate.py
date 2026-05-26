"""AprilTag pair 3D distance check using aligned depth + RGB intrinsics (validation only)."""

from __future__ import annotations

import cv2
import numpy as np

from apriltag_scale import AprilTagScaleOutcome, _dict_id, _tag_center_px
from rgbd_geometry import backproject_xyz_from_uvz


def _sample_depth_median(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
    z_min: float,
    z_max: float,
) -> float | None:
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = max(0, int(radius))
    y1, y2 = max(0, vi - r), min(h, vi + r + 1)
    x1, x2 = max(0, ui - r), min(w, ui + r + 1)
    roi = depth_m[y1:y2, x1:x2].astype(np.float64)
    valid = np.isfinite(roi) & (roi >= z_min) & (roi <= z_max)
    if not np.any(valid):
        return None
    return float(np.median(roi[valid]))


def _triangulate_tag_center_rgbd(
    depth_m: np.ndarray,
    K: np.ndarray,
    corners_one: np.ndarray,
    sample_radius: int,
    z_min: float,
    z_max: float,
) -> tuple[float, float, float] | None:
    u, v = _tag_center_px(corners_one)
    z = _sample_depth_median(depth_m, u, v, sample_radius, z_min, z_max)
    if z is None:
        return None
    return backproject_xyz_from_uvz(u, v, z, K)


def compute_apriltag_distance_validation_rgbd(
    gray_u8: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    *,
    dictionary: str = "APRILTAG_36H11",
    known_spacing_m: float = 1.0,
    tag_id_a: int | None = None,
    tag_id_b: int | None = None,
    sample_radius: int = 2,
    z_min_m: float = 0.05,
    z_max_m: float = 40.0,
    draw_on_bgr: np.ndarray | None = None,
) -> AprilTagScaleOutcome:
    """
    Two AprilTags on the **RGB-aligned** image: 3D distance from depth + ``K`` vs ``known_spacing_m``.

    Returns ``AprilTagScaleOutcome`` with ``scale=None`` (no automatic scaling of detections).
    """
    known_spacing_m = float(known_spacing_m)
    if known_spacing_m <= 0:
        return AprilTagScaleOutcome(
            None, None, known_spacing_m, None, "known_spacing_m must be positive"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(_dict_id(dictionary))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray_u8)

    if ids is None or len(ids) < 2:
        return AprilTagScaleOutcome(
            None,
            None,
            known_spacing_m,
            None,
            f"need>=2 tags, got {0 if ids is None else len(ids)}",
        )

    flat = [(int(ids[i][0]), corners[i]) for i in range(len(ids))]
    flat.sort(key=lambda t: t[0])

    pick: list[tuple[int, np.ndarray]] = []
    if tag_id_a is not None and tag_id_b is not None:
        dmap = {tid: c for tid, c in flat}
        if tag_id_a not in dmap or tag_id_b not in dmap:
            return AprilTagScaleOutcome(
                None,
                None,
                known_spacing_m,
                None,
                f"tag_id pair {tag_id_a},{tag_id_b} not both visible",
            )
        pick = [(tag_id_a, dmap[tag_id_a]), (tag_id_b, dmap[tag_id_b])]
    else:
        pick = [flat[0], flat[1]]

    tid0, c0 = pick[0]
    tid1, c1 = pick[1]

    p0 = _triangulate_tag_center_rgbd(depth_m, K, c0, sample_radius, z_min_m, z_max_m)
    p1 = _triangulate_tag_center_rgbd(depth_m, K, c1, sample_radius, z_min_m, z_max_m)
    if p0 is None or p1 is None:
        return AprilTagScaleOutcome(
            None,
            None,
            known_spacing_m,
            (tid0, tid1),
            "invalid depth at one or both tag centers",
        )

    v0 = np.array(p0, dtype=np.float64)
    v1 = np.array(p1, dtype=np.float64)
    dist = float(np.linalg.norm(v1 - v0))
    if dist < 1e-6:
        return AprilTagScaleOutcome(
            None, None, known_spacing_m, (tid0, tid1), "measured 3D distance ~0"
        )

    rel = abs(dist - known_spacing_m) / known_spacing_m if known_spacing_m > 0 else 0.0
    notes = f"rgbd_verify rel_err={rel:.5f}"

    if draw_on_bgr is not None:
        for corn in (c0, c1):
            pts = corn[0].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(draw_on_bgr, [pts], True, (255, 128, 0), 2)
        u0, v0p = _tag_center_px(c0)
        u1, v1p = _tag_center_px(c1)
        cv2.line(
            draw_on_bgr,
            (int(round(u0)), int(round(v0p))),
            (int(round(u1)), int(round(v1p))),
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(draw_on_bgr, (int(round(u0)), int(round(v0p))), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(draw_on_bgr, (int(round(u1)), int(round(v1p))), 4, (0, 255, 255), -1, cv2.LINE_AA)

    return AprilTagScaleOutcome(
        scale=None,
        measured_distance_m=dist,
        known_spacing_m=known_spacing_m,
        tag_ids=(tid0, tid1),
        notes=notes,
    )
