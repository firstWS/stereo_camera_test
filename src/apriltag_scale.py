"""AprilTag pair spacing -> metric scale for stereo Q triangulation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from stereo_types import DepthEstimate
from triangulate import xyz_from_disparity_pixel

# OpenCV predefined AprilTag dictionaries (same family as common printed tags).
APRILTAG_DICTIONARIES: dict[str, int] = {
    "APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class AprilTagScaleOutcome:
    """If ``scale`` is not None, multiply raw ``(X,Y,Z)`` from ``Q`` by ``scale`` for metric body."""

    scale: float | None
    measured_distance_m: float | None
    known_spacing_m: float
    tag_ids: tuple[int, int] | None
    notes: str


def _dict_id(name: str) -> int:
    key = name.strip().upper().replace("-", "_")
    if key not in APRILTAG_DICTIONARIES:
        allowed = ", ".join(sorted(APRILTAG_DICTIONARIES))
        raise ValueError(f"Unknown AprilTag dictionary {name!r}. Use one of: {allowed}")
    return APRILTAG_DICTIONARIES[key]


def _tag_center_px(corners_one: np.ndarray) -> tuple[float, float]:
    c = corners_one.reshape(-1, 2)
    return float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))


def _sample_disp_median(
    disp: np.ndarray, u: float, v: float, radius: int, min_disp: float
) -> float | None:
    h, w = disp.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = max(0, int(radius))
    y1, y2 = max(0, vi - r), min(h, vi + r + 1)
    x1, x2 = max(0, ui - r), min(w, ui + r + 1)
    roi = disp[y1:y2, x1:x2].astype(np.float32)
    valid = roi > float(min_disp)
    if not np.any(valid):
        return None
    return float(np.median(roi[valid]))


def _triangulate_center(
    disp: np.ndarray,
    Q: np.ndarray,
    corners_one: np.ndarray,
    sample_radius: int,
    min_disp: float,
) -> tuple[float, float, float] | None:
    u, v = _tag_center_px(corners_one)
    d = _sample_disp_median(disp, u, v, sample_radius, min_disp)
    if d is None:
        return None
    try:
        return xyz_from_disparity_pixel(u, v, d, Q)
    except ValueError:
        return None


def scale_depth_estimate(est: DepthEstimate, s: float) -> DepthEstimate:
    if not est.valid:
        return est
    return DepthEstimate(
        track=est.track,
        X=est.X * s,
        Y=est.Y * s,
        Z=est.Z * s,
        disparity=est.disparity,
        valid=est.valid,
        valid_pixel_ratio=est.valid_pixel_ratio,
        notes=est.notes,
    )


def compute_apriltag_metric_scale(
    gray_left: np.ndarray,
    disparity: np.ndarray,
    Q: np.ndarray,
    *,
    dictionary: str = "APRILTAG_36H11",
    known_spacing_m: float = 1.0,
    tag_id_a: int | None = None,
    tag_id_b: int | None = None,
    sample_radius: int = 2,
    min_disp: float = 1.0,
    draw_on_bgr: np.ndarray | None = None,
) -> AprilTagScaleOutcome:
    """
    Detect two AprilTags on the **rectified** left image, triangulate tag centers via ``Q``,
    and return ``scale = known_spacing_m / measured_3d_distance`` (same units as ``Q`` triangulation).

    If ``tag_id_a`` / ``tag_id_b`` are None, uses the two lowest tag IDs among detections.
    """
    known_spacing_m = float(known_spacing_m)
    if known_spacing_m <= 0:
        return AprilTagScaleOutcome(
            None, None, known_spacing_m, None, "known_spacing_m must be positive"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(_dict_id(dictionary))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray_left)

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

    p0 = _triangulate_center(disparity, Q, c0, sample_radius, min_disp)
    p1 = _triangulate_center(disparity, Q, c1, sample_radius, min_disp)
    if p0 is None or p1 is None:
        return AprilTagScaleOutcome(
            None,
            None,
            known_spacing_m,
            (tid0, tid1),
            "invalid disparity at one or both tag centers",
        )

    v0 = np.array(p0, dtype=np.float64)
    v1 = np.array(p1, dtype=np.float64)
    dist = float(np.linalg.norm(v1 - v0))
    if dist < 1e-6:
        return AprilTagScaleOutcome(
            None, None, known_spacing_m, (tid0, tid1), "measured 3D distance ~0"
        )

    scale = known_spacing_m / dist

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
        scale=scale,
        measured_distance_m=dist,
        known_spacing_m=known_spacing_m,
        tag_ids=(tid0, tid1),
        notes="ok",
    )
