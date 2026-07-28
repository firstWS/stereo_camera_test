"""OpenCV overlays for Object Anchor keypoints and estimated pose."""

from __future__ import annotations

import cv2
import numpy as np

from object_anchor_config import ObjectAnchorConfig
from object_anchor_pose import ObjectPoseEstimate


def draw_object_anchor_keypoints(
    image_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    config: ObjectAnchorConfig,
    *,
    confidences: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    show_confidence_visibility: bool = False,
    show_names: bool = False,
    show_labels: bool = True,
    skeleton_crossed: bool = False,
) -> np.ndarray:
    canvas = image_bgr.copy()
    points = np.asarray(keypoints_xy, dtype=np.float64)
    count = len(points)
    confidence_values = (
        np.ones(count, dtype=np.float64)
        if confidences is None
        else np.asarray(confidences, dtype=np.float64).reshape(-1)
    )
    visibility_values = (
        np.full(count, 2, dtype=np.int32)
        if visibility is None
        else np.asarray(visibility, dtype=np.int32).reshape(-1)
    )
    finite = np.all(np.isfinite(points), axis=1)

    for start, end in config.skeleton:
        if start >= count or end >= count:
            continue
        if not finite[start] or not finite[end]:
            continue
        if visibility_values[start] <= 0 or visibility_values[end] <= 0:
            continue
        p1 = tuple(np.rint(points[start]).astype(int))
        p2 = tuple(np.rint(points[end]).astype(int))
        line_color = (20, 20, 230) if skeleton_crossed else (255, 180, 40)
        cv2.line(canvas, p1, p2, line_color, 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        if not finite[index]:
            continue
        center = tuple(np.rint(point).astype(int))
        visible = int(visibility_values[index])
        confident = confidence_values[index] >= config.pose_settings.confidence_threshold
        color = (60, 220, 60) if visible == 2 and confident else (0, 190, 255)
        if visible <= 0 or not confident:
            color = (120, 120, 120)
        cv2.circle(canvas, center, 6, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 8, (20, 20, 20), 1, cv2.LINE_AA)
        if show_labels:
            name = config.keypoints[index].name if show_names else ""
            label = f"{index}:{name}" if name else str(index)
            if show_confidence_visibility:
                label = f"{label} c={confidence_values[index]:.2f} v={visible}"
            cv2.putText(
                canvas,
                label,
                (center[0] + 9, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def draw_keypoint_legend(
    image_bgr: np.ndarray,
    config: ObjectAnchorConfig,
    *,
    origin: tuple[int, int] = (25, 300),
) -> np.ndarray:
    """Draw the fixed ID/name/XYZ mapping away from projected keypoints."""
    canvas = image_bgr.copy()
    x, y = origin
    cv2.putText(
        canvas,
        "KEYPOINT ORDER (object XYZ, meter)",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for row, keypoint in enumerate(config.keypoints, start=1):
        xyz = ", ".join(f"{value:+.4f}" for value in keypoint.xyz)
        text = f"{keypoint.keypoint_id}: {keypoint.name}  ({xyz})"
        cv2.putText(
            canvas,
            text,
            (x, y + row * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_object_pose_axes(
    image_bgr: np.ndarray,
    pose: ObjectPoseEstimate,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    axis_length_m: float = 0.08,
) -> np.ndarray:
    canvas = image_bgr.copy()
    if pose.rvec is None or pose.tvec is None:
        return canvas
    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    )
    cv2.drawFrameAxes(
        canvas,
        np.asarray(camera_matrix, dtype=np.float64),
        distortion,
        pose.rvec,
        pose.tvec,
        float(axis_length_m),
        3,
    )
    return canvas
