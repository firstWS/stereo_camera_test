"""2D/3D correspondence filtering and solvePnPRansac pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from object_anchor_config import ObjectAnchorPoseSettings


@dataclass(frozen=True)
class FilteredCorrespondences:
    object_points: np.ndarray
    image_points: np.ndarray
    original_indices: np.ndarray


@dataclass
class ObjectPoseEstimate:
    valid: bool
    reason: str
    T_camera_object: np.ndarray | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    rotation_matrix: np.ndarray | None = None
    rpy_deg: tuple[float, float, float] | None = None
    correspondence_indices: tuple[int, ...] = ()
    inlier_indices: tuple[int, ...] = ()
    mean_reprojection_error_px: float | None = None
    max_reprojection_error_px: float | None = None

    @property
    def inlier_count(self) -> int:
        return len(self.inlier_indices)


def filter_keypoint_correspondences(
    keypoints_2d: np.ndarray,
    object_points_3d: np.ndarray,
    *,
    confidences: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    confidence_threshold: float = 0.5,
    min_visibility: int = 1,
) -> FilteredCorrespondences:
    image_points = np.asarray(keypoints_2d, dtype=np.float64)
    object_points = np.asarray(object_points_3d, dtype=np.float64)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("keypoints_2d must have shape (N, 2)")
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points_3d must have shape (N, 3)")
    if len(image_points) != len(object_points):
        raise ValueError("2D and 3D keypoint counts must match")

    confidence_values = (
        np.ones(len(image_points), dtype=np.float64)
        if confidences is None
        else np.asarray(confidences, dtype=np.float64).reshape(-1)
    )
    visibility_values = (
        np.full(len(image_points), 2, dtype=np.int32)
        if visibility is None
        else np.asarray(visibility, dtype=np.int32).reshape(-1)
    )
    if len(confidence_values) != len(image_points) or len(visibility_values) != len(image_points):
        raise ValueError("confidence and visibility counts must match keypoint count")

    valid = np.all(np.isfinite(image_points), axis=1)
    valid &= np.all(np.isfinite(object_points), axis=1)
    valid &= image_points[:, 0] >= 0.0
    valid &= image_points[:, 1] >= 0.0
    valid &= np.isfinite(confidence_values)
    valid &= confidence_values >= float(confidence_threshold)
    valid &= visibility_values >= int(min_visibility)
    indices = np.flatnonzero(valid)
    return FilteredCorrespondences(
        object_points=object_points[indices],
        image_points=image_points[indices],
        original_indices=indices,
    )


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    values = np.rad2deg([roll, pitch, yaw])
    return float(values[0]), float(values[1]), float(values[2])


def rpy_deg_to_rotation_matrix(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Build R = Rz(yaw) @ Ry(pitch) @ Rx(roll) from degrees."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cx, sx = float(np.cos(roll)), float(np.sin(roll))
    cy, sy = float(np.cos(pitch)), float(np.sin(pitch))
    cz, sz = float(np.cos(yaw)), float(np.sin(yaw))
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def estimate_object_pose(
    keypoints_2d: np.ndarray,
    object_points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    confidences: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    settings: ObjectAnchorPoseSettings | None = None,
) -> ObjectPoseEstimate:
    settings = settings or ObjectAnchorPoseSettings()
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    )
    correspondences = filter_keypoint_correspondences(
        keypoints_2d,
        object_points_3d,
        confidences=confidences,
        visibility=visibility,
        confidence_threshold=settings.confidence_threshold,
        min_visibility=settings.min_visibility,
    )
    original_indices = tuple(int(index) for index in correspondences.original_indices)
    if len(correspondences.object_points) < settings.min_correspondences:
        return ObjectPoseEstimate(
            valid=False,
            reason=(
                f"insufficient_correspondences:{len(correspondences.object_points)}"
                f"<{settings.min_correspondences}"
            ),
            correspondence_indices=original_indices,
        )

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            correspondences.object_points,
            correspondences.image_points,
            K,
            distortion,
            iterationsCount=settings.ransac_iterations,
            reprojectionError=settings.ransac_reprojection_error_px,
            confidence=settings.ransac_confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        return ObjectPoseEstimate(
            valid=False,
            reason=f"solvepnp_error:{exc.code}",
            correspondence_indices=original_indices,
        )
    if not success or rvec is None or tvec is None:
        return ObjectPoseEstimate(
            valid=False,
            reason="solvepnp_failed",
            correspondence_indices=original_indices,
        )

    local_inliers = (
        np.empty(0, dtype=np.int32)
        if inliers is None
        else np.asarray(inliers, dtype=np.int32).reshape(-1)
    )
    inlier_indices = tuple(
        int(correspondences.original_indices[index]) for index in local_inliers
    )
    if len(local_inliers) < settings.min_inliers:
        return ObjectPoseEstimate(
            valid=False,
            reason=f"insufficient_inliers:{len(local_inliers)}<{settings.min_inliers}",
            rvec=rvec,
            tvec=tvec,
            correspondence_indices=original_indices,
            inlier_indices=inlier_indices,
        )

    if settings.refine_lm and hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                correspondences.object_points[local_inliers],
                correspondences.image_points[local_inliers],
                K,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(
        correspondences.object_points[local_inliers], rvec, tvec, K, distortion
    )
    residuals = np.linalg.norm(
        projected.reshape(-1, 2) - correspondences.image_points[local_inliers], axis=1
    )
    mean_error = float(np.mean(residuals))
    max_error = float(np.max(residuals))
    rotation, _ = cv2.Rodrigues(rvec)
    T_camera_object = np.eye(4, dtype=np.float64)
    T_camera_object[:3, :3] = rotation
    T_camera_object[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

    reason = "ok"
    valid = True
    if not np.all(np.isfinite(T_camera_object)):
        valid, reason = False, "non_finite_pose"
    elif float(tvec[2, 0]) <= 0.0:
        valid, reason = False, "object_behind_camera"
    elif mean_error > settings.max_mean_reprojection_error_px:
        valid = False
        reason = (
            f"reprojection_error:{mean_error:.3f}>"
            f"{settings.max_mean_reprojection_error_px:.3f}"
        )

    return ObjectPoseEstimate(
        valid=valid,
        reason=reason,
        T_camera_object=T_camera_object,
        rvec=rvec,
        tvec=tvec,
        rotation_matrix=rotation,
        rpy_deg=rotation_matrix_to_rpy_deg(rotation),
        correspondence_indices=original_indices,
        inlier_indices=inlier_indices,
        mean_reprojection_error_px=mean_error,
        max_reprojection_error_px=max_error,
    )
