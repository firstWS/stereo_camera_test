"""Persistent sparse map for Phase 4.7-A stereo+IMU SLAM-lite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .stereo_imu_calibration import StereoImuCalibration
from .stereo_imu_vio_lite import (
    StereoImuVioConfig,
    detect_landmarks,
    match_stereo_points,
    triangulate_stereo_points,
)

POSE_SOURCE_VIO_TRACKING = "VIO_TRACKING"
POSE_SOURCE_MAP_TRACKING = "MAP_TRACKING"
POSE_SOURCE_VIO_PROPAGATED = "VIO_PROPAGATED"
POSE_SOURCE_MAP_RELOCALIZED = "MAP_RELOCALIZED"
POSE_SOURCE_TRACKING_LOST = "TRACKING_LOST"
POSE_SOURCE_INIT = "INIT"


@dataclass
class SlamMapPoint:
    map_point_id: int
    position_xyz: np.ndarray
    descriptor: np.ndarray
    first_seen_frame: int
    last_seen_frame: int
    observation_count: int = 1


@dataclass
class SlamKeyframe:
    frame_number: int
    device_timestamp_us: int
    T_slam_camera: np.ndarray
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    map_point_ids: list[int]


@dataclass(frozen=True)
class MapLocalizationResult:
    success: bool
    T_slam_camera: np.ndarray | None
    match_count: int
    inlier_count: int
    failure_reason: str | None = None


@dataclass
class SlamMapConfig:
    orb_features: int = 400
    min_map_matches: int = 8
    min_map_inliers: int = 6
    reprojection_error_px: float = 4.0
    keyframe_interval_frames: int = 8
    keyframe_translation_m: float = 0.04
    keyframe_rotation_deg: float = 4.0
    max_map_points: int = 2000
    stale_frames: int = 120
    descriptor_match_ratio: float = 0.75


@dataclass
class SlamMap:
    config: SlamMapConfig
    keyframes: list[SlamKeyframe] = field(default_factory=list)
    map_points: dict[int, SlamMapPoint] = field(default_factory=dict)
    _next_point_id: int = 1

    @property
    def map_point_count(self) -> int:
        return len(self.map_points)

    @property
    def keyframe_count(self) -> int:
        return len(self.keyframes)

    def _extract_orb(self, left_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orb = cv2.ORB_create(nfeatures=self.config.orb_features)
        keypoints, descriptors = orb.detectAndCompute(left_gray, None)
        if keypoints is None or descriptors is None or len(keypoints) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts, descriptors

    def _camera_to_slam(self, T_slam_camera: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
        R = T_slam_camera[:3, :3]
        t = T_slam_camera[:3, 3]
        return (R @ points_camera.T).T + t.reshape(1, 3)

    def _slam_to_camera(self, T_slam_camera: np.ndarray, points_slam: np.ndarray) -> np.ndarray:
        R = T_slam_camera[:3, :3]
        t = T_slam_camera[:3, 3]
        return (R.T @ (points_slam - t.reshape(1, 3)).T).T

    def add_keyframe(
        self,
        *,
        frame_number: int,
        device_timestamp_us: int,
        T_slam_camera: np.ndarray,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        calib: StereoImuCalibration,
        vio_config: StereoImuVioConfig,
    ) -> int:
        keypoints_xy, descriptors = self._extract_orb(left_gray)
        map_point_ids: list[int] = []

        landmark_state = detect_landmarks(left_gray, right_gray, calib, vio_config)
        if landmark_state is not None and landmark_state.points_3d.shape[0] > 0:
            world_pts = self._camera_to_slam(T_slam_camera, landmark_state.points_3d)
            orb = cv2.ORB_create(nfeatures=self.config.orb_features)
            kp_landmarks = [
                cv2.KeyPoint(float(x), float(y), 7.0)
                for x, y in landmark_state.left_pts.reshape(-1, 2)
            ]
            _, landmark_desc = orb.compute(left_gray, kp_landmarks)
            if landmark_desc is not None:
                for idx in range(min(world_pts.shape[0], landmark_desc.shape[0])):
                    point_id = self._next_point_id
                    self._next_point_id += 1
                    self.map_points[point_id] = SlamMapPoint(
                        map_point_id=point_id,
                        position_xyz=world_pts[idx].astype(np.float64),
                        descriptor=landmark_desc[idx].copy(),
                        first_seen_frame=frame_number,
                        last_seen_frame=frame_number,
                    )
                    map_point_ids.append(point_id)

        if keypoints_xy.shape[0] > 0 and descriptors is not None:
            left_matched, right_matched = match_stereo_points(
                left_gray,
                right_gray,
                keypoints_xy,
                search_px=vio_config.disparity_search_px,
            )
            if len(left_matched) > 0:
                pts3 = triangulate_stereo_points(left_matched, right_matched, calib.p1, calib.p2)
                valid = np.isfinite(pts3).all(axis=1)
                if np.any(valid):
                    world_pts = self._camera_to_slam(T_slam_camera, pts3[valid])
                    valid_indices = np.flatnonzero(valid)
                    for local_idx, world_pt in zip(valid_indices, world_pts):
                        if descriptors.shape[0] <= local_idx:
                            continue
                        point_id = self._next_point_id
                        self._next_point_id += 1
                        self.map_points[point_id] = SlamMapPoint(
                            map_point_id=point_id,
                            position_xyz=world_pt.astype(np.float64),
                            descriptor=descriptors[local_idx].copy(),
                            first_seen_frame=frame_number,
                            last_seen_frame=frame_number,
                        )
                        map_point_ids.append(point_id)

        self.keyframes.append(
            SlamKeyframe(
                frame_number=frame_number,
                device_timestamp_us=device_timestamp_us,
                T_slam_camera=T_slam_camera.copy(),
                keypoints_xy=keypoints_xy,
                descriptors=descriptors if descriptors is not None else np.zeros((0, 32), dtype=np.uint8),
                map_point_ids=map_point_ids,
            )
        )
        self._prune_map_points(frame_number)
        return len(map_point_ids)

    def should_insert_keyframe(
        self,
        *,
        frame_number: int,
        T_slam_camera: np.ndarray,
    ) -> bool:
        if not self.keyframes:
            return True
        last = self.keyframes[-1]
        if frame_number - last.frame_number >= self.config.keyframe_interval_frames:
            return True
        delta = np.linalg.inv(last.T_slam_camera) @ T_slam_camera
        trans = float(np.linalg.norm(delta[:3, 3]))
        trace = float(np.trace(delta[:3, :3]))
        rot_deg = float(np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))))
        return trans >= self.config.keyframe_translation_m or rot_deg >= self.config.keyframe_rotation_deg

    def localize_with_map(
        self,
        *,
        left_gray: np.ndarray,
        calib: StereoImuCalibration,
        min_matches: int | None = None,
        min_inliers: int | None = None,
    ) -> MapLocalizationResult:
        if not self.map_points:
            return MapLocalizationResult(False, None, 0, 0, "empty_map")

        _, descriptors = self._extract_orb(left_gray)
        if descriptors is None or descriptors.shape[0] == 0:
            return MapLocalizationResult(False, None, 0, 0, "no_current_descriptors")

        map_ids = list(self.map_points.keys())
        map_desc = np.vstack([self.map_points[mid].descriptor for mid in map_ids]).astype(np.uint8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(descriptors, map_desc, k=2)
        good_matches: list[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.descriptor_match_ratio * n.distance:
                good_matches.append(m)

        min_match_req = min_matches if min_matches is not None else self.config.min_map_matches
        min_inlier_req = min_inliers if min_inliers is not None else self.config.min_map_inliers
        if len(good_matches) < min_match_req:
            return MapLocalizationResult(False, None, len(good_matches), 0, "insufficient_map_matches")

        keypoints_xy, _ = self._extract_orb(left_gray)
        object_pts = []
        image_pts = []
        for match in good_matches:
            if match.queryIdx >= keypoints_xy.shape[0]:
                continue
            map_id = map_ids[match.trainIdx]
            object_pts.append(self.map_points[map_id].position_xyz)
            image_pts.append(keypoints_xy[match.queryIdx])

        if len(object_pts) < min_match_req:
            return MapLocalizationResult(False, None, len(object_pts), 0, "insufficient_correspondences")

        object_pts_arr = np.asarray(object_pts, dtype=np.float64)
        image_pts_arr = np.asarray(image_pts, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_pts_arr,
            image_pts_arr,
            calib.k_left,
            calib.d_left,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.config.reprojection_error_px,
            confidence=0.99,
        )
        inlier_count = int(len(inliers)) if inliers is not None else 0
        if not ok or inlier_count < min_inlier_req:
            return MapLocalizationResult(False, None, len(object_pts), inlier_count, "solvepnp_failed")

        rmat, _ = cv2.Rodrigues(rvec)
        T_cam_slam = np.eye(4, dtype=np.float64)
        T_cam_slam[:3, :3] = rmat
        T_cam_slam[:3, 3] = tvec.reshape(3)
        T_slam_camera = np.linalg.inv(T_cam_slam)
        return MapLocalizationResult(True, T_slam_camera, len(object_pts), inlier_count, None)

    def relocalize(
        self,
        *,
        left_gray: np.ndarray,
        calib: StereoImuCalibration,
    ) -> MapLocalizationResult:
        return self.localize_with_map(
            left_gray=left_gray,
            calib=calib,
            min_matches=max(6, self.config.min_map_matches - 2),
            min_inliers=max(5, self.config.min_map_inliers - 1),
        )

    def update_observations(self, frame_number: int, match_count: int) -> None:
        if match_count <= 0:
            return
        for point in self.map_points.values():
            if frame_number - point.last_seen_frame <= 3:
                point.last_seen_frame = frame_number
                point.observation_count += 1

    def _prune_map_points(self, frame_number: int) -> None:
        if len(self.map_points) <= self.config.max_map_points:
            stale_ids = [
                pid
                for pid, point in self.map_points.items()
                if frame_number - point.last_seen_frame > self.config.stale_frames
            ]
            for pid in stale_ids:
                del self.map_points[pid]
            return
        ranked = sorted(
            self.map_points.items(),
            key=lambda item: (item[1].observation_count, item[1].last_seen_frame),
        )
        while len(self.map_points) > self.config.max_map_points:
            pid, _ = ranked.pop(0)
            self.map_points.pop(pid, None)
