"""Phase 4.5-A lightweight stereo visual odometry + IMU propagation baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .rgbd_odometry import accumulate_odom_pose, transform_is_finite
from .stereo_imu_calibration import StereoImuCalibration

STEREO_IMU_VIO_LITE_ALGORITHM_ID = "stereo_imu_vio_lite"

DEFAULT_MAX_FEATURES = 300
DEFAULT_MIN_VISUAL_INLIERS = 12
DEFAULT_MIN_STEREO_POINTS = 20
DEFAULT_MAX_PROPAGATED_ONLY_FRAMES = 5
DEFAULT_DISPARITY_SEARCH = 128
DEFAULT_REPROJ_ERROR_PX = 4.0
DEFAULT_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class ImuSampleRecord:
    device_timestamp_us: int
    accel_m_s2: np.ndarray
    gyro_rad_s: np.ndarray


@dataclass(frozen=True)
class StereoImuVioConfig:
    max_features: int = DEFAULT_MAX_FEATURES
    min_visual_inliers: int = DEFAULT_MIN_VISUAL_INLIERS
    min_stereo_points: int = DEFAULT_MIN_STEREO_POINTS
    max_propagated_only_frames: int = DEFAULT_MAX_PROPAGATED_ONLY_FRAMES
    disparity_search_px: int = DEFAULT_DISPARITY_SEARCH
    reprojection_error_px: float = DEFAULT_REPROJ_ERROR_PX
    gravity_m_s2: float = DEFAULT_GRAVITY_M_S2
    lk_win_size: int = 21
    lk_max_level: int = 3


@dataclass
class LandmarkState:
    points_3d: np.ndarray
    left_pts: np.ndarray


@dataclass(frozen=True)
class VisualUpdateResult:
    success: bool
    transform_target_source: np.ndarray | None
    visual_inliers: int
    stereo_points: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class ImuPropagationResult:
    transform_target_source: np.ndarray
    samples_used: int
    finite: bool


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = axis / norm
    x, y, z = k
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def integrate_gyro_rotation(gyro_rad_s: np.ndarray, dt_sec: float) -> np.ndarray:
    omega = np.asarray(gyro_rad_s, dtype=np.float64)
    angle = float(np.linalg.norm(omega) * dt_sec)
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = omega / (np.linalg.norm(omega) + 1e-12)
    return rotation_matrix_from_axis_angle(axis, angle)


def propagate_imu_between_timestamps(
    samples: Sequence[ImuSampleRecord],
    t_start_us: int,
    t_end_us: int,
    calib: StereoImuCalibration,
    *,
    velocity_m_s: np.ndarray | None = None,
    gravity_m_s2: float = DEFAULT_GRAVITY_M_S2,
) -> ImuPropagationResult:
    """Integrate gyro (rotation) and accel (translation aid) between two timestamps."""
    if t_end_us <= t_start_us:
        identity = np.eye(4, dtype=np.float64)
        return ImuPropagationResult(identity, 0, True)

    r_cam_gyro = calib.t_left_gyro[:3, :3]
    relevant = [s for s in samples if t_start_us < s.device_timestamp_us <= t_end_us]
    if not relevant:
        identity = np.eye(4, dtype=np.float64)
        return ImuPropagationResult(identity, 0, True)

    r_delta = np.eye(3, dtype=np.float64)
    t_delta = np.zeros(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64) if velocity_m_s is None else np.asarray(velocity_m_s, dtype=np.float64).copy()
    prev_ts = t_start_us
    used = 0

    for sample in relevant:
        dt = (sample.device_timestamp_us - prev_ts) / 1_000_000.0
        if dt <= 0.0:
            continue
        gyro_cam = r_cam_gyro @ sample.gyro_rad_s
        accel_cam = r_cam_gyro @ sample.accel_m_s2
        dR = integrate_gyro_rotation(gyro_cam, dt)
        r_delta = dR @ r_delta
        v = v + accel_cam * dt
        t_delta = t_delta + v * dt
        prev_ts = sample.device_timestamp_us
        used += 1

    tail_dt = (t_end_us - prev_ts) / 1_000_000.0
    if tail_dt > 0.0 and relevant:
        last = relevant[-1]
        gyro_cam = r_cam_gyro @ last.gyro_rad_s
        accel_cam = r_cam_gyro @ last.accel_m_s2
        dR = integrate_gyro_rotation(gyro_cam, tail_dt)
        r_delta = dR @ r_delta
        v = v + accel_cam * tail_dt
        t_delta = t_delta + v * tail_dt

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r_delta
    transform[:3, 3] = t_delta
    return ImuPropagationResult(transform, used, transform_is_finite(transform))


def triangulate_stereo_points(
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    if left_pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts4 = cv2.triangulatePoints(
        p1,
        p2,
        left_pts.reshape(-1, 2).T.astype(np.float64),
        right_pts.reshape(-1, 2).T.astype(np.float64),
    )
    pts3 = (pts4[:3] / pts4[3:4]).T
    valid = np.isfinite(pts3).all(axis=1) & (pts3[:, 2] > 0.05) & (pts3[:, 2] < 20.0)
    pts3[~valid] = np.nan
    return pts3


def match_stereo_points(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_pts: np.ndarray,
    *,
    search_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = left_gray.shape[:2]
    left_out: list[list[float]] = []
    right_out: list[list[float]] = []
    win = 5
    for pt in np.asarray(left_pts, dtype=np.float32).reshape(-1, 2):
        u, v = int(round(float(pt[0]))), int(round(float(pt[1])))
        if u < win or v < win or u >= w - win or v >= h - win:
            continue
        patch = left_gray[v - win : v + win + 1, u - win : u + win + 1]
        best_score = -1.0
        best_u = None
        v0 = max(win, min(h - win - 1, v))
        for ur in range(max(win, u - search_px), min(w - win, u + 1)):
            rp = right_gray[v0 - win : v0 + win + 1, ur - win : ur + win + 1]
            if rp.shape != patch.shape:
                continue
            score = float(cv2.matchTemplate(rp.astype(np.float32), patch.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_score = score
                best_u = ur
        if best_u is not None and best_score > 0.3:
            left_out.append([float(u), float(v)])
            right_out.append([float(best_u), float(v0)])
    if not left_out:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.asarray(left_out, dtype=np.float32), np.asarray(right_out, dtype=np.float32)


def detect_landmarks(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    calib: StereoImuCalibration,
    config: StereoImuVioConfig,
) -> LandmarkState | None:
    pts = cv2.goodFeaturesToTrack(
        left_gray,
        maxCorners=config.max_features,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if pts is None or len(pts) < config.min_stereo_points:
        return None
    left_matched, right_matched = match_stereo_points(
        left_gray,
        right_gray,
        pts,
        search_px=config.disparity_search_px,
    )
    if len(left_matched) < config.min_stereo_points:
        return None
    points_3d = triangulate_stereo_points(left_matched, right_matched, calib.p1, calib.p2)
    valid = np.isfinite(points_3d).all(axis=1)
    if int(valid.sum()) < config.min_stereo_points:
        return None
    return LandmarkState(points_3d=points_3d[valid], left_pts=left_matched[valid])


def estimate_visual_motion(
    prev_left: np.ndarray,
    curr_left: np.ndarray,
    landmarks: LandmarkState,
    calib: StereoImuCalibration,
    config: StereoImuVioConfig,
) -> VisualUpdateResult:
    if landmarks.points_3d.shape[0] < config.min_visual_inliers:
        return VisualUpdateResult(False, None, 0, int(landmarks.points_3d.shape[0]), "insufficient_landmarks")

    lk_params = dict(
        winSize=(config.lk_win_size, config.lk_win_size),
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_left,
        curr_left,
        landmarks.left_pts.reshape(-1, 1, 2).astype(np.float32),
        None,
        **lk_params,
    )
    if next_pts is None or status is None:
        return VisualUpdateResult(False, None, 0, int(landmarks.points_3d.shape[0]), "optical_flow_failed")

    mask = status.reshape(-1).astype(bool)
    object_pts = landmarks.points_3d[mask]
    image_pts = next_pts.reshape(-1, 2)[mask]
    stereo_points = int(object_pts.shape[0])
    if stereo_points < config.min_visual_inliers:
        return VisualUpdateResult(False, None, 0, stereo_points, "insufficient_tracked_points")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts.astype(np.float64),
        image_pts.astype(np.float64),
        calib.k_left,
        calib.d_left,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=config.reprojection_error_px,
        confidence=0.99,
    )
    if not ok or inliers is None or len(inliers) < config.min_visual_inliers:
        return VisualUpdateResult(False, None, 0, stereo_points, "solvepnp_failed")

    rmat, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rmat
    transform[:3, 3] = tvec.reshape(3)
    if not transform_is_finite(transform):
        return VisualUpdateResult(False, None, 0, stereo_points, "non_finite_transform")

    return VisualUpdateResult(True, transform, int(len(inliers)), stereo_points, None)


def rectify_stereo_pair(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    calib: StereoImuCalibration,
    map1_left: np.ndarray,
    map2_left: np.ndarray,
    map1_right: np.ndarray,
    map2_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_rect = cv2.remap(left_gray, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_gray, map1_right, map2_right, cv2.INTER_LINEAR)
    return left_rect, right_rect


def build_rectification_maps(calib: StereoImuCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w, h = calib.image_size
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        calib.k_left, calib.d_left, calib.r1, calib.p1, (w, h), cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        calib.k_right, calib.d_right, calib.r2, calib.p2, (w, h), cv2.CV_32FC1
    )
    return map1_l, map2_l, map1_r, map2_r


@dataclass(frozen=True)
class StereoImuVioFrameInput:
    frame_number: int
    device_timestamp_us: int
    left_gray: np.ndarray
    right_gray: np.ndarray
    native_left_frame_number: int
    native_right_frame_number: int


@dataclass(frozen=True)
class StereoImuTrajectorySample:
    frame_number: int
    device_timestamp_us: int
    valid: bool
    state: str
    native_left_frame_number: int
    native_right_frame_number: int
    tx: float | None
    ty: float | None
    tz: float | None
    qw: float | None
    qx: float | None
    qy: float | None
    qz: float | None
    visual_inliers: int
    stereo_points: int
    imu_samples_used: int
    visual_update_success: bool
    imu_propagated: bool
    propagated_only: bool
    failure_reason: str | None = None


STATE_VISUAL_IMU = "visual_imu"
STATE_PROPAGATED_ONLY = "propagated_only"
STATE_INVALID = "invalid"
STATE_INIT = "init"


def pose_to_quaternion_translation(T: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    from .rgbd_odometry_continuous import pose_to_quaternion_translation as _pose_to_qt

    return _pose_to_qt(T)


def run_stereo_imu_vio_lite(
    frames: Sequence[StereoImuVioFrameInput],
    imu_samples: Sequence[ImuSampleRecord],
    calib: StereoImuCalibration,
    config: StereoImuVioConfig | None = None,
) -> list[StereoImuTrajectorySample]:
    if not frames:
        raise ValueError("frames must not be empty")
    cfg = config or StereoImuVioConfig()
    map1_l, map2_l, map1_r, map2_r = build_rectification_maps(calib)

    samples: list[StereoImuTrajectorySample] = []
    t_odom = np.eye(4, dtype=np.float64)
    landmarks: LandmarkState | None = None
    prev_left_rect: np.ndarray | None = None
    prev_ts = frames[0].device_timestamp_us
    propagated_streak = 0

    first = frames[0]
    left0, right0 = rectify_stereo_pair(
        first.left_gray, first.right_gray, calib, map1_l, map2_l, map1_r, map2_r
    )
    landmarks = detect_landmarks(left0, right0, calib, cfg)
    prev_left_rect = left0
    tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(t_odom)
    samples.append(
        StereoImuTrajectorySample(
            frame_number=first.frame_number,
            device_timestamp_us=first.device_timestamp_us,
            valid=landmarks is not None,
            state=STATE_INIT if landmarks is not None else STATE_INVALID,
            native_left_frame_number=first.native_left_frame_number,
            native_right_frame_number=first.native_right_frame_number,
            tx=tx,
            ty=ty,
            tz=tz,
            qw=qw,
            qx=qx,
            qy=qy,
            qz=qz,
            visual_inliers=0,
            stereo_points=0 if landmarks is None else int(landmarks.points_3d.shape[0]),
            imu_samples_used=0,
            visual_update_success=landmarks is not None,
            imu_propagated=False,
            propagated_only=False,
            failure_reason=None if landmarks is not None else "init_landmarks_failed",
        )
    )

    for frame in frames[1:]:
        left_rect, right_rect = rectify_stereo_pair(
            frame.left_gray, frame.right_gray, calib, map1_l, map2_l, map1_r, map2_r
        )
        imu_result = propagate_imu_between_timestamps(
            imu_samples,
            prev_ts,
            frame.device_timestamp_us,
            calib,
            gravity_m_s2=cfg.gravity_m_s2,
        )
        t_imu_rel = imu_result.transform_target_source
        t_pred = accumulate_odom_pose(t_odom, t_imu_rel) if transform_is_finite(t_imu_rel) else t_odom.copy()

        visual = VisualUpdateResult(False, None, 0, 0, "no_landmarks")
        if landmarks is not None and prev_left_rect is not None:
            visual = estimate_visual_motion(prev_left_rect, left_rect, landmarks, calib, cfg)

        use_visual = visual.success and visual.transform_target_source is not None
        propagated_only = False
        failure_reason = visual.failure_reason

        if use_visual:
            t_odom = accumulate_odom_pose(t_odom, visual.transform_target_source)
            propagated_streak = 0
            state = STATE_VISUAL_IMU
            valid = True
            imu_propagated = imu_result.samples_used > 0
        elif imu_result.finite and propagated_streak < cfg.max_propagated_only_frames:
            t_odom = t_pred
            propagated_streak += 1
            propagated_only = True
            state = STATE_PROPAGATED_ONLY
            valid = True
            imu_propagated = True
            failure_reason = visual.failure_reason or "visual_update_failed"
        else:
            state = STATE_INVALID
            valid = False
            imu_propagated = imu_result.samples_used > 0
            failure_reason = visual.failure_reason or "propagation_limit_exceeded"

        if valid:
            tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(t_odom)
        else:
            tx = ty = tz = qw = qx = qy = qz = None

        samples.append(
            StereoImuTrajectorySample(
                frame_number=frame.frame_number,
                device_timestamp_us=frame.device_timestamp_us,
                valid=valid,
                state=state,
                native_left_frame_number=frame.native_left_frame_number,
                native_right_frame_number=frame.native_right_frame_number,
                tx=tx,
                ty=ty,
                tz=tz,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                visual_inliers=visual.visual_inliers,
                stereo_points=visual.stereo_points,
                imu_samples_used=imu_result.samples_used,
                visual_update_success=use_visual,
                imu_propagated=imu_propagated,
                propagated_only=propagated_only,
                failure_reason=failure_reason,
            )
        )

        if valid:
            new_landmarks = detect_landmarks(left_rect, right_rect, calib, cfg)
            if new_landmarks is not None:
                landmarks = new_landmarks
            elif landmarks is not None and use_visual:
                landmarks.left_pts = landmarks.left_pts
            prev_left_rect = left_rect
            prev_ts = frame.device_timestamp_us
        elif propagated_only:
            prev_left_rect = left_rect
            prev_ts = frame.device_timestamp_us

    return samples
