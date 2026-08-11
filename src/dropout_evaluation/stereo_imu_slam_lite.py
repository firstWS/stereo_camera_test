"""Phase 4.7-A stereo+IMU SLAM-lite built on the existing VIO frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .rgbd_odometry import accumulate_odom_pose, transform_is_finite
from .stereo_imu_calibration import StereoImuCalibration
from .stereo_imu_slam_map import (
    POSE_SOURCE_INIT,
    POSE_SOURCE_MAP_RELOCALIZED,
    POSE_SOURCE_MAP_TRACKING,
    POSE_SOURCE_TRACKING_LOST,
    POSE_SOURCE_VIO_PROPAGATED,
    POSE_SOURCE_VIO_TRACKING,
    SlamMap,
    SlamMapConfig,
)
from .stereo_imu_vio_lite import (
    ImuSampleRecord,
    StereoImuVioConfig,
    StereoImuVioFrameInput,
    VisualUpdateResult,
    build_rectification_maps,
    detect_landmarks,
    estimate_visual_motion,
    pose_to_quaternion_translation,
    propagate_imu_between_timestamps,
    rectify_stereo_pair,
)

STEREO_IMU_SLAM_LITE_ALGORITHM_ID = "stereo_imu_slam_lite"

STATE_SLAM_INIT = "slam_init"
STATE_SLAM_VIO_TRACKING = "slam_vio_tracking"
STATE_SLAM_MAP_TRACKING = "slam_map_tracking"
STATE_SLAM_VIO_PROPAGATED = "slam_vio_propagated"
STATE_SLAM_MAP_RELOCALIZED = "slam_map_relocalized"
STATE_SLAM_TRACKING_LOST = "slam_tracking_lost"


@dataclass(frozen=True)
class StereoImuSlamConfig:
    vio: StereoImuVioConfig
    map: SlamMapConfig


@dataclass(frozen=True)
class StereoImuSlamTrajectorySample:
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
    frontend_visual_success: bool
    imu_samples_used: int
    keyframe_count: int
    map_point_count: int
    map_match_count: int
    map_inlier_count: int
    map_update_success: bool
    relocalization_attempted: bool
    relocalization_success: bool
    pose_source: str
    failure_reason: str | None = None


def _state_from_pose_source(pose_source: str) -> str:
    mapping = {
        POSE_SOURCE_INIT: STATE_SLAM_INIT,
        POSE_SOURCE_VIO_TRACKING: STATE_SLAM_VIO_TRACKING,
        POSE_SOURCE_MAP_TRACKING: STATE_SLAM_MAP_TRACKING,
        POSE_SOURCE_VIO_PROPAGATED: STATE_SLAM_VIO_PROPAGATED,
        POSE_SOURCE_MAP_RELOCALIZED: STATE_SLAM_MAP_RELOCALIZED,
        POSE_SOURCE_TRACKING_LOST: STATE_SLAM_TRACKING_LOST,
    }
    return mapping.get(pose_source, STATE_SLAM_TRACKING_LOST)


def run_stereo_imu_slam_lite(
    frames: Sequence[StereoImuVioFrameInput],
    imu_samples: Sequence[ImuSampleRecord],
    calib: StereoImuCalibration,
    config: StereoImuSlamConfig | None = None,
) -> tuple[list[StereoImuSlamTrajectorySample], SlamMap, dict[str, int]]:
    if not frames:
        raise ValueError("frames must not be empty")

    cfg = config or StereoImuSlamConfig(vio=StereoImuVioConfig(), map=SlamMapConfig())
    vio_cfg = cfg.vio
    slam_map = SlamMap(cfg.map)
    map1_l, map2_l, map1_r, map2_r = build_rectification_maps(calib)

    samples: list[StereoImuSlamTrajectorySample] = []
    t_slam = np.eye(4, dtype=np.float64)
    t_vio = np.eye(4, dtype=np.float64)
    landmarks = None
    prev_left_rect = None
    prev_ts = frames[0].device_timestamp_us
    propagated_streak = 0

    counters = {
        "keyframes_created": 0,
        "map_update_attempts": 0,
        "map_update_successes": 0,
        "map_based_pose_update_count": 0,
        "relocalization_attempts": 0,
        "relocalization_successes": 0,
    }

    first = frames[0]
    left0, right0 = rectify_stereo_pair(
        first.left_gray, first.right_gray, calib, map1_l, map2_l, map1_r, map2_r
    )
    landmarks = detect_landmarks(left0, right0, calib, vio_cfg)
    prev_left_rect = left0
    init_ok = landmarks is not None
    if init_ok:
        slam_map.add_keyframe(
            frame_number=first.frame_number,
            device_timestamp_us=first.device_timestamp_us,
            T_slam_camera=t_slam,
            left_gray=left0,
            right_gray=right0,
            calib=calib,
            vio_config=vio_cfg,
        )
        counters["keyframes_created"] += 1

    tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(t_slam)
    samples.append(
        StereoImuSlamTrajectorySample(
            frame_number=first.frame_number,
            device_timestamp_us=first.device_timestamp_us,
            valid=init_ok,
            state=STATE_SLAM_INIT if init_ok else STATE_SLAM_TRACKING_LOST,
            native_left_frame_number=first.native_left_frame_number,
            native_right_frame_number=first.native_right_frame_number,
            tx=tx if init_ok else None,
            ty=ty if init_ok else None,
            tz=tz if init_ok else None,
            qw=qw if init_ok else None,
            qx=qx if init_ok else None,
            qy=qy if init_ok else None,
            qz=qz if init_ok else None,
            frontend_visual_success=init_ok,
            imu_samples_used=0,
            keyframe_count=slam_map.keyframe_count,
            map_point_count=slam_map.map_point_count,
            map_match_count=0,
            map_inlier_count=0,
            map_update_success=False,
            relocalization_attempted=False,
            relocalization_success=False,
            pose_source=POSE_SOURCE_INIT if init_ok else POSE_SOURCE_TRACKING_LOST,
            failure_reason=None if init_ok else "init_landmarks_failed",
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
            gravity_m_s2=vio_cfg.gravity_m_s2,
        )
        t_imu_rel = imu_result.transform_target_source
        t_vio_pred = accumulate_odom_pose(t_vio, t_imu_rel) if transform_is_finite(t_imu_rel) else t_vio.copy()

        visual = VisualUpdateResult(False, None, 0, 0, "no_landmarks")
        if landmarks is not None and prev_left_rect is not None:
            visual = estimate_visual_motion(prev_left_rect, left_rect, landmarks, calib, vio_cfg)

        use_visual = visual.success and visual.transform_target_source is not None
        if use_visual:
            t_vio = accumulate_odom_pose(t_vio, visual.transform_target_source)
            propagated_streak = 0
        elif imu_result.finite and propagated_streak < vio_cfg.max_propagated_only_frames:
            t_vio = t_vio_pred
            propagated_streak += 1
        else:
            propagated_streak += 1

        counters["map_update_attempts"] += 1
        map_result = slam_map.localize_with_map(left_gray=left_rect, calib=calib)
        map_update_success = map_result.success and map_result.T_slam_camera is not None
        relocalization_attempted = False
        relocalization_success = False
        pose_source = POSE_SOURCE_TRACKING_LOST
        valid = False
        failure_reason = visual.failure_reason

        if map_update_success:
            t_slam = map_result.T_slam_camera.copy()
            pose_source = POSE_SOURCE_MAP_TRACKING
            valid = True
            counters["map_update_successes"] += 1
            counters["map_based_pose_update_count"] += 1
            slam_map.update_observations(frame.frame_number, map_result.match_count)
        elif use_visual:
            t_slam = accumulate_odom_pose(t_slam, visual.transform_target_source)
            pose_source = POSE_SOURCE_VIO_TRACKING
            valid = True
        elif imu_result.finite and propagated_streak <= vio_cfg.max_propagated_only_frames:
            t_slam = accumulate_odom_pose(t_slam, t_imu_rel) if transform_is_finite(t_imu_rel) else t_slam
            pose_source = POSE_SOURCE_VIO_PROPAGATED
            valid = True
            failure_reason = visual.failure_reason or "visual_update_failed"
        else:
            relocalization_attempted = True
            counters["relocalization_attempts"] += 1
            reloc = slam_map.relocalize(left_gray=left_rect, calib=calib)
            if reloc.success and reloc.T_slam_camera is not None:
                t_slam = reloc.T_slam_camera.copy()
                pose_source = POSE_SOURCE_MAP_RELOCALIZED
                relocalization_success = True
                valid = True
                counters["relocalization_successes"] += 1
                counters["map_based_pose_update_count"] += 1
                map_result = reloc
                map_update_success = True
            else:
                pose_source = POSE_SOURCE_TRACKING_LOST
                failure_reason = visual.failure_reason or reloc.failure_reason or "tracking_lost"

        if valid and slam_map.should_insert_keyframe(
            frame_number=frame.frame_number,
            T_slam_camera=t_slam,
        ):
            slam_map.add_keyframe(
                frame_number=frame.frame_number,
                device_timestamp_us=frame.device_timestamp_us,
                T_slam_camera=t_slam,
                left_gray=left_rect,
                right_gray=right_rect,
                calib=calib,
                vio_config=vio_cfg,
            )
            counters["keyframes_created"] += 1

        if valid:
            tx, ty, tz, qw, qx, qy, qz = pose_to_quaternion_translation(t_slam)
        else:
            tx = ty = tz = qw = qx = qy = qz = None

        samples.append(
            StereoImuSlamTrajectorySample(
                frame_number=frame.frame_number,
                device_timestamp_us=frame.device_timestamp_us,
                valid=valid,
                state=_state_from_pose_source(pose_source),
                native_left_frame_number=frame.native_left_frame_number,
                native_right_frame_number=frame.native_right_frame_number,
                tx=tx,
                ty=ty,
                tz=tz,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                frontend_visual_success=use_visual,
                imu_samples_used=imu_result.samples_used,
                keyframe_count=slam_map.keyframe_count,
                map_point_count=slam_map.map_point_count,
                map_match_count=map_result.match_count,
                map_inlier_count=map_result.inlier_count,
                map_update_success=map_update_success,
                relocalization_attempted=relocalization_attempted,
                relocalization_success=relocalization_success,
                pose_source=pose_source,
                failure_reason=failure_reason,
            )
        )

        if valid or pose_source == POSE_SOURCE_VIO_PROPAGATED:
            new_landmarks = detect_landmarks(left_rect, right_rect, calib, vio_cfg)
            if new_landmarks is not None:
                landmarks = new_landmarks
            prev_left_rect = left_rect
            prev_ts = frame.device_timestamp_us

    return samples, slam_map, counters
