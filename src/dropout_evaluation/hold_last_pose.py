"""HOLD_LAST_POSE Phase 3 dropout baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from .dropout_protocol import DropoutWindow, FrameTimestamp, is_runtime_tag_masked
from .evaluation_metrics import PoseEstimate, PoseTrackingState
from .runtime_apriltag import RUNTIME_APRILTAG_SOURCE, RuntimeAprilTagPose

HOLD_LAST_POSE_ALGORITHM_ID = "hold_last_pose"
DEFAULT_POSE_POLICY = "constant_anchor_pose"
DEFAULT_MAX_ANCHOR_AGE_FRAMES = 120


class HoldLastPoseStatus(str, Enum):
    OK = "OK"
    INVALID = "INVALID"


@dataclass(frozen=True)
class HoldLastPoseConfig:
    max_anchor_age_frames: int = DEFAULT_MAX_ANCHOR_AGE_FRAMES
    algorithm_id: str = HOLD_LAST_POSE_ALGORITHM_ID
    pose_policy: str = DEFAULT_POSE_POLICY


@dataclass(frozen=True)
class HoldLastPoseProvenance:
    window_id: str
    algorithm_id: str
    anchor_frame: int | None
    anchor_device_timestamp_us: int | None
    anchor_age_frames: int | None
    anchor_age_sec: float | None
    recovery_requested_frame: int | None
    recovery_actual_frame: int | None
    recovery_latency_frames: int | None
    recovery_latency_sec: float | None
    anchor_source: str | None
    recovery_source: str | None
    pose_policy: str
    status: HoldLastPoseStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "algorithm_id": self.algorithm_id,
            "anchor_frame": self.anchor_frame,
            "anchor_device_timestamp_us": self.anchor_device_timestamp_us,
            "anchor_age_frames": self.anchor_age_frames,
            "anchor_age_sec": self.anchor_age_sec,
            "recovery_requested_frame": self.recovery_requested_frame,
            "recovery_actual_frame": self.recovery_actual_frame,
            "recovery_latency_frames": self.recovery_latency_frames,
            "recovery_latency_sec": self.recovery_latency_sec,
            "anchor_source": self.anchor_source,
            "recovery_source": self.recovery_source,
            "pose_policy": self.pose_policy,
            "status": self.status.value,
        }


@dataclass
class HoldLastPoseWindowResult:
    window_id: str
    status: HoldLastPoseStatus
    provenance: HoldLastPoseProvenance
    candidates: list[PoseEstimate]


def select_pre_window_anchor(
    *,
    window: DropoutWindow,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    max_anchor_age_frames: int,
) -> RuntimeAprilTagPose | None:
    """Latest valid runtime tag pose strictly before window start timestamp."""
    eligible = [
        pose
        for pose in runtime_poses
        if pose.valid and pose.device_timestamp_us < window.start_device_timestamp_us
    ]
    if not eligible:
        return None
    anchor = max(eligible, key=lambda pose: pose.device_timestamp_us)
    age_frames = window.start_frame - anchor.frame_number
    if age_frames > max_anchor_age_frames:
        return None
    return anchor


def find_recovery_runtime_pose(
    *,
    window: DropoutWindow,
    runtime_poses: Sequence[RuntimeAprilTagPose],
) -> tuple[RuntimeAprilTagPose | None, int | None, float | None]:
    if window.recovery_frame is None or window.recovery_device_timestamp_us is None:
        return None, None, None
    requested_ts = window.recovery_device_timestamp_us
    for pose in runtime_poses:
        if pose.valid and pose.device_timestamp_us >= window.boundary_timestamp_us:
            latency_frames = pose.frame_number - window.recovery_frame
            latency_sec = (pose.device_timestamp_us - requested_ts) / 1_000_000.0
            return pose, latency_frames, latency_sec
    return None, None, None


def generate_hold_last_pose_candidates(
    *,
    window: DropoutWindow,
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    config: HoldLastPoseConfig | None = None,
) -> HoldLastPoseWindowResult:
    """Generate HOLD_LAST_POSE candidates without access to offline reference."""
    cfg = config or HoldLastPoseConfig()
    anchor = select_pre_window_anchor(
        window=window,
        runtime_poses=runtime_poses,
        max_anchor_age_frames=cfg.max_anchor_age_frames,
    )
    recovery_pose, recovery_latency_frames, recovery_latency_sec = find_recovery_runtime_pose(
        window=window,
        runtime_poses=runtime_poses,
    )

    if anchor is None:
        provenance = HoldLastPoseProvenance(
            window_id=window.window_id,
            algorithm_id=cfg.algorithm_id,
            anchor_frame=None,
            anchor_device_timestamp_us=None,
            anchor_age_frames=None,
            anchor_age_sec=None,
            recovery_requested_frame=window.recovery_frame,
            recovery_actual_frame=None,
            recovery_latency_frames=None,
            recovery_latency_sec=None,
            anchor_source=None,
            recovery_source=None,
            pose_policy=cfg.pose_policy,
            status=HoldLastPoseStatus.INVALID,
        )
        return HoldLastPoseWindowResult(
            window_id=window.window_id,
            status=HoldLastPoseStatus.INVALID,
            provenance=provenance,
            candidates=[],
        )

    anchor_age_frames = window.start_frame - anchor.frame_number
    anchor_age_sec = (window.start_device_timestamp_us - anchor.device_timestamp_us) / 1_000_000.0
    T_anchor = anchor.T_world_camera.copy()
    T_recovery = recovery_pose.T_world_camera.copy() if recovery_pose is not None else None

    last_frame = window.recovery_frame or window.end_frame
    if recovery_pose is not None:
        last_frame = max(last_frame, recovery_pose.frame_number)

    candidates: list[PoseEstimate] = []
    for frame in frame_timestamps:
        if frame.frame_number < anchor.frame_number or frame.frame_number > last_frame:
            continue

        if frame.frame_number == anchor.frame_number:
            state = PoseTrackingState.TAG_ANCHORED
            T = T_anchor.copy()
            valid = True
        elif is_runtime_tag_masked(frame.device_timestamp_us, window):
            state = PoseTrackingState.LOCAL_TRACKING
            T = T_anchor.copy()
            valid = True
        elif recovery_pose is not None and frame.frame_number == recovery_pose.frame_number:
            state = PoseTrackingState.RELOCALIZED
            T = T_recovery.copy() if T_recovery is not None else None
            valid = T_recovery is not None
        elif recovery_pose is not None and frame.frame_number > recovery_pose.frame_number:
            state = PoseTrackingState.TAG_ANCHORED
            T = T_recovery.copy() if T_recovery is not None else None
            valid = T_recovery is not None
        elif window.recovery_frame is not None and frame.frame_number >= window.recovery_frame:
            state = PoseTrackingState.RELOCALIZING
            T = T_anchor.copy()
            valid = True
        else:
            continue

        candidates.append(
            PoseEstimate(
                frame_number=frame.frame_number,
                device_timestamp_us=frame.device_timestamp_us,
                T_world_camera=T,
                valid=valid,
                state=state,
                algorithm_id=cfg.algorithm_id,
            )
        )

    provenance = HoldLastPoseProvenance(
        window_id=window.window_id,
        algorithm_id=cfg.algorithm_id,
        anchor_frame=anchor.frame_number,
        anchor_device_timestamp_us=anchor.device_timestamp_us,
        anchor_age_frames=anchor_age_frames,
        anchor_age_sec=anchor_age_sec,
        recovery_requested_frame=window.recovery_frame,
        recovery_actual_frame=recovery_pose.frame_number if recovery_pose else None,
        recovery_latency_frames=recovery_latency_frames,
        recovery_latency_sec=recovery_latency_sec,
        anchor_source=RUNTIME_APRILTAG_SOURCE,
        recovery_source=RUNTIME_APRILTAG_SOURCE if recovery_pose else None,
        pose_policy=cfg.pose_policy,
        status=HoldLastPoseStatus.OK,
    )
    return HoldLastPoseWindowResult(
        window_id=window.window_id,
        status=HoldLastPoseStatus.OK,
        provenance=provenance,
        candidates=candidates,
    )
