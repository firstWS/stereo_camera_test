"""Phase 4.2-C RGB-D continuous local trajectory to Phase 3 PoseEstimate adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .dropout_protocol import DropoutWindow, FrameTimestamp, is_runtime_tag_masked
from .evaluation_metrics import PoseEstimate, PoseTrackingState, RecoveryTiming
from .hold_last_pose import select_pre_window_anchor
from .rgbd_odometry import RGBD_ODOMETRY_ALGORITHM_ID
from .rgbd_odometry_continuous import TrajectorySample
from .runtime_apriltag import RUNTIME_APRILTAG_SOURCE, RuntimeAprilTagPose


class RgbdAdapterStatus(str, Enum):
    OK = "OK"
    INVALID = "INVALID"


@dataclass(frozen=True)
class RgbdOdometryAdapterConfig:
    algorithm_id: str = RGBD_ODOMETRY_ALGORITHM_ID


@dataclass(frozen=True)
class SegmentAlignmentState:
    segment_id: int
    T_world_odom: np.ndarray | None = None
    alignment_frame: int | None = None
    alignment_timestamp_us: int | None = None


@dataclass(frozen=True)
class FrameReplayState:
    frame_number: int
    device_timestamp_us: int
    segment_id: int
    local_valid: bool
    masked: bool
    tag_valid: bool
    joint_updated: bool
    T_odom_camera: np.ndarray | None
    T_world_camera: np.ndarray | None
    world_valid: bool
    bridge_recovered: bool


@dataclass(frozen=True)
class SessionReplayResult:
    frames: dict[int, FrameReplayState]
    recovery_actual_frame: int | None
    recovery_actual_timestamp_us: int | None
    protocol_runtime_anchor_frame: int | None
    world_alignment_frame_before_dropout: int | None
    world_alignment_timestamp_before_dropout: int | None
    alignment_age_frames_at_dropout: int | None
    alignment_age_sec_at_dropout: float | None
    alignment_segment_id: int | None
    segment_reset_during_dropout: bool
    dropout_segment_id: int | None


@dataclass(frozen=True)
class RgbdAdapterProvenance:
    window_id: str
    algorithm_id: str
    protocol_runtime_anchor_frame: int | None
    world_alignment_frame_before_dropout: int | None
    world_alignment_timestamp_us: int | None
    alignment_age_frames: int | None
    alignment_age_sec: float | None
    alignment_segment_id: int | None
    segment_reset_during_dropout: bool
    local_invalid_frames_in_dropout: int
    world_valid_frames_in_dropout: int
    runtime_tag_used_during_dropout: bool
    reference_used_by_candidate: bool
    tag_texture_visible: bool
    recovery_requested_frame: int | None
    recovery_actual_frame: int | None
    recovery_latency_frames: int | None
    recovery_latency_sec: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "algorithm_id": self.algorithm_id,
            "protocol_runtime_anchor_frame": self.protocol_runtime_anchor_frame,
            "world_alignment_frame_before_dropout": self.world_alignment_frame_before_dropout,
            "world_alignment_timestamp_us": self.world_alignment_timestamp_us,
            "alignment_age_frames": self.alignment_age_frames,
            "alignment_age_sec": self.alignment_age_sec,
            "alignment_segment_id": self.alignment_segment_id,
            "segment_reset_during_dropout": self.segment_reset_during_dropout,
            "local_invalid_frames_in_dropout": self.local_invalid_frames_in_dropout,
            "world_valid_frames_in_dropout": self.world_valid_frames_in_dropout,
            "runtime_tag_used_during_dropout": self.runtime_tag_used_during_dropout,
            "reference_used_by_candidate": self.reference_used_by_candidate,
            "tag_texture_visible": self.tag_texture_visible,
            "recovery_requested_frame": self.recovery_requested_frame,
            "recovery_actual_frame": self.recovery_actual_frame,
            "recovery_latency_frames": self.recovery_latency_frames,
            "recovery_latency_sec": self.recovery_latency_sec,
        }


@dataclass(frozen=True)
class WindowDiagnosticSummary:
    window_id: str
    dropout_start_frame: int
    dropout_end_frame: int
    protocol_anchor_frame: int | None
    alignment_frame_before_dropout: int | None
    alignment_age_frames: int | None
    alignment_age_sec: float | None
    segment_id: int | None
    expected_dropout_frames: int
    world_valid_frames_in_dropout: int
    world_invalid_frames_in_dropout: int
    world_availability_ratio: float
    segment_reset_during_dropout: bool
    first_invalid_frame: int | None
    first_local_bridge_recovery_frame: int | None
    recovery_requested_frame: int | None
    recovery_actual_frame: int | None
    recovery_latency_frames: int | None
    recovery_latency_sec: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "dropout_start_frame": self.dropout_start_frame,
            "dropout_end_frame": self.dropout_end_frame,
            "protocol_anchor_frame": self.protocol_anchor_frame,
            "alignment_frame_before_dropout": self.alignment_frame_before_dropout,
            "alignment_age_frames": self.alignment_age_frames,
            "alignment_age_sec": self.alignment_age_sec,
            "segment_id": self.segment_id,
            "expected_dropout_frames": self.expected_dropout_frames,
            "world_valid_frames_in_dropout": self.world_valid_frames_in_dropout,
            "world_invalid_frames_in_dropout": self.world_invalid_frames_in_dropout,
            "world_availability_ratio": self.world_availability_ratio,
            "segment_reset_during_dropout": self.segment_reset_during_dropout,
            "first_invalid_frame": self.first_invalid_frame,
            "first_local_bridge_recovery_frame": self.first_local_bridge_recovery_frame,
            "recovery_requested_frame": self.recovery_requested_frame,
            "recovery_actual_frame": self.recovery_actual_frame,
            "recovery_latency_frames": self.recovery_latency_frames,
            "recovery_latency_sec": self.recovery_latency_sec,
        }


@dataclass
class RgbdAdapterWindowResult:
    window_id: str
    status: RgbdAdapterStatus
    provenance: RgbdAdapterProvenance
    candidates: list[PoseEstimate]
    diagnostic: WindowDiagnosticSummary
    replay: SessionReplayResult


def trajectory_sample_to_T_odom(sample: TrajectorySample) -> np.ndarray | None:
    if not sample.valid:
        return None
    if sample.qw is None or sample.tx is None:
        return None
    qw, qx, qy, qz = sample.qw, sample.qx, sample.qy, sample.qz
    tx, ty, tz = sample.tx, sample.ty, sample.tz
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def compute_joint_alignment(T_world_camera_tag: np.ndarray, T_odom_camera: np.ndarray) -> np.ndarray:
    """T_world_odom = T_world_camera_tag @ inv(T_odom_camera)."""
    T_tag = np.asarray(T_world_camera_tag, dtype=np.float64)
    T_odom = np.asarray(T_odom_camera, dtype=np.float64)
    return T_tag @ np.linalg.inv(T_odom)


def compute_world_pose(T_world_odom: np.ndarray, T_odom_camera: np.ndarray) -> np.ndarray:
    """T_world_camera_candidate = T_world_odom @ T_odom_camera."""
    return np.asarray(T_world_odom, dtype=np.float64) @ np.asarray(T_odom_camera, dtype=np.float64)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def load_local_trajectory_from_csv(path: str | Any) -> list[TrajectorySample]:
    from pathlib import Path

    samples: list[TrajectorySample] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                TrajectorySample(
                    frame_number=int(row["frame_number"]),
                    device_timestamp_us=int(row["device_timestamp_us"]),
                    valid=_parse_bool(row["valid"]),
                    tracking_state=str(row["tracking_state"]),
                    segment_id=int(row["segment_id"]),
                    segment_start=_parse_bool(row["segment_start"]),
                    continuity_from_previous_segment=_parse_bool(row["continuity_from_previous_segment"]),
                    tx=_parse_optional_float(row.get("tx")),
                    ty=_parse_optional_float(row.get("ty")),
                    tz=_parse_optional_float(row.get("tz")),
                    qw=_parse_optional_float(row.get("qw")),
                    qx=_parse_optional_float(row.get("qx")),
                    qy=_parse_optional_float(row.get("qy")),
                    qz=_parse_optional_float(row.get("qz")),
                    source_frame=int(row["source_frame"]) if row.get("source_frame") else None,
                    pair_gap_frames=int(row["pair_gap_frames"]) if row.get("pair_gap_frames") else None,
                    bridge_recovered=_parse_bool(row.get("bridge_recovered", "false")),
                )
            )
    samples.sort(key=lambda item: item.frame_number)
    return samples


def _runtime_pose_index(runtime_poses: Sequence[RuntimeAprilTagPose]) -> dict[int, RuntimeAprilTagPose]:
    return {pose.frame_number: pose for pose in runtime_poses}


def _local_trajectory_index(local_trajectory: Sequence[TrajectorySample]) -> dict[int, TrajectorySample]:
    return {sample.frame_number: sample for sample in local_trajectory}


def replay_session_for_window(
    *,
    window: DropoutWindow,
    local_trajectory: Sequence[TrajectorySample],
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
) -> SessionReplayResult:
    """Causal session replay for one dropout window (session start through last frame)."""
    local_by_frame = _local_trajectory_index(local_trajectory)
    runtime_by_frame = _runtime_pose_index(runtime_poses)

    alignment_states: dict[int, SegmentAlignmentState] = {}
    frame_states: dict[int, FrameReplayState] = {}

    recovery_actual_frame: int | None = None
    recovery_actual_timestamp_us: int | None = None
    segment_reset_during_dropout = False
    dropout_segment_id: int | None = None
    world_alignment_frame_before_dropout: int | None = None
    world_alignment_timestamp_before_dropout: int | None = None
    alignment_age_frames_at_dropout: int | None = None
    alignment_age_sec_at_dropout: float | None = None
    alignment_segment_id: int | None = None

    protocol_anchor = select_pre_window_anchor(
        window=window,
        runtime_poses=runtime_poses,
        max_anchor_age_frames=10_000,
    )
    protocol_runtime_anchor_frame = protocol_anchor.frame_number if protocol_anchor is not None else None

    previous_segment_id: int | None = None

    for frame in sorted(frame_timestamps, key=lambda item: item.frame_number):
        local = local_by_frame.get(frame.frame_number)
        if local is None:
            continue

        masked = is_runtime_tag_masked(frame.device_timestamp_us, window)
        runtime_pose = runtime_by_frame.get(frame.frame_number)
        tag_valid = runtime_pose is not None and runtime_pose.valid

        seg_state = alignment_states.setdefault(
            local.segment_id,
            SegmentAlignmentState(segment_id=local.segment_id),
        )

        joint_updated = False
        if not masked and tag_valid and local.valid and runtime_pose is not None:
            T_odom = trajectory_sample_to_T_odom(local)
            if T_odom is not None:
                seg_state = SegmentAlignmentState(
                    segment_id=local.segment_id,
                    T_world_odom=compute_joint_alignment(runtime_pose.T_world_camera, T_odom),
                    alignment_frame=frame.frame_number,
                    alignment_timestamp_us=frame.device_timestamp_us,
                )
                alignment_states[local.segment_id] = seg_state
                joint_updated = True
                if (
                    frame.device_timestamp_us >= window.boundary_timestamp_us
                    and recovery_actual_frame is None
                ):
                    recovery_actual_frame = frame.frame_number
                    recovery_actual_timestamp_us = frame.device_timestamp_us

        if masked and previous_segment_id is not None and local.segment_id != previous_segment_id:
            segment_reset_during_dropout = True

        if frame.frame_number == window.start_frame:
            dropout_segment_id = local.segment_id
            alignment_segment_id = local.segment_id
            active_alignment = alignment_states.get(local.segment_id)
            if active_alignment is not None and active_alignment.T_world_odom is not None:
                world_alignment_frame_before_dropout = active_alignment.alignment_frame
                world_alignment_timestamp_before_dropout = active_alignment.alignment_timestamp_us
                if active_alignment.alignment_frame is not None:
                    alignment_age_frames_at_dropout = window.start_frame - active_alignment.alignment_frame
                if (
                    active_alignment.alignment_timestamp_us is not None
                    and world_alignment_timestamp_before_dropout is not None
                ):
                    alignment_age_sec_at_dropout = (
                        window.start_device_timestamp_us - active_alignment.alignment_timestamp_us
                    ) / 1_000_000.0

        T_odom = trajectory_sample_to_T_odom(local) if local.valid else None
        active_alignment = alignment_states.get(local.segment_id)
        world_valid = False
        T_world: np.ndarray | None = None
        if local.valid and T_odom is not None and active_alignment is not None and active_alignment.T_world_odom is not None:
            T_world = compute_world_pose(active_alignment.T_world_odom, T_odom)
            world_valid = True

        frame_states[frame.frame_number] = FrameReplayState(
            frame_number=frame.frame_number,
            device_timestamp_us=frame.device_timestamp_us,
            segment_id=local.segment_id,
            local_valid=local.valid,
            masked=masked,
            tag_valid=tag_valid,
            joint_updated=joint_updated,
            T_odom_camera=T_odom,
            T_world_camera=T_world.copy() if T_world is not None else None,
            world_valid=world_valid,
            bridge_recovered=local.bridge_recovered,
        )
        previous_segment_id = local.segment_id

    return SessionReplayResult(
        frames=frame_states,
        recovery_actual_frame=recovery_actual_frame,
        recovery_actual_timestamp_us=recovery_actual_timestamp_us,
        protocol_runtime_anchor_frame=protocol_runtime_anchor_frame,
        world_alignment_frame_before_dropout=world_alignment_frame_before_dropout,
        world_alignment_timestamp_before_dropout=world_alignment_timestamp_before_dropout,
        alignment_age_frames_at_dropout=alignment_age_frames_at_dropout,
        alignment_age_sec_at_dropout=alignment_age_sec_at_dropout,
        alignment_segment_id=alignment_segment_id,
        segment_reset_during_dropout=segment_reset_during_dropout,
        dropout_segment_id=dropout_segment_id,
    )


def _output_last_frame(window: DropoutWindow, recovery_actual_frame: int | None) -> int:
    last_frame = window.recovery_frame or window.end_frame
    if recovery_actual_frame is not None:
        last_frame = max(last_frame, recovery_actual_frame)
    return last_frame


def _tracking_state_for_frame(
    *,
    frame_number: int,
    replay_state: FrameReplayState,
    window: DropoutWindow,
    recovery_actual_frame: int | None,
) -> PoseTrackingState:
    if not replay_state.local_valid or not replay_state.world_valid:
        return PoseTrackingState.TRACKING_LOST
    if recovery_actual_frame is not None and frame_number == recovery_actual_frame:
        return PoseTrackingState.RELOCALIZED
    if recovery_actual_frame is not None and frame_number > recovery_actual_frame:
        return PoseTrackingState.TAG_ANCHORED
    if replay_state.masked:
        return PoseTrackingState.LOCAL_TRACKING
    if window.recovery_frame is not None and frame_number >= window.recovery_frame:
        if recovery_actual_frame is None or frame_number < recovery_actual_frame:
            return PoseTrackingState.RELOCALIZING
    return PoseTrackingState.TAG_ANCHORED


def generate_rgbd_odometry_candidates(
    *,
    window: DropoutWindow,
    local_trajectory: Sequence[TrajectorySample],
    runtime_poses: Sequence[RuntimeAprilTagPose],
    frame_timestamps: Sequence[FrameTimestamp],
    config: RgbdOdometryAdapterConfig | None = None,
) -> RgbdAdapterWindowResult:
    """Generate Phase 3 PoseEstimate candidates from continuous RGB-D local trajectory."""
    cfg = config or RgbdOdometryAdapterConfig()
    replay = replay_session_for_window(
        window=window,
        local_trajectory=local_trajectory,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
    )

    has_alignment = replay.world_alignment_frame_before_dropout is not None
    if not has_alignment:
        provenance = RgbdAdapterProvenance(
            window_id=window.window_id,
            algorithm_id=cfg.algorithm_id,
            protocol_runtime_anchor_frame=replay.protocol_runtime_anchor_frame,
            world_alignment_frame_before_dropout=None,
            world_alignment_timestamp_us=None,
            alignment_age_frames=None,
            alignment_age_sec=None,
            alignment_segment_id=replay.alignment_segment_id,
            segment_reset_during_dropout=replay.segment_reset_during_dropout,
            local_invalid_frames_in_dropout=0,
            world_valid_frames_in_dropout=0,
            runtime_tag_used_during_dropout=False,
            reference_used_by_candidate=False,
            tag_texture_visible=True,
            recovery_requested_frame=window.recovery_frame,
            recovery_actual_frame=None,
            recovery_latency_frames=None,
            recovery_latency_sec=None,
        )
        diagnostic = _build_window_diagnostic(window=window, replay=replay, candidates=[])
        return RgbdAdapterWindowResult(
            window_id=window.window_id,
            status=RgbdAdapterStatus.INVALID,
            provenance=provenance,
            candidates=[],
            diagnostic=diagnostic,
            replay=replay,
        )

    output_start = replay.protocol_runtime_anchor_frame or (window.start_frame - 1)
    output_end = _output_last_frame(window, replay.recovery_actual_frame)

    candidates: list[PoseEstimate] = []
    for frame in sorted(frame_timestamps, key=lambda item: item.frame_number):
        if frame.frame_number < output_start or frame.frame_number > output_end:
            continue
        replay_state = replay.frames.get(frame.frame_number)
        if replay_state is None:
            continue
        state = _tracking_state_for_frame(
            frame_number=frame.frame_number,
            replay_state=replay_state,
            window=window,
            recovery_actual_frame=replay.recovery_actual_frame,
        )
        valid = replay_state.world_valid and state != PoseTrackingState.TRACKING_LOST
        candidates.append(
            PoseEstimate(
                frame_number=frame.frame_number,
                device_timestamp_us=frame.device_timestamp_us,
                T_world_camera=replay_state.T_world_camera.copy() if valid and replay_state.T_world_camera is not None else None,
                valid=valid,
                state=state,
                algorithm_id=cfg.algorithm_id,
            )
        )

    recovery_latency_frames: int | None = None
    recovery_latency_sec: float | None = None
    if window.recovery_frame is not None and replay.recovery_actual_frame is not None:
        recovery_latency_frames = replay.recovery_actual_frame - window.recovery_frame
    if (
        window.recovery_device_timestamp_us is not None
        and replay.recovery_actual_timestamp_us is not None
    ):
        recovery_latency_sec = (
            replay.recovery_actual_timestamp_us - window.recovery_device_timestamp_us
        ) / 1_000_000.0

    masked_states = [state for fn, state in replay.frames.items() if state.masked]
    local_invalid_in_dropout = sum(1 for state in masked_states if not state.local_valid)
    world_valid_in_dropout = sum(1 for state in masked_states if state.world_valid)

    provenance = RgbdAdapterProvenance(
        window_id=window.window_id,
        algorithm_id=cfg.algorithm_id,
        protocol_runtime_anchor_frame=replay.protocol_runtime_anchor_frame,
        world_alignment_frame_before_dropout=replay.world_alignment_frame_before_dropout,
        world_alignment_timestamp_us=replay.world_alignment_timestamp_before_dropout,
        alignment_age_frames=replay.alignment_age_frames_at_dropout,
        alignment_age_sec=replay.alignment_age_sec_at_dropout,
        alignment_segment_id=replay.alignment_segment_id,
        segment_reset_during_dropout=replay.segment_reset_during_dropout,
        local_invalid_frames_in_dropout=local_invalid_in_dropout,
        world_valid_frames_in_dropout=world_valid_in_dropout,
        runtime_tag_used_during_dropout=False,
        reference_used_by_candidate=False,
        tag_texture_visible=True,
        recovery_requested_frame=window.recovery_frame,
        recovery_actual_frame=replay.recovery_actual_frame,
        recovery_latency_frames=recovery_latency_frames,
        recovery_latency_sec=recovery_latency_sec,
    )
    diagnostic = _build_window_diagnostic(window=window, replay=replay, candidates=candidates)
    return RgbdAdapterWindowResult(
        window_id=window.window_id,
        status=RgbdAdapterStatus.OK,
        provenance=provenance,
        candidates=candidates,
        diagnostic=diagnostic,
        replay=replay,
    )


def _build_window_diagnostic(
    *,
    window: DropoutWindow,
    replay: SessionReplayResult,
    candidates: Sequence[PoseEstimate],
) -> WindowDiagnosticSummary:
    masked_states = [state for fn, state in sorted(replay.frames.items()) if state.masked]
    expected_dropout_frames = len(masked_states)
    world_valid_in_dropout = sum(1 for state in masked_states if state.world_valid)
    world_invalid_in_dropout = expected_dropout_frames - world_valid_in_dropout
    availability = (
        world_valid_in_dropout / expected_dropout_frames if expected_dropout_frames else 0.0
    )

    first_invalid: int | None = None
    first_bridge: int | None = None
    for state in masked_states:
        if not state.local_valid and first_invalid is None:
            first_invalid = state.frame_number
        if state.bridge_recovered and first_bridge is None:
            first_bridge = state.frame_number

    recovery_latency_frames: int | None = None
    recovery_latency_sec: float | None = None
    if window.recovery_frame is not None and replay.recovery_actual_frame is not None:
        recovery_latency_frames = replay.recovery_actual_frame - window.recovery_frame
    if (
        window.recovery_device_timestamp_us is not None
        and replay.recovery_actual_timestamp_us is not None
    ):
        recovery_latency_sec = (
            replay.recovery_actual_timestamp_us - window.recovery_device_timestamp_us
        ) / 1_000_000.0

    return WindowDiagnosticSummary(
        window_id=window.window_id,
        dropout_start_frame=window.start_frame,
        dropout_end_frame=window.end_frame,
        protocol_anchor_frame=replay.protocol_runtime_anchor_frame,
        alignment_frame_before_dropout=replay.world_alignment_frame_before_dropout,
        alignment_age_frames=replay.alignment_age_frames_at_dropout,
        alignment_age_sec=replay.alignment_age_sec_at_dropout,
        segment_id=replay.dropout_segment_id,
        expected_dropout_frames=expected_dropout_frames,
        world_valid_frames_in_dropout=world_valid_in_dropout,
        world_invalid_frames_in_dropout=world_invalid_in_dropout,
        world_availability_ratio=availability,
        segment_reset_during_dropout=replay.segment_reset_during_dropout,
        first_invalid_frame=first_invalid,
        first_local_bridge_recovery_frame=first_bridge,
        recovery_requested_frame=window.recovery_frame,
        recovery_actual_frame=replay.recovery_actual_frame,
        recovery_latency_frames=recovery_latency_frames,
        recovery_latency_sec=recovery_latency_sec,
    )


def resolve_adapter_recovery_timing(
    *,
    window: DropoutWindow,
    result: RgbdAdapterWindowResult,
) -> RecoveryTiming:
    return RecoveryTiming(
        recovery_requested_frame=window.recovery_frame,
        recovery_requested_device_timestamp_us=window.recovery_device_timestamp_us,
        recovery_actual_frame=result.provenance.recovery_actual_frame,
        recovery_actual_device_timestamp_us=result.replay.recovery_actual_timestamp_us,
    )
