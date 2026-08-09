"""Evaluation metrics for Phase 3 dropout windows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .dropout_protocol import DropoutWindow, SuccessThresholds, is_runtime_tag_masked


class PoseTrackingState(str, Enum):
    TAG_ANCHORED = "TAG_ANCHORED"
    LOCAL_TRACKING = "LOCAL_TRACKING"
    RELOCALIZING = "RELOCALIZING"
    RELOCALIZED = "RELOCALIZED"
    TRACKING_LOST = "TRACKING_LOST"


class MetricStatus(str, Enum):
    OK = "OK"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class PoseReference:
    frame_number: int
    device_timestamp_us: int
    T_world_camera: np.ndarray | None
    valid: bool
    interpolated: bool = False
    quality: str = ""


@dataclass
class PoseEstimate:
    frame_number: int
    device_timestamp_us: int
    T_world_camera: np.ndarray | None
    valid: bool
    state: PoseTrackingState = PoseTrackingState.LOCAL_TRACKING
    algorithm_id: str = ""


@dataclass(frozen=True)
class CupObservation:
    frame_number: int
    device_timestamp_us: int
    semantic_id: str
    P_camera: np.ndarray
    valid: bool


@dataclass
class DistributionSummary:
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    max: float | None = None
    end_of_dropout_error: float | None = None
    last_valid_before_recovery_error: float | None = None
    sample_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "max": self.max,
            "end_of_dropout_error": self.end_of_dropout_error,
            "last_valid_before_recovery_error": self.last_valid_before_recovery_error,
            "sample_count": self.sample_count,
        }


@dataclass
class AvailabilityMetrics:
    expected_frames: int
    valid_pose_frames: int
    availability_ratio: float
    lost_event_count: int
    longest_invalid_gap_frames: int
    longest_invalid_gap_sec: float
    tracking_lost_frames: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_frames": self.expected_frames,
            "valid_pose_frames": self.valid_pose_frames,
            "availability_ratio": self.availability_ratio,
            "lost_event_count": self.lost_event_count,
            "longest_invalid_gap_frames": self.longest_invalid_gap_frames,
            "longest_invalid_gap_sec": self.longest_invalid_gap_sec,
            "tracking_lost_frames": self.tracking_lost_frames,
        }


@dataclass
class RecoveryMetrics:
    recovery_requested_frame: int | None
    recovery_actual_frame: int | None
    recovery_latency_frames: int | None
    recovery_latency_sec: float | None
    pre_recovery_translation_error: float | None
    pre_recovery_rotation_error: float | None
    reanchor_translation_jump: float | None
    reanchor_rotation_jump: float | None
    post_recovery_translation_error: float | None
    post_recovery_rotation_error: float | None
    status: MetricStatus = MetricStatus.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovery_requested_frame": self.recovery_requested_frame,
            "recovery_actual_frame": self.recovery_actual_frame,
            "recovery_latency_frames": self.recovery_latency_frames,
            "recovery_latency_sec": self.recovery_latency_sec,
            "pre_recovery_translation_error": self.pre_recovery_translation_error,
            "pre_recovery_rotation_error": self.pre_recovery_rotation_error,
            "reanchor_translation_jump": self.reanchor_translation_jump,
            "reanchor_rotation_jump": self.reanchor_rotation_jump,
            "post_recovery_translation_error": self.post_recovery_translation_error,
            "post_recovery_rotation_error": self.post_recovery_rotation_error,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class RecoveryTiming:
    recovery_requested_frame: int | None
    recovery_requested_device_timestamp_us: int | None
    recovery_actual_frame: int | None
    recovery_actual_device_timestamp_us: int | None


@dataclass
class CupWorldMetrics:
    status: MetricStatus
    expected_observations: int
    candidate_world_valid_count: int
    availability_ratio: float | None
    position_error: DistributionSummary = field(default_factory=DistributionSummary)
    first_valid_cup2_world_error: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "expected_observations": self.expected_observations,
            "candidate_world_valid_count": self.candidate_world_valid_count,
            "availability_ratio": self.availability_ratio,
            "position_error": self.position_error.as_dict(),
            "first_valid_cup2_world_error": self.first_valid_cup2_world_error,
        }


@dataclass
class PoseWindowMetrics:
    translation_error: DistributionSummary = field(default_factory=DistributionSummary)
    rotation_error_deg: DistributionSummary = field(default_factory=DistributionSummary)
    availability: AvailabilityMetrics | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "translation_error_m": self.translation_error.as_dict(),
            "rotation_error_deg": self.rotation_error_deg.as_dict(),
        }
        if self.availability is not None:
            payload["availability"] = self.availability.as_dict()
        return payload


@dataclass
class EvaluationResult:
    window_id: str
    algorithm_id: str
    status: MetricStatus
    pose: PoseWindowMetrics
    recovery: RecoveryMetrics
    cup2_world: CupWorldMetrics
    threshold_comparison: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "algorithm_id": self.algorithm_id,
            "status": self.status.value,
            "pose": self.pose.as_dict(),
            "recovery": self.recovery.as_dict(),
            "cup2_world": self.cup2_world.as_dict(),
            "threshold_comparison": self.threshold_comparison,
        }


def _as_transform(T: np.ndarray | None) -> np.ndarray | None:
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64)


def translation_error(reference: PoseReference, candidate: PoseEstimate) -> float | None:
    if not reference.valid or not candidate.valid:
        return None
    T_ref = _as_transform(reference.T_world_camera)
    T_cand = _as_transform(candidate.T_world_camera)
    if T_ref is None or T_cand is None:
        return None
    return float(np.linalg.norm(T_ref[:3, 3] - T_cand[:3, 3]))


def rotation_error_deg(reference: PoseReference, candidate: PoseEstimate) -> float | None:
    if not reference.valid or not candidate.valid:
        return None
    T_ref = _as_transform(reference.T_world_camera)
    T_cand = _as_transform(candidate.T_world_camera)
    if T_ref is None or T_cand is None:
        return None
    R_err = T_ref[:3, :3].T @ T_cand[:3, :3]
    trace = float(np.trace(R_err))
    cos_angle = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def _percentile(values: Sequence[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, percentile))


def _distribution_summary(
    values: Sequence[float],
    *,
    end_of_dropout_value: float | None = None,
    last_valid_before_recovery_value: float | None = None,
) -> DistributionSummary:
    if not values:
        return DistributionSummary(
            end_of_dropout_error=end_of_dropout_value,
            last_valid_before_recovery_error=last_valid_before_recovery_value,
            sample_count=0,
        )
    return DistributionSummary(
        median=float(np.median(values)),
        p90=_percentile(values, 90.0),
        p95=_percentile(values, 95.0),
        max=float(np.max(values)),
        end_of_dropout_error=end_of_dropout_value,
        last_valid_before_recovery_error=last_valid_before_recovery_value,
        sample_count=len(values),
    )


def _pose_by_frame(poses: Sequence[PoseReference | PoseEstimate]) -> dict[int, PoseReference | PoseEstimate]:
    return {pose.frame_number: pose for pose in poses}


def _masked_frame_numbers(window: DropoutWindow, poses: Sequence[PoseReference | PoseEstimate]) -> list[int]:
    masked: list[int] = []
    for pose in poses:
        if is_runtime_tag_masked(pose.device_timestamp_us, window):
            masked.append(pose.frame_number)
    masked.sort()
    return masked


def compute_pose_window_metrics(
    *,
    window: DropoutWindow,
    references: Sequence[PoseReference],
    candidates: Sequence[PoseEstimate],
) -> PoseWindowMetrics:
    ref_by_frame = _pose_by_frame(references)
    cand_by_frame = _pose_by_frame(candidates)
    masked_frames = _masked_frame_numbers(window, references)

    translation_values: list[float] = []
    rotation_values: list[float] = []
    for frame_number in masked_frames:
        reference = ref_by_frame.get(frame_number)
        candidate = cand_by_frame.get(frame_number)
        if reference is None or candidate is None:
            continue
        t_err = translation_error(reference, candidate)
        r_err = rotation_error_deg(reference, candidate)
        if t_err is not None:
            translation_values.append(t_err)
        if r_err is not None:
            rotation_values.append(r_err)

    end_frame = window.end_frame
    end_reference = ref_by_frame.get(end_frame)
    end_candidate = cand_by_frame.get(end_frame)
    end_t = (
        translation_error(end_reference, end_candidate)
        if end_reference is not None and end_candidate is not None and end_candidate.valid
        else None
    )
    end_r = (
        rotation_error_deg(end_reference, end_candidate)
        if end_reference is not None and end_candidate is not None and end_candidate.valid
        else None
    )

    last_valid_t: float | None = None
    last_valid_r: float | None = None
    for frame_number in reversed(masked_frames):
        reference = ref_by_frame.get(frame_number)
        candidate = cand_by_frame.get(frame_number)
        if reference is None or candidate is None or not candidate.valid:
            continue
        t_err = translation_error(reference, candidate)
        r_err = rotation_error_deg(reference, candidate)
        if t_err is not None:
            last_valid_t = t_err
        if r_err is not None:
            last_valid_r = r_err
        if last_valid_t is not None and last_valid_r is not None:
            break

    availability = compute_availability_metrics(window=window, candidates=candidates)
    return PoseWindowMetrics(
        translation_error=_distribution_summary(
            translation_values,
            end_of_dropout_value=end_t,
            last_valid_before_recovery_value=last_valid_t,
        ),
        rotation_error_deg=_distribution_summary(
            rotation_values,
            end_of_dropout_value=end_r,
            last_valid_before_recovery_value=last_valid_r,
        ),
        availability=availability,
    )


def compute_availability_metrics(
    *,
    window: DropoutWindow,
    candidates: Sequence[PoseEstimate],
) -> AvailabilityMetrics:
    masked_candidates = [pose for pose in candidates if is_runtime_tag_masked(pose.device_timestamp_us, window)]
    expected_frames = len(masked_candidates)
    valid_pose_frames = sum(1 for pose in masked_candidates if pose.valid)
    availability_ratio = valid_pose_frames / expected_frames if expected_frames else 0.0

    lost_event_count = 0
    longest_invalid_gap_frames = 0
    longest_invalid_gap_sec = 0.0
    tracking_lost_frames = 0

    current_gap = 0
    current_gap_sec = 0.0
    previous_ts: int | None = None
    for pose in masked_candidates:
        if pose.state == PoseTrackingState.TRACKING_LOST:
            tracking_lost_frames += 1
        if not pose.valid:
            current_gap += 1
            if previous_ts is not None:
                current_gap_sec += max((pose.device_timestamp_us - previous_ts) / 1_000_000.0, 0.0)
            longest_invalid_gap_frames = max(longest_invalid_gap_frames, current_gap)
            longest_invalid_gap_sec = max(longest_invalid_gap_sec, current_gap_sec)
        else:
            if current_gap > 0:
                lost_event_count += 1
            current_gap = 0
            current_gap_sec = 0.0
        previous_ts = pose.device_timestamp_us
    if current_gap > 0:
        lost_event_count += 1

    return AvailabilityMetrics(
        expected_frames=expected_frames,
        valid_pose_frames=valid_pose_frames,
        availability_ratio=availability_ratio,
        lost_event_count=lost_event_count,
        longest_invalid_gap_frames=longest_invalid_gap_frames,
        longest_invalid_gap_sec=longest_invalid_gap_sec,
        tracking_lost_frames=tracking_lost_frames,
    )


def _last_valid_candidate_before(
    candidates: Sequence[PoseEstimate],
    before_frame: int,
) -> PoseEstimate | None:
    eligible = [candidate for candidate in candidates if candidate.valid and candidate.frame_number < before_frame]
    if not eligible:
        return None
    return max(eligible, key=lambda candidate: candidate.frame_number)


def _find_actual_recovery_frame(candidates: Sequence[PoseEstimate]) -> int | None:
    relocalized = [
        candidate
        for candidate in candidates
        if candidate.state == PoseTrackingState.RELOCALIZED and candidate.valid
    ]
    if not relocalized:
        return None
    return min(relocalized, key=lambda candidate: candidate.frame_number).frame_number


def resolve_recovery_timing(
    *,
    window: DropoutWindow,
    candidates: Sequence[PoseEstimate],
) -> RecoveryTiming:
    actual_frame = _find_actual_recovery_frame(candidates)
    actual_ts: int | None = None
    if actual_frame is not None:
        actual_candidate = _pose_by_frame(candidates).get(actual_frame)
        if actual_candidate is not None:
            actual_ts = actual_candidate.device_timestamp_us
    return RecoveryTiming(
        recovery_requested_frame=window.recovery_frame,
        recovery_requested_device_timestamp_us=window.recovery_device_timestamp_us,
        recovery_actual_frame=actual_frame,
        recovery_actual_device_timestamp_us=actual_ts,
    )


def _candidate_reanchor_jump(
    before: PoseEstimate,
    after: PoseEstimate,
) -> tuple[float | None, float | None]:
    if (
        not before.valid
        or not after.valid
        or before.T_world_camera is None
        or after.T_world_camera is None
    ):
        return None, None
    T_before = _as_transform(before.T_world_camera)
    T_after = _as_transform(after.T_world_camera)
    assert T_before is not None and T_after is not None
    reanchor_t = float(np.linalg.norm(T_after[:3, 3] - T_before[:3, 3]))
    R_jump = T_before[:3, :3].T @ T_after[:3, :3]
    trace = float(np.trace(R_jump))
    cos_angle = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    reanchor_r = math.degrees(math.acos(cos_angle))
    return reanchor_t, reanchor_r


def compute_recovery_metrics(
    *,
    window: DropoutWindow,
    references: Sequence[PoseReference],
    candidates: Sequence[PoseEstimate],
    recovery_timing: RecoveryTiming | None = None,
) -> RecoveryMetrics:
    timing = recovery_timing or resolve_recovery_timing(window=window, candidates=candidates)
    if timing.recovery_requested_frame is None:
        return RecoveryMetrics(
            recovery_requested_frame=None,
            recovery_actual_frame=None,
            recovery_latency_frames=None,
            recovery_latency_sec=None,
            pre_recovery_translation_error=None,
            pre_recovery_rotation_error=None,
            reanchor_translation_jump=None,
            reanchor_rotation_jump=None,
            post_recovery_translation_error=None,
            post_recovery_rotation_error=None,
            status=MetricStatus.INVALID,
        )

    requested_frame = timing.recovery_requested_frame
    actual_frame = timing.recovery_actual_frame
    if actual_frame is None:
        return RecoveryMetrics(
            recovery_requested_frame=requested_frame,
            recovery_actual_frame=None,
            recovery_latency_frames=None,
            recovery_latency_sec=None,
            pre_recovery_translation_error=None,
            pre_recovery_rotation_error=None,
            reanchor_translation_jump=None,
            reanchor_rotation_jump=None,
            post_recovery_translation_error=None,
            post_recovery_rotation_error=None,
            status=MetricStatus.OK,
        )

    recovery_latency_frames = actual_frame - requested_frame
    recovery_latency_sec: float | None = None
    if (
        timing.recovery_requested_device_timestamp_us is not None
        and timing.recovery_actual_device_timestamp_us is not None
    ):
        recovery_latency_sec = (
            timing.recovery_actual_device_timestamp_us
            - timing.recovery_requested_device_timestamp_us
        ) / 1_000_000.0

    ref_by_frame = _pose_by_frame(references)
    cand_by_frame = _pose_by_frame(candidates)
    pre_candidate = _last_valid_candidate_before(candidates, actual_frame)
    actual_candidate = cand_by_frame.get(actual_frame)
    actual_reference = ref_by_frame.get(actual_frame)

    pre_t: float | None = None
    pre_r: float | None = None
    if pre_candidate is not None:
        pre_reference = ref_by_frame.get(pre_candidate.frame_number)
        if pre_reference is not None:
            pre_t = translation_error(pre_reference, pre_candidate)
            pre_r = rotation_error_deg(pre_reference, pre_candidate)

    post_t: float | None = None
    post_r: float | None = None
    if actual_reference is not None and actual_candidate is not None and actual_candidate.valid:
        post_t = translation_error(actual_reference, actual_candidate)
        post_r = rotation_error_deg(actual_reference, actual_candidate)

    reanchor_t: float | None = None
    reanchor_r: float | None = None
    if pre_candidate is not None and actual_candidate is not None:
        reanchor_t, reanchor_r = _candidate_reanchor_jump(pre_candidate, actual_candidate)

    return RecoveryMetrics(
        recovery_requested_frame=requested_frame,
        recovery_actual_frame=actual_frame,
        recovery_latency_frames=recovery_latency_frames,
        recovery_latency_sec=recovery_latency_sec,
        pre_recovery_translation_error=pre_t,
        pre_recovery_rotation_error=pre_r,
        reanchor_translation_jump=reanchor_t,
        reanchor_rotation_jump=reanchor_r,
        post_recovery_translation_error=post_t,
        post_recovery_rotation_error=post_r,
        status=MetricStatus.OK,
    )


def transform_point_camera_to_world(T_world_camera: np.ndarray, P_camera: np.ndarray) -> np.ndarray:
    T = _as_transform(T_world_camera)
    P = np.asarray(P_camera, dtype=np.float64).reshape(3)
    assert T is not None
    homogeneous = np.array([P[0], P[1], P[2], 1.0], dtype=np.float64)
    world = T @ homogeneous
    return world[:3]


def cup_world_position_error(
    *,
    reference_pose: PoseReference,
    candidate_pose: PoseEstimate,
    observation: CupObservation,
) -> float | None:
    if not observation.valid or observation.semantic_id != "cup2":
        return None
    if not reference_pose.valid or not candidate_pose.valid:
        return None
    if reference_pose.T_world_camera is None or candidate_pose.T_world_camera is None:
        return None
    P_world_ref = transform_point_camera_to_world(reference_pose.T_world_camera, observation.P_camera)
    P_world_cand = transform_point_camera_to_world(candidate_pose.T_world_camera, observation.P_camera)
    return float(np.linalg.norm(P_world_ref - P_world_cand))


def compute_cup2_world_metrics(
    *,
    window: DropoutWindow,
    references: Sequence[PoseReference],
    candidates: Sequence[PoseEstimate],
    observations: Sequence[CupObservation],
    cup2_semantic_id: str = "cup2",
) -> CupWorldMetrics:
    ref_by_frame = _pose_by_frame(references)
    cand_by_frame = _pose_by_frame(candidates)

    window_observations = [
        obs
        for obs in observations
        if obs.semantic_id == cup2_semantic_id
        and is_runtime_tag_masked(obs.device_timestamp_us, window)
        and obs.valid
    ]
    if not window_observations:
        return CupWorldMetrics(
            status=MetricStatus.NOT_APPLICABLE,
            expected_observations=0,
            candidate_world_valid_count=0,
            availability_ratio=None,
        )

    errors: list[float] = []
    candidate_world_valid_count = 0
    first_valid_error: float | None = None
    for observation in sorted(window_observations, key=lambda item: item.frame_number):
        reference = ref_by_frame.get(observation.frame_number)
        candidate = cand_by_frame.get(observation.frame_number)
        if reference is None or candidate is None:
            continue
        if candidate.valid and candidate.T_world_camera is not None:
            candidate_world_valid_count += 1
        err = cup_world_position_error(
            reference_pose=reference,
            candidate_pose=candidate,
            observation=observation,
        )
        if err is not None:
            errors.append(err)
            if first_valid_error is None and candidate.valid:
                first_valid_error = err

    availability_ratio = (
        candidate_world_valid_count / len(window_observations) if window_observations else None
    )
    return CupWorldMetrics(
        status=MetricStatus.OK,
        expected_observations=len(window_observations),
        candidate_world_valid_count=candidate_world_valid_count,
        availability_ratio=availability_ratio,
        position_error=_distribution_summary(errors),
        first_valid_cup2_world_error=first_valid_error,
    )


def compare_thresholds(
    *,
    pose_metrics: PoseWindowMetrics,
    cup2_metrics: CupWorldMetrics,
    thresholds: SuccessThresholds,
) -> dict[str, Any]:
    availability = pose_metrics.availability.availability_ratio if pose_metrics.availability else None
    lost_events = pose_metrics.availability.lost_event_count if pose_metrics.availability else None

    cup2_median = cup2_metrics.position_error.median
    cup2_p90 = cup2_metrics.position_error.p90

    return {
        "pose_availability_pass": (
            availability is not None and availability >= thresholds.pose_availability_min
        ),
        "major_tracking_lost_pass": (
            lost_events is not None and lost_events <= thresholds.major_tracking_lost_max
        ),
        "cup2_median_pass": (
            cup2_metrics.status == MetricStatus.NOT_APPLICABLE
            or (cup2_median is not None and cup2_median <= thresholds.cup2_world_median_max_m)
        ),
        "cup2_p90_pass": (
            cup2_metrics.status == MetricStatus.NOT_APPLICABLE
            or (cup2_p90 is not None and cup2_p90 <= thresholds.cup2_world_p90_max_m)
        ),
    }


def evaluate_window(
    *,
    window: DropoutWindow,
    references: Sequence[PoseReference],
    candidates: Sequence[PoseEstimate],
    observations: Sequence[CupObservation] | None = None,
    algorithm_id: str = "",
    thresholds: SuccessThresholds | None = None,
    cup2_semantic_id: str = "cup2",
    recovery_timing: RecoveryTiming | None = None,
) -> EvaluationResult:
    pose_metrics = compute_pose_window_metrics(
        window=window,
        references=references,
        candidates=candidates,
    )
    recovery_metrics = compute_recovery_metrics(
        window=window,
        references=references,
        candidates=candidates,
        recovery_timing=recovery_timing,
    )
    cup2_metrics = compute_cup2_world_metrics(
        window=window,
        references=references,
        candidates=candidates,
        observations=observations or [],
        cup2_semantic_id=cup2_semantic_id,
    )

    threshold_comparison: dict[str, Any] = {}
    if thresholds is not None:
        threshold_comparison = compare_thresholds(
            pose_metrics=pose_metrics,
            cup2_metrics=cup2_metrics,
            thresholds=thresholds,
        )

    status = MetricStatus.OK
    if pose_metrics.availability and pose_metrics.availability.expected_frames == 0:
        status = MetricStatus.INVALID

    return EvaluationResult(
        window_id=window.window_id,
        algorithm_id=algorithm_id,
        status=status,
        pose=pose_metrics,
        recovery=recovery_metrics,
        cup2_world=cup2_metrics,
        threshold_comparison=threshold_comparison,
    )


def make_transform(translation: Sequence[float], rotation_deg: float = 0.0, axis: str = "z") -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    if rotation_deg != 0.0:
        theta = math.radians(rotation_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        if axis == "z":
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        elif axis == "y":
            R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        else:
            R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
        T[:3, :3] = R
    return T


def apply_known_translation_offset(
    references: Sequence[PoseReference],
    offset_m: Sequence[float],
) -> list[PoseEstimate]:
    offset = np.asarray(offset_m, dtype=np.float64)
    estimates: list[PoseEstimate] = []
    for reference in references:
        T = None
        if reference.valid and reference.T_world_camera is not None:
            T = reference.T_world_camera.copy()
            T[:3, 3] = T[:3, 3] + offset
        estimates.append(
            PoseEstimate(
                frame_number=reference.frame_number,
                device_timestamp_us=reference.device_timestamp_us,
                T_world_camera=T,
                valid=reference.valid,
                state=PoseTrackingState.LOCAL_TRACKING,
                algorithm_id="synthetic_offset",
            )
        )
    return estimates


def apply_known_rotation_offset_deg(
    references: Sequence[PoseReference],
    rotation_deg: float,
    *,
    axis: str = "z",
) -> list[PoseEstimate]:
    theta = math.radians(rotation_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == "z":
        R_offset = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    elif axis == "y":
        R_offset = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    else:
        R_offset = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    estimates: list[PoseEstimate] = []
    for reference in references:
        T = None
        if reference.valid and reference.T_world_camera is not None:
            T = reference.T_world_camera.copy()
            T[:3, :3] = T[:3, :3] @ R_offset
        estimates.append(
            PoseEstimate(
                frame_number=reference.frame_number,
                device_timestamp_us=reference.device_timestamp_us,
                T_world_camera=T,
                valid=reference.valid,
                state=PoseTrackingState.LOCAL_TRACKING,
                algorithm_id="synthetic_offset",
            )
        )
    return estimates


def references_to_perfect_candidates(
    references: Sequence[PoseReference],
    *,
    algorithm_id: str = "perfect_reference",
) -> list[PoseEstimate]:
    return [
        PoseEstimate(
            frame_number=reference.frame_number,
            device_timestamp_us=reference.device_timestamp_us,
            T_world_camera=None if reference.T_world_camera is None else reference.T_world_camera.copy(),
            valid=reference.valid,
            state=PoseTrackingState.TAG_ANCHORED,
            algorithm_id=algorithm_id,
        )
        for reference in references
    ]
