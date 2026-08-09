"""Lightweight offline cup MOT: Hungarian assignment with track lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

INFINITE_COST = 1e6


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DELETED = "DELETED"


@dataclass(frozen=True)
class CupDetectionRecord:
    frame_number: int
    device_timestamp_us: int | str
    detection_index: int
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    depth_valid: bool = False
    camera_x: float | str = ""
    camera_y: float | str = ""
    camera_z: float | str = ""
    world_valid: bool = False
    world_x: float | str = ""
    world_y: float | str = ""
    world_z: float | str = ""
    file_name: str = ""
    image_width: int = 0
    image_height: int = 0
    depth_frame_number: int | str = ""
    depth_device_timestamp_us: int | str = ""
    rgb_depth_delta_us: int | str = ""


@dataclass(frozen=True)
class TrackAssignment:
    frame_number: int
    device_timestamp_us: int | str
    detection_index: int
    track_id: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class _Track:
    track_id: str
    state: TrackState
    last_bbox: tuple[float, float, float, float]
    last_frame: int
    first_frame: int
    image_width: int
    image_height: int
    hit_count: int = 0
    consecutive_hits: int = 0
    miss_streak: int = 0
    lost_frames: int = 0
    lost_count: int = 0
    reactivation_count: int = 0
    center_velocity: tuple[float, float] = (0.0, 0.0)
    last_center: tuple[float, float] | None = None
    world_valid: bool = False
    world_position: tuple[float, float, float] | None = None

    def predicted_bbox(self, frame_number: int) -> tuple[float, float, float, float]:
        if self.last_center is None or frame_number <= self.last_frame:
            return self.last_bbox
        dt = float(frame_number - self.last_frame)
        vx, vy = self.center_velocity
        dx = vx * dt
        dy = vy * dt
        x1, y1, x2, y2 = self.last_bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


@dataclass(frozen=True)
class TrackSummary:
    track_id: str
    first_seen_frame: int
    last_seen_frame: int
    detection_count: int
    mean_confidence: float
    representative_frame: int
    final_state: str
    lost_count: int
    reactivation_count: int
    max_gap_frames: int
    semantic_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "detection_count": self.detection_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "representative_frame": self.representative_frame,
            "final_state": self.final_state,
            "lost_count": self.lost_count,
            "reactivation_count": self.reactivation_count,
            "max_gap_frames": self.max_gap_frames,
            "semantic_hint": self.semantic_hint,
        }


@dataclass(frozen=True)
class AssociationConfig:
    w_iou: float = 0.35
    w_center: float = 0.35
    w_scale: float = 0.20
    w_world: float = 0.10
    center_dist_threshold: float = 0.25
    scale_diff_threshold: float = 0.45
    world_dist_threshold_m: float = 0.15
    max_assignment_cost: float = 0.85
    reactivation_max_cost: float = 0.70
    min_confirm_hits: int = 2
    max_tentative_miss_frames: int = 2
    max_lost_frames: int = 30
    use_world_cue: bool = True

    @property
    def max_miss_frames(self) -> int:
        """Backward-compatible alias for tests/config migration."""
        return self.max_lost_frames


def association_config_from_mapping(raw: Mapping[str, Any] | None) -> AssociationConfig:
    if not raw:
        return AssociationConfig()
    defaults = AssociationConfig()
    max_lost = raw.get("max_lost_frames", raw.get("max_miss_frames", defaults.max_lost_frames))
    return AssociationConfig(
        w_iou=float(raw.get("w_iou", defaults.w_iou)),
        w_center=float(raw.get("w_center", defaults.w_center)),
        w_scale=float(raw.get("w_scale", defaults.w_scale)),
        w_world=float(raw.get("w_world", defaults.w_world)),
        center_dist_threshold=float(
            raw.get("center_dist_threshold", defaults.center_dist_threshold)
        ),
        scale_diff_threshold=float(
            raw.get("scale_diff_threshold", defaults.scale_diff_threshold)
        ),
        world_dist_threshold_m=float(
            raw.get("world_dist_threshold_m", defaults.world_dist_threshold_m)
        ),
        max_assignment_cost=float(
            raw.get("max_assignment_cost", defaults.max_assignment_cost)
        ),
        reactivation_max_cost=float(
            raw.get("reactivation_max_cost", defaults.reactivation_max_cost)
        ),
        min_confirm_hits=int(raw.get("min_confirm_hits", defaults.min_confirm_hits)),
        max_tentative_miss_frames=int(
            raw.get("max_tentative_miss_frames", defaults.max_tentative_miss_frames)
        ),
        max_lost_frames=int(max_lost),
        use_world_cue=bool(raw.get("use_world_cue", defaults.use_world_cue)),
    )


def order_cup_boxes(boxes: Sequence[Any]) -> list[Any]:
    return sorted(
        boxes,
        key=lambda box: (
            -float(box.confidence),
            float(box.xyxy[0]),
            float(box.xyxy[1]),
            float(box.xyxy[2]),
            float(box.xyxy[3]),
        ),
    )


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_size(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def normalized_center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> float:
    if image_width <= 0 or image_height <= 0:
        return 1.0
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    dx = (ax - bx) / float(image_width)
    dy = (ay - by) / float(image_height)
    return (dx * dx + dy * dy) ** 0.5


def bbox_scale_difference(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    aw, ah = bbox_size(a)
    bw, bh = bbox_size(b)
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 1.0
    width_ratio = abs(aw - bw) / max(aw, bw)
    height_ratio = abs(ah - bh) / max(ah, bh)
    return max(width_ratio, height_ratio)


def _world_position(det: CupDetectionRecord) -> tuple[float, float, float] | None:
    if not det.world_valid:
        return None
    try:
        return (float(det.world_x), float(det.world_y), float(det.world_z))
    except (TypeError, ValueError):
        return None


def association_cost(
    det: CupDetectionRecord,
    track: _Track,
    *,
    frame_number: int,
    config: AssociationConfig,
    reactivation: bool = False,
) -> float:
    image_width = det.image_width or track.image_width
    image_height = det.image_height or track.image_height
    predicted_bbox = track.predicted_bbox(frame_number)
    iou = bbox_iou(det.bbox, predicted_bbox)
    center_dist = normalized_center_distance(
        det.bbox,
        predicted_bbox,
        image_width=image_width,
        image_height=image_height,
    )
    scale_diff = bbox_scale_difference(det.bbox, track.last_bbox)
    center_term = min(1.0, center_dist / config.center_dist_threshold)
    scale_term = min(1.0, scale_diff / config.scale_diff_threshold)
    cost = (
        config.w_iou * (1.0 - iou)
        + config.w_center * center_term
        + config.w_scale * scale_term
    )
    world_dist: float | None = None
    if config.use_world_cue and track.world_valid:
        det_world = _world_position(det)
        if det_world is not None and track.world_position is not None:
            world_dist = float(
                np.linalg.norm(np.asarray(det_world) - np.asarray(track.world_position))
            )
            world_term = min(1.0, world_dist / config.world_dist_threshold_m)
            cost += config.w_world * world_term
    strong_world_match = (
        world_dist is not None and world_dist <= config.world_dist_threshold_m * 0.35
    )
    max_cost = config.reactivation_max_cost if reactivation else config.max_assignment_cost
    if not strong_world_match:
        center_gate = config.center_dist_threshold * (0.85 if reactivation else 1.0)
        scale_gate = config.scale_diff_threshold * (0.85 if reactivation else 1.0)
        if center_dist > center_gate:
            return INFINITE_COST
        if scale_diff > scale_gate:
            return INFINITE_COST
    if cost > max_cost:
        return INFINITE_COST
    return cost


def match_score(
    detection_bbox: tuple[float, float, float, float],
    track_bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    config: AssociationConfig,
) -> float:
    """Legacy score helper retained for unit tests."""
    iou = bbox_iou(detection_bbox, track_bbox)
    center_dist = normalized_center_distance(
        detection_bbox,
        track_bbox,
        image_width=image_width,
        image_height=image_height,
    )
    center_similarity = max(0.0, 1.0 - center_dist / config.center_dist_threshold)
    score = config.w_iou * iou + config.w_center * center_similarity
    if iou >= 0.25:
        return max(score, 0.35)
    if center_dist <= config.center_dist_threshold:
        return max(center_similarity, score)
    return 0.0


def _format_track_id(track_number: int) -> str:
    return f"track_{track_number:04d}"


def _update_track_motion(track: _Track, det: CupDetectionRecord, frame_number: int) -> None:
    new_center = bbox_center(det.bbox)
    if track.last_center is not None and frame_number > track.last_frame:
        dt = float(frame_number - track.last_frame)
        if dt > 0.0:
            vx = (new_center[0] - track.last_center[0]) / dt
            vy = (new_center[1] - track.last_center[1]) / dt
            track.center_velocity = (vx, vy)
    track.last_center = new_center
    track.last_bbox = det.bbox
    track.last_frame = frame_number
    track.image_width = det.image_width or track.image_width
    track.image_height = det.image_height or track.image_height
    world = _world_position(det)
    if world is not None:
        track.world_valid = True
        track.world_position = world


def _hungarian_assign(
    detections: Sequence[CupDetectionRecord],
    tracks: Sequence[_Track],
    *,
    frame_number: int,
    config: AssociationConfig,
) -> list[tuple[int, int]]:
    if not detections or not tracks:
        return []
    cost_matrix = np.full((len(detections), len(tracks)), INFINITE_COST, dtype=np.float64)
    for det_index, det in enumerate(detections):
        for track_index, track in enumerate(tracks):
            reactivation = track.state == TrackState.LOST
            cost_matrix[det_index, track_index] = association_cost(
                det,
                track,
                frame_number=frame_number,
                config=config,
                reactivation=reactivation,
            )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs: list[tuple[int, int]] = []
    for det_index, track_index in zip(row_ind, col_ind, strict=True):
        if cost_matrix[det_index, track_index] < INFINITE_COST:
            pairs.append((det_index, track_index))
    return pairs


def _age_track_on_miss(track: _Track, frame_number: int, cfg: AssociationConfig) -> None:
    if track.state == TrackState.DELETED:
        return
    if frame_number <= track.last_frame:
        return
    track.miss_streak += 1
    track.consecutive_hits = 0
    if track.state == TrackState.TENTATIVE:
        if track.hit_count >= 1:
            track.state = TrackState.LOST
            track.lost_count += 1
            track.lost_frames = frame_number - track.last_frame
        elif track.miss_streak > cfg.max_tentative_miss_frames:
            track.state = TrackState.DELETED
        return
    if track.state == TrackState.CONFIRMED:
        track.state = TrackState.LOST
        track.lost_count += 1
        track.lost_frames = frame_number - track.last_frame
        return
    if track.state == TrackState.LOST:
        track.lost_frames = frame_number - track.last_frame
        if track.lost_frames > cfg.max_lost_frames:
            track.state = TrackState.DELETED


def associate_detections_to_tracks(
    detections: Sequence[CupDetectionRecord],
    *,
    config: AssociationConfig | None = None,
) -> tuple[list[TrackAssignment], list[TrackSummary], dict[str, Any]]:
    cfg = config or AssociationConfig()
    by_frame: dict[int, list[CupDetectionRecord]] = {}
    for det in detections:
        by_frame.setdefault(int(det.frame_number), []).append(det)
    if not by_frame:
        return [], [], {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "tentative_only_tracks": 0,
            "reactivated_tracks": 0,
            "deleted_tracks": 0,
        }
    frame_numbers = range(min(by_frame), max(by_frame) + 1)

    tracks: list[_Track] = []
    next_track_number = 1
    assignments: list[TrackAssignment] = []
    track_stats: dict[str, dict[str, Any]] = {}

    for frame_number in frame_numbers:
        frame_dets = sorted(
            by_frame.get(frame_number, []),
            key=lambda d: d.detection_index,
        )
        candidate_tracks = [
            track
            for track in tracks
            if track.state in {TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST}
            and (
                track.state != TrackState.LOST
                or (frame_number - track.last_frame) <= cfg.max_lost_frames
            )
        ]
        pairs = _hungarian_assign(
            frame_dets,
            candidate_tracks,
            frame_number=frame_number,
            config=cfg,
        )
        matched_det: set[int] = set()
        matched_track: set[str] = set()

        for det_index, track_index in pairs:
            det = frame_dets[det_index]
            track = candidate_tracks[track_index]
            matched_det.add(det.detection_index)
            matched_track.add(track.track_id)
            if track.state == TrackState.LOST:
                track.reactivation_count += 1
            track.hit_count += 1
            track.consecutive_hits += 1
            track.miss_streak = 0
            track.lost_frames = 0
            if track.hit_count >= cfg.min_confirm_hits:
                track.state = TrackState.CONFIRMED
            elif track.state == TrackState.LOST:
                track.state = TrackState.TENTATIVE
            _update_track_motion(track, det, frame_number)
            assignments.append(
                TrackAssignment(
                    frame_number=frame_number,
                    device_timestamp_us=det.device_timestamp_us,
                    detection_index=det.detection_index,
                    track_id=track.track_id,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )
            stats = track_stats[track.track_id]
            stats["frames"].append(frame_number)
            stats["confidences"].append(det.confidence)

        for det in frame_dets:
            if det.detection_index in matched_det:
                continue
            track_id = _format_track_id(next_track_number)
            next_track_number += 1
            new_track = _Track(
                track_id=track_id,
                state=TrackState.TENTATIVE,
                last_bbox=det.bbox,
                last_frame=frame_number,
                first_frame=frame_number,
                image_width=det.image_width,
                image_height=det.image_height,
                hit_count=1,
                consecutive_hits=1,
            )
            _update_track_motion(new_track, det, frame_number)
            tracks.append(new_track)
            assignments.append(
                TrackAssignment(
                    frame_number=frame_number,
                    device_timestamp_us=det.device_timestamp_us,
                    detection_index=det.detection_index,
                    track_id=track_id,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )
            track_stats[track_id] = {
                "frames": [frame_number],
                "confidences": [det.confidence],
                "lost_count": 0,
                "reactivation_count": 0,
                "final_state": TrackState.TENTATIVE.value,
            }

        for track in tracks:
            if track.track_id in matched_track:
                continue
            if frame_number <= track.last_frame:
                continue
            _age_track_on_miss(track, frame_number, cfg)

    for track in tracks:
        stats = track_stats.setdefault(
            track.track_id,
            {
                "frames": [],
                "confidences": [],
                "lost_count": track.lost_count,
                "reactivation_count": track.reactivation_count,
            },
        )
        stats["lost_count"] = track.lost_count
        stats["reactivation_count"] = track.reactivation_count
        stats["final_state"] = track.state.value

    summaries, aggregate = _build_track_summaries(track_stats)
    return assignments, summaries, aggregate


def _build_track_summaries(
    track_stats: dict[str, dict[str, Any]],
) -> tuple[list[TrackSummary], dict[str, Any]]:
    if not track_stats:
        return [], {
            "total_tracks": 0,
            "confirmed_tracks": 0,
            "tentative_only_tracks": 0,
            "reactivated_tracks": 0,
            "deleted_tracks": 0,
        }
    earliest_first_frame = min(min(stats["frames"]) for stats in track_stats.values())
    summaries: list[TrackSummary] = []
    for track_id in sorted(track_stats):
        stats = track_stats[track_id]
        frames = stats["frames"]
        confidences = stats["confidences"]
        first_seen = min(frames)
        last_seen = max(frames)
        representative = frames[len(frames) // 2]
        mean_conf = sum(confidences) / len(confidences)
        gaps = [frames[index + 1] - frames[index] for index in range(len(frames) - 1)]
        max_gap = max(gaps) if gaps else 0
        if first_seen == earliest_first_frame and len(frames) >= 2:
            hint = "cup1_candidate"
        elif first_seen > earliest_first_frame:
            hint = "cup2_candidate"
        else:
            hint = ""
        summaries.append(
            TrackSummary(
                track_id=track_id,
                first_seen_frame=first_seen,
                last_seen_frame=last_seen,
                detection_count=len(frames),
                mean_confidence=mean_conf,
                representative_frame=representative,
                final_state=str(stats.get("final_state", TrackState.CONFIRMED.value)),
                lost_count=int(stats.get("lost_count", 0)),
                reactivation_count=int(stats.get("reactivation_count", 0)),
                max_gap_frames=max_gap,
                semantic_hint=hint,
            )
        )
    aggregate = {
        "total_tracks": len(summaries),
        "confirmed_tracks": sum(
            1 for summary in summaries if summary.final_state == TrackState.CONFIRMED.value
        ),
        "tentative_only_tracks": sum(
            1
            for summary in summaries
            if summary.final_state == TrackState.TENTATIVE.value and summary.detection_count <= 1
        ),
        "reactivated_tracks": sum(1 for summary in summaries if summary.reactivation_count > 0),
        "deleted_tracks": sum(
            1 for summary in summaries if summary.final_state == TrackState.DELETED.value
        ),
    }
    return summaries, aggregate
