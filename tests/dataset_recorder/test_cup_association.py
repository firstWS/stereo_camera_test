from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.cup_association import (  # noqa: E402
    AssociationConfig,
    CupDetectionRecord,
    TrackState,
    associate_detections_to_tracks,
    association_cost,
    bbox_iou,
    match_score,
    normalized_center_distance,
    order_cup_boxes,
)
from stereo_types import BBox  # noqa: E402


def _det(
    frame_number: int,
    detection_index: int,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
    world: tuple[float, float, float] | None = None,
) -> CupDetectionRecord:
    return CupDetectionRecord(
        frame_number=frame_number,
        device_timestamp_us=frame_number * 1000,
        detection_index=detection_index,
        class_id=41,
        label="cup",
        confidence=confidence,
        bbox=bbox,
        image_width=640,
        image_height=480,
        world_valid=world is not None,
        world_x=world[0] if world else "",
        world_y=world[1] if world else "",
        world_z=world[2] if world else "",
    )


def _run(detections: list[CupDetectionRecord], config: AssociationConfig | None = None):
    assignments, summaries, aggregate = associate_detections_to_tracks(
        detections,
        config=config,
    )
    return assignments, summaries, aggregate


def test_order_cup_boxes_is_confidence_desc_then_bbox() -> None:
    boxes = [
        BBox(xyxy=(10, 10, 30, 30), confidence=0.7, class_id=41, label="cup"),
        BBox(xyxy=(100, 100, 140, 140), confidence=0.9, class_id=41, label="cup"),
        BBox(xyxy=(50, 50, 80, 80), confidence=0.9, class_id=41, label="cup"),
    ]
    ordered = order_cup_boxes(boxes)
    assert [box.confidence for box in ordered] == [0.9, 0.9, 0.7]
    assert ordered[0].xyxy[0] < ordered[1].xyxy[0]


def test_two_detections_same_frame_produce_two_rows() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140), confidence=0.9),
        _det(0, 1, (300, 100, 340, 140), confidence=0.8),
    ]
    assignments, summaries, _ = _run(detections)
    assert len(assignments) == 2
    assert {row.track_id for row in assignments} == {"track_0001", "track_0002"}
    assert len(summaries) == 2


def test_confidence_order_change_does_not_change_track_ids() -> None:
    frame0 = [
        _det(0, 0, (100, 100, 140, 140), confidence=0.9),
        _det(0, 1, (300, 100, 340, 140), confidence=0.5),
    ]
    frame1 = [
        _det(1, 0, (300, 100, 340, 140), confidence=0.95),
        _det(1, 1, (100, 100, 140, 140), confidence=0.6),
    ]
    assignments, _, _ = _run(frame0 + frame1)
    by_frame = {}
    for row in assignments:
        by_frame.setdefault(row.frame_number, {})[row.detection_index] = row.track_id
    assert by_frame[0][0] == by_frame[1][1]
    assert by_frame[0][1] == by_frame[1][0]


def test_same_cup_keeps_track_id_across_frames() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(1, 0, (110, 105, 150, 145)),
        _det(2, 0, (120, 110, 160, 150)),
    ]
    assignments, summaries, _ = _run(detections)
    assert {row.track_id for row in assignments} == {"track_0001"}
    assert summaries[0].detection_count == 3
    assert summaries[0].final_state == TrackState.CONFIRMED.value


def test_bbox_motion_association_uses_center_when_iou_low() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(1, 0, (180, 100, 220, 140)),
    ]
    assignments, _, _ = _run(detections)
    assert len({row.track_id for row in assignments}) == 1


def test_short_miss_reactivates_same_track() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(1, 0, (102, 102, 142, 142)),
        _det(4, 0, (110, 105, 150, 145)),
    ]
    assignments, summaries, _ = _run(detections)
    assert len({row.track_id for row in assignments}) == 1
    assert summaries[0].reactivation_count >= 1


def test_16_frame_miss_reactivates_same_track() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(1, 0, (102, 102, 142, 142)),
        _det(18, 0, (250, 100, 290, 140)),
    ]
    assignments, summaries, _ = _run(detections, config=AssociationConfig(max_lost_frames=30))
    assert len({row.track_id for row in assignments}) == 1
    assert summaries[0].reactivation_count >= 1


def test_miss_beyond_max_lost_frames_creates_new_track() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(40, 0, (250, 100, 290, 140)),
    ]
    assignments, summaries, _ = _run(detections, config=AssociationConfig(max_lost_frames=30))
    assert len(summaries) == 2
    assert summaries[0].track_id != summaries[1].track_id


def test_new_cup_gets_new_track() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(5, 0, (400, 100, 440, 140)),
    ]
    assignments, summaries, _ = _run(detections)
    assert len(summaries) == 2
    assert summaries[0].first_seen_frame == 0
    assert summaries[1].first_seen_frame == 5


def test_one_detection_not_assigned_to_two_tracks() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(0, 1, (300, 100, 340, 140)),
        _det(1, 0, (105, 105, 145, 145)),
        _det(1, 1, (305, 105, 345, 145)),
    ]
    assignments, _, _ = _run(detections)
    per_frame = {}
    for row in assignments:
        key = (row.frame_number, row.detection_index)
        assert key not in per_frame
        per_frame[key] = row.track_id
    assert len(per_frame) == 4


def test_track_ids_are_deterministic() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(0, 1, (300, 100, 340, 140)),
        _det(1, 0, (105, 105, 145, 145)),
        _det(1, 1, (305, 105, 345, 145)),
    ]
    first, _, _ = _run(detections)
    second, _, _ = _run(detections)
    assert [(r.frame_number, r.detection_index, r.track_id) for r in first] == [
        (r.frame_number, r.detection_index, r.track_id) for r in second
    ]


def test_cup2_partial_entry_and_reactivation() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140), confidence=0.9),
        _det(0, 1, (300, 100, 340, 140), confidence=0.4),
        _det(1, 0, (105, 105, 145, 145), confidence=0.9),
        _det(3, 0, (110, 110, 150, 150), confidence=0.9),
        _det(3, 1, (305, 105, 345, 145), confidence=0.35),
        _det(4, 1, (310, 110, 350, 150), confidence=0.3),
    ]
    assignments, summaries, _ = _run(detections)
    cup2_ids = {
        row.track_id
        for row in assignments
        if row.detection_index == 1 and row.frame_number in {0, 3, 4}
    }
    assert len(cup2_ids) == 1


def test_scenario_a_like_fragmentation_reduces_with_long_lost_window() -> None:
    detections = [
        _det(frame, 0, (100, 100, 140, 140))
        for frame in (1, 50, 100, 150, 200, 250, 300)
    ]
    detections.extend(
        [
            _det(203, 1, (300, 100, 340, 140), confidence=0.29),
            _det(207, 1, (305, 105, 345, 145), confidence=0.28),
            _det(223, 1, (310, 110, 350, 150), confidence=0.41),
            _det(233, 1, (315, 115, 355, 155), confidence=0.40),
            _det(239, 1, (320, 120, 360, 160), confidence=0.43),
            _det(248, 1, (325, 125, 365, 165), confidence=0.78),
            _det(300, 1, (330, 130, 370, 170), confidence=0.77),
        ]
    )
    assignments, summaries, aggregate = _run(detections, config=AssociationConfig(max_lost_frames=30))
    cup2_track_ids = {
        row.track_id
        for row in assignments
        if row.detection_index == 1 and row.frame_number >= 203
    }
    assert len(cup2_track_ids) <= 2
    assert aggregate["reactivated_tracks"] >= 1


def test_tentative_requires_confirmation() -> None:
    detections = [_det(0, 0, (100, 100, 140, 140))]
    _, summaries, _ = _run(detections, config=AssociationConfig(min_confirm_hits=2))
    assert summaries[0].final_state == TrackState.TENTATIVE.value


def test_confirmed_track_goes_lost_then_reactivated() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(1, 0, (102, 102, 142, 142)),
        _det(5, 0, (110, 110, 150, 150)),
    ]
    _, summaries, _ = _run(detections)
    assert summaries[0].lost_count >= 1
    assert summaries[0].reactivation_count >= 1


def test_track_summary_hints_without_auto_semantic_truth() -> None:
    detections = [
        _det(0, 0, (100, 100, 140, 140)),
        _det(1, 0, (105, 105, 145, 145)),
        _det(5, 0, (400, 100, 440, 140)),
    ]
    _, summaries, _ = _run(detections)
    hints = {summary.track_id: summary.semantic_hint for summary in summaries}
    assert hints["track_0001"] == "cup1_candidate"
    assert hints["track_0002"] == "cup2_candidate"


def test_world_cue_helps_low_iou_same_object() -> None:
    cfg = AssociationConfig(use_world_cue=True, w_world=0.5)
    det = _det(10, 0, (300, 100, 340, 140), world=(1.0, 0.2, 2.0))
    from dataset_recorder.cup_association import _Track

    track = _Track(
        track_id="track_0001",
        state=TrackState.LOST,
        last_bbox=(100, 100, 140, 140),
        last_frame=0,
        first_frame=0,
        image_width=640,
        image_height=480,
        world_valid=True,
        world_position=(1.0, 0.2, 2.0),
    )
    cost = association_cost(det, track, frame_number=10, config=cfg, reactivation=True)
    assert cost <= cfg.reactivation_max_cost


def test_world_cue_rejects_different_objects() -> None:
    cfg = AssociationConfig(use_world_cue=True, w_world=0.5)
    det = _det(10, 0, (300, 100, 340, 140), world=(2.0, 0.2, 2.0))
    from dataset_recorder.cup_association import _Track

    track = _Track(
        track_id="track_0001",
        state=TrackState.CONFIRMED,
        last_bbox=(100, 100, 140, 140),
        last_frame=9,
        first_frame=0,
        image_width=640,
        image_height=480,
        world_valid=True,
        world_position=(0.5, 0.2, 2.0),
    )
    cost = association_cost(det, track, frame_number=10, config=cfg)
    assert cost >= 1e5 or cost > cfg.max_assignment_cost


def test_match_score_legacy_helper() -> None:
    bbox_a = (100.0, 100.0, 140.0, 140.0)
    bbox_b = (102.0, 102.0, 142.0, 142.0)
    bbox_far = (250.0, 100.0, 290.0, 140.0)
    cfg = AssociationConfig()
    assert match_score(bbox_a, bbox_b, image_width=640, image_height=480, config=cfg) > 0.3
    assert match_score(bbox_a, bbox_far, image_width=640, image_height=480, config=cfg) < 0.35


def test_bbox_iou_and_center_distance_helpers() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    assert bbox_iou(a, b) > 0.1
    assert normalized_center_distance(a, b, image_width=100, image_height=100) < 0.2
