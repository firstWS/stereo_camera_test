from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_config import load_object_anchor_config  # noqa: E402
from yolo_pose_sanitize import (  # noqa: E402
    PoseLabelValidationError,
    cuboid_face_keypoint_ids,
    find_skeleton_crossings,
    infer_named_view,
    sanitize_yolo_pose_line,
)


def _line(*, bbox: str = "0.5 0.5 0.5 0.5", points: list[str] | None = None) -> str:
    points = points or ["0.1 0.1 2"] * 8
    return "0 " + bbox + " " + " ".join(points)


def test_hidden_out_of_range_keypoint_is_zeroed() -> None:
    points = ["0.1 0.1 2"] * 8
    points[3] = "0.795114 2.243069 0"
    result = sanitize_yolo_pose_line(_line(points=points))
    np.testing.assert_array_equal(result.keypoints_xyv[3], [0.0, 0.0, 0.0])
    assert result.hidden_keypoints_rewritten == 1
    assert len(result.to_yolo_line().split()) == 29


def test_visible_out_of_range_keypoint_is_error() -> None:
    points = ["0.1 0.1 2"] * 8
    points[3] = "0.795114 2.243069 1"
    with pytest.raises(PoseLabelValidationError, match="visible keypoint 3"):
        sanitize_yolo_pose_line(_line(points=points))


def test_wrong_value_count_is_error() -> None:
    with pytest.raises(PoseLabelValidationError, match="expected 29 values"):
        sanitize_yolo_pose_line("0 0.5 0.5 0.2 0.2")


def test_front_only_four_keypoint_row_has_17_values() -> None:
    points = ["0.1 0.1 2"] * 4
    result = sanitize_yolo_pose_line(
        _line(points=points), expected_keypoints=4
    )
    assert result.keypoints_xyv.shape == (4, 3)
    assert len(result.to_yolo_line().split()) == 17


@pytest.mark.parametrize(
    "bbox",
    ["1.1 0.5 0.5 0.5", "0.5 -0.1 0.5 0.5", "0.5 0.5 0 0.5"],
)
def test_invalid_bbox_is_error(bbox: str) -> None:
    with pytest.raises(PoseLabelValidationError, match="bbox"):
        sanitize_yolo_pose_line(_line(bbox=bbox))


def test_tissue_box_face_ids_come_from_fixed_xyz() -> None:
    config = load_object_anchor_config(
        ROOT / "configs" / "object_anchors" / "tissue_box_01.yaml"
    )
    faces = cuboid_face_keypoint_ids(config.object_points)
    assert faces == {
        "left": (0, 3, 5, 6),
        "right": (1, 2, 4, 7),
        "front": (0, 1, 2, 3),
        "back": (4, 5, 6, 7),
        "bottom": (2, 3, 6, 7),
        "top": (0, 1, 4, 5),
    }
    assert infer_named_view("tissue_box_top_1") == "top"


def test_crossed_top_face_skeleton_detects_swapped_back_points() -> None:
    config = load_object_anchor_config(
        ROOT / "configs" / "object_anchors" / "tissue_box_01.yaml"
    )
    points = np.zeros((8, 3), dtype=np.float64)
    points[[0, 1, 4, 5], 2] = 2
    points[0, :2] = [0.1, 0.9]
    points[1, :2] = [0.9, 0.9]
    points[4, :2] = [0.1, 0.1]
    points[5, :2] = [0.9, 0.1]
    crossings = find_skeleton_crossings(points, config.skeleton)
    assert crossings == [((0, 5), (1, 4))]

    points[[4, 5], :2] = points[[5, 4], :2]
    assert find_skeleton_crossings(points, config.skeleton) == []
