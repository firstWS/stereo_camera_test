from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.object_annotations import (  # noqa: E402
    build_track_to_semantic_map,
    semantic_id_for_track,
    validate_object_annotations,
    write_object_annotations,
)


def test_missing_annotation_maps_to_unknown() -> None:
    assert build_track_to_semantic_map(None) == {}
    assert semantic_id_for_track("track_0001", {}) == "unknown"


def test_annotation_applies_cup1_cup2() -> None:
    payload = {
        "schema_version": 1,
        "objects": {
            "cup1": {
                "description": "red mug",
                "visible_initially": True,
                "track_id": "track_0001",
            },
            "cup2": {
                "description": "transparent cup",
                "visible_initially": False,
                "track_id": "track_0002",
            },
        },
    }
    assert validate_object_annotations(payload) == []
    mapping = build_track_to_semantic_map(payload)
    assert semantic_id_for_track("track_0001", mapping) == "cup1"
    assert semantic_id_for_track("track_0002", mapping) == "cup2"
    assert semantic_id_for_track("track_0003", mapping) == "unknown"


def test_semantic_and_track_id_remain_separate(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "objects": {
            "cup1": {"track_id": "track_0002"},
            "cup2": {"track_id": "track_0001"},
        },
    }
    write_object_annotations(tmp_path, payload)
    mapping = build_track_to_semantic_map(payload)
    assert mapping["track_0001"] == "cup2"
    assert mapping["track_0002"] == "cup1"
