from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.cup_association import AssociationConfig  # noqa: E402
from dataset_recorder.derive_observations import derive_observations  # noqa: E402
from dataset_recorder.object_annotations import write_object_annotations  # noqa: E402
from dataset_recorder.reader import DatasetReader  # noqa: E402
from apriltag_world import AprilTagWorldObservation, AprilTagWorldResult  # noqa: E402
from stereo_types import BBox, DetectionResult  # noqa: E402

_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures.py"
import importlib.util

_FIXTURES_SPEC = importlib.util.spec_from_file_location("dataset_recorder_fixtures", _FIXTURES_PATH)
assert _FIXTURES_SPEC and _FIXTURES_SPEC.loader
_FIXTURES = importlib.util.module_from_spec(_FIXTURES_SPEC)
_FIXTURES_SPEC.loader.exec_module(_FIXTURES)
build_synthetic_session = _FIXTURES.build_synthetic_session


class FakeDetector:
    def __init__(self, boxes_by_frame: dict[int, list[BBox]] | None = None) -> None:
        self._boxes_by_frame = boxes_by_frame or {}

    def predict(self, bgr: np.ndarray, frame_number: int | None = None) -> DetectionResult:
        h, w = bgr.shape[:2]
        if frame_number is not None and frame_number in self._boxes_by_frame:
            boxes = self._boxes_by_frame[frame_number]
        elif self._boxes_by_frame:
            boxes = next(iter(self._boxes_by_frame.values()))
        else:
            boxes = [
                BBox(
                    xyxy=(w * 0.25, h * 0.25, w * 0.75, h * 0.75),
                    confidence=0.9,
                    class_id=41,
                    label="cup",
                )
            ]
        return DetectionResult(boxes=boxes, image_shape_hw=(h, w))


class FrameAwareDetector:
    def __init__(self, boxes_by_frame: dict[int, list[BBox]]) -> None:
        self._boxes_by_frame = boxes_by_frame
        self._call_index = 0

    def predict(self, bgr: np.ndarray) -> DetectionResult:
        h, w = bgr.shape[:2]
        boxes = self._boxes_by_frame.get(self._call_index, [])
        self._call_index += 1
        return DetectionResult(boxes=boxes, image_shape_hw=(h, w))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_derive_writes_manifest_and_layered_csvs(synthetic_session: Path) -> None:
    manifest = derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FakeDetector(),
    )
    reader = DatasetReader(synthetic_session)
    assert manifest["apriltag_rows"] == 0
    assert manifest["apriltag_enabled"] is False
    assert manifest["cup_detection_rows"] == 5
    assert manifest["cup_track_rows"] == 5
    assert manifest["cup_observation_rows"] == 5
    assert (synthetic_session / "derived" / "manifest.json").is_file()
    assert (synthetic_session / "derived" / "cups" / "detections.csv").is_file()
    assert (synthetic_session / "derived" / "cups" / "tracks.csv").is_file()
    assert (synthetic_session / "derived" / "cups" / "observations.csv").is_file()
    assert (synthetic_session / "derived" / "cups" / "track_summary.json").is_file()
    assert reader.derived_manifest() is not None


def test_two_cups_in_one_frame_create_two_detection_rows(synthetic_session: Path) -> None:
    h, w = 48, 64
    boxes = [
        BBox(xyxy=(10, 10, 30, 30), confidence=0.9, class_id=41, label="cup"),
        BBox(xyxy=(35, 10, 55, 30), confidence=0.7, class_id=41, label="cup"),
    ]
    detector = FrameAwareDetector({index: boxes for index in range(5)})
    manifest = derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=detector,
    )
    detections = _read_csv(synthetic_session / "derived" / "cups" / "detections.csv")
    assert manifest["cup_detection_rows"] == 10
    frame0 = [row for row in detections if row["frame_number"] == "0"]
    assert len(frame0) == 2
    assert frame0[0]["detection_index"] == "0"
    assert frame0[1]["detection_index"] == "1"
    assert float(frame0[0]["confidence"]) > float(frame0[1]["confidence"])


def test_annotation_missing_sets_semantic_unknown(synthetic_session: Path) -> None:
    derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FakeDetector(),
    )
    observations = _read_csv(synthetic_session / "derived" / "cups" / "observations.csv")
    assert observations
    assert all(row["semantic_id"] == "unknown" for row in observations)


def test_annotation_applies_semantic_ids(synthetic_session: Path) -> None:
    h, w = 48, 64
    boxes_by_frame = {
        0: [
            BBox(xyxy=(10, 10, 30, 30), confidence=0.9, class_id=41, label="cup"),
            BBox(xyxy=(35, 10, 55, 30), confidence=0.5, class_id=41, label="cup"),
        ],
        1: [
            BBox(xyxy=(12, 12, 32, 32), confidence=0.9, class_id=41, label="cup"),
            BBox(xyxy=(37, 12, 57, 32), confidence=0.4, class_id=41, label="cup"),
        ],
        2: [
            BBox(xyxy=(14, 14, 34, 34), confidence=0.9, class_id=41, label="cup"),
        ],
        3: [
            BBox(xyxy=(16, 16, 36, 36), confidence=0.9, class_id=41, label="cup"),
            BBox(xyxy=(39, 16, 59, 36), confidence=0.35, class_id=41, label="cup"),
        ],
        4: [
            BBox(xyxy=(18, 18, 38, 38), confidence=0.9, class_id=41, label="cup"),
            BBox(xyxy=(41, 18, 61, 38), confidence=0.3, class_id=41, label="cup"),
        ],
    }
    write_object_annotations(
        synthetic_session,
        {
            "schema_version": 1,
            "objects": {
                "cup1": {"track_id": "track_0001"},
                "cup2": {"track_id": "track_0002"},
            },
        },
    )
    derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FrameAwareDetector(boxes_by_frame),
        association_config=AssociationConfig(max_lost_frames=30),
    )
    observations = _read_csv(synthetic_session / "derived" / "cups" / "observations.csv")
    semantic_by_track = {row["track_id"]: row["semantic_id"] for row in observations}
    assert semantic_by_track["track_0001"] == "cup1"
    assert semantic_by_track["track_0002"] == "cup2"


def test_depth_xyz_preserved_in_detection_and_observation(synthetic_session: Path) -> None:
    derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FakeDetector(),
    )
    detections = _read_csv(synthetic_session / "derived" / "cups" / "detections.csv")
    observations = _read_csv(synthetic_session / "derived" / "cups" / "observations.csv")
    assert detections[0]["depth_valid"] == "True"
    assert detections[0]["camera_z"]
    assert observations[0]["camera_z"] == detections[0]["camera_z"]
    assert detections[0]["depth_frame_number"] != ""
    assert detections[0]["rgb_depth_delta_us"] != ""


def test_nearest_depth_timestamp_used_when_frame_numbers_differ(tmp_path: Path) -> None:
    session = build_synthetic_session(tmp_path, frame_count=3)
    depth_index = session / "streams" / "depth" / "index.csv"
    rows = _read_csv(depth_index)
    rows[0]["frame_number"] = "10"
    rows[0]["device_timestamp_us"] = "1"
    rows[1]["frame_number"] = "11"
    rows[1]["device_timestamp_us"] = "33334"
    rows[2]["frame_number"] = "12"
    rows[2]["device_timestamp_us"] = "66667"
    with depth_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    derive_observations(
        session,
        apriltag_config={"enabled": False},
        detector=FakeDetector(),
    )
    detections = _read_csv(session / "derived" / "cups" / "detections.csv")
    assert detections[0]["depth_frame_number"] == "10"
    assert int(detections[0]["rgb_depth_delta_us"]) <= 33333


def test_apriltag_disabled_warning_in_manifest(synthetic_session: Path) -> None:
    manifest = derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FakeDetector(),
    )
    assert any("apriltag_world.enabled is false" in warning for warning in manifest["warnings"])


def test_derive_calls_estimate_apriltag_world_with_gray_k_cfg(
    synthetic_session: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, ...], tuple[int, ...], bool]] = []

    def fake_estimate(
        gray: np.ndarray,
        k_matrix: np.ndarray,
        cfg: object,
        *,
        draw_on_bgr: np.ndarray | None = None,
        pose_state: object | None = None,
        frame_number: int | None = None,
        device_timestamp_us: int | None = None,
    ) -> AprilTagWorldResult:
        calls.append((gray.ndim, gray.shape, k_matrix.shape, bool(getattr(cfg, "enabled", False))))
        return AprilTagWorldResult([], "mock")

    import importlib

    derive_module = importlib.import_module("dataset_recorder.derive_observations")
    monkeypatch.setattr(derive_module, "estimate_apriltag_world", fake_estimate)
    derive_observations(
        synthetic_session,
        apriltag_config={
            "enabled": True,
            "tag_size_m": 0.1,
            "tags": {0: {"position": [0.0, 0.0, 0.0]}},
        },
        detector=FakeDetector(),
    )
    assert len(calls) == 5
    assert all(ndim == 2 for ndim, _, _, _ in calls)
    assert all(enabled is True for _, _, _, enabled in calls)


def test_derive_writes_apriltag_reference_when_enabled(
    synthetic_session: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.1, 0.2, 2.5]

    def fake_estimate(
        gray: np.ndarray,
        k_matrix: np.ndarray,
        cfg: object,
        *,
        draw_on_bgr: np.ndarray | None = None,
        pose_state: object | None = None,
        frame_number: int | None = None,
        device_timestamp_us: int | None = None,
    ) -> AprilTagWorldResult:
        return AprilTagWorldResult(
            [
                AprilTagWorldObservation(
                    tag_id=0,
                    T_camera_tag=np.eye(4),
                    T_world_tag=np.eye(4),
                    T_world_camera=T.copy(),
                    reprojection_error_px=0.05,
                )
            ],
            "ok",
        )

    import importlib

    derive_module = importlib.import_module("dataset_recorder.derive_observations")
    monkeypatch.setattr(derive_module, "estimate_apriltag_world", fake_estimate)
    manifest = derive_observations(
        synthetic_session,
        apriltag_config={
            "enabled": True,
            "tag_size_m": 0.1,
            "tags": {0: {"position": [0.0, 0.0, 0.0]}},
        },
        apriltag_reference_config={"enabled": True, "window_frames": 9, "gap_interp_max_frames": 3},
        detector=FakeDetector(),
    )
    ref_csv = synthetic_session / "derived" / "reference" / "apriltag_pose_smoothed.csv"
    ref_manifest = synthetic_session / "derived" / "reference" / "manifest.json"
    assert ref_csv.is_file()
    assert ref_manifest.is_file()
    assert manifest["apriltag_reference"]["method"] == "centered_se3"
    rows = _read_csv(ref_csv)
    assert len(rows) == 5
    assert rows[0]["reference_valid"] == "True"
    assert rows[0]["raw_tx"] == "0.1"


def test_derive_skips_reference_when_disabled(
    synthetic_session: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    T = np.eye(4, dtype=np.float64)

    def fake_estimate(
        gray: np.ndarray,
        k_matrix: np.ndarray,
        cfg: object,
        *,
        draw_on_bgr: np.ndarray | None = None,
        pose_state: object | None = None,
        frame_number: int | None = None,
        device_timestamp_us: int | None = None,
    ) -> AprilTagWorldResult:
        return AprilTagWorldResult(
            [
                AprilTagWorldObservation(
                    tag_id=0,
                    T_camera_tag=np.eye(4),
                    T_world_tag=np.eye(4),
                    T_world_camera=T,
                    reprojection_error_px=0.05,
                )
            ],
            "ok",
        )

    import importlib

    derive_module = importlib.import_module("dataset_recorder.derive_observations")
    monkeypatch.setattr(derive_module, "estimate_apriltag_world", fake_estimate)
    manifest = derive_observations(
        synthetic_session,
        apriltag_config={
            "enabled": True,
            "tag_size_m": 0.1,
            "tags": {0: {"position": [0.0, 0.0, 0.0]}},
        },
        apriltag_reference_config={"enabled": False},
        detector=FakeDetector(),
    )
    assert manifest["apriltag_reference"] is None
    assert not (synthetic_session / "derived" / "reference" / "apriltag_pose_smoothed.csv").exists()


def test_derive_records_cup_depth_provenance(synthetic_session: Path) -> None:
    manifest = derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        cup_depth_config={
            "method": "robust_near_quantile",
            "near_quantile": 0.25,
            "min_near_points": 1,
        },
        detector=FakeDetector(),
    )
    cup_depth = manifest["cup_depth"]
    assert cup_depth["method"] == "robust_near_quantile"
    assert cup_depth["near_quantile"] == pytest.approx(0.25)
    assert cup_depth["min_near_points"] == 1
    detections = _read_csv(synthetic_session / "derived" / "cups" / "detections.csv")
    assert detections
    assert detections[0]["depth_valid"] == "True"
    assert "camera_x" in detections[0]


def test_track_summary_contains_candidate_hints(synthetic_session: Path) -> None:
    h, w = 48, 64
    boxes_by_frame = {
        0: [BBox(xyxy=(10, 10, 30, 30), confidence=0.9, class_id=41, label="cup")],
        1: [BBox(xyxy=(12, 12, 32, 32), confidence=0.9, class_id=41, label="cup")],
        2: [BBox(xyxy=(14, 14, 34, 34), confidence=0.9, class_id=41, label="cup")],
        3: [
            BBox(xyxy=(16, 16, 36, 36), confidence=0.9, class_id=41, label="cup"),
            BBox(xyxy=(39, 16, 59, 36), confidence=0.4, class_id=41, label="cup"),
        ],
        4: [
            BBox(xyxy=(18, 18, 38, 38), confidence=0.9, class_id=41, label="cup"),
            BBox(xyxy=(41, 18, 61, 38), confidence=0.35, class_id=41, label="cup"),
        ],
    }
    derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FrameAwareDetector(boxes_by_frame),
    )
    summary = json.loads(
        (synthetic_session / "derived" / "cups" / "track_summary.json").read_text(encoding="utf-8")
    )
    hints = {entry["track_id"]: entry["semantic_hint"] for entry in summary["tracks"]}
    assert hints["track_0001"] == "cup1_candidate"
    assert hints.get("track_0002") == "cup2_candidate"


def test_detection_index_differs_from_track_id(synthetic_session: Path) -> None:
    h, w = 48, 64
    boxes = [
        BBox(xyxy=(10, 10, 30, 30), confidence=0.9, class_id=41, label="cup"),
        BBox(xyxy=(35, 10, 55, 30), confidence=0.7, class_id=41, label="cup"),
    ]
    derive_observations(
        synthetic_session,
        apriltag_config={"enabled": False},
        detector=FrameAwareDetector({index: boxes for index in range(5)}),
    )
    tracks = _read_csv(synthetic_session / "derived" / "cups" / "tracks.csv")
    for row in tracks:
        assert row["track_id"].startswith("track_")
        assert row["detection_index"] in {"0", "1"}
