"""Tests for runtime AprilTag pose persistence in observations.csv."""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apriltag_world import AprilTagWorldObservation, AprilTagWorldResult  # noqa: E402
from dataset_recorder.apriltag_reference import _quaternion_to_rotation, _rotation_to_quaternion  # noqa: E402
from dataset_recorder.apriltag_runtime_pose import (  # noqa: E402
    RUNTIME_POSE_CSV_FIELDS,
    apriltag_observation_pose_columns,
    invalid_runtime_pose_columns,
    runtime_pose_columns_from_transform,
    transform_from_runtime_pose_columns,
)
from dataset_recorder.derive_observations import derive_observations  # noqa: E402
from stereo_types import BBox, DetectionResult  # noqa: E402

_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures.py"
import importlib.util

_FIXTURES_SPEC = importlib.util.spec_from_file_location("dataset_recorder_fixtures", _FIXTURES_PATH)
assert _FIXTURES_SPEC and _FIXTURES_SPEC.loader
_FIXTURES = importlib.util.module_from_spec(_FIXTURES_SPEC)
_FIXTURES_SPEC.loader.exec_module(_FIXTURES)
build_synthetic_session = _FIXTURES.build_synthetic_session


class FakeDetector:
    def predict(self, bgr: np.ndarray, frame_number: int | None = None) -> DetectionResult:
        h, w = bgr.shape[:2]
        return DetectionResult(
            boxes=[
                BBox(
                    xyxy=(w * 0.25, h * 0.25, w * 0.75, h * 0.75),
                    confidence=0.9,
                    class_id=41,
                    label="cup",
                )
            ],
            image_shape_hw=(h, w),
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_transform() -> np.ndarray:
    R = _quaternion_to_rotation(np.array([0.6, 0.2, 0.3, 0.7], dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [0.12, -0.34, 2.51]
    return T


def test_runtime_pose_columns_roundtrip_translation_and_quaternion() -> None:
    T = _sample_transform()
    columns = runtime_pose_columns_from_transform(T)
    assert columns["pose_valid"] == "True"
    assert float(columns["world_tx"]) == pytest.approx(T[0, 3])
    assert float(columns["world_ty"]) == pytest.approx(T[1, 3])
    assert float(columns["world_tz"]) == pytest.approx(T[2, 3])
    q = np.array(
        [
            float(columns["world_qw"]),
            float(columns["world_qx"]),
            float(columns["world_qy"]),
            float(columns["world_qz"]),
        ],
        dtype=np.float64,
    )
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
    restored = transform_from_runtime_pose_columns(columns)
    assert restored[:3, 3] == pytest.approx(T[:3, 3])
    assert restored[:3, :3] == pytest.approx(T[:3, :3], abs=1e-9)
    assert np.linalg.det(restored[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_invalid_pose_columns_use_empty_values_not_identity() -> None:
    columns = invalid_runtime_pose_columns()
    assert columns["pose_valid"] == "False"
    for field in RUNTIME_POSE_CSV_FIELDS[1:]:
        assert columns[field] == ""


def test_visible_and_pose_valid_semantics_are_separate() -> None:
    visible_without_pose = apriltag_observation_pose_columns(visible=True, T_world_camera=None)
    assert visible_without_pose["pose_valid"] == "False"
    for field in RUNTIME_POSE_CSV_FIELDS[1:]:
        assert visible_without_pose[field] == ""

    invisible = apriltag_observation_pose_columns(visible=False, T_world_camera=_sample_transform())
    assert invisible["pose_valid"] == "False"
    for field in RUNTIME_POSE_CSV_FIELDS[1:]:
        assert invisible[field] == ""


def test_transform_direction_p_world_equals_t_world_camera_times_p_camera() -> None:
    T = _sample_transform()
    p_camera = np.array([0.2, -0.1, 1.5, 1.0], dtype=np.float64)
    p_world = T @ p_camera
    restored = transform_from_runtime_pose_columns(runtime_pose_columns_from_transform(T))
    assert restored @ p_camera == pytest.approx(p_world, abs=1e-9)


def test_derive_persists_runtime_pose_columns(
    synthetic_session: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    T = _sample_transform()

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

    derive_module = importlib.import_module("dataset_recorder.derive_observations")
    monkeypatch.setattr(derive_module, "estimate_apriltag_world", fake_estimate)
    derive_observations(
        synthetic_session,
        apriltag_config={
            "enabled": True,
            "tag_size_m": 0.1,
            "tags": {0: {"position": [0.0, 0.0, 0.0]}},
        },
        apriltag_reference_config={"enabled": True, "window_frames": 9, "gap_interp_max_frames": 3},
        detector=FakeDetector(),
    )
    rows = _read_csv(synthetic_session / "derived" / "apriltag" / "observations.csv")
    assert rows
    assert all(field in rows[0] for field in RUNTIME_POSE_CSV_FIELDS)
    valid_rows = [row for row in rows if row["pose_valid"] == "True"]
    assert valid_rows
    assert float(valid_rows[0]["world_tx"]) == pytest.approx(T[0, 3])
    q = _rotation_to_quaternion(T[:3, :3])
    assert float(valid_rows[0]["world_qw"]) == pytest.approx(q[0])
    assert float(valid_rows[0]["world_qx"]) == pytest.approx(q[1])
    assert float(valid_rows[0]["world_qy"]) == pytest.approx(q[2])
    assert float(valid_rows[0]["world_qz"]) == pytest.approx(q[3])


def test_reference_builder_unchanged_when_runtime_pose_columns_persisted(
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

    derive_module = importlib.import_module("dataset_recorder.derive_observations")
    monkeypatch.setattr(derive_module, "estimate_apriltag_world", fake_estimate)
    derive_observations(
        synthetic_session,
        apriltag_config={
            "enabled": True,
            "tag_size_m": 0.1,
            "tags": {0: {"position": [0.0, 0.0, 0.0]}},
        },
        apriltag_reference_config={"enabled": True, "window_frames": 9, "gap_interp_max_frames": 3},
        detector=FakeDetector(),
    )
    ref_rows = _read_csv(synthetic_session / "derived" / "reference" / "apriltag_pose_smoothed.csv")
    assert len(ref_rows) == 5
    assert ref_rows[0]["reference_valid"] == "True"
    assert ref_rows[0]["raw_tx"] == "0.1"
    assert ref_rows[0]["raw_ty"] == "0.2"
    assert ref_rows[0]["raw_tz"] == "2.5"
