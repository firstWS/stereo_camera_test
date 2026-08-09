"""Tests for offline AprilTag reference trajectory smoothing."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apriltag_world import rotation_delta_deg  # noqa: E402
from dataset_recorder.apriltag_reference import (  # noqa: E402
    REFERENCE_CSV_FIELDS,
    AprilTagPoseFrame,
    AprilTagReferenceConfig,
    _interpolate_short_gaps,
    _quaternion_to_rotation,
    _rotation_to_quaternion,
    _slerp,
    _smooth_centered_se3,
    build_and_write_apriltag_reference,
    build_apriltag_reference_trajectory,
    reference_config_from_mapping,
    write_apriltag_reference_csv,
)


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _frame(fn: int, T: np.ndarray | None, *, reproj: float | None = 0.1) -> AprilTagPoseFrame:
    return AprilTagPoseFrame(
        frame_number=fn,
        device_timestamp_us=fn * 33_333,
        source_valid=T is not None,
        T_world_camera=T,
        reprojection_error_px=reproj,
    )


def _cfg(**kwargs) -> AprilTagReferenceConfig:
    return AprilTagReferenceConfig(**kwargs)


def test_constant_se3_pose_returns_identical_reference() -> None:
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    t = np.array([0.5, -0.2, 2.6], dtype=np.float64)
    T = _T(R, t)
    frames = [_frame(i, T) for i in range(1, 21)]
    rows = build_apriltag_reference_trajectory(frames, _cfg())
    valid = [r for r in rows if r.reference_valid]
    assert len(valid) == 20
    for row in valid:
        assert row.ref_tx == pytest.approx(t[0], abs=1e-9)
        assert row.ref_ty == pytest.approx(t[1], abs=1e-9)
        assert row.ref_tz == pytest.approx(t[2], abs=1e-9)
        assert rotation_delta_deg(R, _quaternion_to_rotation(np.array([row.ref_qw, row.ref_qx, row.ref_qy, row.ref_qz]))) == pytest.approx(0.0, abs=1e-6)


def test_translation_jitter_is_smoothed() -> None:
    R = np.eye(3, dtype=np.float64)
    frames = []
    for i in range(1, 31):
        t = np.array([0.5 + (0.2 if i % 2 else -0.2), 0.0, 2.0], dtype=np.float64)
        frames.append(_frame(i, _T(R, t)))
    rows = build_apriltag_reference_trajectory(frames, _cfg())
    raw_d = [
        float(np.linalg.norm(np.array([rows[i].raw_tx, rows[i].raw_ty, rows[i].raw_tz]) - np.array([rows[i - 1].raw_tx, rows[i - 1].raw_ty, rows[i - 1].raw_tz])))
        for i in range(1, len(rows))
        if rows[i].raw_tx is not None and rows[i - 1].raw_tx is not None
    ]
    ref_d = [
        float(np.linalg.norm(np.array([rows[i].ref_tx, rows[i].ref_ty, rows[i].ref_tz]) - np.array([rows[i - 1].ref_tx, rows[i - 1].ref_ty, rows[i - 1].ref_tz])))
        for i in range(1, len(rows))
        if rows[i].reference_valid and rows[i - 1].reference_valid
    ]
    assert np.median(raw_d) > np.median(ref_d)
    assert np.percentile(ref_d, 90) < np.percentile(raw_d, 90)


def test_rotation_jitter_is_smoothed() -> None:
    frames = []
    for i in range(1, 31):
        yaw = math.radians(5.0 if i % 2 else -5.0)
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        frames.append(_frame(i, _T(R, np.array([0.0, 0.0, 2.0]))))
    rows = build_apriltag_reference_trajectory(frames, _cfg())
    raw_angles = []
    ref_angles = []
    for i in range(1, len(rows)):
        if rows[i].raw_qw is None:
            continue
        R0 = _quaternion_to_rotation(np.array([rows[i - 1].raw_qw, rows[i - 1].raw_qx, rows[i - 1].raw_qy, rows[i - 1].raw_qz]))
        R1 = _quaternion_to_rotation(np.array([rows[i].raw_qw, rows[i].raw_qx, rows[i].raw_qy, rows[i].raw_qz]))
        raw_angles.append(rotation_delta_deg(R0, R1))
        if rows[i].reference_valid and rows[i - 1].reference_valid:
            S0 = _quaternion_to_rotation(np.array([rows[i - 1].ref_qw, rows[i - 1].ref_qx, rows[i - 1].ref_qy, rows[i - 1].ref_qz]))
            S1 = _quaternion_to_rotation(np.array([rows[i].ref_qw, rows[i].ref_qx, rows[i].ref_qy, rows[i].ref_qz]))
            ref_angles.append(rotation_delta_deg(S0, S1))
    assert np.median(raw_angles) > np.median(ref_angles)


def test_centered_window_nine_uses_reduced_edge_samples() -> None:
    R = np.eye(3, dtype=np.float64)
    frames = [_frame(i, _T(R, np.array([float(i), 0.0, 2.0]))) for i in range(1, 21)]
    rows = build_apriltag_reference_trajectory(frames, _cfg(window_frames=9))
    assert rows[0].window_valid_sample_count == 5
    assert rows[9].window_valid_sample_count == 9
    assert rows[-1].window_valid_sample_count == 5


def test_quaternion_hemisphere_alignment() -> None:
    q = np.array([0.6, 0.2, 0.3, 0.7], dtype=np.float64)
    q /= np.linalg.norm(q)
    q_neg = -q
    mid = _slerp(q, q_neg, 0.5)
    assert float(np.dot(mid, q)) > 0.0


def test_smoothed_rotation_determinant_is_plus_one() -> None:
    frames = []
    for i in range(1, 21):
        yaw = math.radians(i * 0.5)
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        frames.append(_frame(i, _T(R, np.array([0.1 * i, 0.0, 2.0]))))
    rows = build_apriltag_reference_trajectory(frames, _cfg())
    for row in rows:
        if not row.reference_valid:
            continue
        R = _quaternion_to_rotation(np.array([row.ref_qw, row.ref_qx, row.ref_qy, row.ref_qz]))
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_short_gap_interpolation_allowed() -> None:
    R = np.eye(3, dtype=np.float64)
    frames = [
        _frame(1, _T(R, np.array([0.0, 0.0, 2.0]))),
        _frame(2, None),
        _frame(3, _T(R, np.array([0.2, 0.0, 2.0]))),
    ]
    rows = build_apriltag_reference_trajectory(frames, _cfg(gap_interp_max_frames=3))
    assert rows[1].interpolated is True
    assert rows[1].reference_valid is True
    assert rows[1].reference_quality == "INTERPOLATED"


def test_four_frame_gap_is_not_interpolated() -> None:
    R = np.eye(3, dtype=np.float64)
    frames = [
        _frame(1, _T(R, np.array([0.0, 0.0, 2.0]))),
        _frame(2, None),
        _frame(3, None),
        _frame(4, None),
        _frame(5, None),
        _frame(6, _T(R, np.array([0.4, 0.0, 2.0]))),
    ]
    rows = build_apriltag_reference_trajectory(frames, _cfg(gap_interp_max_frames=3))
    for row in rows[1:5]:
        assert row.interpolated is False
        assert row.reference_valid is False
        assert row.reference_quality == "INSUFFICIENT_SUPPORT"


def test_translation_interpolation_is_linear() -> None:
    seq = _interpolate_short_gaps(
        [
            _frame(1, _T(np.eye(3), np.array([0.0, 0.0, 0.0]))),
            _frame(2, None),
            _frame(3, _T(np.eye(3), np.array([2.0, 0.0, 0.0]))),
        ],
        max_gap_frames=3,
    )
    assert seq[1]["valid"] is True
    np.testing.assert_allclose(seq[1]["t"], [1.0, 0.0, 0.0], atol=1e-9)


def test_rotation_interpolation_uses_slerp() -> None:
    R0 = np.eye(3, dtype=np.float64)
    yaw = math.radians(90.0)
    R1 = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    seq = _interpolate_short_gaps(
        [
            _frame(1, _T(R0, np.zeros(3))),
            _frame(2, None),
            _frame(3, _T(R1, np.zeros(3))),
        ],
        max_gap_frames=3,
    )
    expected = _quaternion_to_rotation(_slerp(_rotation_to_quaternion(R0), _rotation_to_quaternion(R1), 0.5))
    np.testing.assert_allclose(seq[1]["R"], expected, atol=1e-9)


def test_invalid_samples_excluded_from_smoothing_window() -> None:
    R = np.eye(3, dtype=np.float64)
    seq = [
        {"frame_number": 1, "valid": True, "t": np.array([0.0, 0.0, 0.0]), "R": R, "source_valid": True, "interpolated": False},
        {"frame_number": 2, "valid": False, "t": None, "R": None, "source_valid": False, "interpolated": False},
        {"frame_number": 3, "valid": True, "t": np.array([2.0, 0.0, 0.0]), "R": R, "source_valid": True, "interpolated": False},
    ]
    out = _smooth_centered_se3(seq, window_frames=3, min_valid_samples=1)
    assert out[1]["ref_valid"] is False
    assert out[1]["window_valid_sample_count"] == 0


def test_insufficient_samples_marks_reference_invalid() -> None:
    R = np.eye(3, dtype=np.float64)
    frames = [_frame(1, _T(R, np.zeros(3)))]
    rows = build_apriltag_reference_trajectory(frames, _cfg(window_frames=9, min_valid_samples=5))
    assert rows[0].reference_valid is False
    assert rows[0].reference_quality == "INSUFFICIENT_SUPPORT"


def test_output_is_deterministic() -> None:
    R = np.eye(3, dtype=np.float64)
    frames = [_frame(i, _T(R, np.array([math.sin(i), 0.0, 2.0]))) for i in range(1, 16)]
    a = build_apriltag_reference_trajectory(frames, _cfg())
    b = build_apriltag_reference_trajectory(frames, _cfg())
    assert [r.as_dict() for r in a] == [r.as_dict() for r in b]


def test_schema_and_provenance_fields_present() -> None:
    R = np.eye(3, dtype=np.float64)
    rows = build_apriltag_reference_trajectory([_frame(1, _T(R, np.zeros(3)))], _cfg())
    row = rows[0].as_dict()
    for field in REFERENCE_CSV_FIELDS:
        assert field in row
    assert row["smoothing_method"] == "centered_se3"
    assert row["window_frames"] == 9
    assert row["gap_interp_max_frames"] == 3


def test_reference_config_from_mapping_requires_odd_window() -> None:
    with pytest.raises(ValueError, match="odd"):
        reference_config_from_mapping({"window_frames": 8})


def test_study_reproduction_jitter_reduction() -> None:
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    frames = []
    for i in range(1, 41):
        offset = 0.25 if i % 2 == 0 else 0.0
        frames.append(_frame(i, _T(R, np.array([0.5 + offset, -0.2, 2.6 + offset * 0.1]))))
    rows = build_apriltag_reference_trajectory(frames, _cfg())
    raw = []
    ref = []
    for i in range(1, len(rows)):
        if rows[i].raw_tx is None:
            continue
        raw.append(float(np.linalg.norm(np.array([rows[i].raw_tx, rows[i].raw_ty, rows[i].raw_tz]) - np.array([rows[i - 1].raw_tx, rows[i - 1].raw_ty, rows[i - 1].raw_tz]))))
        if rows[i].reference_valid and rows[i - 1].reference_valid:
            ref.append(float(np.linalg.norm(np.array([rows[i].ref_tx, rows[i].ref_ty, rows[i].ref_tz]) - np.array([rows[i - 1].ref_tx, rows[i - 1].ref_ty, rows[i - 1].ref_tz]))))
    assert np.percentile(ref, 90) < np.percentile(raw, 90)
    assert np.max(ref) < np.max(raw)


def test_build_and_write_reference_creates_files(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    R = np.eye(3, dtype=np.float64)
    frames = [_frame(i, _T(R, np.zeros(3))) for i in range(1, 6)]
    summary = build_and_write_apriltag_reference(session, frames, _cfg())
    csv_path = session / "derived" / "reference" / "apriltag_pose_smoothed.csv"
    manifest_path = session / "derived" / "reference" / "manifest.json"
    assert csv_path.is_file()
    assert manifest_path.is_file()
    assert summary["method"] == "centered_se3"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 5


def test_write_csv_preserves_raw_pose_separately(tmp_path: Path) -> None:
    path = tmp_path / "ref.csv"
    R = np.eye(3, dtype=np.float64)
    rows = build_apriltag_reference_trajectory(
        [_frame(1, _T(R, np.array([1.0, 2.0, 3.0])))],
        _cfg(),
    )
    write_apriltag_reference_csv(path, rows)
    saved = list(csv.DictReader(path.open(encoding="utf-8")))[0]
    assert float(saved["raw_tx"]) == pytest.approx(1.0)
    assert float(saved["ref_tx"]) == pytest.approx(1.0)
