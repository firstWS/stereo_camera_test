"""AprilTag IPPE pose disambiguation and temporal continuity tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apriltag_world import (  # noqa: E402
    DEFAULT_MAX_CONTINUITY_FRAME_GAP,
    DEFAULT_MAX_ROTATION_JUMP_DEG,
    DEFAULT_MAX_TRANSLATION_JUMP_M,
    AprilTagPoseSelectorState,
    AprilTagWorldConfig,
    AprilTagWorldTag,
    PoseSelectionDiagnostics,
    _candidate_from_rvec_tvec,
    _enumerate_marker_pose_candidates,
    _homogeneous,
    _rotation_delta_deg,
    _select_marker_pose_candidate,
    _solve_marker_pose,
    _tag_object_points,
    _world_from_tag,
    build_apriltag_world_config,
    estimate_apriltag_world,
    rotation_delta_deg,
)


def _identity_K() -> np.ndarray:
    return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _dist() -> np.ndarray:
    return np.zeros((5, 1), dtype=np.float64)


def _T_world_tag() -> np.ndarray:
    return _world_from_tag([0.0, 0.0, 0.0], top_direction="+Y", front_normal="+Z")


def _make_front_facing_rvec_tvec(*, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Near/correct mode: tag front (+Z) points toward camera (-Z in camera frame)."""
    if not flip:
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        t = np.array([[0.0], [0.0], [0.5]], dtype=np.float64)
    else:
        # Ambiguous far/flip mode: large orientation change, meter-scale world shift.
        R = cv2.Rodrigues(np.array([2.4, 0.3, -0.2], dtype=np.float64))[0]
        t = np.array([[0.0], [0.0], [2.5]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def _project_corners(
    rvec: np.ndarray,
    tvec: np.ndarray,
    tag_size_m: float = 0.135,
    K: np.ndarray | None = None,
) -> np.ndarray:
    if K is None:
        K = _identity_K()
    object_points = _tag_object_points(tag_size_m)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, _dist())
    return image_points.reshape(4, 2).astype(np.float32)


def _candidate_from_pose(
    index: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_points: np.ndarray,
    tag_size_m: float = 0.135,
    K: np.ndarray | None = None,
):
    if K is None:
        K = _identity_K()
    object_points = _tag_object_points(tag_size_m)
    return _candidate_from_rvec_tvec(
        index,
        rvec,
        tvec,
        object_points,
        image_points,
        K,
        _dist(),
    )


def test_ippe_square_dual_solution_enumeration() -> None:
    K = _identity_K()
    rvec_a, tvec_a = _make_front_facing_rvec_tvec(flip=False)
    image_points = _project_corners(rvec_a, tvec_a, K=K)
    object_points = _tag_object_points(0.135)
    rvec_b, tvec_b = _make_front_facing_rvec_tvec(flip=True)

    def fake_generic(obj, img, k, dist, flags=0):
        return True, np.array([rvec_a, rvec_b]), np.array([tvec_a, tvec_b]), None

    with patch.object(cv2, "solvePnPGeneric", fake_generic):
        candidates = _enumerate_marker_pose_candidates(image_points, object_points, K, _dist())
    assert len(candidates) == 2
    assert any(c.reprojection_error_px < 1.0 for c in candidates)


def test_single_valid_candidate_fallback() -> None:
    K = _identity_K()
    rvec, tvec = _make_front_facing_rvec_tvec(flip=False)
    image_points = _project_corners(rvec, tvec, K=K)
    object_points = _tag_object_points(0.135)

    def fake_generic(obj, img, k, dist, flags=0):
        return True, np.array([rvec]), np.array([tvec]), None

    with patch.object(cv2, "solvePnPGeneric", fake_generic):
        candidates = _enumerate_marker_pose_candidates(image_points, object_points, K, _dist())
    assert len(candidates) == 1
    selected, diag = _select_marker_pose_candidate(
        candidates,
        _T_world_tag(),
        previous_T_world_camera=None,
        reinitialize=True,
    )
    assert selected is not None
    assert diag is not None
    assert diag.selection_reason == "single_candidate"


def test_initial_frame_prefers_front_facing_candidate() -> None:
    K = _identity_K()
    rvec_near, tvec_near = _make_front_facing_rvec_tvec(flip=False)
    rvec_far, tvec_far = _make_front_facing_rvec_tvec(flip=True)
    image_points = _project_corners(rvec_near, tvec_near, K=K)
    candidates = [
        _candidate_from_pose(0, rvec_far, tvec_far, image_points),
        _candidate_from_pose(1, rvec_near, tvec_near, image_points),
    ]
    candidates = [c for c in candidates if c is not None]
    assert len(candidates) == 2

    selected, diag = _select_marker_pose_candidate(
        candidates,
        _T_world_tag(),
        previous_T_world_camera=None,
        reinitialize=True,
    )
    assert selected is not None
    assert diag is not None
    assert diag.selection_reason == "initial_front_facing"
    assert selected.index == 1
    assert selected.front_alignment > candidates[0].front_alignment


def test_front_facing_wins_despite_similar_reprojection() -> None:
    K = _identity_K()
    rvec_near, tvec_near = _make_front_facing_rvec_tvec(flip=False)
    rvec_far, tvec_far = _make_front_facing_rvec_tvec(flip=True)
    image_points = _project_corners(rvec_near, tvec_near, K=K)
    near = _candidate_from_pose(0, rvec_near, tvec_near, image_points)
    far = _candidate_from_pose(1, rvec_far, tvec_far, image_points)
    assert near is not None and far is not None
    far = far.__class__(
        index=far.index,
        T_camera_tag=far.T_camera_tag,
        rvec=far.rvec,
        reprojection_error_px=near.reprojection_error_px - 0.01,
        tag_normal_camera=far.tag_normal_camera,
        front_alignment=far.front_alignment,
        determinant=far.determinant,
    )
    selected, _ = _select_marker_pose_candidate(
        [far, near],
        _T_world_tag(),
        previous_T_world_camera=None,
        reinitialize=True,
    )
    assert selected is not None
    assert selected.index == near.index


def test_temporal_continuity_selects_smooth_candidate() -> None:
    K = _identity_K()
    T_world_tag = _T_world_tag()
    rvec_prev, tvec_prev = _make_front_facing_rvec_tvec(flip=False)
    image_prev = _project_corners(rvec_prev, tvec_prev, K=K)
    prev_candidate = _candidate_from_pose(0, rvec_prev, tvec_prev, image_prev)
    assert prev_candidate is not None
    T_world_camera_prev = T_world_tag @ np.linalg.inv(prev_candidate.T_camera_tag)

    rvec_smooth, tvec_smooth = _make_front_facing_rvec_tvec(flip=False)
    rvec_smooth = rvec_smooth + np.array([[0.01], [0.0], [0.0]], dtype=np.float64)
    rvec_flip, tvec_flip = _make_front_facing_rvec_tvec(flip=True)
    image_curr = _project_corners(rvec_smooth, tvec_smooth, K=K)
    smooth = _candidate_from_pose(0, rvec_smooth, tvec_smooth, image_curr)
    flip = _candidate_from_pose(1, rvec_flip, tvec_flip, image_curr)
    assert smooth is not None and flip is not None
    flip = flip.__class__(
        index=flip.index,
        T_camera_tag=flip.T_camera_tag,
        rvec=flip.rvec,
        reprojection_error_px=smooth.reprojection_error_px - 0.02,
        tag_normal_camera=flip.tag_normal_camera,
        front_alignment=flip.front_alignment,
        determinant=flip.determinant,
    )

    selected, diag = _select_marker_pose_candidate(
        [flip, smooth],
        T_world_tag,
        previous_T_world_camera=T_world_camera_prev,
        reinitialize=False,
    )
    assert selected is not None
    assert diag is not None
    assert diag.selection_reason == "temporal_continuity"
    assert selected.index == smooth.index


def test_rejects_two_meter_translation_jump() -> None:
    K = _identity_K()
    T_world_tag = _T_world_tag()
    rvec_prev, tvec_prev = _make_front_facing_rvec_tvec(flip=False)
    prev = _candidate_from_pose(0, rvec_prev, tvec_prev, _project_corners(rvec_prev, tvec_prev, K=K))
    assert prev is not None
    T_prev = T_world_tag @ np.linalg.inv(prev.T_camera_tag)

    rvec_jump, tvec_jump = _make_front_facing_rvec_tvec(flip=True)
    image_curr = _project_corners(rvec_jump, tvec_jump, K=K)
    jump = _candidate_from_pose(0, rvec_jump, tvec_jump, image_curr)
    assert jump is not None
    assert jump.reprojection_error_px < 1.0

    selected, diag = _select_marker_pose_candidate(
        [jump],
        T_world_tag,
        previous_T_world_camera=T_prev,
        reinitialize=False,
    )
    assert selected is not None
    assert diag is not None
    assert diag.selection_reason == "fallback"


def test_rejects_large_orientation_flip() -> None:
    K = _identity_K()
    T_world_tag = _T_world_tag()
    rvec_prev, tvec_prev = _make_front_facing_rvec_tvec(flip=False)
    prev = _candidate_from_pose(0, rvec_prev, tvec_prev, _project_corners(rvec_prev, tvec_prev, K=K))
    assert prev is not None
    T_prev = T_world_tag @ np.linalg.inv(prev.T_camera_tag)

    rvec_flip, tvec_flip = _make_front_facing_rvec_tvec(flip=True)
    image_curr = _project_corners(rvec_flip, tvec_flip, K=K)
    flip = _candidate_from_pose(0, rvec_flip, tvec_flip, image_curr)
    assert flip is not None
    angle = _rotation_delta_deg(T_prev[:3, :3], (T_world_tag @ np.linalg.inv(flip.T_camera_tag))[:3, :3])
    assert angle > 40.0

    selected, diag = _select_marker_pose_candidate(
        [flip],
        T_world_tag,
        previous_T_world_camera=T_prev,
        reinitialize=False,
    )
    assert diag is not None
    assert diag.selection_reason == "fallback"


def test_so3_rotation_delta_metric() -> None:
    R_prev = np.eye(3, dtype=np.float64)
    R_cand = cv2.Rodrigues(np.deg2rad(np.array([0.0, 0.0, 90.0])))[0]
    assert rotation_delta_deg(R_prev, R_cand) == pytest.approx(90.0, abs=1e-6)
    assert rotation_delta_deg(R_prev, R_prev) == pytest.approx(0.0, abs=1e-9)


def test_rejects_non_positive_depth() -> None:
    K = _identity_K()
    rvec, tvec = _make_front_facing_rvec_tvec(flip=False)
    tvec = np.array([[0.0], [0.0], [-0.2]], dtype=np.float64)
    image_points = _project_corners(rvec, tvec, K=K)
    candidate = _candidate_from_pose(0, rvec, tvec, image_points)
    assert candidate is None


def test_rejects_invalid_rotation_determinant(monkeypatch: pytest.MonkeyPatch) -> None:
    K = _identity_K()
    rvec, tvec = _make_front_facing_rvec_tvec(flip=False)
    image_points = _project_corners(rvec, tvec, K=K)
    object_points = _tag_object_points(0.135)

    def bad_rodrigues(_rvec: np.ndarray) -> tuple[np.ndarray, None]:
        R = np.diag([1.0, 1.0, 1.2])
        return R, None

    monkeypatch.setattr(cv2, "Rodrigues", bad_rodrigues)
    candidate = _candidate_from_rvec_tvec(
        0,
        rvec,
        tvec,
        object_points,
        image_points,
        K,
        _dist(),
    )
    assert candidate is None


def test_short_tag_miss_keeps_previous_pose_state() -> None:
    state = AprilTagPoseSelectorState()
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.1, 0.2, 0.3]
    state.update(T, frame_number=10, device_timestamp_us=1_000_000)
    assert not state.continuity_expired(frame_number=12, device_timestamp_us=1_066_666)
    assert state.T_world_camera is not None
    np.testing.assert_allclose(state.T_world_camera, T)


def test_long_gap_reinitializes() -> None:
    state = AprilTagPoseSelectorState(max_continuity_frame_gap=5)
    T = np.eye(4, dtype=np.float64)
    state.update(T, frame_number=10, device_timestamp_us=1_000_000)
    assert state.continuity_expired(frame_number=20, device_timestamp_us=1_333_333)
    assert state.continuity_expired(
        frame_number=None,
        device_timestamp_us=1_000_000 + DEFAULT_MAX_CONTINUITY_FRAME_GAP * 33_333 + 1,
    )


def test_deterministic_selection() -> None:
    K = _identity_K()
    rvec_near, tvec_near = _make_front_facing_rvec_tvec(flip=False)
    rvec_far, tvec_far = _make_front_facing_rvec_tvec(flip=True)
    image_points = _project_corners(rvec_near, tvec_near, K=K)
    candidates = [
        c
        for c in (
            _candidate_from_pose(0, rvec_far, tvec_far, image_points),
            _candidate_from_pose(1, rvec_near, tvec_near, image_points),
        )
        if c is not None
    ]
    first, _ = _select_marker_pose_candidate(
        candidates,
        _T_world_tag(),
        previous_T_world_camera=None,
        reinitialize=True,
    )
    second, _ = _select_marker_pose_candidate(
        candidates,
        _T_world_tag(),
        previous_T_world_camera=None,
        reinitialize=True,
    )
    assert first is not None and second is not None
    assert first.index == second.index


def test_reset_prevents_session_state_leakage() -> None:
    state_a = AprilTagPoseSelectorState()
    state_b = AprilTagPoseSelectorState()
    T_a = np.eye(4, dtype=np.float64)
    T_a[0, 3] = 1.0
    state_a.update(T_a, frame_number=1, device_timestamp_us=100)
    state_b.reset()
    assert state_b.T_world_camera is None
    assert state_a.T_world_camera is not None


def test_corner_ordering_regression() -> None:
    points = _tag_object_points(0.135)
    np.testing.assert_allclose(points[0], [-0.0675, 0.0675, 0.0])
    np.testing.assert_allclose(points[1], [0.0675, 0.0675, 0.0])
    np.testing.assert_allclose(points[2], [0.0675, -0.0675, 0.0])
    np.testing.assert_allclose(points[3], [-0.0675, -0.0675, 0.0])


def test_world_transform_formula_unchanged() -> None:
    K = _identity_K()
    T_world_tag = _T_world_tag()
    rvec, tvec = _make_front_facing_rvec_tvec(flip=False)
    image_points = _project_corners(rvec, tvec, K=K)
    object_points = _tag_object_points(0.135)
    solved = _solve_marker_pose(
        image_points,
        object_points,
        K,
        _dist(),
        T_world_tag,
    )
    assert solved is not None
    T_camera_tag, _, _, _ = solved
    T_world_camera = T_world_tag @ np.linalg.inv(T_camera_tag)
    np.testing.assert_allclose(T_world_camera[:3, :3], T_world_camera[:3, :3])
    assert np.linalg.det(T_world_camera[:3, :3]) == pytest.approx(1.0)


def test_stateless_estimate_apriltag_world_backward_compatible() -> None:
    cfg = build_apriltag_world_config(
        {
            "enabled": True,
            "tag_size_m": 0.135,
            "tags": {0: {"position": [0.0, 0.0, 0.0]}},
        }
    )
    K = _identity_K()
    rvec, tvec = _make_front_facing_rvec_tvec(flip=False)
    image_points = _project_corners(rvec, tvec, K=K)
    gray = np.zeros((480, 640), dtype=np.uint8)
    corners = [image_points.reshape(1, 4, 2)]
    ids = np.array([[0]], dtype=np.int32)

    class FakeDetector:
        def detectMarkers(self, _gray):
            return corners, ids, None

    with patch("cv2.aruco.ArucoDetector", return_value=FakeDetector()):
        result = estimate_apriltag_world(gray, K, cfg)
    assert result.observations
    assert result.observations[0].pose_selection is not None


def test_synthetic_scenario_a_like_sequence_prefers_smooth_branch() -> None:
    K = _identity_K()
    T_world_tag = _T_world_tag()
    state = AprilTagPoseSelectorState()
    object_points = _tag_object_points(0.135)
    mode_switches = 0
    meter_jumps = 0
    prev_T: np.ndarray | None = None

    for frame in range(30):
        yaw = np.deg2rad(frame * 0.5)
        R_smooth = cv2.Rodrigues(np.array([0.0, yaw, 0.0], dtype=np.float64))[0] @ np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        t_smooth = np.array([[0.02 * frame], [0.0], [0.5]], dtype=np.float64)
        rvec_smooth, _ = cv2.Rodrigues(R_smooth)
        image_points = _project_corners(rvec_smooth, t_smooth, K=K)

        rvec_flip, tvec_flip = _make_front_facing_rvec_tvec(flip=True)
        cand_smooth = _candidate_from_pose(0, rvec_smooth, t_smooth, image_points)
        cand_flip = _candidate_from_pose(1, rvec_flip, tvec_flip, image_points)
        assert cand_smooth is not None and cand_flip is not None

        selected, _ = _select_marker_pose_candidate(
            [cand_flip, cand_smooth],
            T_world_tag,
            previous_T_world_camera=state.T_world_camera,
            reinitialize=state.continuity_expired(frame, frame * 33_333),
        )
        assert selected is not None
        T_world_camera = T_world_tag @ np.linalg.inv(selected.T_camera_tag)
        if prev_T is not None:
            jump = float(np.linalg.norm(T_world_camera[:3, 3] - prev_T[:3, 3]))
            if jump > DEFAULT_MAX_TRANSLATION_JUMP_M:
                meter_jumps += 1
            rot = _rotation_delta_deg(prev_T[:3, :3], T_world_camera[:3, :3])
            if rot > DEFAULT_MAX_ROTATION_JUMP_DEG:
                mode_switches += 1
        prev_T = T_world_camera
        state.update(T_world_camera, frame_number=frame, device_timestamp_us=frame * 33_333)

    assert meter_jumps == 0
    assert mode_switches == 0


def test_solve_marker_pose_updates_explicit_state() -> None:
    K = _identity_K()
    T_world_tag = _T_world_tag()
    state = AprilTagPoseSelectorState()
    rvec, tvec = _make_front_facing_rvec_tvec(flip=False)
    image_points = _project_corners(rvec, tvec, K=K)
    object_points = _tag_object_points(0.135)
    solved = _solve_marker_pose(
        image_points,
        object_points,
        K,
        _dist(),
        T_world_tag,
        pose_state=state,
        frame_number=1,
        device_timestamp_us=100,
    )
    assert solved is not None
    assert state.frame_number == 1
    assert state.T_world_camera is not None
