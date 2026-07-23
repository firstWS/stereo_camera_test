from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apriltag_world import AprilTagWorldObservation, AprilTagWorldResult  # noqa: E402
from object_anchor_detector import ObjectAnchorDetection  # noqa: E402
from object_anchor_geometry import find_skeleton_crossings  # noqa: E402
from object_anchor_pose import ObjectPoseEstimate, rpy_deg_to_rotation_matrix  # noqa: E402
from object_anchor_runtime import ObjectAnchorFrameResult  # noqa: E402
from object_anchor_world import (  # noqa: E402
    ObjectAnchorWorldFrameResult,
    WorldPoseRegistrationCollector,
    average_transforms,
    estimate_object_anchor_world_pose,
    rotation_delta_deg,
)


def _transform(
    translation: tuple[float, float, float],
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rpy_deg_to_rotation_matrix(*rpy_deg)
    result[:3, 3] = translation
    return result


def _anchor_result(T_camera_object: np.ndarray) -> ObjectAnchorFrameResult:
    detection = ObjectAnchorDetection(
        keypoints_xy=np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float64),
        keypoint_confidences=np.ones(4),
    )
    pose = ObjectPoseEstimate(
        valid=True,
        reason="ok",
        T_camera_object=T_camera_object,
        rotation_matrix=T_camera_object[:3, :3],
        tvec=T_camera_object[:3, 3].reshape(3, 1),
    )
    return ObjectAnchorFrameResult(detection, pose, np.zeros((30, 30, 3), dtype=np.uint8))


def test_skeleton_crossing_detects_only_crossed_quadrilateral() -> None:
    skeleton = ((0, 1), (1, 2), (2, 3), (3, 0))
    normal = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float64)
    crossed = np.array([[0, 0], [2, 1], [2, 0], [0, 1]], dtype=np.float64)
    assert find_skeleton_crossings(normal, skeleton) == ()
    assert find_skeleton_crossings(crossed, skeleton)


def test_world_pose_uses_documented_transform_direction() -> None:
    T_world_tag = _transform((1.0, 2.0, 0.0), (0.0, 0.0, 10.0))
    T_camera_tag = _transform((0.2, -0.1, 1.5), (2.0, -4.0, 8.0))
    T_world_camera = T_world_tag @ np.linalg.inv(T_camera_tag)
    T_camera_object = _transform((-0.1, 0.05, 1.0), (5.0, 3.0, -7.0))
    observation = AprilTagWorldObservation(
        tag_id=0,
        T_camera_tag=T_camera_tag,
        T_world_tag=T_world_tag,
        T_world_camera=T_world_camera,
        reprojection_error_px=0.2,
    )
    result = estimate_object_anchor_world_pose(
        AprilTagWorldResult([observation], "ok"),
        _anchor_result(T_camera_object),
    )
    assert result.valid and result.accepted
    np.testing.assert_allclose(result.T_world_camera, T_world_camera, atol=1e-10)
    np.testing.assert_allclose(
        result.T_world_object,
        T_world_tag @ np.linalg.inv(T_camera_tag) @ T_camera_object,
        atol=1e-10,
    )


def test_transform_average_handles_quaternion_sign_and_wraparound() -> None:
    transforms = [
        _transform((1.0, 2.0, 3.0), (0.0, 0.0, 179.0)),
        _transform((1.02, 1.98, 3.01), (0.0, 0.0, -179.0)),
    ]
    averaged = average_transforms(transforms)
    np.testing.assert_allclose(averaged[:3, 3], [1.01, 1.99, 3.005], atol=1e-12)
    assert rotation_delta_deg(averaged[:3, :3], transforms[0][:3, :3]) < 2.0


def test_registration_collector_rejects_position_outlier() -> None:
    collector = WorldPoseRegistrationCollector(
        target_frames=3,
        min_seed_frames=2,
        max_position_outlier_m=0.10,
        max_rotation_outlier_deg=10.0,
    )
    for x in (1.00, 1.01):
        assert collector.add(
            ObjectAnchorWorldFrameResult(
                valid=True,
                accepted=True,
                reason="ok",
                T_world_object=_transform((x, 2.0, 0.5)),
            )
        )
    assert not collector.add(
        ObjectAnchorWorldFrameResult(
            valid=True,
            accepted=True,
            reason="ok",
            T_world_object=_transform((2.0, 2.0, 0.5)),
        )
    )
    assert collector.excluded["registration_position_outlier"] == 1
    assert collector.add(
        ObjectAnchorWorldFrameResult(
            valid=True,
            accepted=True,
            reason="ok",
            T_world_object=_transform((0.99, 2.0, 0.5)),
        )
    )
    assert collector.complete
    np.testing.assert_allclose(collector.finalize()[:3, 3], [1.0, 2.0, 0.5], atol=1e-12)
