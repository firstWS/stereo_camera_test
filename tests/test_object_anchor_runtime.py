from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_pose import ObjectPoseEstimate, rpy_deg_to_rotation_matrix  # noqa: E402
from object_anchor_config import load_object_anchor_config  # noqa: E402
from object_anchor_detector import ObjectAnchorDetection, ObjectAnchorDetector  # noqa: E402
from object_anchor_registration import (  # noqa: E402
    load_world_pose_registration,
    save_world_pose_registration,
)
from object_anchor_runtime import (  # noqa: E402
    ObjectAnchorRuntime,
    ObjectAnchorRuntimeSettings,
    ObjectPoseTemporalValidator,
    build_optional_object_anchor_runtime,
)


class _FakeDetector(ObjectAnchorDetector):
    def __init__(self, detection: ObjectAnchorDetection) -> None:
        self.detection = detection

    def predict(self, bgr: np.ndarray) -> list[ObjectAnchorDetection]:
        return [self.detection]


def _pose(translation: tuple[float, float, float], yaw_deg: float = 0.0) -> ObjectPoseEstimate:
    rotation = rpy_deg_to_rotation_matrix(0.0, 0.0, yaw_deg)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return ObjectPoseEstimate(
        valid=True,
        reason="ok",
        T_camera_object=transform,
        rvec=rvec,
        tvec=tvec,
        rotation_matrix=rotation,
        rpy_deg=(0.0, 0.0, yaw_deg),
    )


def test_missing_model_disables_only_object_anchor() -> None:
    runtime, reason = build_optional_object_anchor_runtime(
        {
            "enabled": True,
            "config_path": "configs/object_anchors/tissue_box_01.yaml",
            "model_path": "models/object_anchor/not_ready/best.pt",
        },
        repo_root=ROOT,
    )
    assert runtime is None
    assert reason.startswith("model_not_found:")


def test_temporal_validator_rejects_translation_and_rotation_jumps() -> None:
    validator = ObjectPoseTemporalValidator(
        ObjectAnchorRuntimeSettings(
            enabled=True,
            max_translation_jump_m=0.20,
            max_rotation_jump_deg=20.0,
        )
    )
    assert validator.validate(_pose((0.0, 0.0, 1.0))).valid
    translation_jump = validator.validate(_pose((0.5, 0.0, 1.0)))
    assert not translation_jump.valid
    assert translation_jump.reason.startswith("translation_jump:")
    rotation_jump = validator.validate(_pose((0.0, 0.0, 1.0), yaw_deg=45.0))
    assert not rotation_jump.valid
    assert rotation_jump.reason.startswith("rotation_jump:")


def test_registration_round_trip(tmp_path: Path) -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_deg_to_rotation_matrix(2.0, -3.0, 15.0)
    transform[:3, 3] = [1.2, 0.8, 1.1]
    path = save_world_pose_registration(
        tmp_path / "tissue_box_01_registration.yaml",
        object_id="tissue_box_01",
        T_world_object=transform,
        source="apriltag_rigid_board",
        metadata={"tag_id": 0},
    )
    restored = load_world_pose_registration(path, expected_object_id="tissue_box_01")
    np.testing.assert_allclose(restored, transform, atol=1e-12)


def test_live_runtime_detection_to_overlay() -> None:
    config = load_object_anchor_config(
        ROOT / "configs" / "object_anchors" / "tissue_box_01.yaml"
    )
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    rotation = rpy_deg_to_rotation_matrix(8.0, -12.0, 18.0)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.array([[0.08], [-0.04], [1.2]])
    keypoints, _ = cv2.projectPoints(
        config.object_points, rvec, tvec, K, np.zeros((8, 1))
    )
    detection = ObjectAnchorDetection(
        keypoints_xy=keypoints.reshape(-1, 2),
        keypoint_confidences=np.full(8, 0.98),
        bbox_xyxy=(500.0, 300.0, 780.0, 500.0),
        score=0.99,
        label="tissue_box",
    )
    runtime = ObjectAnchorRuntime(
        config,
        _FakeDetector(detection),
        ObjectAnchorRuntimeSettings(enabled=True, draw=True),
    )
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    result = runtime.process(image, K, np.zeros((8, 1)))
    assert result.pose.valid, result.pose.reason
    assert result.pose.inlier_count == 8
    assert np.count_nonzero(result.overlay_bgr) > 0
    lines = runtime.overlay_lines(result)
    assert any("T_cam_obj=" in line for line in lines)
    assert any("RPY=" in line for line in lines)
    assert any("reproj=" in line for line in lines)


def test_front_only_runtime_detection_to_overlay() -> None:
    config = load_object_anchor_config(
        ROOT / "configs" / "object_anchors" / "tissue_box_01_front_only.yaml"
    )
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    rotation = rpy_deg_to_rotation_matrix(82.0, -12.0, 18.0)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.array([[0.08], [-0.04], [1.2]])
    keypoints, _ = cv2.projectPoints(
        config.object_points, rvec, tvec, K, np.zeros((5, 1))
    )
    points = keypoints.reshape(-1, 2)
    detection = ObjectAnchorDetection(
        keypoints_xy=points,
        keypoint_confidences=np.full(4, 0.98),
        bbox_xyxy=(
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ),
        score=0.99,
        label="tissue_box",
    )
    runtime = ObjectAnchorRuntime(
        config,
        _FakeDetector(detection),
        ObjectAnchorRuntimeSettings(enabled=True, draw=True),
    )
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    result = runtime.process(image, K, np.zeros((5, 1)))
    assert result.pose.valid, result.pose.reason
    assert result.pose.inlier_count == 4
    assert np.count_nonzero(result.overlay_bgr) > 0
    assert "kpts=4/4 inliers=4" in runtime.overlay_lines(result)
