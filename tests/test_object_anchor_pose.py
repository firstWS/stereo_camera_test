from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_config import load_object_anchor_config  # noqa: E402
from object_anchor_pose import (  # noqa: E402
    estimate_object_pose,
    rpy_deg_to_rotation_matrix,
)


class ObjectAnchorPoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_object_anchor_config(
            ROOT / "configs" / "object_anchors" / "tissue_box_01.yaml"
        )
        cls.K = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cls.rotation = rpy_deg_to_rotation_matrix(8.0, -12.0, 18.0)
        cls.rvec, _ = cv2.Rodrigues(cls.rotation)
        cls.tvec = np.array([[0.08], [-0.04], [1.20]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            cls.config.object_points,
            cls.rvec,
            cls.tvec,
            cls.K,
            np.zeros((5, 1)),
        )
        cls.keypoints = projected.reshape(-1, 2)

    def test_recovers_synthetic_pose(self) -> None:
        pose = estimate_object_pose(
            self.keypoints,
            self.config.object_points,
            self.K,
            confidences=np.ones(8),
            visibility=np.full(8, 2),
            settings=self.config.pose_settings,
        )
        self.assertTrue(pose.valid, pose.reason)
        self.assertIsNotNone(pose.tvec)
        self.assertIsNotNone(pose.rotation_matrix)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=1e-5)
        np.testing.assert_allclose(pose.rotation_matrix, self.rotation, atol=1e-5)
        self.assertLess(pose.mean_reprojection_error_px or 999.0, 1e-3)

    def test_config_geometry_and_skeleton_order(self) -> None:
        expected_points = np.array(
            [
                [-0.1175, -0.0575, 0.0550],
                [0.1175, -0.0575, 0.0550],
                [0.1175, -0.0575, -0.0550],
                [-0.1175, -0.0575, -0.0550],
                [0.1175, 0.0575, 0.0550],
                [-0.1175, 0.0575, 0.0550],
                [-0.1175, 0.0575, -0.0550],
                [0.1175, 0.0575, -0.0550],
            ]
        )
        expected_skeleton = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 5), (1, 4), (2, 7), (3, 6),
        )
        np.testing.assert_array_equal(self.config.object_points, expected_points)
        self.assertEqual(self.config.skeleton, expected_skeleton)

    def test_filters_low_confidence_and_hidden_keypoints(self) -> None:
        keypoints = self.keypoints.copy()
        keypoints[6] += np.array([100.0, -100.0])
        confidence = np.ones(8)
        visibility = np.full(8, 2)
        confidence[6] = 0.1
        visibility[7] = 0
        pose = estimate_object_pose(
            keypoints,
            self.config.object_points,
            self.K,
            confidences=confidence,
            visibility=visibility,
            settings=self.config.pose_settings,
        )
        self.assertTrue(pose.valid, pose.reason)
        self.assertNotIn(6, pose.correspondence_indices)
        self.assertNotIn(7, pose.correspondence_indices)
        self.assertEqual(len(pose.correspondence_indices), 6)

    def test_rejects_fewer_than_four_keypoints(self) -> None:
        visibility = np.array([2, 2, 2, 0, 0, 0, 0, 0])
        pose = estimate_object_pose(
            self.keypoints,
            self.config.object_points,
            self.K,
            confidences=np.ones(8),
            visibility=visibility,
            settings=self.config.pose_settings,
        )
        self.assertFalse(pose.valid)
        self.assertTrue(pose.reason.startswith("insufficient_correspondences"))

    def test_accepts_four_visible_front_face_keypoints(self) -> None:
        visibility = np.array([2, 2, 2, 2, 0, 0, 0, 0])
        pose = estimate_object_pose(
            self.keypoints,
            self.config.object_points,
            self.K,
            confidences=np.ones(8),
            visibility=visibility,
            settings=self.config.pose_settings,
        )
        self.assertTrue(pose.valid, pose.reason)
        self.assertEqual(pose.inlier_count, 4)


class FrontOnlyObjectAnchorPoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_object_anchor_config(
            ROOT / "configs" / "object_anchors" / "tissue_box_01_front_only.yaml"
        )
        cls.K = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cls.rotation = rpy_deg_to_rotation_matrix(82.0, -12.0, 18.0)
        cls.rvec, _ = cv2.Rodrigues(cls.rotation)
        cls.tvec = np.array([[0.08], [-0.04], [1.20]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            cls.config.object_points,
            cls.rvec,
            cls.tvec,
            cls.K,
            np.zeros((5, 1)),
        )
        cls.keypoints = projected.reshape(-1, 2)

    def test_config_is_front_only_rectangle(self) -> None:
        self.assertEqual(self.config.anchor_mode, "front_only")
        self.assertEqual(
            self.config.keypoint_names,
            (
                "front_top_left",
                "front_top_right",
                "front_bottom_right",
                "front_bottom_left",
            ),
        )
        self.assertEqual(self.config.skeleton, ((0, 1), (1, 2), (2, 3), (3, 0)))
        self.assertAlmostEqual(np.ptp(self.config.object_points[:, 0]), 0.235)
        self.assertAlmostEqual(np.ptp(self.config.object_points[:, 2]), 0.110)
        self.assertTrue(np.allclose(self.config.object_points[:, 1], -0.0575))

    def test_recovers_front_only_synthetic_pose(self) -> None:
        pose = estimate_object_pose(
            self.keypoints,
            self.config.object_points,
            self.K,
            confidences=np.ones(4),
            visibility=np.full(4, 2),
            settings=self.config.pose_settings,
        )
        self.assertTrue(pose.valid, pose.reason)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=1e-5)
        np.testing.assert_allclose(pose.rotation_matrix, self.rotation, atol=1e-5)
        self.assertEqual(pose.inlier_count, 4)


if __name__ == "__main__":
    unittest.main()
