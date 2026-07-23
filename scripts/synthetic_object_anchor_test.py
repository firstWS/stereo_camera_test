#!/usr/bin/env python3
"""Project synthetic tissue-box keypoints, recover pose, and save a debug overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from object_anchor_config import load_object_anchor_config  # noqa: E402
from object_anchor_pose import estimate_object_pose, rpy_deg_to_rotation_matrix  # noqa: E402
from object_anchor_visualizer import (  # noqa: E402
    draw_keypoint_legend,
    draw_object_anchor_keypoints,
    draw_object_pose_axes,
)


def _rotation_error_deg(estimated: np.ndarray, expected: np.ndarray) -> float:
    relative = estimated @ expected.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/object_anchors/tissue_box_01_front_only.yaml",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_object_anchor_config(config_path)
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros((5, 1), dtype=np.float64)
    expected_rpy_deg = (
        (82.0, -12.0, 18.0)
        if config.anchor_mode == "front_only"
        else (8.0, -12.0, 18.0)
    )
    expected_rotation = rpy_deg_to_rotation_matrix(*expected_rpy_deg)
    expected_rvec, _ = cv2.Rodrigues(expected_rotation)
    expected_tvec = np.array([[0.08], [-0.04], [1.20]], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        config.object_points, expected_rvec, expected_tvec, K, distortion
    )
    keypoints = projected.reshape(-1, 2)
    rng = np.random.default_rng(20260721)
    keypoints += rng.normal(0.0, 0.30, size=keypoints.shape)
    confidences = np.full(len(keypoints), 0.98, dtype=np.float64)
    visibility = np.full(len(keypoints), 2, dtype=np.int32)

    # The 8-point mode can lose one corner; front_only requires all four points.
    if len(keypoints) > config.pose_settings.min_correspondences:
        keypoints[-2] += np.array([35.0, -30.0])
        confidences[-2] = 0.10
        visibility[-2] = 0

    pose = estimate_object_pose(
        keypoints,
        config.object_points,
        K,
        dist_coeffs=distortion,
        confidences=confidences,
        visibility=visibility,
        settings=config.pose_settings,
    )
    if not pose.valid or pose.T_camera_object is None or pose.rotation_matrix is None:
        raise SystemExit(f"Synthetic pose recovery failed: {pose.reason}")

    translation_error_m = float(np.linalg.norm(pose.tvec.reshape(3) - expected_tvec.reshape(3)))
    rotation_error_deg = _rotation_error_deg(pose.rotation_matrix, expected_rotation)
    canvas = np.full((800, 1280, 3), 245, dtype=np.uint8)
    canvas = draw_object_anchor_keypoints(
        canvas,
        keypoints,
        config,
        confidences=confidences,
        visibility=visibility,
    )
    canvas = draw_object_pose_axes(canvas, pose, K, dist_coeffs=distortion)
    canvas = draw_keypoint_legend(canvas, config, origin=(25, 285))
    lines = [
        f"Object Anchor: {config.object_id}",
        f"valid={pose.valid} reason={pose.reason}",
        f"tvec_m={np.array2string(pose.tvec.reshape(3), precision=5)}",
        f"rpy_deg={np.array2string(np.asarray(pose.rpy_deg), precision=3)}",
        f"expected_rpy_deg={np.array2string(np.asarray(expected_rpy_deg), precision=3)}",
        f"inliers={pose.inlier_count}/{len(pose.correspondence_indices)}",
        f"mean_reprojection_error_px={pose.mean_reprojection_error_px:.4f}",
        f"translation_error_m={translation_error_m:.6f}",
        f"rotation_error_deg={rotation_error_deg:.4f}",
    ]
    for row, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (25, 35 + row * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "out" / "object_anchor" / f"synthetic_{config.anchor_mode}_pose.png"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise SystemExit(f"Could not write {output_path}")

    print("T_camera_object=")
    print(np.array2string(pose.T_camera_object, precision=6, suppress_small=True))
    for line in lines[2:]:
        print(line)
    print(f"overlay={output_path}")


if __name__ == "__main__":
    main()
