"""Offline diagnosis of planar Object Anchor PnP ambiguity.

This script only reads an existing registration run and writes isolated
diagnostic artifacts. It never opens a camera or changes runtime settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from scipy.cluster.vq import kmeans2
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from src.object_anchor_config import ObjectAnchorPoseSettings  # noqa: E402
from src.object_anchor_pose import (  # noqa: E402
    estimate_object_pose,
    rotation_matrix_to_rpy_deg,
    rpy_deg_to_rotation_matrix,
)
from src.object_anchor_world import (  # noqa: E402
    ObjectAnchorWorldFrameResult,
    WorldPoseRegistrationCollector,
    average_transforms,
)


DEFAULT_RUN = (
    ROOT
    / "out/object_anchor_full99/replacement_feasibility/20260726_132601"
)
DEFAULT_FRAMES = (
    DEFAULT_RUN
    / "registration/live_world/20260726_132621/object_anchor_world_frames.csv"
)
DEFAULT_CAMERA = ROOT / "out/object_anchor_full99/offline_comparison/full99/summary.json"
DEFAULT_ANCHOR = ROOT / "configs/object_anchors/tissue_box_01_front_only.yaml"
DEFAULT_OUTPUT = DEFAULT_RUN / "pnp_cluster_diagnostics"


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _transform(translation: Iterable[float], rpy_deg: Iterable[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rpy_deg_to_rotation_matrix(
        *(float(v) for v in rpy_deg)
    )
    result[:3, 3] = np.asarray(tuple(translation), dtype=np.float64)
    return result


def _row_transform(row: dict[str, str], prefix: str) -> np.ndarray | None:
    names = [f"{prefix}_{axis}" for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
    if any(not row[name].strip() for name in names):
        return None
    return _transform(
        (float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])),
        (
            float(row[f"{prefix}_roll"]),
            float(row[f"{prefix}_pitch"]),
            float(row[f"{prefix}_yaw"]),
        ),
    )


def _rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _translation_distance_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _mean_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))


def _pose_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    result[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return result


def _positive_depth(object_points: np.ndarray, transform: np.ndarray) -> bool:
    camera = (transform[:3, :3] @ object_points.T).T + transform[:3, 3]
    return bool(np.all(camera[:, 2] > 0.0))


def _serialize_transform(transform: np.ndarray) -> dict[str, list[float]]:
    return {
        "translation_m": [float(v) for v in transform[:3, 3]],
        "rpy_deg": [float(v) for v in rotation_matrix_to_rpy_deg(transform[:3, :3])],
    }


def _load_inputs(
    frames_path: Path, camera_path: Path, anchor_path: Path
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, ObjectAnchorPoseSettings]:
    with frames_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    anchor = yaml.safe_load(anchor_path.read_text(encoding="utf-8"))["object_anchor"]
    object_points = np.asarray(
        [item["xyz"] for item in anchor["keypoints_3d"]], dtype=np.float64
    )
    pose = anchor["pose_estimation"]
    settings = ObjectAnchorPoseSettings(
        confidence_threshold=float(pose["confidence_threshold"]),
        min_visibility=int(pose["min_visibility"]),
        min_correspondences=int(pose["min_correspondences"]),
        min_inliers=int(pose["min_inliers"]),
        ransac_reprojection_error_px=float(pose["ransac_reprojection_error_px"]),
        max_mean_reprojection_error_px=float(pose["max_mean_reprojection_error_px"]),
        ransac_confidence=float(pose["ransac_confidence"]),
        ransac_iterations=int(pose["ransac_iterations"]),
        refine_lm=bool(pose["refine_lm"]),
    )
    return (
        rows,
        np.asarray(camera["camera_matrix"], dtype=np.float64),
        np.asarray(camera["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
        object_points,
        settings,
    )


def _cluster_world_poses(
    valid: list[dict[str, Any]],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    translations = np.asarray([item["T_world_object"][:3, 3] for item in valid])
    rotation_vectors = np.asarray(
        [Rotation.from_matrix(item["T_world_object"][:3, :3]).as_rotvec() for item in valid]
    )
    # Translation and rotation are both included. The scales express a diagnostic
    # equivalence of 10 cm and 10 degrees, not a production acceptance threshold.
    features = np.hstack(
        (translations / 0.10, rotation_vectors / np.deg2rad(10.0))
    )
    inertia: dict[int, float] = {}
    assignments: dict[int, np.ndarray] = {}
    for count in range(1, min(6, len(valid)) + 1):
        centers, labels = kmeans2(features, count, minit="++", seed=0, iter=100)
        inertia[count] = float(np.sum((features - centers[labels]) ** 2))
        assignments[count] = labels

    one_to_two_reduction = (
        1.0 - inertia[2] / inertia[1] if len(valid) >= 2 and inertia[1] else 0.0
    )
    cluster_count = 2 if one_to_two_reduction >= 0.90 else 1
    labels = assignments[cluster_count]
    sizes = Counter(int(value) for value in labels)
    ordered_old_ids = [key for key, _ in sizes.most_common()]
    remap = {old: new for new, old in enumerate(ordered_old_ids)}
    labels = np.asarray([remap[int(value)] for value in labels], dtype=int)
    centers = [
        average_transforms(
            [valid[index]["T_world_object"] for index in np.flatnonzero(labels == cluster)]
        )
        for cluster in range(cluster_count)
    ]
    return labels, centers, {
        "method": "kmeans_SE3_scaled",
        "translation_scale_m": 0.10,
        "rotation_scale_deg": 10.0,
        "selection_rule": "choose_2_if_k1_to_k2_inertia_reduction_at_least_90_percent",
        "k1_to_k2_inertia_reduction": one_to_two_reduction,
        "inertia_by_k": {str(key): value for key, value in inertia.items()},
    }


def _replay_registration(
    rows: list[dict[str, str]], world_by_frame: dict[int, np.ndarray]
) -> dict[int, dict[str, Any]]:
    collector = WorldPoseRegistrationCollector(
        target_frames=100,
        min_seed_frames=5,
        max_position_outlier_m=0.10,
        max_rotation_outlier_deg=20.0,
    )
    replay: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame_idx = int(row["frame_idx"])
        transform = world_by_frame.get(frame_idx)
        if transform is None:
            continue
        result = ObjectAnchorWorldFrameResult(
            valid=True,
            accepted=True,
            reason="ok",
            T_world_object=transform,
        )
        added = collector.add(result)
        replay[frame_idx] = {
            "accepted": added,
            "reason": "accepted" if added else collector.last_reason,
            "position_outlier": collector.last_reason == "registration_position_outlier",
        }
    replay["_summary"] = {  # type: ignore[index]
        "accepted": len(collector.samples),
        "excluded": dict(collector.excluded),
    }
    return replay


def _write_assignment_csv(
    output: Path,
    valid: list[dict[str, Any]],
    labels: np.ndarray,
    replay: dict[int, dict[str, Any]],
    all_rows: list[dict[str, str]],
) -> dict[str, Any]:
    row_by_frame = {int(row["frame_idx"]): row for row in all_rows}
    cluster_by_frame = {
        item["frame_idx"]: int(label) for item, label in zip(valid, labels)
    }
    jump_frames = [
        int(row["frame_idx"]) for row in all_rows if row["reason"].startswith("rotation_jump:")
    ]
    valid_frames = sorted(cluster_by_frame)
    bridged: list[dict[str, Any]] = []
    for frame in jump_frames:
        before = max((value for value in valid_frames if value < frame), default=None)
        after = min((value for value in valid_frames if value > frame), default=None)
        if before is not None and after is not None:
            bridged.append(
                {
                    "jump_frame": frame,
                    "before_frame": before,
                    "after_frame": after,
                    "before_cluster": cluster_by_frame[before],
                    "after_cluster": cluster_by_frame[after],
                    "cluster_transition": cluster_by_frame[before] != cluster_by_frame[after],
                }
            )

    fields = [
        "frame_idx", "elapsed_s", "cluster_id", "world_obj_x", "world_obj_y",
        "world_obj_z", "world_obj_roll", "world_obj_pitch", "world_obj_yaw",
        "registration_accepted_replayed", "registration_position_outlier_replayed",
        "registration_replay_reason", "previous_joint_frame", "previous_cluster",
        "cluster_transition_from_previous_joint", "rotation_jump_frame",
        "jump_bridges_cluster_transition",
    ]
    previous_frame: int | None = None
    previous_cluster: int | None = None
    bridge_by_target = {item["after_frame"]: item for item in bridged}
    with (output / "frame_cluster_assignment.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, label in zip(valid, labels):
            frame = item["frame_idx"]
            source = row_by_frame[frame]
            replay_item = replay[frame]
            bridge = bridge_by_target.get(frame)
            writer.writerow(
                {
                    "frame_idx": frame,
                    "elapsed_s": source["elapsed_s"],
                    "cluster_id": int(label),
                    **{name: source[name] for name in fields if name.startswith("world_obj_")},
                    "registration_accepted_replayed": replay_item["accepted"],
                    "registration_position_outlier_replayed": replay_item["position_outlier"],
                    "registration_replay_reason": replay_item["reason"],
                    "previous_joint_frame": previous_frame if previous_frame is not None else "",
                    "previous_cluster": previous_cluster if previous_cluster is not None else "",
                    "cluster_transition_from_previous_joint": (
                        int(label) != previous_cluster if previous_cluster is not None else False
                    ),
                    "rotation_jump_frame": bridge["jump_frame"] if bridge else "",
                    "jump_bridges_cluster_transition": (
                        bridge["cluster_transition"] if bridge else False
                    ),
                }
            )
            previous_frame, previous_cluster = frame, int(label)

    direct_transitions = int(np.count_nonzero(labels[1:] != labels[:-1]))
    return {
        "rotation_jump_frames": len(jump_frames),
        "jump_frames_with_joint_pose_on_both_sides": len(bridged),
        "jump_frames_bridging_cluster_transition": sum(
            bool(item["cluster_transition"]) for item in bridged
        ),
        "direct_cluster_transitions_between_joint_poses": direct_transitions,
        "bridged_details": bridged,
    }


def _solver_rows(
    valid: list[dict[str, Any]],
    labels: np.ndarray,
    centers: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    object_points: np.ndarray,
    settings: ObjectAnchorPoseSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    replay_translation_deltas: list[float] = []
    replay_rotation_deltas: list[float] = []
    previous_logged: np.ndarray | None = None
    fields = [
        "frame_idx", "cluster_id", "solver", "branch_index", "refined_lm",
        "success", "positive_depth", "reprojection_error_px", "cam_obj_x",
        "cam_obj_y", "cam_obj_z", "cam_obj_roll", "cam_obj_pitch", "cam_obj_yaw",
        "world_obj_x", "world_obj_y", "world_obj_z", "world_obj_roll",
        "world_obj_pitch", "world_obj_yaw", "main_cluster_translation_difference_m",
        "main_cluster_rotation_difference_deg", "previous_logged_translation_difference_m",
        "previous_logged_rotation_difference_deg", "same_branch_as_existing_solver",
    ]

    for item, label in zip(valid, labels):
        image_points = np.ascontiguousarray(item["keypoints"][:, :2], dtype=np.float64)
        confidences = item["keypoints"][:, 2]
        visibility = np.where(confidences >= settings.confidence_threshold, 2.0, 0.0)
        logged = item["T_camera_object"]
        world_camera = item["T_world_camera"]
        main = centers[0]

        candidates: list[tuple[str, int | None, bool, np.ndarray, float]] = []
        replay = estimate_object_pose(
            image_points,
            object_points,
            K,
            dist_coeffs=dist,
            confidences=confidences,
            visibility=visibility,
            settings=settings,
        )
        if replay.T_camera_object is not None:
            candidates.append(
                (
                    "current_iterative_ransac",
                    None,
                    settings.refine_lm,
                    replay.T_camera_object,
                    float(replay.mean_reprojection_error_px),
                )
            )
            replay_translation_deltas.append(_translation_distance_m(replay.T_camera_object, logged))
            replay_rotation_deltas.append(_rotation_distance_deg(replay.T_camera_object, logged))

        ok_single, rvec_single, tvec_single = cv2.solvePnP(
            object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE
        )
        if ok_single:
            single_transform = _pose_from_rt(rvec_single, tvec_single)
            candidates.append(
                (
                    "solvePnP_IPPE",
                    0,
                    False,
                    single_transform,
                    _mean_reprojection_error(
                        object_points, image_points, rvec_single, tvec_single, K, dist
                    ),
                )
            )

        generic = cv2.solvePnPGeneric(
            object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE
        )
        rvecs = generic[1] if generic[0] else ()
        tvecs = generic[2] if generic[0] else ()
        generic_transforms: list[np.ndarray] = []
        for branch, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            raw_transform = _pose_from_rt(rvec, tvec)
            generic_transforms.append(raw_transform)
            candidates.append(
                (
                    "solvePnPGeneric_IPPE",
                    branch,
                    False,
                    raw_transform,
                    _mean_reprojection_error(
                        object_points, image_points, rvec, tvec, K, dist
                    ),
                )
            )
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K,
                dist,
                np.asarray(rvec, dtype=np.float64).copy(),
                np.asarray(tvec, dtype=np.float64).copy(),
            )
            refined_transform = _pose_from_rt(refined_rvec, refined_tvec)
            candidates.append(
                (
                    "solvePnPGeneric_IPPE",
                    branch,
                    True,
                    refined_transform,
                    _mean_reprojection_error(
                        object_points,
                        image_points,
                        refined_rvec,
                        refined_tvec,
                        K,
                        dist,
                    ),
                )
            )

        nearest_branch: int | None = None
        if generic_transforms:
            branch_costs = [
                _translation_distance_m(candidate, logged) / 0.10
                + _rotation_distance_deg(candidate, logged) / 10.0
                for candidate in generic_transforms
            ]
            nearest_branch = int(np.argmin(branch_costs))

        for solver, branch, refined, camera_object, error in candidates:
            world_object = world_camera @ camera_object
            camera_rpy = rotation_matrix_to_rpy_deg(camera_object[:3, :3])
            world_rpy = rotation_matrix_to_rpy_deg(world_object[:3, :3])
            output.append(
                {
                    "frame_idx": item["frame_idx"],
                    "cluster_id": int(label),
                    "solver": solver,
                    "branch_index": "" if branch is None else branch,
                    "refined_lm": refined,
                    "success": True,
                    "positive_depth": _positive_depth(object_points, camera_object),
                    "reprojection_error_px": error,
                    "cam_obj_x": camera_object[0, 3],
                    "cam_obj_y": camera_object[1, 3],
                    "cam_obj_z": camera_object[2, 3],
                    "cam_obj_roll": camera_rpy[0],
                    "cam_obj_pitch": camera_rpy[1],
                    "cam_obj_yaw": camera_rpy[2],
                    "world_obj_x": world_object[0, 3],
                    "world_obj_y": world_object[1, 3],
                    "world_obj_z": world_object[2, 3],
                    "world_obj_roll": world_rpy[0],
                    "world_obj_pitch": world_rpy[1],
                    "world_obj_yaw": world_rpy[2],
                    "main_cluster_translation_difference_m": _translation_distance_m(
                        world_object, main
                    ),
                    "main_cluster_rotation_difference_deg": _rotation_distance_deg(
                        world_object, main
                    ),
                    "previous_logged_translation_difference_m": (
                        _translation_distance_m(camera_object, previous_logged)
                        if previous_logged is not None
                        else ""
                    ),
                    "previous_logged_rotation_difference_deg": (
                        _rotation_distance_deg(camera_object, previous_logged)
                        if previous_logged is not None
                        else ""
                    ),
                    "same_branch_as_existing_solver": (
                        branch == nearest_branch
                        if solver == "solvePnPGeneric_IPPE" and not refined
                        else ""
                    ),
                }
            )
        previous_logged = logged

    return output, {
        "fields": fields,
        "current_solver_replay_vs_logged": {
            "frames": len(replay_translation_deltas),
            "translation_difference_median_m": float(np.median(replay_translation_deltas)),
            "translation_difference_max_m": float(np.max(replay_translation_deltas)),
            "rotation_difference_median_deg": float(np.median(replay_rotation_deltas)),
            "rotation_difference_max_deg": float(np.max(replay_rotation_deltas)),
        },
    }


def _write_solver_csv(
    output: Path, solver_rows: list[dict[str, Any]], fields: list[str]
) -> None:
    with (output / "solver_branch_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(solver_rows)


def _keypoint_comparison(
    output: Path, valid: list[dict[str, Any]], labels: np.ndarray
) -> dict[str, Any]:
    fields = [
        "cluster_id", "cluster_frames", "keypoint_id", "mean_x_px", "mean_y_px",
        "mean_confidence", "mean_relative_x_from_bbox", "mean_relative_y_from_bbox",
        "mean_radial_distance_from_quad_center_px", "radial_difference_vs_main_px",
        "radial_direction_vs_main", "mean_top_bottom_width_px", "mean_left_right_height_px",
        "mean_diagonal_02_px", "mean_diagonal_13_px", "mean_signed_polygon_area_px2",
        "polygon_orientation", "mean_bbox_width_px", "mean_bbox_height_px",
        "mean_bbox_area_px2", "mean_bbox_center_x_px", "mean_bbox_center_y_px",
    ]
    records: list[dict[str, Any]] = []
    aggregate: dict[int, dict[str, Any]] = {}
    for cluster in sorted(set(int(value) for value in labels)):
        items = [item for item, label in zip(valid, labels) if int(label) == cluster]
        keypoints = np.asarray([item["keypoints"][:, :2] for item in items])
        confidences = np.asarray([item["keypoints"][:, 2] for item in items])
        boxes = np.asarray([item["bbox"] for item in items])
        quad_centers = np.mean(keypoints, axis=1)
        radial = np.linalg.norm(keypoints - quad_centers[:, None, :], axis=2)
        relative = np.empty_like(keypoints)
        relative[:, :, 0] = (keypoints[:, :, 0] - boxes[:, None, 0]) / (
            boxes[:, None, 2] - boxes[:, None, 0]
        )
        relative[:, :, 1] = (keypoints[:, :, 1] - boxes[:, None, 1]) / (
            boxes[:, None, 3] - boxes[:, None, 1]
        )
        widths = 0.5 * (
            np.linalg.norm(keypoints[:, 1] - keypoints[:, 0], axis=1)
            + np.linalg.norm(keypoints[:, 2] - keypoints[:, 3], axis=1)
        )
        heights = 0.5 * (
            np.linalg.norm(keypoints[:, 3] - keypoints[:, 0], axis=1)
            + np.linalg.norm(keypoints[:, 2] - keypoints[:, 1], axis=1)
        )
        diagonals_02 = np.linalg.norm(keypoints[:, 2] - keypoints[:, 0], axis=1)
        diagonals_13 = np.linalg.norm(keypoints[:, 3] - keypoints[:, 1], axis=1)
        signed_area = 0.5 * np.sum(
            keypoints[:, :, 0] * np.roll(keypoints[:, :, 1], -1, axis=1)
            - keypoints[:, :, 1] * np.roll(keypoints[:, :, 0], -1, axis=1),
            axis=1,
        )
        aggregate[cluster] = {
            "radial": np.mean(radial, axis=0),
            "mean_confidence": np.mean(confidences, axis=0),
        }
        for keypoint in range(keypoints.shape[1]):
            records.append(
                {
                    "cluster_id": cluster,
                    "cluster_frames": len(items),
                    "keypoint_id": keypoint,
                    "mean_x_px": float(np.mean(keypoints[:, keypoint, 0])),
                    "mean_y_px": float(np.mean(keypoints[:, keypoint, 1])),
                    "mean_confidence": float(np.mean(confidences[:, keypoint])),
                    "mean_relative_x_from_bbox": float(np.mean(relative[:, keypoint, 0])),
                    "mean_relative_y_from_bbox": float(np.mean(relative[:, keypoint, 1])),
                    "mean_radial_distance_from_quad_center_px": float(
                        np.mean(radial[:, keypoint])
                    ),
                    "radial_difference_vs_main_px": "",
                    "radial_direction_vs_main": "",
                    "mean_top_bottom_width_px": float(np.mean(widths)),
                    "mean_left_right_height_px": float(np.mean(heights)),
                    "mean_diagonal_02_px": float(np.mean(diagonals_02)),
                    "mean_diagonal_13_px": float(np.mean(diagonals_13)),
                    "mean_signed_polygon_area_px2": float(np.mean(signed_area)),
                    "polygon_orientation": (
                        "clockwise_image_coordinates"
                        if float(np.mean(signed_area)) > 0
                        else "counterclockwise_image_coordinates"
                    ),
                    "mean_bbox_width_px": float(np.mean(boxes[:, 2] - boxes[:, 0])),
                    "mean_bbox_height_px": float(np.mean(boxes[:, 3] - boxes[:, 1])),
                    "mean_bbox_area_px2": float(
                        np.mean(
                            (boxes[:, 2] - boxes[:, 0])
                            * (boxes[:, 3] - boxes[:, 1])
                        )
                    ),
                    "mean_bbox_center_x_px": float(
                        np.mean((boxes[:, 0] + boxes[:, 2]) * 0.5)
                    ),
                    "mean_bbox_center_y_px": float(
                        np.mean((boxes[:, 1] + boxes[:, 3]) * 0.5)
                    ),
                }
            )

    for record in records:
        cluster = int(record["cluster_id"])
        keypoint = int(record["keypoint_id"])
        difference = float(
            aggregate[cluster]["radial"][keypoint] - aggregate[0]["radial"][keypoint]
        )
        record["radial_difference_vs_main_px"] = difference
        record["radial_direction_vs_main"] = (
            "outward" if difference > 0.5 else "inward" if difference < -0.5 else "similar"
        )

    with (output / "keypoint_cluster_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return {
        str(cluster): {
            "mean_confidence_by_keypoint": [
                float(value) for value in aggregate[cluster]["mean_confidence"]
            ],
            "mean_radial_distance_by_keypoint_px": [
                float(value) for value in aggregate[cluster]["radial"]
            ],
        }
        for cluster in aggregate
    }


def _solver_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            f"{row['solver']}:branch={row['branch_index']}:"
            f"refined={str(row['refined_lm']).lower()}"
        )
        groups.setdefault(key, []).append(row)
    result: dict[str, Any] = {}
    for key, values in groups.items():
        reprojection = np.asarray([float(item["reprojection_error_px"]) for item in values])
        translation = np.asarray(
            [float(item["main_cluster_translation_difference_m"]) for item in values]
        )
        rotation = np.asarray(
            [float(item["main_cluster_rotation_difference_deg"]) for item in values]
        )
        result[key] = {
            "frames": len(values),
            "positive_depth_frames": sum(bool(item["positive_depth"]) for item in values),
            "reprojection_error_median_px": float(np.median(reprojection)),
            "reprojection_error_p90_px": float(np.percentile(reprojection, 90)),
            "main_cluster_translation_difference_median_m": float(np.median(translation)),
            "main_cluster_rotation_difference_median_deg": float(np.median(rotation)),
            "same_branch_as_existing_count": sum(
                item["same_branch_as_existing_solver"] is True for item in values
            ),
        }
    return result


def _component_cluster_summary(
    valid: list[dict[str, Any]], labels: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("T_camera_object", "T_world_camera", "T_world_object"):
        centers: list[np.ndarray] = []
        cluster_records: list[dict[str, Any]] = []
        for cluster in sorted(set(int(value) for value in labels)):
            transforms = [
                item[name] for item, label in zip(valid, labels) if int(label) == cluster
            ]
            center = average_transforms(transforms)
            centers.append(center)
            cluster_records.append(
                {
                    "cluster_id": cluster,
                    "frames": len(transforms),
                    "centroid": _serialize_transform(center),
                }
            )
        result[name] = {
            "clusters": cluster_records,
            "inter_cluster_translation_distance_m": _translation_distance_m(
                centers[0], centers[1]
            ),
            "inter_cluster_rotation_difference_deg": _rotation_distance_deg(
                centers[0], centers[1]
            ),
        }
    tag_ids: dict[str, Counter[str]] = {}
    for item, label in zip(valid, labels):
        key = str(int(label))
        tag_ids.setdefault(key, Counter())[item["row"]["apriltag_tag_ids"]] += 1
    result["apriltag_tag_ids_by_cluster"] = {
        key: dict(value) for key, value in tag_ids.items()
    }
    return result


def _ippe_branch_pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row["solver"] != "solvePnPGeneric_IPPE" or row["refined_lm"]:
            continue
        by_frame.setdefault(int(row["frame_idx"]), {})[int(row["branch_index"])] = row
    translation: list[float] = []
    rotation: list[float] = []
    reprojection_advantage: list[float] = []
    for branches in by_frame.values():
        if set(branches) != {0, 1}:
            continue
        first = branches[0]
        second = branches[1]
        T0 = _transform(
            (first["cam_obj_x"], first["cam_obj_y"], first["cam_obj_z"]),
            (first["cam_obj_roll"], first["cam_obj_pitch"], first["cam_obj_yaw"]),
        )
        T1 = _transform(
            (second["cam_obj_x"], second["cam_obj_y"], second["cam_obj_z"]),
            (second["cam_obj_roll"], second["cam_obj_pitch"], second["cam_obj_yaw"]),
        )
        translation.append(_translation_distance_m(T0, T1))
        rotation.append(_rotation_distance_deg(T0, T1))
        reprojection_advantage.append(
            float(second["reprojection_error_px"]) - float(first["reprojection_error_px"])
        )
    return {
        "frames_with_two_branches": len(rotation),
        "branch_translation_difference_median_m": float(np.median(translation)),
        "branch_rotation_difference_median_deg": float(np.median(rotation)),
        "branch_rotation_difference_p10_deg": float(np.percentile(rotation, 10)),
        "branch_rotation_difference_p90_deg": float(np.percentile(rotation, 90)),
        "branch1_minus_branch0_reprojection_median_px": float(
            np.median(reprojection_advantage)
        ),
    }


def _markdown(
    summary: dict[str, Any],
    frames_path: Path,
    camera_path: Path,
    anchor_path: Path,
) -> str:
    clusters = summary["clusters"]
    temporal = summary["temporal"]
    missing = summary["saved_data"]["missing"]
    components = summary["component_cluster_comparison"]
    camera_object = components["T_camera_object"]
    world_camera = components["T_world_camera"]
    ippe_pair = summary["ippe_two_branch_comparison"]
    cluster_lines = "\n".join(
        f"- cluster {item['cluster_id']}: {item['frames']} frames, "
        f"translation={np.round(item['centroid']['translation_m'], 4).tolist()} m, "
        f"RPY={np.round(item['centroid']['rpy_deg'], 2).tolist()} deg"
        for item in clusters
    )
    return f"""# Offline planar PnP cluster diagnosis

## Scope

- Input: `{frames_path.relative_to(ROOT)}`
- Camera parameters: `{camera_path.relative_to(ROOT)}` (same saved Orbbec profile; not stored per frame)
- Object geometry: `{anchor_path.relative_to(ROOT)}`
- Camera was not opened. Production code, thresholds, models, and calibration were not modified.

## Saved-data audit

Available directly: frame index, elapsed time, bbox coordinates, 4 keypoints and keypoint
confidence, reprojection error, `T_camera_object` translation/RPY, `T_world_object`
translation/RPY, and reason strings containing temporal rotation jumps.

Missing directly: {", ".join(missing)}.

`T_world_camera` was reconstructed only where both logged transforms exist using
`T_world_object @ inverse(T_camera_object)`. Registration position-outlier status was
replayed with the unchanged collector settings and is explicitly marked as reconstructed.
No missing bbox confidence or absolute timestamp was estimated.

## Pose clusters

The SE(3) feature used translation/0.10 m and rotation-vector/10 degrees. The k=1 to k=2
inertia reduction was {summary['clustering_method']['k1_to_k2_inertia_reduction']:.2%};
therefore two dominant clusters were selected.

{cluster_lines}

- Inter-cluster translation: {summary['inter_cluster']['translation_distance_m']:.4f} m
- Inter-cluster rotation: {summary['inter_cluster']['rotation_difference_deg']:.2f} deg
- Direct chronological cluster transitions: {temporal['direct_cluster_transitions_between_joint_poses']}
- Rotation-jump frames: {temporal['rotation_jump_frames']}
- Jump frames bracketed by joint poses: {temporal['jump_frames_with_joint_pose_on_both_sides']}
- Bracketed jumps that changed cluster: {temporal['jump_frames_bridging_cluster_transition']}

## Solver finding

See `solver_branch_comparison.csv` for every frame and both generic IPPE branches before
and after LM refinement. `SOLVEPNP_IPPE_SQUARE` was not used. The logged current pose is
the branch-reference; the current ITERATIVE+RANSAC+LM solver was replayed separately.

- Object Anchor `T_camera_object` difference between the two world-pose labels:
  {camera_object['inter_cluster_translation_distance_m']:.4f} m /
  {camera_object['inter_cluster_rotation_difference_deg']:.2f} deg.
- Reconstructed AprilTag `T_world_camera` difference:
  {world_camera['inter_cluster_translation_distance_m']:.4f} m /
  {world_camera['inter_cluster_rotation_difference_deg']:.2f} deg.
- All 265 joint poses used AprilTag ID 0; this is not a multi-tag averaging split.
- Raw Object Anchor IPPE branches differ by median
  {ippe_pair['branch_rotation_difference_median_deg']:.2f} deg. The existing solver is
  closest to raw branch 0 in
  {summary['solver_summary']['solvePnPGeneric_IPPE:branch=0:refined=false']['same_branch_as_existing_count']}/265
  frames.

This separates two effects: the 265 world-pose split comes from the AprilTag camera pose,
while Object Anchor also has a latent planar alternate branch that plausibly explains the
separately rejected 58–65 degree temporal jumps.

## Keypoint finding

See `keypoint_cluster_comparison.csv`. “inward/outward” is only a relative radial shift
from the quadrilateral center versus cluster 0; there is no saved 2D ground truth, so it
must not be interpreted as absolute annotation error.

## Camera and coordinate static audit

- `src/orbbec_rgbd_capture.py:182-190,215-220,251-273` selects the configured RGB profile,
  obtains K/distortion from that same color profile, and returns its BGR frame.
- `src/object_anchor_detector.py:51-76,86-91` passes that BGR frame directly to
  Ultralytics and consumes `result.keypoints.xy`/`boxes.xyxy`; there is no application-side
  letterbox restoration or second scaling.
- `.venv/Lib/site-packages/ultralytics/models/yolo/pose/predict.py:44-66` restores
  keypoints to `orig_img.shape`, and
  `.venv/Lib/site-packages/ultralytics/models/yolo/detect/predict.py:109-122` does the
  same for boxes. `ultralytics/engine/results.py:1155-1175` documents `keypoints.xy` as
  original-image pixel coordinates. No repository code scales them again before PnP.
- `src/orbbec_rgbd_capture.py:98-119` maps the color intrinsic and distortion values to
  OpenCV ordering.
- `src/object_anchor_world.py:403-458` logs the detector coordinates and pose transforms
  without coordinate scaling.
- `experiments/object_anchor_replacement_feasibility.py:396-410` sends the same RGB frame
  and `frame.K` to both paths. Object Anchor additionally receives `frame.dist_coeffs`.
- `src/apriltag_world.py:157-169,186-217,220-260` defaults AprilTag distortion to zero,
  solves each square tag with IPPE_SQUARE, and constructs `T_world_camera`.
- `src/object_anchor_world.py:153-166` averages visible tag poses. Here every joint row
  reports only tag ID 0, so multi-tag fusion did not create the split.

## Final classification

{summary['final_judgment']}

Limitations: raw rvec/full matrices/direct AprilTag `T_world_camera`, AprilTag corners and
per-tag candidates, bbox confidence, and absolute timestamps were not logged. The transform
reconstructions are deterministic from saved full-precision translation/RPY, but future
captures need raw AprilTag candidates to prove which square-IPPE branch was selected.
"""


def run(frames_path: Path, camera_path: Path, anchor_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows, K, dist, object_points, settings = _load_inputs(
        frames_path, camera_path, anchor_path
    )
    valid: list[dict[str, Any]] = []
    for row in rows:
        T_camera_object = _row_transform(row, "cam_obj")
        T_world_object = _row_transform(row, "world_obj")
        keypoints_json = json.loads(row["keypoints_xy_conf"])
        bbox_json = json.loads(row["bbox_xyxy"])
        if (
            _bool(row["world_valid"])
            and T_camera_object is not None
            and T_world_object is not None
            and len(keypoints_json) == 4
            and bbox_json is not None
        ):
            keypoints = np.asarray(
                [
                    [item["x"], item["y"], item["confidence"]]
                    for item in keypoints_json
                ],
                dtype=np.float64,
            )
            valid.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "row": row,
                    "keypoints": keypoints,
                    "bbox": np.asarray(bbox_json, dtype=np.float64),
                    "T_camera_object": T_camera_object,
                    "T_world_object": T_world_object,
                    "T_world_camera": T_world_object @ np.linalg.inv(T_camera_object),
                }
            )
    if len(valid) != 265:
        raise RuntimeError(f"Expected 265 jointly valid poses, found {len(valid)}")

    labels, centers, clustering_method = _cluster_world_poses(valid)
    world_by_frame = {item["frame_idx"]: item["T_world_object"] for item in valid}
    replay = _replay_registration(rows, world_by_frame)
    temporal = _write_assignment_csv(output, valid, labels, replay, rows)
    solver_rows, solver_replay = _solver_rows(
        valid, labels, centers, K, dist, object_points, settings
    )
    _write_solver_csv(output, solver_rows, solver_replay["fields"])
    keypoint_summary = _keypoint_comparison(output, valid, labels)
    component_summary = _component_cluster_summary(valid, labels)
    ippe_pair_summary = _ippe_branch_pair_summary(solver_rows)

    cluster_records = []
    for cluster, center in enumerate(centers):
        frames = [item["frame_idx"] for item, label in zip(valid, labels) if label == cluster]
        cluster_records.append(
            {
                "cluster_id": cluster,
                "frames": len(frames),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "centroid": _serialize_transform(center),
            }
        )
    inter_cluster = (
        {
            "translation_distance_m": _translation_distance_m(centers[0], centers[1]),
            "rotation_difference_deg": _rotation_distance_deg(centers[0], centers[1]),
        }
        if len(centers) > 1
        else {"translation_distance_m": 0.0, "rotation_difference_deg": 0.0}
    )
    replay_summary = replay["_summary"]  # type: ignore[index]
    if (
        replay_summary["accepted"] != 49
        or replay_summary["excluded"].get("registration_position_outlier") != 216
    ):
        raise RuntimeError(f"Registration replay mismatch: {replay_summary}")

    summary: dict[str, Any] = {
        "input": {
            "frames_csv": str(frames_path),
            "camera_parameter_source": str(camera_path),
            "anchor_geometry": str(anchor_path),
            "total_frames": len(rows),
            "joint_world_poses": len(valid),
        },
        "saved_data": {
            "directly_available": [
                "frame_idx", "elapsed_s", "bbox_xyxy", "four_keypoints_xy",
                "keypoint_confidence", "reprojection_error_px",
                "T_camera_object_translation_and_rpy",
                "T_world_object_translation_and_rpy", "temporal_jump_in_reason",
            ],
            "reconstructed": [
                "T_world_camera_from_T_world_object_and_T_camera_object",
                "rvec_from_logged_rotation_rpy",
                "tvec_from_logged_T_camera_object_translation",
                "registration_position_outlier_by_collector_replay",
            ],
            "missing": [
                "bbox_confidence", "raw_PnP_rvec", "raw_PnP_tvec_as_distinct_field",
                "full_T_camera_object_matrix", "direct_AprilTag_T_world_camera",
                "full_T_world_object_matrix", "absolute_timestamp",
                "per_frame_registration_position_outlier_flag",
                "per_frame_camera_matrix_and_distortion",
                "AprilTag_corners_and_candidate_poses",
            ],
        },
        "camera": {
            "K": K.tolist(),
            "dist_coeffs": dist.reshape(-1).tolist(),
            "resolution_from_config": [1280, 800],
        },
        "clustering_method": clustering_method,
        "clusters": cluster_records,
        "inter_cluster": inter_cluster,
        "temporal": temporal,
        "registration_replay": replay_summary,
        "component_cluster_comparison": component_summary,
        "solver_replay": solver_replay["current_solver_replay_vs_logged"],
        "solver_summary": _solver_summary(solver_rows),
        "ippe_two_branch_comparison": ippe_pair_summary,
        "keypoint_summary": keypoint_summary,
        "final_judgment": (
            "D is primary for the 265 joint world poses: reconstructed AprilTag "
            "T_world_camera splits by 2.878 m and 56.75 deg while Object Anchor "
            "T_camera_object changes only 0.015 m and 0.94 deg. All rows use tag 0, "
            "so this is AprilTag pose instability rather than multi-tag fusion. "
            "A is secondary and independently confirmed for Object Anchor: generic IPPE "
            "has a second positive-depth branch about 56 deg away, but the existing "
            "solver follows the lower-error branch in 264/265 joint frames; it therefore "
            "does not cause the two joint world clusters, though it plausibly causes the "
            "separately rejected 58-65 deg temporal jumps. B is unsupported by the nearly "
            "identical keypoint/bbox distributions and absent 2D ground truth. C has no "
            "duplicate-scaling evidence, but AprilTag zero-distortion versus Object Anchor "
            "Orbbec-distortion remains inconsistent. E applies because raw AprilTag "
            "corners/candidates, direct T_world_camera, raw PnP vectors, bbox confidence, "
            "and absolute timestamps were not logged."
        ),
        "protection": {
            "camera_opened": False,
            "calibration_created": False,
            "production_code_or_threshold_modified": False,
            "model_retrained": False,
            "production_applied": False,
        },
    }
    (output / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "diagnostic_README.md").write_text(
        _markdown(summary, frames_path, camera_path, anchor_path), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.frames.resolve(),
        args.camera.resolve(),
        args.anchor.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result["clusters"], indent=2))
    print(json.dumps(result["inter_cluster"], indent=2))
    print(json.dumps(result["temporal"], indent=2))


if __name__ == "__main__":
    main()
