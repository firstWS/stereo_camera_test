#!/usr/bin/env python3
"""Isolated AprilTag-vs-Object-Anchor camera-pose comparison experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from apriltag_world import (  # noqa: E402
    AprilTagWorldResult,
    build_apriltag_world_config,
    estimate_apriltag_world,
)
from apriltag_scale import _dict_id  # noqa: E402
from detect import pick_primary_box  # noqa: E402
from object_anchor_registration import load_world_pose_registration  # noqa: E402
from object_anchor_runtime import build_optional_object_anchor_runtime  # noqa: E402
from object_anchor_world import average_transforms, rotation_delta_deg  # noqa: E402
from orbbec_rgbd_capture import OrbbecRGBDCapture  # noqa: E402
from repeatability_run import build_detector  # noqa: E402
from rgbd_geometry import depth_estimate_rgbd_bbox  # noqa: E402

FRAME_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "elapsed_s",
    "fps",
    "anchor_detected",
    "anchor_bbox_confidence",
    "valid_keypoints",
    "pnp_raw_computed",
    "pnp_operational_valid",
    "pnp_reason",
    "pnp_inliers",
    "reprojection_error_px",
    "reprojection_bucket",
    "skeleton_crossed",
    "operational_temporal_jump",
    "raw_rotation_delta_deg",
    "raw_rotation_jump",
    "apriltag_detected",
    "apriltag_tag_ids",
    "apriltag_reprojection_error_px",
    "tag_cam_x",
    "tag_cam_y",
    "tag_cam_z",
    "anchor_cam_x",
    "anchor_cam_y",
    "anchor_cam_z",
    "translation_error_cm",
    "rotation_error_deg",
)

CUP_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "cup_detected",
    "cup_confidence",
    "cup_valid_depth",
    "cup_valid_depth_ratio",
    "cup_camera_x",
    "cup_camera_y",
    "cup_camera_z",
    "cup_world_tag_x",
    "cup_world_tag_y",
    "cup_world_tag_z",
    "cup_world_object_x",
    "cup_world_object_y",
    "cup_world_object_z",
    "cup_world_difference_cm",
)

BUCKETS = ("0_to_3", "3_to_5", "5_to_7_5", "7_5_to_10", "over_10")

APRILTAG_OBSERVATION_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "tag_id",
    "candidate_index",
    "detected_candidate_count",
    "rejected_candidate_count",
    "corners_xy_json",
    "polygon_area_px2",
    "mean_width_px",
    "mean_height_px",
    "shortest_edge_px",
    "center_distance_px",
    "near_image_boundary",
    "partial_occlusion_suspected",
    "occlusion_assessment",
    "zero_success",
    "zero_rvec_json",
    "zero_tvec_json",
    "zero_reprojection_error_px",
    "zero_T_camera_tag_json",
    "zero_T_world_camera_json",
    "distortion_success",
    "distortion_rvec_json",
    "distortion_tvec_json",
    "distortion_reprojection_error_px",
    "distortion_T_camera_tag_json",
    "distortion_T_world_camera_json",
    "translation_difference_m",
    "rotation_difference_deg",
    "zero_temporal_rotation_jump",
    "zero_temporal_translation_jump",
    "distortion_temporal_rotation_jump",
    "distortion_temporal_translation_jump",
)

APRILTAG_FRAME_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "elapsed_s",
    "fps",
    "rgb_width",
    "rgb_height",
    "camera_matrix_json",
    "orbbec_dist_coeffs_json",
    "detected_candidate_count",
    "configured_tag_count",
    "detected_ids_json",
    "rejected_candidate_count",
    "rejected_candidates_xy_json",
    "zero_pose_valid",
    "zero_T_world_camera_json",
    "zero_reprojection_error_px",
    "zero_translation_delta_m",
    "zero_rotation_delta_deg",
    "zero_translation_jump",
    "zero_rotation_jump",
    "zero_cluster_id",
    "zero_cluster_transition",
    "distortion_pose_valid",
    "distortion_T_world_camera_json",
    "distortion_reprojection_error_px",
    "distortion_translation_delta_m",
    "distortion_rotation_delta_deg",
    "distortion_translation_jump",
    "distortion_rotation_jump",
    "distortion_cluster_id",
    "distortion_cluster_transition",
    "zero_vs_distortion_translation_m",
    "zero_vs_distortion_rotation_deg",
    "mean_tag_area_px2",
    "mean_tag_width_px",
    "mean_tag_height_px",
    "minimum_tag_edge_px",
    "mean_tag_center_distance_px",
    "any_near_image_boundary",
    "any_partial_occlusion_suspected",
    "anchor_detected",
    "anchor_pnp_raw_computed",
    "anchor_pnp_operational_valid",
    "anchor_pnp_reason",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def reprojection_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "no_raw_pnp"
    if value <= 3.0:
        return "0_to_3"
    if value <= 5.0:
        return "3_to_5"
    if value <= 7.5:
        return "5_to_7_5"
    if value <= 10.0:
        return "7_5_to_10"
    return "over_10"


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def compare_transforms(
    T_world_camera_tag: np.ndarray | None,
    T_world_camera_object: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if T_world_camera_tag is None or T_world_camera_object is None:
        return None, None
    translation_m = float(
        np.linalg.norm(
            T_world_camera_tag[:3, 3] - T_world_camera_object[:3, 3]
        )
    )
    rotation_deg = rotation_delta_deg(
        T_world_camera_tag[:3, :3],
        T_world_camera_object[:3, :3],
    )
    return translation_m, rotation_deg


def _json_array(value: np.ndarray | list[Any] | tuple[Any, ...]) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) * 0.5
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def solve_apriltag_candidate(
    corners_xy: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, Any] | None:
    """Replay the production square-tag solver with explicit distortion."""
    image_points = np.ascontiguousarray(corners_xy, dtype=np.float64).reshape(4, 2)
    object_points = _tag_object_points(tag_size_m)
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            distortion,
            flags=flags,
        )
    except cv2.error:
        success = False
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
    if not success:
        return None
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        distortion,
    )
    reprojection_error = float(
        np.mean(
            np.linalg.norm(
                projected.reshape(-1, 2) - image_points,
                axis=1,
            )
        )
    )
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
        "reprojection_error_px": reprojection_error,
        "T_camera_tag": _homogeneous(rotation, tvec),
    }


def tag_pixel_geometry(
    corners_xy: np.ndarray,
    image_shape: tuple[int, int],
    *,
    boundary_margin_px: float,
    occlusion_edge_ratio: float,
) -> dict[str, Any]:
    corners = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    height, width = image_shape
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    area = float(abs(cv2.contourArea(corners.astype(np.float32))))
    center = np.mean(corners, axis=0)
    image_center = np.asarray([width * 0.5, height * 0.5], dtype=np.float64)
    near_boundary = bool(
        np.any(corners[:, 0] <= boundary_margin_px)
        or np.any(corners[:, 1] <= boundary_margin_px)
        or np.any(corners[:, 0] >= width - 1 - boundary_margin_px)
        or np.any(corners[:, 1] >= height - 1 - boundary_margin_px)
    )
    shortest = float(np.min(edges))
    longest = float(np.max(edges))
    convex = bool(cv2.isContourConvex(corners.astype(np.float32)))
    suspected = bool(
        near_boundary
        or not convex
        or (longest > 0.0 and shortest / longest < occlusion_edge_ratio)
    )
    return {
        "polygon_area_px2": area,
        "mean_width_px": float((edges[0] + edges[2]) * 0.5),
        "mean_height_px": float((edges[1] + edges[3]) * 0.5),
        "shortest_edge_px": shortest,
        "center_distance_px": float(np.linalg.norm(center - image_center)),
        "near_image_boundary": near_boundary,
        "partial_occlusion_suspected": suspected,
        "occlusion_assessment": (
            "heuristic_only_boundary_or_geometry"
            if suspected
            else "not_observed_but_not_directly_measurable"
        ),
    }


def pose_cluster_summary(
    transforms: list[np.ndarray],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    min_samples: int,
) -> dict[str, Any]:
    """Small deterministic DBSCAN for SE(3) without a runtime dependency."""
    count = len(transforms)
    if count == 0:
        return {
            "pose_count": 0,
            "cluster_count": 0,
            "noise_count": 0,
            "cluster_sizes": [],
            "largest_cluster_ratio": 0.0,
            "labels": [],
        }
    neighbors: list[list[int]] = []
    for index, first in enumerate(transforms):
        current: list[int] = []
        for other, second in enumerate(transforms):
            if (
                float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
                <= translation_threshold_m
                and rotation_delta_deg(first[:3, :3], second[:3, :3])
                <= rotation_threshold_deg
            ):
                current.append(other)
        neighbors.append(current)
    labels = np.full(count, -1, dtype=np.int32)
    visited = np.zeros(count, dtype=bool)
    cluster = 0
    for index in range(count):
        if visited[index]:
            continue
        visited[index] = True
        if len(neighbors[index]) < min_samples:
            continue
        labels[index] = cluster
        seeds = list(neighbors[index])
        seed_set = set(seeds)
        cursor = 0
        while cursor < len(seeds):
            candidate = seeds[cursor]
            cursor += 1
            if not visited[candidate]:
                visited[candidate] = True
                if len(neighbors[candidate]) >= min_samples:
                    for value in neighbors[candidate]:
                        if value not in seed_set:
                            seeds.append(value)
                            seed_set.add(value)
            if labels[candidate] < 0:
                labels[candidate] = cluster
        cluster += 1
    sizes = sorted(
        [int(np.count_nonzero(labels == value)) for value in range(cluster)],
        reverse=True,
    )
    return {
        "pose_count": count,
        "cluster_count": cluster,
        "noise_count": int(np.count_nonzero(labels < 0)),
        "cluster_sizes": sizes,
        "largest_cluster_ratio": (sizes[0] / count if sizes else 0.0),
        "labels": [int(value) for value in labels],
    }


def transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    point = np.ones(4, dtype=np.float64)
    point[:3] = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    return (np.asarray(transform, dtype=np.float64).reshape(4, 4) @ point)[:3]


def prepare_registration_config(
    base_config_path: Path,
    output_root: Path,
    max_frames: int,
) -> dict[str, str]:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse experiment root: {output_root}")
    output_root.mkdir(parents=True)
    registration_dir = output_root / "registration"
    registration_dir.mkdir()
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    registration_path = registration_dir / "tissue_box_01_world_pose.yaml"
    config["object_anchor"]["registration_file"] = _relative(registration_path)
    config["object_anchor"]["world_validation"]["session_dir"] = _relative(
        registration_dir / "live_world"
    )
    config["repeatability"]["max_frames"] = int(max_frames)
    config["repeatability"]["warmup_frames"] = 0
    config["repeatability"]["output_csv"] = _relative(
        registration_dir / "repeatability.csv"
    )
    config["preview"]["snapshot_dir"] = _relative(registration_dir / "snapshots")
    config_path = registration_dir / "registration_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    metadata = {
        "experiment_root": _relative(output_root),
        "registration_config": _relative(config_path),
        "registration_file": _relative(registration_path),
        "max_registration_frames": str(max_frames),
    }
    (output_root / "experiment_paths.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def _tag_transform(result: AprilTagWorldResult | None) -> np.ndarray | None:
    if result is None or not result.observations:
        return None
    return average_transforms(
        [observation.T_world_camera for observation in result.observations]
    )


def _xyz_fields(prefix: str, transform: np.ndarray | None) -> dict[str, float | str]:
    if transform is None:
        return {f"{prefix}_{axis}": "" for axis in ("x", "y", "z")}
    return {
        f"{prefix}_x": float(transform[0, 3]),
        f"{prefix}_y": float(transform[1, 3]),
        f"{prefix}_z": float(transform[2, 3]),
    }


def _point_fields(prefix: str, point: np.ndarray | None) -> dict[str, float | str]:
    if point is None:
        return {f"{prefix}_{axis}": "" for axis in ("x", "y", "z")}
    return {
        f"{prefix}_x": float(point[0]),
        f"{prefix}_y": float(point[1]),
        f"{prefix}_z": float(point[2]),
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _draw_comparison(
    image: np.ndarray,
    *,
    cup_bbox: Any,
    cup_confidence: float | None,
    translation_cm: float | None,
    rotation_deg: float | None,
    cup_delta_cm: float | None,
    bucket: str,
) -> np.ndarray:
    output = image.copy()
    if cup_bbox is not None:
        x1, y1, x2, y2 = np.rint(cup_bbox.xyxy).astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            output,
            f"cup {cup_confidence:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    lines = [
        f"reproj_bucket={bucket}",
        f"camera_delta_cm={translation_cm:.2f}" if translation_cm is not None else "camera_delta_cm=n/a",
        f"rotation_delta_deg={rotation_deg:.2f}" if rotation_deg is not None else "rotation_delta_deg=n/a",
        f"cup_world_delta_cm={cup_delta_cm:.2f}" if cup_delta_cm is not None else "cup_world_delta_cm=n/a",
    ]
    for index, text in enumerate(lines):
        y = 28 + 24 * index
        cv2.putText(
            output,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return output


def _bucket_rows(
    frame_rows: list[dict[str, Any]],
    cup_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cup_by_frame = {int(row["frame_idx"]): row for row in cup_rows}
    output: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        selected = [row for row in frame_rows if row["reprojection_bucket"] == bucket]
        translation = [
            float(row["translation_error_cm"])
            for row in selected
            if row["translation_error_cm"] != ""
        ]
        rotation = [
            float(row["rotation_error_deg"])
            for row in selected
            if row["rotation_error_deg"] != ""
        ]
        cup_difference = [
            float(cup_by_frame[int(row["frame_idx"])]["cup_world_difference_cm"])
            for row in selected
            if cup_by_frame[int(row["frame_idx"])]["cup_world_difference_cm"] != ""
        ]
        output.append(
            {
                "reprojection_bucket": bucket,
                "frames": len(selected),
                "translation_error_median_cm": distribution(translation)["median"],
                "translation_error_p90_cm": distribution(translation)["p90"],
                "rotation_error_median_deg": distribution(rotation)["median"],
                "rotation_error_p90_deg": distribution(rotation)["p90"],
                "cup_world_difference_median_cm": distribution(cup_difference)["median"],
                "cup_world_difference_p90_cm": distribution(cup_difference)["p90"],
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stability_summary(
    frame_rows: list[dict[str, Any]],
    method: str,
    *,
    translation_cluster_threshold_m: float,
    rotation_cluster_threshold_deg: float,
    cluster_min_samples: int,
) -> dict[str, Any]:
    valid_rows = [row for row in frame_rows if bool(row[f"{method}_pose_valid"])]
    transforms = [
        np.asarray(json.loads(str(row[f"{method}_T_world_camera_json"])), dtype=np.float64)
        for row in valid_rows
    ]
    reprojection = [
        float(row[f"{method}_reprojection_error_px"]) for row in valid_rows
    ]
    translation_delta = [
        float(row[f"{method}_translation_delta_m"])
        for row in valid_rows
        if row[f"{method}_translation_delta_m"] != ""
    ]
    rotation_delta = [
        float(row[f"{method}_rotation_delta_deg"])
        for row in valid_rows
        if row[f"{method}_rotation_delta_deg"] != ""
    ]
    clusters = pose_cluster_summary(
        transforms,
        translation_threshold_m=translation_cluster_threshold_m,
        rotation_threshold_deg=rotation_cluster_threshold_deg,
        min_samples=cluster_min_samples,
    )
    previous_row: dict[str, Any] | None = None
    previous_label: int | None = None
    for row, label in zip(valid_rows, clusters["labels"]):
        row[f"{method}_cluster_id"] = label
        row[f"{method}_cluster_transition"] = bool(
            previous_row is not None
            and int(previous_row["frame_idx"]) == int(row["frame_idx"]) - 1
            and previous_label is not None
            and previous_label >= 0
            and label >= 0
            and previous_label != label
        )
        previous_row = row
        previous_label = label
    for row in frame_rows:
        row.setdefault(f"{method}_cluster_id", "")
        row.setdefault(f"{method}_cluster_transition", False)
    return {
        "total_frames": len(frame_rows),
        "detection_frames": len(valid_rows),
        "detection_rate": len(valid_rows) / max(len(frame_rows), 1),
        "reprojection_error_px": distribution(reprojection),
        "frame_to_frame_translation_delta_m": distribution(translation_delta),
        "frame_to_frame_rotation_delta_deg": distribution(rotation_delta),
        "rotation_jump_ge_30deg": sum(
            bool(row[f"{method}_rotation_jump"]) for row in valid_rows
        ),
        "translation_jump_ge_50cm": sum(
            bool(row[f"{method}_translation_jump"]) for row in valid_rows
        ),
        "pose_clusters": clusters,
    }


def _tag_size_by_jump(
    frame_rows: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    usable = [
        row
        for row in frame_rows
        if bool(row[f"{method}_pose_valid"]) and row["mean_tag_area_px2"] != ""
    ]
    jumped = [
        row
        for row in usable
        if bool(row[f"{method}_rotation_jump"])
        or bool(row[f"{method}_translation_jump"])
        or bool(row[f"{method}_cluster_transition"])
    ]
    normal = [row for row in usable if row not in jumped]

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "frames": len(rows),
            "area_px2": distribution([float(row["mean_tag_area_px2"]) for row in rows]),
            "width_px": distribution([float(row["mean_tag_width_px"]) for row in rows]),
            "height_px": distribution([float(row["mean_tag_height_px"]) for row in rows]),
            "shortest_edge_px": distribution(
                [float(row["minimum_tag_edge_px"]) for row in rows]
            ),
            "center_distance_px": distribution(
                [float(row["mean_tag_center_distance_px"]) for row in rows]
            ),
            "near_boundary_frames": sum(
                bool(row["any_near_image_boundary"]) for row in rows
            ),
            "partial_occlusion_suspected_frames": sum(
                bool(row["any_partial_occlusion_suspected"]) for row in rows
            ),
        }

    jumped_metrics = metrics(jumped)
    normal_metrics = metrics(normal)
    jump_area = jumped_metrics["area_px2"]["median"]
    normal_area = normal_metrics["area_px2"]["median"]
    median_smaller = bool(
        jumped
        and normal
        and jump_area is not None
        and normal_area is not None
        and jump_area < normal_area * 0.75
    )
    strict_small_only = bool(
        jumped
        and normal
        and max(float(row["mean_tag_area_px2"]) for row in jumped)
        < float(
            np.percentile(
                [float(row["mean_tag_area_px2"]) for row in normal],
                25,
            )
        )
    )
    return {
        "jump_frames": jumped_metrics,
        "normal_frames": normal_metrics,
        "jump_tag_area_median_lt_75pct_normal": median_smaller,
        "all_jump_or_transition_areas_below_normal_p25": strict_small_only,
    }


def classify_apriltag_diagnostic(
    zero: dict[str, Any],
    distortion: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    size_comparison: dict[str, Any],
) -> dict[str, Any]:
    zero_event_frames = {
        int(row["frame_idx"])
        for row in frame_rows
        if bool(row["zero_rotation_jump"])
        or bool(row["zero_translation_jump"])
        or bool(row.get("zero_cluster_transition", False))
    }
    distortion_event_frames = {
        int(row["frame_idx"])
        for row in frame_rows
        if bool(row["distortion_rotation_jump"])
        or bool(row["distortion_translation_jump"])
        or bool(row.get("distortion_cluster_transition", False))
    }
    union = zero_event_frames | distortion_event_frames
    overlap_ratio = (
        len(zero_event_frames & distortion_event_frames) / len(union) if union else 1.0
    )
    zero_stable = (
        zero["pose_clusters"]["cluster_count"] == 1
        and zero["pose_clusters"]["largest_cluster_ratio"] >= 0.95
    )
    distortion_stable = (
        distortion["pose_clusters"]["cluster_count"] == 1
        and distortion["pose_clusters"]["largest_cluster_ratio"] >= 0.95
    )
    small_tag_only = bool(
        size_comparison["zero"][
            "all_jump_or_transition_areas_below_normal_p25"
        ]
        or size_comparison["distortion"][
            "all_jump_or_transition_areas_below_normal_p25"
        ]
    )
    if distortion_stable and not zero_stable:
        classification = "distortion_mismatch_primary"
    elif union and overlap_ratio >= 0.80:
        classification = (
            "small_tag_corner_localization_primary"
            if small_tag_only
            else "corner_localization_primary"
        )
    elif zero_stable and distortion_stable:
        classification = "both_stable_no_distortion_advantage"
    else:
        classification = "mixed_or_inconclusive"
    return {
        "classification": classification,
        "zero_stable_single_cluster": zero_stable,
        "distortion_stable_single_cluster": distortion_stable,
        "jump_frame_overlap_ratio": overlap_ratio,
        "zero_jump_or_cluster_transition_frames": sorted(zero_event_frames),
        "distortion_jump_or_cluster_transition_frames": sorted(
            distortion_event_frames
        ),
        "small_tag_only_jump_evidence": small_tag_only,
        "registration_config_preparation_allowed": distortion_stable,
    }


def _prepare_distortion_registration_config(
    config: dict[str, Any],
    output_dir: Path,
    dist_coeffs: np.ndarray,
) -> Path:
    prepared = json.loads(json.dumps(config))
    registration_dir = output_dir / "distortion_registration_candidate"
    registration_file = registration_dir / "tissue_box_01_world_pose.yaml"
    prepared["apriltag_world"]["dist_coeffs"] = [
        float(value) for value in np.asarray(dist_coeffs).reshape(-1)
    ]
    prepared["object_anchor"]["registration_file"] = _relative(registration_file)
    prepared["object_anchor"]["world_validation"]["session_dir"] = _relative(
        registration_dir / "live_world"
    )
    prepared["repeatability"]["max_frames"] = 600
    prepared["repeatability"]["warmup_frames"] = 0
    prepared["repeatability"]["output_csv"] = _relative(
        registration_dir / "repeatability.csv"
    )
    prepared["preview"]["snapshot_dir"] = _relative(registration_dir / "snapshots")
    prepared.setdefault("apriltag_intrinsic_diagnostics", {})[
        "prepared_only_do_not_auto_run"
    ] = True
    path = output_dir / "distortion_registration_ready.yaml"
    path.write_text(
        yaml.safe_dump(prepared, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def run_apriltag_intrinsic_diagnostics(
    *,
    config_path: Path,
    output_dir: Path,
    frames: int = 300,
) -> dict[str, Any]:
    """Live-only isolated comparison; callers must ensure the scene is ready."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    diagnostic = config.get("apriltag_intrinsic_diagnostics") or {}
    if not bool(diagnostic.get("enabled", False)):
        raise RuntimeError("apriltag_intrinsic_diagnostics.enabled must be true")
    if int(frames) != 300:
        raise ValueError("first AprilTag intrinsic diagnostic must use exactly 300 frames")
    apriltag_config = build_apriltag_world_config(config.get("apriltag_world") or {})
    if not apriltag_config.enabled:
        raise RuntimeError("AprilTag world detection must be enabled")
    anchor_runtime, anchor_status = build_optional_object_anchor_runtime(
        config.get("object_anchor"),
        repo_root=ROOT,
    )
    if anchor_runtime is None:
        raise RuntimeError(f"Object Anchor unavailable: {anchor_status}")
    expected_model = (
        "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt"
    )
    if str(config["object_anchor"].get("model_path", "")).replace("\\", "/") != expected_model:
        raise RuntimeError("intrinsic diagnostic requires the unchanged Full99 model")
    orbbec_config = config.get("orbbec")
    if not isinstance(orbbec_config, dict):
        raise RuntimeError("orbbec config block is required")

    boundary_margin_px = float(diagnostic.get("boundary_margin_px", 10.0))
    occlusion_edge_ratio = float(diagnostic.get("occlusion_edge_ratio", 0.40))
    translation_jump_m = float(diagnostic.get("translation_jump_m", 0.50))
    rotation_jump_deg = float(diagnostic.get("rotation_jump_deg", 30.0))
    translation_cluster_m = float(
        diagnostic.get("cluster_translation_threshold_m", 0.25)
    )
    rotation_cluster_deg = float(
        diagnostic.get("cluster_rotation_threshold_deg", 20.0)
    )
    cluster_min_samples = int(diagnostic.get("cluster_min_samples", 3))

    (output_dir / "diagnostic_config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    frame_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    previous: dict[str, tuple[int, np.ndarray] | None] = {
        "zero": None,
        "distortion": None,
    }
    jump_images_saved = 0
    normal_image_saved = False
    first_dist_coeffs: np.ndarray | None = None
    first_camera_matrix: np.ndarray | None = None
    first_resolution: tuple[int, int] | None = None
    cap = OrbbecRGBDCapture(orbbec_config)
    cap.start()
    start = time.perf_counter()
    previous_time: float | None = None
    fps_ema = 0.0
    try:
        while len(frame_rows) < frames:
            ok, frame = cap.read_rgbd()
            if not ok or frame is None:
                if (
                    time.perf_counter() - start
                    > float(orbbec_config.get("startup_timeout_s", 20.0))
                    and not frame_rows
                ):
                    raise RuntimeError("no synchronized Orbbec RGB-D frame before timeout")
                continue
            now = time.perf_counter()
            if previous_time is not None:
                instant = 1.0 / max(now - previous_time, 1e-6)
                fps_ema = instant if fps_ema <= 0.0 else 0.9 * fps_ema + 0.1 * instant
            previous_time = now
            frame_idx = len(frame_rows)
            timestamp = datetime.now(timezone.utc).isoformat()
            rgb = frame.bgr
            height, width = rgb.shape[:2]
            K = np.asarray(frame.K, dtype=np.float64).reshape(3, 3)
            dist = np.asarray(frame.dist_coeffs, dtype=np.float64).reshape(-1, 1)
            if first_dist_coeffs is None:
                first_dist_coeffs = dist.copy()
                first_camera_matrix = K.copy()
                first_resolution = (width, height)
            elif (
                not np.allclose(first_dist_coeffs, dist)
                or not np.allclose(first_camera_matrix, K)
                or first_resolution != (width, height)
            ):
                raise RuntimeError("RGB intrinsic, distortion, or resolution changed mid-run")

            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(_dict_id(apriltag_config.dictionary)),
                cv2.aruco.DetectorParameters(),
            )
            corners, ids, rejected = detector.detectMarkers(gray)
            detected_ids = (
                [int(value) for value in ids.reshape(-1)] if ids is not None else []
            )
            zero_observations: list[dict[str, Any]] = []
            distortion_observations: list[dict[str, Any]] = []
            pending_observations: list[dict[str, Any]] = []
            geometries: list[dict[str, Any]] = []
            for candidate_index, tag_id in enumerate(detected_ids):
                tag = apriltag_config.tags.get(tag_id)
                if tag is None:
                    continue
                points = corners[candidate_index].reshape(4, 2).astype(np.float64)
                geometry = tag_pixel_geometry(
                    points,
                    (height, width),
                    boundary_margin_px=boundary_margin_px,
                    occlusion_edge_ratio=occlusion_edge_ratio,
                )
                geometries.append(geometry)
                solved: dict[str, dict[str, Any] | None] = {
                    "zero": solve_apriltag_candidate(
                        points,
                        apriltag_config.tag_size_m,
                        K,
                        np.zeros_like(dist),
                    ),
                    "distortion": solve_apriltag_candidate(
                        points,
                        apriltag_config.tag_size_m,
                        K,
                        dist,
                    ),
                }
                world: dict[str, np.ndarray | None] = {}
                for method in ("zero", "distortion"):
                    candidate = solved[method]
                    if candidate is None:
                        world[method] = None
                        continue
                    try:
                        T_world_camera = tag.T_world_tag @ np.linalg.inv(
                            candidate["T_camera_tag"]
                        )
                    except np.linalg.LinAlgError:
                        world[method] = None
                        continue
                    world[method] = T_world_camera
                    observations = (
                        zero_observations
                        if method == "zero"
                        else distortion_observations
                    )
                    observations.append(
                        {
                            "tag_id": tag_id,
                            "T_world_camera": T_world_camera,
                            "reprojection_error_px": candidate[
                                "reprojection_error_px"
                            ],
                        }
                    )
                difference_translation, difference_rotation = compare_transforms(
                    world["zero"], world["distortion"]
                )
                pending_observations.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_utc": timestamp,
                        "tag_id": tag_id,
                        "candidate_index": candidate_index,
                        "detected_candidate_count": len(detected_ids),
                        "rejected_candidate_count": len(rejected),
                        "corners_xy_json": _json_array(points),
                        **geometry,
                        "zero_success": solved["zero"] is not None,
                        "zero_rvec_json": (
                            _json_array(solved["zero"]["rvec"])
                            if solved["zero"] is not None
                            else ""
                        ),
                        "zero_tvec_json": (
                            _json_array(solved["zero"]["tvec"])
                            if solved["zero"] is not None
                            else ""
                        ),
                        "zero_reprojection_error_px": (
                            solved["zero"]["reprojection_error_px"]
                            if solved["zero"] is not None
                            else ""
                        ),
                        "zero_T_camera_tag_json": (
                            _json_array(solved["zero"]["T_camera_tag"])
                            if solved["zero"] is not None
                            else ""
                        ),
                        "zero_T_world_camera_json": (
                            _json_array(world["zero"]) if world["zero"] is not None else ""
                        ),
                        "distortion_success": solved["distortion"] is not None,
                        "distortion_rvec_json": (
                            _json_array(solved["distortion"]["rvec"])
                            if solved["distortion"] is not None
                            else ""
                        ),
                        "distortion_tvec_json": (
                            _json_array(solved["distortion"]["tvec"])
                            if solved["distortion"] is not None
                            else ""
                        ),
                        "distortion_reprojection_error_px": (
                            solved["distortion"]["reprojection_error_px"]
                            if solved["distortion"] is not None
                            else ""
                        ),
                        "distortion_T_camera_tag_json": (
                            _json_array(solved["distortion"]["T_camera_tag"])
                            if solved["distortion"] is not None
                            else ""
                        ),
                        "distortion_T_world_camera_json": (
                            _json_array(world["distortion"])
                            if world["distortion"] is not None
                            else ""
                        ),
                        "translation_difference_m": (
                            difference_translation
                            if difference_translation is not None
                            else ""
                        ),
                        "rotation_difference_deg": (
                            difference_rotation if difference_rotation is not None else ""
                        ),
                    }
                )

            fused: dict[str, np.ndarray | None] = {
                "zero": (
                    average_transforms(
                        [item["T_world_camera"] for item in zero_observations]
                    )
                    if zero_observations
                    else None
                ),
                "distortion": (
                    average_transforms(
                        [item["T_world_camera"] for item in distortion_observations]
                    )
                    if distortion_observations
                    else None
                ),
            }
            deltas: dict[str, dict[str, float | bool | str]] = {}
            for method in ("zero", "distortion"):
                transform = fused[method]
                translation_delta: float | str = ""
                rotation_delta: float | str = ""
                previous_item = previous[method]
                if (
                    transform is not None
                    and previous_item is not None
                    and previous_item[0] == frame_idx - 1
                ):
                    translation_delta = float(
                        np.linalg.norm(
                            transform[:3, 3] - previous_item[1][:3, 3]
                        )
                    )
                    rotation_delta = rotation_delta_deg(
                        transform[:3, :3], previous_item[1][:3, :3]
                    )
                if transform is not None:
                    previous[method] = (frame_idx, transform.copy())
                else:
                    previous[method] = None
                deltas[method] = {
                    "translation": translation_delta,
                    "rotation": rotation_delta,
                    "translation_jump": (
                        translation_delta != ""
                        and float(translation_delta) >= translation_jump_m
                    ),
                    "rotation_jump": (
                        rotation_delta != ""
                        and float(rotation_delta) >= rotation_jump_deg
                    ),
                }
            zero_vs_dist_translation, zero_vs_dist_rotation = compare_transforms(
                fused["zero"], fused["distortion"]
            )
            for row in pending_observations:
                row.update(
                    {
                        "zero_temporal_rotation_jump": deltas["zero"][
                            "rotation_jump"
                        ],
                        "zero_temporal_translation_jump": deltas["zero"][
                            "translation_jump"
                        ],
                        "distortion_temporal_rotation_jump": deltas["distortion"][
                            "rotation_jump"
                        ],
                        "distortion_temporal_translation_jump": deltas["distortion"][
                            "translation_jump"
                        ],
                    }
                )
            observation_rows.extend(pending_observations)

            anchor_result = anchor_runtime.process(rgb, K, dist)
            areas = [float(item["polygon_area_px2"]) for item in geometries]
            widths = [float(item["mean_width_px"]) for item in geometries]
            heights = [float(item["mean_height_px"]) for item in geometries]
            shortest = [float(item["shortest_edge_px"]) for item in geometries]
            center_distances = [
                float(item["center_distance_px"]) for item in geometries
            ]
            frame_row: dict[str, Any] = {
                "frame_idx": frame_idx,
                "timestamp_utc": timestamp,
                "elapsed_s": now - start,
                "fps": fps_ema,
                "rgb_width": width,
                "rgb_height": height,
                "camera_matrix_json": _json_array(K),
                "orbbec_dist_coeffs_json": _json_array(dist.reshape(-1)),
                "detected_candidate_count": len(detected_ids),
                "configured_tag_count": len(pending_observations),
                "detected_ids_json": json.dumps(detected_ids),
                "rejected_candidate_count": len(rejected),
                "rejected_candidates_xy_json": json.dumps(
                    [
                        np.asarray(candidate).reshape(-1, 2).tolist()
                        for candidate in rejected
                    ],
                    separators=(",", ":"),
                ),
                "mean_tag_area_px2": float(np.mean(areas)) if areas else "",
                "mean_tag_width_px": float(np.mean(widths)) if widths else "",
                "mean_tag_height_px": float(np.mean(heights)) if heights else "",
                "minimum_tag_edge_px": float(np.min(shortest)) if shortest else "",
                "mean_tag_center_distance_px": (
                    float(np.mean(center_distances)) if center_distances else ""
                ),
                "any_near_image_boundary": any(
                    bool(item["near_image_boundary"]) for item in geometries
                ),
                "any_partial_occlusion_suspected": any(
                    bool(item["partial_occlusion_suspected"]) for item in geometries
                ),
                "anchor_detected": anchor_result.detection is not None,
                "anchor_pnp_raw_computed": (
                    anchor_result.pose.T_camera_object is not None
                ),
                "anchor_pnp_operational_valid": anchor_result.pose.valid,
                "anchor_pnp_reason": anchor_result.pose.reason,
                "zero_vs_distortion_translation_m": (
                    zero_vs_dist_translation
                    if zero_vs_dist_translation is not None
                    else ""
                ),
                "zero_vs_distortion_rotation_deg": (
                    zero_vs_dist_rotation if zero_vs_dist_rotation is not None else ""
                ),
            }
            for method, observations in (
                ("zero", zero_observations),
                ("distortion", distortion_observations),
            ):
                frame_row.update(
                    {
                        f"{method}_pose_valid": fused[method] is not None,
                        f"{method}_T_world_camera_json": (
                            _json_array(fused[method])
                            if fused[method] is not None
                            else ""
                        ),
                        f"{method}_reprojection_error_px": (
                            float(
                                np.mean(
                                    [
                                        item["reprojection_error_px"]
                                        for item in observations
                                    ]
                                )
                            )
                            if observations
                            else ""
                        ),
                        f"{method}_translation_delta_m": deltas[method][
                            "translation"
                        ],
                        f"{method}_rotation_delta_deg": deltas[method]["rotation"],
                        f"{method}_translation_jump": deltas[method][
                            "translation_jump"
                        ],
                        f"{method}_rotation_jump": deltas[method]["rotation_jump"],
                    }
                )
            frame_rows.append(frame_row)

            jumped = any(
                bool(deltas[method]["rotation_jump"])
                or bool(deltas[method]["translation_jump"])
                for method in ("zero", "distortion")
            )
            if jumped and jump_images_saved < 50:
                _write_image(
                    output_dir / "jump_frames" / f"frame_{frame_idx:03d}_raw.jpg",
                    rgb,
                )
                jump_images_saved += 1
            elif not jumped and geometries and not normal_image_saved:
                _write_image(output_dir / "representative_normal_raw.jpg", rgb)
                normal_image_saved = True
    finally:
        cap.release()
        cv2.destroyAllWindows()

    _write_csv(
        output_dir / "apriltag_observation_comparison.csv",
        observation_rows,
        APRILTAG_OBSERVATION_FIELDS,
    )
    zero_summary = _stability_summary(
        frame_rows,
        "zero",
        translation_cluster_threshold_m=translation_cluster_m,
        rotation_cluster_threshold_deg=rotation_cluster_deg,
        cluster_min_samples=cluster_min_samples,
    )
    distortion_summary = _stability_summary(
        frame_rows,
        "distortion",
        translation_cluster_threshold_m=translation_cluster_m,
        rotation_cluster_threshold_deg=rotation_cluster_deg,
        cluster_min_samples=cluster_min_samples,
    )
    size_comparison = {
        "zero": _tag_size_by_jump(frame_rows, "zero"),
        "distortion": _tag_size_by_jump(frame_rows, "distortion"),
    }
    judgment = classify_apriltag_diagnostic(
        zero_summary,
        distortion_summary,
        frame_rows,
        size_comparison,
    )
    _write_csv(
        output_dir / "apriltag_frame_summary.csv",
        frame_rows,
        APRILTAG_FRAME_FIELDS,
    )
    registration_config: str | None = None
    if (
        judgment["registration_config_preparation_allowed"]
        and bool(diagnostic.get("prepare_registration_if_stable", True))
    ):
        assert first_dist_coeffs is not None
        registration_config = _relative(
            _prepare_distortion_registration_config(
                config,
                output_dir,
                first_dist_coeffs,
            )
        )
    summary = {
        "frames": len(frame_rows),
        "output": _relative(output_dir),
        "same_detection_corners_for_both_methods": True,
        "camera": {
            "rgb_resolution": list(first_resolution) if first_resolution else None,
            "camera_matrix": (
                first_camera_matrix.tolist() if first_camera_matrix is not None else None
            ),
            "orbbec_dist_coeffs": (
                first_dist_coeffs.reshape(-1).tolist()
                if first_dist_coeffs is not None
                else None
            ),
        },
        "thresholds_diagnostic_only": {
            "rotation_jump_deg": rotation_jump_deg,
            "translation_jump_m": translation_jump_m,
            "cluster_translation_threshold_m": translation_cluster_m,
            "cluster_rotation_threshold_deg": rotation_cluster_deg,
            "cluster_min_samples": cluster_min_samples,
        },
        "zero_distortion": zero_summary,
        "orbbec_distortion": distortion_summary,
        "tag_size_jump_comparison": size_comparison,
        "judgment": judgment,
        "distortion_registration_ready_config": registration_config,
        "registration_automatically_executed": False,
        "protection": {
            "production_apriltag_code_modified": False,
            "production_object_anchor_code_modified": False,
            "production_calibration_modified": False,
            "cup_code_modified": False,
            "object_anchor_model_or_solver_or_threshold_changed": False,
        },
    }
    (output_dir / "apriltag_intrinsic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "frame_diagnostics.csv",
        frame_rows,
        APRILTAG_FRAME_FIELDS,
    )
    (output_dir / "pose_cluster_summary.json").write_text(
        json.dumps(
            {
                "zero_distortion": zero_summary["pose_clusters"],
                "orbbec_distortion": distortion_summary["pose_clusters"],
                "cluster_definition": {
                    "translation_threshold_m": translation_cluster_m,
                    "rotation_threshold_deg": rotation_cluster_deg,
                    "min_samples": cluster_min_samples,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    jump_rows = [
        {
            "frame_idx": row["frame_idx"],
            "timestamp_utc": row["timestamp_utc"],
            "zero_translation_delta_m": row["zero_translation_delta_m"],
            "zero_rotation_delta_deg": row["zero_rotation_delta_deg"],
            "zero_translation_jump": row["zero_translation_jump"],
            "zero_rotation_jump": row["zero_rotation_jump"],
            "zero_cluster_transition": row["zero_cluster_transition"],
            "distortion_translation_delta_m": row[
                "distortion_translation_delta_m"
            ],
            "distortion_rotation_delta_deg": row["distortion_rotation_delta_deg"],
            "distortion_translation_jump": row["distortion_translation_jump"],
            "distortion_rotation_jump": row["distortion_rotation_jump"],
            "distortion_cluster_transition": row[
                "distortion_cluster_transition"
            ],
            "both_methods_event": (
                (
                    bool(row["zero_translation_jump"])
                    or bool(row["zero_rotation_jump"])
                    or bool(row["zero_cluster_transition"])
                )
                and (
                    bool(row["distortion_translation_jump"])
                    or bool(row["distortion_rotation_jump"])
                    or bool(row["distortion_cluster_transition"])
                )
            ),
            "mean_tag_area_px2": row["mean_tag_area_px2"],
            "mean_tag_width_px": row["mean_tag_width_px"],
            "mean_tag_height_px": row["mean_tag_height_px"],
            "minimum_tag_edge_px": row["minimum_tag_edge_px"],
            "mean_tag_center_distance_px": row["mean_tag_center_distance_px"],
            "any_near_image_boundary": row["any_near_image_boundary"],
            "any_partial_occlusion_suspected": row[
                "any_partial_occlusion_suspected"
            ],
        }
        for row in frame_rows
        if bool(row["zero_translation_jump"])
        or bool(row["zero_rotation_jump"])
        or bool(row["zero_cluster_transition"])
        or bool(row["distortion_translation_jump"])
        or bool(row["distortion_rotation_jump"])
        or bool(row["distortion_cluster_transition"])
    ]
    jump_fields = tuple(jump_rows[0]) if jump_rows else (
        "frame_idx",
        "timestamp_utc",
    )
    _write_csv(output_dir / "jump_analysis.csv", jump_rows, jump_fields)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# AprilTag intrinsic and corner-stability diagnostic",
                "",
                "- Both pose methods use the exact same detected corner arrays and RGB K.",
                "- `zero_distortion` preserves the existing isolated baseline behavior.",
                "- `orbbec_distortion` uses the same coefficients as Object Anchor.",
                "- Object Anchor remains Full99 with its unchanged production PnP path.",
                "- Jump and cluster thresholds are diagnostic-only.",
                "- No registration is executed automatically.",
                f"- Classification: `{judgment['classification']}`",
                f"- Prepared registration config: `{registration_config or 'not prepared'}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_comparison(
    *,
    config_path: Path,
    registration_path: Path,
    output_dir: Path,
    frames: int,
    phase: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registration = load_world_pose_registration(
        registration_path,
        expected_object_id="tissue_box_01",
    )
    cup_detector = build_detector(config)
    anchor_runtime, anchor_status = build_optional_object_anchor_runtime(
        config.get("object_anchor"),
        repo_root=ROOT,
    )
    if anchor_runtime is None:
        raise RuntimeError(f"Object Anchor unavailable: {anchor_status}")
    apriltag_config = build_apriltag_world_config(config.get("apriltag_world") or {})
    if not apriltag_config.enabled:
        raise RuntimeError("AprilTag world detection must be enabled")
    orbbec_config = config.get("orbbec")
    if not isinstance(orbbec_config, dict):
        raise RuntimeError("orbbec config block is required")

    (output_dir / "comparison_config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    cap = OrbbecRGBDCapture(orbbec_config)
    cap.start()
    startup_timeout = float(orbbec_config.get("startup_timeout_s", 20.0))
    z_min = float(orbbec_config.get("roi_z_min_m", 0.05))
    z_max = float(orbbec_config.get("roi_z_max_m", 40.0))
    min_valid_ratio = float(orbbec_config.get("min_valid_depth_ratio", 0.03))
    frame_rows: list[dict[str, Any]] = []
    cup_rows: list[dict[str, Any]] = []
    raw_rotation_previous: np.ndarray | None = None
    start = time.perf_counter()
    previous_time: float | None = None
    fps_ema = 0.0
    successful_example: tuple[np.ndarray, np.ndarray] | None = None
    highest_error_example: tuple[float, np.ndarray, np.ndarray] | None = None

    try:
        while len(frame_rows) < frames:
            ok, frame = cap.read_rgbd()
            if not ok or frame is None:
                if time.perf_counter() - start > startup_timeout and not frame_rows:
                    raise RuntimeError("no synchronized Orbbec RGB-D frame before timeout")
                continue
            now = time.perf_counter()
            if previous_time is not None:
                instantaneous = 1.0 / max(now - previous_time, 1e-6)
                fps_ema = instantaneous if fps_ema <= 0 else 0.9 * fps_ema + 0.1 * instantaneous
            previous_time = now
            index = len(frame_rows)
            timestamp = datetime.now(timezone.utc).isoformat()
            rgb = frame.bgr
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            tag_overlay = rgb.copy()
            tag_result = estimate_apriltag_world(
                gray,
                frame.K,
                apriltag_config,
                draw_on_bgr=tag_overlay,
            )
            tag_transform = _tag_transform(tag_result)
            anchor_result = anchor_runtime.process(
                rgb,
                frame.K,
                frame.dist_coeffs,
                draw_on_bgr=tag_overlay,
            )
            detection = anchor_result.detection
            pose = anchor_result.pose
            valid_keypoints = (
                int(np.count_nonzero(anchor_result.effective_visibility >= 1))
                if anchor_result.effective_visibility is not None
                else 0
            )
            raw_transform = pose.T_camera_object
            object_camera_transform: np.ndarray | None = None
            if raw_transform is not None:
                try:
                    object_camera_transform = registration @ np.linalg.inv(raw_transform)
                except np.linalg.LinAlgError:
                    object_camera_transform = None
            translation_m, rotation_deg = compare_transforms(
                tag_transform,
                object_camera_transform,
            )
            raw_rotation_delta: float | None = None
            if raw_transform is not None:
                current_rotation = raw_transform[:3, :3]
                if raw_rotation_previous is not None:
                    raw_rotation_delta = rotation_delta_deg(
                        current_rotation,
                        raw_rotation_previous,
                    )
                raw_rotation_previous = current_rotation.copy()

            cup_detections = cup_detector.predict(rgb)
            cup_bbox = pick_primary_box(cup_detections)
            cup_estimate = (
                depth_estimate_rgbd_bbox(
                    frame.depth_m,
                    cup_bbox,
                    frame.K,
                    min_valid_ratio=min_valid_ratio,
                    z_min_m=z_min,
                    z_max_m=z_max,
                )
                if cup_bbox is not None
                else None
            )
            cup_camera = (
                np.asarray([cup_estimate.X, cup_estimate.Y, cup_estimate.Z], dtype=np.float64)
                if cup_estimate is not None and cup_estimate.valid
                else None
            )
            cup_world_tag = (
                transform_point(tag_transform, cup_camera)
                if tag_transform is not None and cup_camera is not None
                else None
            )
            cup_world_object = (
                transform_point(object_camera_transform, cup_camera)
                if object_camera_transform is not None and cup_camera is not None
                else None
            )
            cup_delta_cm = (
                float(np.linalg.norm(cup_world_tag - cup_world_object) * 100.0)
                if cup_world_tag is not None and cup_world_object is not None
                else None
            )
            bucket = reprojection_bucket(pose.mean_reprojection_error_px)
            operational_temporal = pose.reason.startswith(
                ("rotation_jump", "translation_jump")
            )
            frame_row: dict[str, Any] = {
                "frame_idx": index,
                "timestamp_utc": timestamp,
                "elapsed_s": now - start,
                "fps": fps_ema,
                "anchor_detected": detection is not None,
                "anchor_bbox_confidence": detection.score if detection is not None else "",
                "valid_keypoints": valid_keypoints,
                "pnp_raw_computed": raw_transform is not None,
                "pnp_operational_valid": pose.valid,
                "pnp_reason": pose.reason,
                "pnp_inliers": pose.inlier_count,
                "reprojection_error_px": (
                    pose.mean_reprojection_error_px
                    if pose.mean_reprojection_error_px is not None
                    else ""
                ),
                "reprojection_bucket": bucket,
                "skeleton_crossed": bool(anchor_result.skeleton_crossings),
                "operational_temporal_jump": operational_temporal,
                "raw_rotation_delta_deg": (
                    raw_rotation_delta if raw_rotation_delta is not None else ""
                ),
                "raw_rotation_jump": (
                    raw_rotation_delta is not None
                    and raw_rotation_delta > anchor_runtime.settings.max_rotation_jump_deg
                ),
                "apriltag_detected": bool(tag_result.observations),
                "apriltag_tag_ids": ",".join(
                    str(value) for value in tag_result.visible_tag_ids
                ),
                "apriltag_reprojection_error_px": (
                    float(
                        np.mean(
                            [
                                observation.reprojection_error_px
                                for observation in tag_result.observations
                            ]
                        )
                    )
                    if tag_result.observations
                    else ""
                ),
                **_xyz_fields("tag_cam", tag_transform),
                **_xyz_fields("anchor_cam", object_camera_transform),
                "translation_error_cm": (
                    translation_m * 100.0 if translation_m is not None else ""
                ),
                "rotation_error_deg": rotation_deg if rotation_deg is not None else "",
            }
            cup_row: dict[str, Any] = {
                "frame_idx": index,
                "timestamp_utc": timestamp,
                "cup_detected": cup_bbox is not None,
                "cup_confidence": cup_bbox.confidence if cup_bbox is not None else "",
                "cup_valid_depth": bool(cup_estimate and cup_estimate.valid),
                "cup_valid_depth_ratio": (
                    cup_estimate.valid_pixel_ratio
                    if cup_estimate is not None
                    and cup_estimate.valid_pixel_ratio is not None
                    else ""
                ),
                **_point_fields("cup_camera", cup_camera),
                **_point_fields("cup_world_tag", cup_world_tag),
                **_point_fields("cup_world_object", cup_world_object),
                "cup_world_difference_cm": cup_delta_cm if cup_delta_cm is not None else "",
            }
            overlay = _draw_comparison(
                anchor_result.overlay_bgr,
                cup_bbox=cup_bbox,
                cup_confidence=cup_bbox.confidence if cup_bbox is not None else None,
                translation_cm=translation_m * 100.0 if translation_m is not None else None,
                rotation_deg=rotation_deg,
                cup_delta_cm=cup_delta_cm,
                bucket=bucket,
            )
            if phase == "preflight":
                _write_image(output_dir / "frames" / f"frame_{index:03d}_raw.jpg", rgb)
                _write_image(
                    output_dir / "frames" / f"frame_{index:03d}_overlay.jpg",
                    overlay,
                )
            if (
                successful_example is None
                and pose.valid
                and tag_transform is not None
                and cup_world_tag is not None
                and cup_world_object is not None
            ):
                successful_example = (rgb.copy(), overlay.copy())
            error_score = translation_m * 100.0 if translation_m is not None else -1.0
            if (
                highest_error_example is None
                or error_score > highest_error_example[0]
            ):
                highest_error_example = (error_score, rgb.copy(), overlay.copy())
            frame_rows.append(frame_row)
            cup_rows.append(cup_row)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    _write_csv(output_dir / "frame_pose_comparison.csv", frame_rows, FRAME_FIELDS)
    _write_csv(output_dir / "cup_world_comparison.csv", cup_rows, CUP_FIELDS)
    bucket_rows = _bucket_rows(frame_rows, cup_rows)
    _write_csv(
        output_dir / "reprojection_bucket_summary.csv",
        bucket_rows,
        tuple(bucket_rows[0]),
    )
    if successful_example is not None:
        _write_image(output_dir / "representative_success_raw.jpg", successful_example[0])
        _write_image(
            output_dir / "representative_success_overlay.jpg",
            successful_example[1],
        )
    if highest_error_example is not None:
        _write_image(
            output_dir / "representative_high_error_raw.jpg",
            highest_error_example[1],
        )
        _write_image(
            output_dir / "representative_high_error_overlay.jpg",
            highest_error_example[2],
        )

    translation_all = [
        float(row["translation_error_cm"])
        for row in frame_rows
        if row["translation_error_cm"] != ""
    ]
    rotation_all = [
        float(row["rotation_error_deg"])
        for row in frame_rows
        if row["rotation_error_deg"] != ""
    ]
    operational_indices = {
        int(row["frame_idx"])
        for row in frame_rows
        if bool(row["pnp_operational_valid"])
    }
    translation_operational = [
        float(row["translation_error_cm"])
        for row in frame_rows
        if row["translation_error_cm"] != ""
        and int(row["frame_idx"]) in operational_indices
    ]
    rotation_operational = [
        float(row["rotation_error_deg"])
        for row in frame_rows
        if row["rotation_error_deg"] != ""
        and int(row["frame_idx"]) in operational_indices
    ]
    cup_all = [
        float(row["cup_world_difference_cm"])
        for row in cup_rows
        if row["cup_world_difference_cm"] != ""
    ]
    cup_operational = [
        float(row["cup_world_difference_cm"])
        for row in cup_rows
        if row["cup_world_difference_cm"] != ""
        and int(row["frame_idx"]) in operational_indices
    ]
    failure_reasons = Counter(
        str(row["pnp_reason"])
        for row in frame_rows
        if not bool(row["pnp_operational_valid"])
    )
    total = len(frame_rows)
    detection_count = sum(bool(row["anchor_detected"]) for row in frame_rows)
    complete_count = sum(int(row["valid_keypoints"]) == 4 for row in frame_rows)
    raw_jump_count = sum(bool(row["raw_rotation_jump"]) for row in frame_rows)
    operational_jump_count = sum(
        bool(row["operational_temporal_jump"]) for row in frame_rows
    )
    operational_translation = distribution(translation_operational)
    operational_rotation = distribution(rotation_operational)
    operational_cup = distribution(cup_operational)
    temporary_criteria = {
        "translation_median_le_3cm": (
            operational_translation["median"] is not None
            and operational_translation["median"] <= 3.0
        ),
        "translation_p90_le_5cm": (
            operational_translation["p90"] is not None
            and operational_translation["p90"] <= 5.0
        ),
        "rotation_median_le_3deg": (
            operational_rotation["median"] is not None
            and operational_rotation["median"] <= 3.0
        ),
        "rotation_p90_le_5deg": (
            operational_rotation["p90"] is not None
            and operational_rotation["p90"] <= 5.0
        ),
        "cup_median_le_3cm": (
            operational_cup["median"] is not None
            and operational_cup["median"] <= 3.0
        ),
        "cup_p90_le_5cm": (
            operational_cup["p90"] is not None
            and operational_cup["p90"] <= 5.0
        ),
        "raw_rotation_jump_rate_lt_1pct": raw_jump_count / max(total, 1) < 0.01,
        "detection_rate_ge_95pct": detection_count / max(total, 1) >= 0.95,
        "complete_keypoint_rate_ge_95pct": complete_count / max(total, 1) >= 0.95,
    }
    summary: dict[str, Any] = {
        "phase": phase,
        "frames": total,
        "config": _relative(config_path),
        "registration": _relative(registration_path),
        "distortion_note": {
            "object_anchor": "Orbbec frame distortion coefficients",
            "apriltag_world": apriltag_config.dist_coeffs.reshape(-1).tolist(),
            "same_rgb_and_camera_matrix": True,
            "same_distortion_coefficients": bool(
                np.allclose(
                    np.asarray(frame.dist_coeffs).reshape(-1)[:5],
                    apriltag_config.dist_coeffs.reshape(-1)[:5],
                )
            ),
        },
        "coverage": {
            "object_anchor_detection": detection_count,
            "four_valid_keypoints": complete_count,
            "pnp_raw_computed": sum(bool(row["pnp_raw_computed"]) for row in frame_rows),
            "pnp_operational_valid": len(operational_indices),
            "apriltag_detection": sum(bool(row["apriltag_detected"]) for row in frame_rows),
            "cup_detection": sum(bool(row["cup_detected"]) for row in cup_rows),
            "cup_valid_depth": sum(bool(row["cup_valid_depth"]) for row in cup_rows),
            "comparable_camera_pose": len(translation_all),
            "comparable_cup_world": len(cup_all),
        },
        "camera_pose_error_all_raw": {
            "translation_cm": distribution(translation_all),
            "rotation_deg": distribution(rotation_all),
        },
        "camera_pose_error_operational_5px": {
            "translation_cm": operational_translation,
            "rotation_deg": operational_rotation,
        },
        "cup_world_difference_all_raw": distribution(cup_all),
        "cup_world_difference_operational_5px": operational_cup,
        "temporal": {
            "operational_jump_count": operational_jump_count,
            "raw_rotation_jump_count": raw_jump_count,
            "raw_rotation_jump_rate": raw_jump_count / max(total, 1),
        },
        "failure_reasons": dict(failure_reasons),
        "reprojection_buckets": bucket_rows,
        "temporary_research_criteria": temporary_criteria,
        "temporary_research_criteria_passed": all(temporary_criteria.values()),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Object Anchor replacement feasibility — {phase}",
                "",
                f"- Frames: {total}",
                f"- Registration: `{_relative(registration_path)}`",
                "- AprilTag camera pose uses transform averaging across configured visible tags.",
                "- Object Anchor camera pose uses registered T_world_tissue @ inverse(T_camera_tissue).",
                "- The same Cup camera point is transformed through both camera poses.",
                "- The operational 5px gate is unchanged; rejected raw poses are diagnostic only.",
                "- AprilTag world currently uses zero distortion while Object Anchor uses Orbbec distortion.",
                "- Temporary criteria are research-only and are not an operating acceptance specification.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-registration-config")
    prepare.add_argument(
        "--base-config",
        default="configs/experiments/orbbec_gemini_object_anchor_full99.yaml",
    )
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--max-frames", type=int, default=600)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--config",
        default="configs/experiments/orbbec_gemini_object_anchor_full99.yaml",
    )
    run.add_argument("--registration", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--frames", type=int, required=True)
    run.add_argument("--phase", choices=("preflight", "comparison"), required=True)
    intrinsic = subparsers.add_parser("run-apriltag-intrinsic-diagnostics")
    intrinsic.add_argument(
        "--config",
        default="configs/experiments/orbbec_gemini_object_anchor_full99.yaml",
    )
    intrinsic.add_argument("--output")
    intrinsic.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    if args.command == "prepare-registration-config":
        result = prepare_registration_config(
            _resolve(args.base_config),
            _resolve(args.output_root),
            args.max_frames,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == "run-apriltag-intrinsic-diagnostics":
        output = (
            _resolve(args.output)
            if args.output
            else (
                ROOT
                / "out/object_anchor_full99/replacement_feasibility"
                / "apriltag_intrinsic_diagnostics"
                / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        )
        run_apriltag_intrinsic_diagnostics(
            config_path=_resolve(args.config),
            output_dir=output,
            frames=args.frames,
        )
        return
    run_comparison(
        config_path=_resolve(args.config),
        registration_path=_resolve(args.registration),
        output_dir=_resolve(args.output),
        frames=args.frames,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
