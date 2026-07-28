#!/usr/bin/env python3
"""Isolated 1st-MVP final AprilTag(ID0)-vs-Object-Anchor quantitative comparison.

Does not modify production AprilTag / Object Anchor / Cup code or default configs.
Live camera capture requires an explicit --i-confirm-scene-ready flag.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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

from apriltag_scale import _dict_id  # noqa: E402
from apriltag_world import (  # noqa: E402
    AprilTagWorldResult,
    build_apriltag_world_config,
    estimate_apriltag_world,
)
from detect import pick_primary_box  # noqa: E402
from object_anchor_registration import save_world_pose_registration  # noqa: E402
from object_anchor_runtime import build_optional_object_anchor_runtime  # noqa: E402
from object_anchor_world import average_transforms, rotation_delta_deg  # noqa: E402
from orbbec_rgbd_capture import OrbbecRGBDCapture  # noqa: E402
from repeatability_run import build_detector  # noqa: E402
from rgbd_geometry import depth_estimate_rgbd_bbox  # noqa: E402

_FEASIBILITY_PATH = ROOT / "experiments" / "object_anchor_replacement_feasibility.py"
_SPEC = importlib.util.spec_from_file_location(
    "object_anchor_replacement_feasibility", _FEASIBILITY_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_FEAS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FEAS)

pose_cluster_summary = _FEAS.pose_cluster_summary
distribution = _FEAS.distribution
compare_transforms = _FEAS.compare_transforms
transform_point = _FEAS.transform_point

REFERENCE_CLUSTER_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "apriltag_detected",
    "apriltag_tag_ids",
    "apriltag_corners_xy_json",
    "apriltag_reprojection_error_px",
    "apriltag_rvec_json",
    "apriltag_tvec_json",
    "T_world_camera_tag_json",
    "pose_cluster_id",
    "in_reference_cluster",
    "branch_outlier",
    "anchor_detected",
    "valid_keypoints",
    "pnp_raw_computed",
    "pnp_operational_valid",
    "pnp_reason",
    "reprojection_error_px",
    "T_camera_object_json",
    "T_camera_object_filtered_json",
    "operational_temporal_valid",
    # Cup camera point must be persisted even if the run aborts before comparison.
    "cup_detected",
    "cup_confidence",
    "cup_valid_depth",
    "P_camera_cup_x",
    "P_camera_cup_y",
    "P_camera_cup_z",
)

REGISTRATION_SAMPLE_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "in_reference_cluster",
    "anchor_operational_valid",
    "reprojection_error_px",
    "T_world_camera_tag_json",
    "T_camera_object_json",
    "T_world_object_json",
    "translation_residual_m",
    "rotation_residual_deg",
    "outlier",
    "outlier_reason",
)

FRAME_COMPARE_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "elapsed_s",
    "cup_motion_phase",
    "apriltag_detected",
    "apriltag_reprojection_error_px",
    "apriltag_branch_id",
    "apriltag_reference_valid",
    "apriltag_branch_outlier",
    "apriltag_exclude_reason",
    "T_world_camera_tag_json",
    "T_camera_tag_json",
    "anchor_detected",
    "valid_keypoints",
    "pnp_raw_computed",
    "pnp_operational_valid",
    "pnp_reason",
    "anchor_reprojection_error_px",
    "operational_temporal_jump",
    "T_camera_object_json",
    "T_camera_object_filtered_json",
    "T_camera_tag_predicted_json",
    "T_world_camera_object_json",
    "object_world_pose_valid",
    "pose_comparable",
    "translation_difference_cm",
    "rotation_difference_deg",
    "tag_frame_to_frame_translation_m",
    "tag_frame_to_frame_rotation_deg",
    "object_frame_to_frame_translation_m",
    "object_frame_to_frame_rotation_deg",
    "tag_temporal_jump",
    "object_temporal_jump",
    "cup_detected",
    "cup_valid_depth",
    "P_camera_cup_x",
    "P_camera_cup_y",
    "P_camera_cup_z",
    "P_world_cup_tag_x",
    "P_world_cup_tag_y",
    "P_world_cup_tag_z",
    "P_world_cup_object_x",
    "P_world_cup_object_y",
    "P_world_cup_object_z",
    "cup_world_difference_cm",
)

CUP_COMPARE_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "cup_motion_phase",
    "cup_detected",
    "cup_confidence",
    "cup_valid_depth",
    "apriltag_branch_id",
    "P_camera_cup_x",
    "P_camera_cup_y",
    "P_camera_cup_z",
    "T_world_camera_tag_json",
    "T_world_camera_object_json",
    "P_world_cup_tag_x",
    "P_world_cup_tag_y",
    "P_world_cup_tag_z",
    "P_world_cup_object_x",
    "P_world_cup_object_y",
    "P_world_cup_object_z",
    "cup_world_difference_cm",
    "apriltag_reference_valid",
    "object_anchor_valid",
    "cup_comparable",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _json_matrix(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    return json.dumps(np.asarray(value, dtype=np.float64).tolist(), separators=(",", ":"))


def causal_filter_pose(
    history_including_current: list[np.ndarray],
    window: int,
) -> np.ndarray:
    """Isolated causal SE(3) filter: translation median + quaternion average."""
    if not history_including_current:
        raise ValueError("empty history")
    selected = history_including_current[-max(1, int(window)) :]
    if len(selected) == 1:
        return np.asarray(selected[0], dtype=np.float64).copy()
    return average_transforms(selected, position_median=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def load_mvp_settings(config: dict[str, Any]) -> dict[str, Any]:
    block = config.get("mvp_final_comparison") or {}
    defaults = {
        "registration_frames": 300,
        "preflight_frames": 20,
        "comparison_frames": 300,
        "stationary_frames": 50,
        "cluster_translation_threshold_m": 0.25,
        "cluster_rotation_threshold_deg": 20.0,
        "cluster_min_samples": 3,
        # Legacy single-cluster gate (disabled for branch-aware MVP).
        "reference_min_detection_rate": 0.95,
        "reference_min_largest_cluster_ratio": 0.95,
        "reference_min_cluster_size": 250,
        "use_single_reference_cluster_gate": False,
        # Per-branch relative registration minima.
        "branch_min_joint_valid": 30,
        "branch_min_inliers": 20,
        "registration_min_joint_valid": 30,
        "registration_min_inliers": 20,
        "registration_max_position_outlier_m": 0.10,
        "registration_max_rotation_outlier_deg": 20.0,
        "registration_residual_translation_median_m": 0.05,
        "registration_residual_translation_p90_m": 0.10,
        "registration_residual_rotation_median_deg": 5.0,
        "registration_residual_rotation_p90_deg": 10.0,
        "preflight_min_tag_reference_valid": 15,
        "preflight_min_object_world_pose": 15,
        "preflight_min_pose_comparable": 15,
        "preflight_min_cup_camera": 5,
        "mvp_min_object_world_pose_rate": 0.80,
        "mvp_min_pose_comparable_rate": 0.75,
        "mvp_max_object_temporal_jump_rate": 0.01,
        "mvp_translation_median_cm": 5.0,
        "mvp_translation_p90_cm": 10.0,
        "mvp_rotation_median_deg": 5.0,
        "mvp_rotation_p90_deg": 10.0,
        "mvp_cup_recommended_frames": 100,
        "mvp_cup_median_cm": 5.0,
        "mvp_cup_p90_cm": 10.0,
        "output_root": "out/object_anchor_full99/mvp_final_comparison",
        # Isolated causal filter (selected offline). Production runtime unchanged.
        "temporal_filter_window": 3,
        "branch_aware": True,
    }
    merged = dict(defaults)
    merged.update(block)
    return merged


def assert_single_apriltag_id0(config: dict[str, Any]) -> None:
    tags = ((config.get("apriltag_world") or {}).get("tags")) or {}
    ids = sorted(int(key) for key in tags.keys())
    if ids != [0]:
        raise ValueError(f"MVP config must use AprilTag ID 0 only, got {ids}")


def tag0_observation(result: AprilTagWorldResult | None) -> Any | None:
    if result is None or not result.observations:
        return None
    for observation in result.observations:
        if int(observation.tag_id) == 0:
            return observation
    return None


def tag0_transform(result: AprilTagWorldResult | None) -> np.ndarray | None:
    observation = tag0_observation(result)
    if observation is None:
        return None
    return np.asarray(observation.T_world_camera, dtype=np.float64).reshape(4, 4)


def detect_tag0_corners_xy(
    gray_u8: np.ndarray,
    dictionary_name: str,
) -> np.ndarray | None:
    """Isolated logging helper. Does not alter production AprilTag pose code."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(_dict_id(dictionary_name))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray_u8)
    if ids is None or len(ids) == 0:
        return None
    for index, tag_id in enumerate(ids.reshape(-1)):
        if int(tag_id) == 0:
            return np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
    return None


def camera_tag_rvec_tvec(T_camera_tag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(np.asarray(T_camera_tag[:3, :3], dtype=np.float64))
    tvec = np.asarray(T_camera_tag[:3, 3], dtype=np.float64).reshape(3)
    return rvec.reshape(3), tvec


def evaluate_reference_cluster_gate(
    *,
    total_frames: int,
    detection_count: int,
    cluster_summary: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    detection_rate = detection_count / max(total_frames, 1)
    sizes = list(cluster_summary.get("cluster_sizes") or [])
    largest_size = int(sizes[0]) if sizes else 0
    largest_ratio = float(cluster_summary.get("largest_cluster_ratio") or 0.0)
    passed = (
        detection_rate >= float(settings["reference_min_detection_rate"])
        and largest_ratio >= float(settings["reference_min_largest_cluster_ratio"])
        and largest_size >= int(settings["reference_min_cluster_size"])
    )
    return {
        "passed": passed,
        "total_frames": total_frames,
        "detection_count": detection_count,
        "detection_rate": detection_rate,
        "largest_cluster_size": largest_size,
        "largest_cluster_ratio": largest_ratio,
        "cluster_count": int(cluster_summary.get("cluster_count") or 0),
        "branch_outlier_count": max(detection_count - largest_size, 0),
        "thresholds": {
            "min_detection_rate": settings["reference_min_detection_rate"],
            "min_largest_cluster_ratio": settings["reference_min_largest_cluster_ratio"],
            "min_cluster_size": settings["reference_min_cluster_size"],
        },
        "note": (
            "Largest SE(3) cluster is an MVP temporary reference only; "
            "not physical ground truth."
        ),
    }


def is_near_reference_cluster(
    transform: np.ndarray,
    reference_centroid: np.ndarray,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> bool:
    translation = float(
        np.linalg.norm(transform[:3, 3] - reference_centroid[:3, 3])
    )
    rotation = rotation_delta_deg(transform[:3, :3], reference_centroid[:3, :3])
    return translation <= translation_threshold_m and rotation <= rotation_threshold_deg


def relative_tag_object(
    T_camera_tag: np.ndarray,
    T_camera_object: np.ndarray,
) -> np.ndarray:
    """Project convention: T_tag_object = inv(T_camera_tag) @ T_camera_object."""
    return np.linalg.inv(T_camera_tag) @ T_camera_object


def predict_camera_tag(
    T_camera_object: np.ndarray,
    T_tag_object_registered: np.ndarray,
) -> np.ndarray:
    return T_camera_object @ np.linalg.inv(T_tag_object_registered)


def assign_branch_id(
    transform: np.ndarray,
    branch_centroids: dict[int, np.ndarray],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> int | None:
    best_id: int | None = None
    best_score = float("inf")
    for branch_id, centroid in branch_centroids.items():
        translation = float(np.linalg.norm(transform[:3, 3] - centroid[:3, 3]))
        rotation = rotation_delta_deg(transform[:3, :3], centroid[:3, :3])
        if translation > translation_threshold_m or rotation > rotation_threshold_deg:
            continue
        score = translation + rotation / 180.0
        if score < best_score:
            best_score = score
            best_id = int(branch_id)
    return best_id


def match_pose_to_prototypes(
    transform: np.ndarray,
    prototypes: dict[int, np.ndarray],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    ambiguous_margin_translation_m: float = 0.05,
    ambiguous_margin_rotation_deg: float = 5.0,
) -> dict[str, Any]:
    """Match a live pose to registration prototypes by SE(3) distance.

    Numeric branch IDs are internal labels only. Callers must use the returned
    prototype id solely as a key into the registration-time prototype/calibration map.
    """
    distances: dict[int, dict[str, float]] = {}
    for branch_id, centroid in prototypes.items():
        translation = float(np.linalg.norm(transform[:3, 3] - centroid[:3, 3]))
        rotation = rotation_delta_deg(transform[:3, :3], centroid[:3, :3])
        distances[int(branch_id)] = {
            "translation_m": translation,
            "rotation_deg": rotation,
            "combined": translation + rotation / 180.0,
        }
    if not distances:
        return {
            "branch_id": None,
            "status": "unknown_no_prototypes",
            "distances": {},
        }
    ordered = sorted(distances.items(), key=lambda item: item[1]["combined"])
    best_id, best = ordered[0]
    within = [
        branch_id
        for branch_id, dist in distances.items()
        if dist["translation_m"] <= translation_threshold_m
        and dist["rotation_deg"] <= rotation_threshold_deg
    ]
    if not within:
        return {
            "branch_id": None,
            "status": "unknown_too_far",
            "distances": {str(k): v for k, v in distances.items()},
            "nearest_internal_id": best_id,
        }
    within_sorted = sorted(within, key=lambda branch: distances[branch]["combined"])
    if len(within_sorted) >= 2:
        top, second_id = within_sorted[0], within_sorted[1]
        top_d, second_d = distances[top], distances[second_id]
        if (
            abs(top_d["translation_m"] - second_d["translation_m"])
            <= ambiguous_margin_translation_m
            and abs(top_d["rotation_deg"] - second_d["rotation_deg"])
            <= ambiguous_margin_rotation_deg
        ):
            return {
                "branch_id": None,
                "status": "unknown_ambiguous",
                "distances": {str(k): v for k, v in distances.items()},
                "candidates": within_sorted,
            }
    return {
        "branch_id": int(within_sorted[0]),
        "status": "matched",
        "distances": {str(k): v for k, v in distances.items()},
    }


def t_camera_tag_from_row(row: dict[str, Any]) -> np.ndarray | None:
    if row.get("apriltag_rvec_json") and row.get("apriltag_tvec_json"):
        rvec = np.asarray(json.loads(str(row["apriltag_rvec_json"])), dtype=np.float64)
        tvec = np.asarray(json.loads(str(row["apriltag_tvec_json"])), dtype=np.float64)
        rotation, _ = cv2.Rodrigues(rvec.reshape(3))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        return transform
    return None


def assign_april_tag_branches(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, np.ndarray]]:
    """Cluster AprilTag T_world_camera into branches without a 95% majority gate."""
    valid_indices: list[int] = []
    transforms: list[np.ndarray] = []
    for index, row in enumerate(rows):
        raw = row.get("T_world_camera_tag_json")
        if not raw:
            row["pose_cluster_id"] = ""
            continue
        valid_indices.append(index)
        transforms.append(np.asarray(json.loads(str(raw)), dtype=np.float64))
    summary = pose_cluster_summary(
        transforms,
        translation_threshold_m=float(settings["cluster_translation_threshold_m"]),
        rotation_threshold_deg=float(settings["cluster_rotation_threshold_deg"]),
        min_samples=int(settings["cluster_min_samples"]),
    )
    labels = list(summary.get("labels") or [])
    centroids: dict[int, np.ndarray] = {}
    for label in sorted({int(value) for value in labels if int(value) >= 0}):
        members = [
            transforms[i] for i, value in enumerate(labels) if int(value) == label
        ]
        if members:
            centroids[label] = average_transforms(members, position_median=True)
    for index, row in enumerate(rows):
        if index not in valid_indices:
            continue
        local = valid_indices.index(index)
        label = int(labels[local])
        row["pose_cluster_id"] = label if label >= 0 else ""
    info = {
        "mode": "branch_aware",
        "single_reference_cluster_gate_used": False,
        "pose_clusters": {
            "cluster_count": summary["cluster_count"],
            "cluster_sizes": summary["cluster_sizes"],
            "noise_count": summary["noise_count"],
            "largest_cluster_ratio": summary["largest_cluster_ratio"],
        },
        "branch_ids": sorted(centroids.keys()),
        "note": (
            "AprilTag SE(3) branches are temporary frames for relative comparison; "
            "not physical ground truth. Branches are never mixed into one calibration."
        ),
    }
    return rows, info, centroids


def register_branch_relative_poses(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Register T_tag_object independently for each AprilTag branch."""
    by_branch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        cluster = row.get("pose_cluster_id")
        if cluster == "" or cluster is None:
            continue
        if not bool(row.get("pnp_operational_valid")):
            continue
        if int(row.get("valid_keypoints") or 0) < 4:
            continue
        filtered_json = row.get("T_camera_object_filtered_json") or row.get(
            "T_camera_object_json"
        )
        if not filtered_json:
            continue
        T_camera_tag = t_camera_tag_from_row(row)
        T_camera_object = np.asarray(json.loads(str(filtered_json)), dtype=np.float64)
        if T_camera_tag is None:
            continue
        T_tag_object = relative_tag_object(T_camera_tag, T_camera_object)
        by_branch.setdefault(int(cluster), []).append(
            {
                "frame_idx": row["frame_idx"],
                "timestamp_utc": row["timestamp_utc"],
                "pose_cluster_id": int(cluster),
                "reprojection_error_px": row.get("reprojection_error_px", ""),
                "T_camera_tag_json": _json_matrix(T_camera_tag),
                "T_camera_object_filtered_json": _json_matrix(T_camera_object),
                "T_tag_object": T_tag_object,
                "T_tag_object_json": _json_matrix(T_tag_object),
                # reuse register_world_poses machinery with this key
                "T_world_object": T_tag_object,
                "in_reference_cluster": True,
                "anchor_operational_valid": True,
                "T_world_camera_tag_json": row.get("T_world_camera_tag_json", ""),
                "T_camera_object_json": filtered_json,
                "T_world_object_json": _json_matrix(T_tag_object),
            }
        )
    results: dict[int, dict[str, Any]] = {}
    for branch_id, candidates in sorted(by_branch.items()):
        registration = register_world_poses(
            candidates,
            max_position_outlier_m=float(settings["registration_max_position_outlier_m"]),
            max_rotation_outlier_deg=float(
                settings["registration_max_rotation_outlier_deg"]
            ),
            min_joint_valid=int(settings["branch_min_joint_valid"]),
            min_inliers=int(settings["branch_min_inliers"]),
        )
        residual = evaluate_registration_residual_gate(
            registration.get("samples") or [], settings
        )
        results[branch_id] = {
            **registration,
            "pose_cluster_id": branch_id,
            "T_tag_object": registration.get("T_world_object"),
            "residual_gate": residual,
            "accepted": bool(registration.get("ok")) and bool(residual.get("passed")),
        }
    return results


def register_world_poses(
    candidates: list[dict[str, Any]],
    *,
    max_position_outlier_m: float,
    max_rotation_outlier_deg: float,
    min_joint_valid: int,
    min_inliers: int,
) -> dict[str, Any]:
    if len(candidates) < min_joint_valid:
        return {
            "ok": False,
            "reason": f"joint_valid_below_minimum:{len(candidates)}<{min_joint_valid}",
            "candidate_count": len(candidates),
            "inlier_count": 0,
            "outlier_count": 0,
            "T_world_object": None,
            "samples": [],
        }
    transforms = [
        np.asarray(item["T_world_object"], dtype=np.float64).reshape(4, 4)
        for item in candidates
    ]
    seed = average_transforms(transforms, position_median=True)
    sample_rows: list[dict[str, Any]] = []
    inliers: list[np.ndarray] = []
    for item, transform in zip(candidates, transforms):
        translation_residual = float(np.linalg.norm(transform[:3, 3] - seed[:3, 3]))
        rotation_residual = rotation_delta_deg(transform[:3, :3], seed[:3, :3])
        outlier = False
        reason = ""
        if translation_residual > max_position_outlier_m:
            outlier = True
            reason = "registration_position_outlier"
        elif rotation_residual > max_rotation_outlier_deg:
            outlier = True
            reason = "registration_rotation_outlier"
        else:
            inliers.append(transform)
        sample_rows.append(
            {
                **item,
                "translation_residual_m": translation_residual,
                "rotation_residual_deg": rotation_residual,
                "outlier": outlier,
                "outlier_reason": reason,
            }
        )
    if len(inliers) < min_inliers:
        return {
            "ok": False,
            "reason": f"inliers_below_minimum:{len(inliers)}<{min_inliers}",
            "candidate_count": len(candidates),
            "inlier_count": len(inliers),
            "outlier_count": len(candidates) - len(inliers),
            "T_world_object": None,
            "samples": sample_rows,
            "excluded_reasons": dict(Counter(row["outlier_reason"] for row in sample_rows if row["outlier"])),
        }
    registered = average_transforms(inliers, position_median=True)
    # Recompute residuals against the final registered pose for reporting.
    for row, transform in zip(sample_rows, transforms):
        row["translation_residual_m"] = float(
            np.linalg.norm(transform[:3, 3] - registered[:3, 3])
        )
        row["rotation_residual_deg"] = rotation_delta_deg(
            transform[:3, :3], registered[:3, :3]
        )
    return {
        "ok": True,
        "reason": "ok",
        "candidate_count": len(candidates),
        "inlier_count": len(inliers),
        "outlier_count": len(candidates) - len(inliers),
        "T_world_object": registered,
        "samples": sample_rows,
        "excluded_reasons": dict(
            Counter(row["outlier_reason"] for row in sample_rows if row["outlier"])
        ),
    }


def evaluate_registration_residual_gate(
    samples: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    inliers = [row for row in samples if not bool(row.get("outlier"))]
    translation = [float(row["translation_residual_m"]) for row in inliers]
    rotation = [float(row["rotation_residual_deg"]) for row in inliers]
    translation_stats = distribution(translation)
    rotation_stats = distribution(rotation)
    checks = {
        "translation_median_le_5cm": (
            translation_stats["median"] is not None
            and translation_stats["median"]
            <= float(settings["registration_residual_translation_median_m"])
        ),
        "translation_p90_le_10cm": (
            translation_stats["p90"] is not None
            and translation_stats["p90"]
            <= float(settings["registration_residual_translation_p90_m"])
        ),
        "rotation_median_le_5deg": (
            rotation_stats["median"] is not None
            and rotation_stats["median"]
            <= float(settings["registration_residual_rotation_median_deg"])
        ),
        "rotation_p90_le_10deg": (
            rotation_stats["p90"] is not None
            and rotation_stats["p90"]
            <= float(settings["registration_residual_rotation_p90_deg"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "final_inlier_count": len(inliers),
        "excluded_count": len(samples) - len(inliers),
        "translation_residual_m": translation_stats,
        "rotation_residual_deg": rotation_stats,
    }


def evaluate_preflight_gate(
    *,
    tag_reference_valid: int,
    object_world_pose: int,
    pose_comparable: int,
    cup_camera: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "tag_reference_valid_ge_15": tag_reference_valid
        >= int(settings["preflight_min_tag_reference_valid"]),
        "object_world_pose_ge_15": object_world_pose
        >= int(settings["preflight_min_object_world_pose"]),
        "pose_comparable_ge_15": pose_comparable
        >= int(settings["preflight_min_pose_comparable"]),
        "cup_camera_ge_5": cup_camera >= int(settings["preflight_min_cup_camera"]),
    }
    return {
        "passed": all(checks.values()),
        "counts": {
            "tag_reference_valid": tag_reference_valid,
            "object_world_pose": object_world_pose,
            "pose_comparable": pose_comparable,
            "cup_camera": cup_camera,
        },
        "checks": checks,
    }


def evaluate_mvp_decision(
    *,
    pose_summary: dict[str, Any],
    cup_summary: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    total = max(int(pose_summary.get("total_frames") or 0), 1)
    object_world = int(pose_summary.get("object_anchor_world_pose_count") or 0)
    comparable = int(pose_summary.get("pose_comparable_count") or 0)
    object_jumps = int(pose_summary.get("object_temporal_jump_count") or 0)
    translation = pose_summary.get("translation_difference_cm") or {}
    rotation = pose_summary.get("rotation_difference_deg") or {}
    cup_diff = cup_summary.get("cup_world_difference_cm") or {}
    cup_comparable = int(cup_summary.get("cup_comparable_count") or 0)

    availability = {
        "object_world_pose_rate_ge_80pct": object_world / total
        >= float(settings["mvp_min_object_world_pose_rate"]),
        "pose_comparable_rate_ge_75pct": comparable / total
        >= float(settings["mvp_min_pose_comparable_rate"]),
        "object_temporal_jump_rate_lt_1pct": object_jumps / total
        < float(settings["mvp_max_object_temporal_jump_rate"]),
    }
    pose_diff = {
        "translation_median_le_5cm": (
            translation.get("median") is not None
            and translation["median"] <= float(settings["mvp_translation_median_cm"])
        ),
        "translation_p90_le_10cm": (
            translation.get("p90") is not None
            and translation["p90"] <= float(settings["mvp_translation_p90_cm"])
        ),
        "rotation_median_le_5deg": (
            rotation.get("median") is not None
            and rotation["median"] <= float(settings["mvp_rotation_median_deg"])
        ),
        "rotation_p90_le_10deg": (
            rotation.get("p90") is not None
            and rotation["p90"] <= float(settings["mvp_rotation_p90_deg"])
        ),
    }
    cup_ok = {
        "cup_comparable_ge_recommended": cup_comparable
        >= int(settings["mvp_cup_recommended_frames"]),
        "cup_median_le_5cm": (
            cup_diff.get("median") is not None
            and cup_diff["median"] <= float(settings["mvp_cup_median_cm"])
        ),
        "cup_p90_le_10cm": (
            cup_diff.get("p90") is not None
            and cup_diff["p90"] <= float(settings["mvp_cup_p90_cm"])
        ),
    }
    pose_core_ok = all(availability.values()) and all(pose_diff.values())
    cup_core_ok = bool(cup_ok["cup_median_le_5cm"] and cup_ok["cup_p90_le_10cm"])
    cup_count_ok = bool(cup_ok["cup_comparable_ge_recommended"])

    if pose_core_ok and cup_core_ok and cup_count_ok:
        decision = "A"
        label = "1st_MVP_complete"
        rationale = (
            "Object Anchor 6DoF availability and AprilTag-reference pose/Cup "
            "differences meet 1st-MVP temporary criteria."
        )
    elif pose_core_ok and not cup_count_ok:
        # Pose MVP can complete even when Cup detections are sparse.
        decision = "B"
        label = "1st_MVP_conditional_complete"
        rationale = (
            "Object Anchor pose MVP criteria met; Cup detector coverage or "
            "downstream sample count is insufficient and tracked separately."
        )
    elif pose_core_ok and cup_count_ok and not cup_core_ok:
        decision = "C"
        label = "1st_MVP_incomplete"
        rationale = (
            "Pose availability met MVP thresholds, but Cup world difference "
            "exceeded temporary criteria with enough Cup samples."
        )
    else:
        decision = "C"
        label = "1st_MVP_incomplete"
        rationale = (
            "Object Anchor vs AprilTag-reference pose difference, registration "
            "quality, availability, or temporal stability did not meet 1st-MVP criteria."
        )

    return {
        "decision": decision,
        "label": label,
        "rationale": rationale,
        "availability_checks": availability,
        "pose_difference_checks": pose_diff,
        "cup_checks": cup_ok,
        "pose_core_passed": pose_core_ok,
        "cup_core_passed": cup_core_ok,
        "cup_count_passed": cup_count_ok,
        "criteria_are_temporary_mvp_only": True,
        "april_tag_is_temporary_reference_not_ground_truth": True,
    }


def _draw_overlay(
    image: np.ndarray,
    *,
    translation_cm: float | None,
    rotation_deg: float | None,
    cup_delta_cm: float | None,
    notes: str,
) -> np.ndarray:
    output = image.copy()
    lines = [
        notes,
        f"camera_delta_cm={translation_cm:.2f}" if translation_cm is not None else "camera_delta_cm=n/a",
        f"rotation_delta_deg={rotation_deg:.2f}" if rotation_deg is not None else "rotation_delta_deg=n/a",
        f"cup_world_delta_cm={cup_delta_cm:.2f}" if cup_delta_cm is not None else "cup_world_delta_cm=n/a",
    ]
    for index, text in enumerate(lines):
        y = 28 + 24 * index
        cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return output


def _capture_session_frames(
    *,
    config: dict[str, Any],
    frames: int,
    registered_world_pose: np.ndarray | None,
    reference_centroid: np.ndarray | None,
    settings: dict[str, Any],
    stationary_frames: int,
    phase_name: str,
    branch_registry: dict[int, dict[str, Any]] | None = None,
    branch_centroids: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
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

    cap = OrbbecRGBDCapture(orbbec_config)
    cap.start()
    startup_timeout = float(orbbec_config.get("startup_timeout_s", 20.0))
    z_min = float(orbbec_config.get("roi_z_min_m", 0.05))
    z_max = float(orbbec_config.get("roi_z_max_m", 40.0))
    min_valid_ratio = float(orbbec_config.get("min_valid_depth_ratio", 0.03))
    translation_thresh = float(settings["cluster_translation_threshold_m"])
    rotation_thresh = float(settings["cluster_rotation_threshold_deg"])
    jump_trans = float((config.get("object_anchor") or {}).get("max_translation_jump_m", 0.25))
    jump_rot = float((config.get("object_anchor") or {}).get("max_rotation_jump_deg", 35.0))

    rows: list[dict[str, Any]] = []
    overlays: list[tuple[str, np.ndarray]] = []
    start = time.perf_counter()
    previous_time: float | None = None
    fps_ema = 0.0
    prev_tag: np.ndarray | None = None
    prev_object: np.ndarray | None = None
    object_pose_history: list[np.ndarray] = []
    filter_window = max(1, int(settings.get("temporal_filter_window", 1)))

    try:
        while len(rows) < frames:
            ok, frame = cap.read_rgbd()
            if not ok or frame is None:
                if time.perf_counter() - start > startup_timeout and not rows:
                    raise RuntimeError("no synchronized Orbbec RGB-D frame before timeout")
                continue
            now = time.perf_counter()
            if previous_time is not None:
                instantaneous = 1.0 / max(now - previous_time, 1e-6)
                fps_ema = instantaneous if fps_ema <= 0 else 0.9 * fps_ema + 0.1 * instantaneous
            previous_time = now
            index = len(rows)
            timestamp = datetime.now(timezone.utc).isoformat()
            rgb = frame.bgr
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            overlay = rgb.copy()
            tag_result = estimate_apriltag_world(
                gray,
                frame.K,
                apriltag_config,
                draw_on_bgr=overlay,
            )
            tag_obs = tag0_observation(tag_result)
            tag_transform = tag0_transform(tag_result)
            anchor_result = anchor_runtime.process(
                rgb,
                frame.K,
                frame.dist_coeffs,
                draw_on_bgr=overlay,
            )
            detection = anchor_result.detection
            pose = anchor_result.pose
            valid_keypoints = (
                int(np.count_nonzero(anchor_result.effective_visibility >= 1))
                if anchor_result.effective_visibility is not None
                else 0
            )
            raw_transform = (
                np.asarray(pose.T_camera_object, dtype=np.float64)
                if pose.T_camera_object is not None
                else None
            )
            filtered_transform: np.ndarray | None = None
            if raw_transform is not None and pose.valid:
                object_pose_history.append(raw_transform.copy())
                filtered_transform = causal_filter_pose(object_pose_history, filter_window)
            T_camera_tag = (
                np.asarray(tag_obs.T_camera_tag, dtype=np.float64)
                if tag_obs is not None
                else None
            )
            object_camera_transform: np.ndarray | None = None
            T_camera_tag_predicted: np.ndarray | None = None
            branch_id: int | None = None
            reference_valid = False
            branch_outlier = False
            exclude_reason = ""

            if tag_transform is None or T_camera_tag is None:
                exclude_reason = "apriltag_not_detected"
            elif branch_centroids:
                # Match against ALL registration prototypes by pose, never by
                # independently re-clustered numeric labels.
                match = match_pose_to_prototypes(
                    tag_transform,
                    branch_centroids,
                    translation_threshold_m=translation_thresh,
                    rotation_threshold_deg=rotation_thresh,
                )
                branch_id = match.get("branch_id")
                if branch_id is None:
                    branch_outlier = True
                    exclude_reason = f"apriltag_branch_{match.get('status', 'unknown')}"
                elif not branch_registry or branch_id not in branch_registry:
                    exclude_reason = f"branch_calibration_missing:{branch_id}"
                elif filtered_transform is None or not pose.valid:
                    exclude_reason = "object_anchor_filtered_pose_unavailable"
                else:
                    T_tag_object = branch_registry[branch_id]["T_tag_object"]
                    T_camera_tag_predicted = predict_camera_tag(
                        filtered_transform, T_tag_object
                    )
                    # World camera from Object Anchor via branch-relative prediction.
                    T_world_tag = tag_transform @ T_camera_tag
                    object_camera_transform = T_world_tag @ np.linalg.inv(
                        T_camera_tag_predicted
                    )
                    reference_valid = True
            elif reference_centroid is not None and registered_world_pose is not None:
                # Legacy single-cluster path retained only if explicitly enabled.
                if is_near_reference_cluster(
                    tag_transform,
                    reference_centroid,
                    translation_threshold_m=translation_thresh,
                    rotation_threshold_deg=rotation_thresh,
                ):
                    pose_for_world = (
                        filtered_transform
                        if filtered_transform is not None
                        else raw_transform
                    )
                    if pose_for_world is not None and pose.valid:
                        object_camera_transform = registered_world_pose @ np.linalg.inv(
                            pose_for_world
                        )
                        reference_valid = True
                else:
                    branch_outlier = True
                    exclude_reason = "apriltag_branch_outlier_vs_reference_cluster"
            else:
                # Registration capture: branch assignment happens offline afterward.
                exclude_reason = ""

            if (
                branch_centroids
                and reference_valid
                and T_camera_tag is not None
                and T_camera_tag_predicted is not None
            ):
                translation_m, rotation_deg = compare_transforms(
                    T_camera_tag,
                    T_camera_tag_predicted,
                )
            else:
                translation_m, rotation_deg = compare_transforms(
                    tag_transform if reference_valid else None,
                    object_camera_transform,
                )
            tag_f2f_t = tag_f2f_r = object_f2f_t = object_f2f_r = None
            tag_jump = object_jump = False
            if tag_transform is not None and prev_tag is not None:
                tag_f2f_t = float(np.linalg.norm(tag_transform[:3, 3] - prev_tag[:3, 3]))
                tag_f2f_r = rotation_delta_deg(tag_transform[:3, :3], prev_tag[:3, :3])
                tag_jump = tag_f2f_t > 0.50 or tag_f2f_r > 30.0
            if object_camera_transform is not None and prev_object is not None:
                object_f2f_t = float(
                    np.linalg.norm(object_camera_transform[:3, 3] - prev_object[:3, 3])
                )
                object_f2f_r = rotation_delta_deg(
                    object_camera_transform[:3, :3], prev_object[:3, :3]
                )
                object_jump = object_f2f_t > jump_trans or object_f2f_r > jump_rot
            if tag_transform is not None:
                prev_tag = tag_transform.copy()
            if object_camera_transform is not None:
                prev_object = object_camera_transform.copy()

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
                if reference_valid and tag_transform is not None and cup_camera is not None
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
            motion_phase = "stationary" if index < stationary_frames else "moving"
            operational_temporal_jump = str(pose.reason).startswith(
                ("rotation_jump", "translation_jump")
            )
            corners = detect_tag0_corners_xy(gray, apriltag_config.dictionary)
            corners_json = _json_matrix(corners) if corners is not None else ""
            rvec_json = ""
            tvec_json = ""
            if tag_obs is not None:
                rvec, tvec = camera_tag_rvec_tvec(tag_obs.T_camera_tag)
                rvec_json = json.dumps(rvec.tolist(), separators=(",", ":"))
                tvec_json = json.dumps(tvec.tolist(), separators=(",", ":"))

            row = {
                "frame_idx": index,
                "timestamp_utc": timestamp,
                "elapsed_s": now - start,
                "fps": fps_ema,
                "phase": phase_name,
                "cup_motion_phase": motion_phase,
                "apriltag_detected": tag_obs is not None,
                "apriltag_tag_ids": ",".join(str(v) for v in tag_result.visible_tag_ids),
                "apriltag_corners_xy_json": corners_json,
                "apriltag_reprojection_error_px": (
                    float(tag_obs.reprojection_error_px) if tag_obs is not None else ""
                ),
                "apriltag_rvec_json": rvec_json,
                "apriltag_tvec_json": tvec_json,
                "T_world_camera_tag_json": _json_matrix(tag_transform),
                "T_camera_tag_json": _json_matrix(T_camera_tag),
                "apriltag_branch_id": branch_id if branch_id is not None else "",
                "pose_cluster_id": branch_id if branch_id is not None else "",
                "apriltag_reference_valid": reference_valid,
                "apriltag_branch_outlier": branch_outlier,
                "apriltag_exclude_reason": exclude_reason,
                "anchor_detected": detection is not None,
                "valid_keypoints": valid_keypoints,
                "pnp_raw_computed": raw_transform is not None,
                "pnp_operational_valid": bool(pose.valid),
                "pnp_reason": pose.reason,
                "reprojection_error_px": (
                    pose.mean_reprojection_error_px
                    if pose.mean_reprojection_error_px is not None
                    else ""
                ),
                "anchor_reprojection_error_px": (
                    pose.mean_reprojection_error_px
                    if pose.mean_reprojection_error_px is not None
                    else ""
                ),
                "T_camera_object_json": _json_matrix(raw_transform),
                "T_camera_object_filtered_json": _json_matrix(filtered_transform),
                "T_camera_tag_predicted_json": _json_matrix(T_camera_tag_predicted),
                "operational_temporal_valid": bool(pose.valid),
                "operational_temporal_jump": operational_temporal_jump,
                "T_world_camera_object_json": _json_matrix(object_camera_transform),
                "object_world_pose_valid": object_camera_transform is not None,
                "pose_comparable": translation_m is not None,
                "translation_difference_cm": (
                    translation_m * 100.0 if translation_m is not None else ""
                ),
                "rotation_difference_deg": rotation_deg if rotation_deg is not None else "",
                "tag_frame_to_frame_translation_m": tag_f2f_t if tag_f2f_t is not None else "",
                "tag_frame_to_frame_rotation_deg": tag_f2f_r if tag_f2f_r is not None else "",
                "object_frame_to_frame_translation_m": (
                    object_f2f_t if object_f2f_t is not None else ""
                ),
                "object_frame_to_frame_rotation_deg": (
                    object_f2f_r if object_f2f_r is not None else ""
                ),
                "tag_temporal_jump": tag_jump,
                "object_temporal_jump": object_jump,
                "cup_detected": cup_bbox is not None,
                "cup_confidence": cup_bbox.confidence if cup_bbox is not None else "",
                "cup_valid_depth": bool(cup_estimate and cup_estimate.valid),
                "P_camera_cup_x": float(cup_camera[0]) if cup_camera is not None else "",
                "P_camera_cup_y": float(cup_camera[1]) if cup_camera is not None else "",
                "P_camera_cup_z": float(cup_camera[2]) if cup_camera is not None else "",
                "P_world_cup_tag_x": float(cup_world_tag[0]) if cup_world_tag is not None else "",
                "P_world_cup_tag_y": float(cup_world_tag[1]) if cup_world_tag is not None else "",
                "P_world_cup_tag_z": float(cup_world_tag[2]) if cup_world_tag is not None else "",
                "P_world_cup_object_x": (
                    float(cup_world_object[0]) if cup_world_object is not None else ""
                ),
                "P_world_cup_object_y": (
                    float(cup_world_object[1]) if cup_world_object is not None else ""
                ),
                "P_world_cup_object_z": (
                    float(cup_world_object[2]) if cup_world_object is not None else ""
                ),
                "cup_world_difference_cm": cup_delta_cm if cup_delta_cm is not None else "",
                "object_anchor_valid": object_camera_transform is not None,
                "cup_comparable": cup_delta_cm is not None,
                "rgb": rgb,
                "overlay": None,
            }
            if cup_bbox is not None:
                x1, y1, x2, y2 = np.rint(cup_bbox.xyxy).astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
            row["overlay"] = _draw_overlay(
                overlay,
                translation_cm=translation_m * 100.0 if translation_m is not None else None,
                rotation_deg=rotation_deg,
                cup_delta_cm=cup_delta_cm,
                notes=phase_name,
            )
            rows.append(row)
            overlays.append((f"{phase_name}_{index:03d}", row["overlay"]))
            if (
                phase_name == "comparison"
                and index + 1 == stationary_frames
            ):
                print(
                    "\n"
                    "FINAL COMPARISON: first 50 fixed frames completed.\n"
                    "Move only the Cup slowly and pause at each position.\n"
                    "Do not move the camera, AprilTag, or Object Anchor.\n",
                    flush=True,
                )
            if preview_enabled(config):
                preview = cv2.resize(row["overlay"], None, fx=0.5, fy=0.5)
                cv2.imshow("mvp_final_comparison", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {"rows": rows, "overlays": overlays}


def preview_enabled(config: dict[str, Any]) -> bool:
    return bool((config.get("preview") or {}).get("enabled", True))


def _assign_reference_cluster(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray | None]:
    valid_indices: list[int] = []
    transforms: list[np.ndarray] = []
    for index, row in enumerate(rows):
        raw = row.get("T_world_camera_tag_json")
        if not raw:
            row["pose_cluster_id"] = ""
            row["in_reference_cluster"] = False
            row["branch_outlier"] = False
            continue
        valid_indices.append(index)
        transforms.append(np.asarray(json.loads(str(raw)), dtype=np.float64))
    summary = pose_cluster_summary(
        transforms,
        translation_threshold_m=float(settings["cluster_translation_threshold_m"]),
        rotation_threshold_deg=float(settings["cluster_rotation_threshold_deg"]),
        min_samples=int(settings["cluster_min_samples"]),
    )
    labels = list(summary.get("labels") or [])
    sizes = list(summary.get("cluster_sizes") or [])
    reference_label = None
    if sizes:
        # Map size rank back to label id with matching count.
        size_to_labels: dict[int, list[int]] = {}
        for label in sorted(set(labels)):
            if label < 0:
                continue
            count = sum(1 for value in labels if value == label)
            size_to_labels.setdefault(count, []).append(label)
        reference_label = size_to_labels[sizes[0]][0]
    reference_transforms = [
        transforms[i]
        for i, label in enumerate(labels)
        if reference_label is not None and label == reference_label
    ]
    centroid = (
        average_transforms(reference_transforms, position_median=True)
        if reference_transforms
        else None
    )
    for index, row in enumerate(rows):
        if index not in valid_indices:
            continue
        local = valid_indices.index(index)
        label = int(labels[local])
        in_ref = reference_label is not None and label == reference_label
        row["pose_cluster_id"] = label
        row["in_reference_cluster"] = in_ref
        row["branch_outlier"] = bool(row["apriltag_detected"]) and not in_ref
    gate = evaluate_reference_cluster_gate(
        total_frames=len(rows),
        detection_count=sum(bool(row["apriltag_detected"]) for row in rows),
        cluster_summary=summary,
        settings=settings,
    )
    gate["reference_cluster_id"] = reference_label
    gate["pose_clusters"] = {
        "cluster_count": summary["cluster_count"],
        "cluster_sizes": summary["cluster_sizes"],
        "noise_count": summary["noise_count"],
        "largest_cluster_ratio": summary["largest_cluster_ratio"],
    }
    return rows, gate, centroid


def _build_registration_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("in_reference_cluster")):
            continue
        if not bool(row.get("pnp_operational_valid")):
            continue
        if int(row.get("valid_keypoints") or 0) < 4:
            continue
        if not bool(row.get("pnp_raw_computed")):
            continue
        tag_json = row.get("T_world_camera_tag_json")
        obj_json = row.get("T_camera_object_json")
        if not tag_json or not obj_json:
            continue
        T_world_camera = np.asarray(json.loads(str(tag_json)), dtype=np.float64)
        T_camera_object = np.asarray(json.loads(str(obj_json)), dtype=np.float64)
        T_world_object = T_world_camera @ T_camera_object
        candidates.append(
            {
                "frame_idx": row["frame_idx"],
                "timestamp_utc": row["timestamp_utc"],
                "in_reference_cluster": True,
                "anchor_operational_valid": True,
                "reprojection_error_px": row.get("reprojection_error_px", ""),
                "T_world_camera_tag_json": tag_json,
                "T_camera_object_json": obj_json,
                "T_world_object_json": _json_matrix(T_world_object),
                "T_world_object": T_world_object,
            }
        )
    return candidates


def _summarize_pose_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    translation = [
        float(row["translation_difference_cm"])
        for row in rows
        if row.get("translation_difference_cm") != ""
    ]
    rotation = [
        float(row["rotation_difference_deg"])
        for row in rows
        if row.get("rotation_difference_deg") != ""
    ]
    return {
        "total_frames": total,
        "apriltag_detection_count": sum(bool(row.get("apriltag_detected")) for row in rows),
        "apriltag_reference_valid_count": sum(
            bool(row.get("apriltag_reference_valid")) for row in rows
        ),
        "apriltag_branch_outlier_count": sum(
            bool(row.get("apriltag_branch_outlier")) for row in rows
        ),
        "object_anchor_detection_count": sum(bool(row.get("anchor_detected")) for row in rows),
        "object_anchor_pnp_count": sum(bool(row.get("pnp_operational_valid")) for row in rows),
        "object_anchor_world_pose_count": sum(
            bool(row.get("object_world_pose_valid")) for row in rows
        ),
        "pose_comparable_count": sum(bool(row.get("pose_comparable")) for row in rows),
        "object_temporal_jump_count": sum(bool(row.get("object_temporal_jump")) for row in rows),
        "translation_difference_cm": distribution(translation),
        "rotation_difference_deg": distribution(rotation),
    }


def _summarize_cup_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        float(row["cup_world_difference_cm"])
        for row in rows
        if row.get("cup_world_difference_cm") != ""
    ]
    stationary = [
        float(row["cup_world_difference_cm"])
        for row in rows
        if row.get("cup_world_difference_cm") != ""
        and row.get("cup_motion_phase") == "stationary"
    ]
    moving = [
        float(row["cup_world_difference_cm"])
        for row in rows
        if row.get("cup_world_difference_cm") != ""
        and row.get("cup_motion_phase") == "moving"
    ]
    return {
        "cup_detection_count": sum(bool(row.get("cup_detected")) for row in rows),
        "cup_valid_depth_count": sum(bool(row.get("cup_valid_depth")) for row in rows),
        "cup_comparable_count": len(comparable),
        "cup_world_difference_cm": distribution(comparable),
        "cup_world_difference_stationary_cm": distribution(stationary),
        "cup_world_difference_moving_cm": distribution(moving),
    }


def _branch_metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    translations = [
        float(row["translation_difference_cm"])
        for row in rows
        if row.get("translation_difference_cm") != ""
    ]
    rotations = [
        float(row["rotation_difference_deg"])
        for row in rows
        if row.get("rotation_difference_deg") != ""
    ]
    return {
        "frames": len(rows),
        "pose_comparable_count": sum(bool(row.get("pose_comparable")) for row in rows),
        "object_anchor_detection_count": sum(bool(row.get("anchor_detected")) for row in rows),
        "object_anchor_pnp_count": sum(bool(row.get("pnp_operational_valid")) for row in rows),
        "object_anchor_filtered_pose_count": sum(
            bool(str(row.get("T_camera_object_filtered_json") or "").strip()) for row in rows
        ),
        "translation_difference_cm": distribution(translations),
        "rotation_difference_deg": distribution(rotations),
        "translation_error_ge_10cm": sum(value >= 10.0 for value in translations),
        "rotation_error_ge_10deg": sum(value >= 10.0 for value in rotations),
        "object_temporal_jump_count": sum(bool(row.get("object_temporal_jump")) for row in rows),
    }


def _summarize_pose_rows_branch_aware(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_branch: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("apriltag_branch_id") if row.get("apriltag_branch_id") != "" else "unassigned")
        by_branch.setdefault(key, []).append(row)
    branches = {key: _branch_metric_block(value) for key, value in sorted(by_branch.items())}
    weighted = _branch_metric_block(rows)
    weighted["total_frames"] = len(rows)
    weighted["apriltag_detection_count"] = sum(bool(row.get("apriltag_detected")) for row in rows)
    weighted["object_anchor_world_pose_count"] = sum(
        bool(row.get("object_world_pose_valid")) for row in rows
    )
    return {
        "total_frames": len(rows),
        "branches": branches,
        "weighted": weighted,
        # Compatibility keys for older decision helpers / readme.
        "object_anchor_world_pose_count": weighted["object_anchor_world_pose_count"],
        "pose_comparable_count": weighted["pose_comparable_count"],
        "object_temporal_jump_count": weighted["object_temporal_jump_count"],
        "translation_difference_cm": weighted["translation_difference_cm"],
        "rotation_difference_deg": weighted["rotation_difference_deg"],
    }


def _summarize_cup_rows_branch_aware(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = _summarize_cup_rows(rows)
    by_branch: dict[str, Any] = {}
    for row in rows:
        key = str(row.get("apriltag_branch_id") if row.get("apriltag_branch_id") != "" else "unassigned")
        by_branch.setdefault(key, []).append(row)
    branch_stats = {}
    for key, values in sorted(by_branch.items()):
        comparable = [
            float(row["cup_world_difference_cm"])
            for row in values
            if row.get("cup_world_difference_cm") != ""
        ]
        stationary = [
            float(row["cup_world_difference_cm"])
            for row in values
            if row.get("cup_world_difference_cm") != ""
            and row.get("cup_motion_phase") == "stationary"
        ]
        moving = [
            float(row["cup_world_difference_cm"])
            for row in values
            if row.get("cup_world_difference_cm") != ""
            and row.get("cup_motion_phase") == "moving"
        ]
        branch_stats[key] = {
            "cup_comparable_count": len(comparable),
            "cup_world_difference_cm": distribution(comparable),
            "cup_world_difference_stationary_cm": distribution(stationary),
            "cup_world_difference_moving_cm": distribution(moving),
        }
    base["branches"] = branch_stats
    base["weighted"] = {
        "cup_comparable_count": base["cup_comparable_count"],
        "cup_world_difference_cm": base["cup_world_difference_cm"],
        "cup_world_difference_stationary_cm": base["cup_world_difference_stationary_cm"],
        "cup_world_difference_moving_cm": base["cup_world_difference_moving_cm"],
    }
    return base


def evaluate_mvp_decision_branch_aware(
    *,
    pose_summary: dict[str, Any],
    cup_summary: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    weighted = pose_summary.get("weighted") or pose_summary
    total = max(int(weighted.get("total_frames") or pose_summary.get("total_frames") or 0), 1)
    translation = weighted.get("translation_difference_cm") or {}
    rotation = weighted.get("rotation_difference_deg") or {}
    jumps = int(weighted.get("object_temporal_jump_count") or 0)
    comparable = int(weighted.get("pose_comparable_count") or 0)
    pose_checks = {
        "translation_median_le_5cm": (
            translation.get("median") is not None
            and translation["median"] <= float(settings["mvp_translation_median_cm"])
        ),
        "translation_p90_le_10cm": (
            translation.get("p90") is not None
            and translation["p90"] <= float(settings["mvp_translation_p90_cm"])
        ),
        "rotation_median_le_5deg": (
            rotation.get("median") is not None
            and rotation["median"] <= float(settings["mvp_rotation_median_deg"])
        ),
        "rotation_p90_le_10deg": (
            rotation.get("p90") is not None
            and rotation["p90"] <= float(settings["mvp_rotation_p90_deg"])
        ),
        "temporal_jump_rate_lt_1pct": jumps / total
        < float(settings["mvp_max_object_temporal_jump_rate"]),
        "pose_comparable_rate_ge_75pct": comparable / total
        >= float(settings["mvp_min_pose_comparable_rate"]),
    }
    cup_diff = (cup_summary.get("weighted") or cup_summary).get("cup_world_difference_cm") or {}
    cup_comparable = int(
        (cup_summary.get("weighted") or cup_summary).get("cup_comparable_count") or 0
    )
    cup_checks = {
        "cup_comparable_ge_recommended": cup_comparable
        >= int(settings["mvp_cup_recommended_frames"]),
        "cup_median_le_5cm": (
            cup_diff.get("median") is not None
            and cup_diff["median"] <= float(settings["mvp_cup_median_cm"])
        ),
        "cup_p90_le_10cm": (
            cup_diff.get("p90") is not None
            and cup_diff["p90"] <= float(settings["mvp_cup_p90_cm"])
        ),
    }
    pose_ok = all(pose_checks.values())
    cup_core_ok = bool(cup_checks["cup_median_le_5cm"] and cup_checks["cup_p90_le_10cm"])
    cup_count_ok = bool(cup_checks["cup_comparable_ge_recommended"])
    if pose_ok and cup_core_ok and cup_count_ok:
        decision, label = "A", "1st_MVP_complete"
        rationale = (
            "Branch-aware Object Anchor pose reproduction and Cup world differences "
            "meet temporary 1st-MVP criteria."
        )
    elif pose_ok and not cup_count_ok:
        decision, label = "B", "1st_MVP_conditional_complete"
        rationale = (
            "Branch-aware Object Anchor pose criteria met. Cup sample count and/or "
            "absolute AprilTag branch selection remain follow-up items."
        )
    else:
        decision, label = "C", "1st_MVP_incomplete"
        rationale = (
            "Within-branch Object Anchor vs AprilTag pose differences exceed "
            "temporary MVP thresholds (absolute branch ambiguity alone is not the reason)."
        )
    return {
        "decision": decision,
        "label": label,
        "rationale": rationale,
        "pose_difference_checks": pose_checks,
        "cup_checks": cup_checks,
        "pose_core_passed": pose_ok,
        "cup_core_passed": cup_core_ok,
        "cup_count_passed": cup_count_ok,
        "evaluated_by_branch_separately": True,
        "april_tag_is_temporary_reference_not_ground_truth": True,
        "criteria_are_temporary_mvp_only": True,
    }


def _save_representative_overlays(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    overlay_dir = output_dir / "representative_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    def save_named(name: str, row: dict[str, Any] | None) -> None:
        if row is None or "overlay" not in row:
            return
        _write_image(overlay_dir / f"{name}.jpg", row["overlay"])

    comparable = [row for row in rows if bool(row.get("pose_comparable"))]
    save_named("normal_pose_comparison", comparable[0] if comparable else None)
    if comparable:
        save_named(
            "largest_translation_difference",
            max(comparable, key=lambda row: float(row["translation_difference_cm"])),
        )
        save_named(
            "largest_rotation_difference",
            max(comparable, key=lambda row: float(row["rotation_difference_deg"])),
        )
    outlier = next((row for row in rows if bool(row.get("apriltag_branch_outlier"))), None)
    save_named("apriltag_branch_outlier", outlier)
    stationary_cup = next(
        (
            row
            for row in rows
            if row.get("cup_motion_phase") == "stationary" and bool(row.get("cup_comparable"))
        ),
        None,
    )
    moving_cup = next(
        (
            row
            for row in rows
            if row.get("cup_motion_phase") == "moving" and bool(row.get("cup_comparable"))
        ),
        None,
    )
    save_named("cup_stationary_comparison", stationary_cup)
    save_named("cup_moving_comparison", moving_cup)


def run_mvp_final(
    *,
    config_path: Path,
    confirm_live: bool,
) -> dict[str, Any]:
    if not confirm_live:
        raise RuntimeError(
            "Refusing to open the camera. Re-run with --i-confirm-scene-ready "
            "after the physical scene is prepared."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_single_apriltag_id0(config)
    settings = load_mvp_settings(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _resolve(settings["output_root"]) / timestamp
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    assert settings.get("temporal_filter_window", 3) == 3 or int(
        settings.get("temporal_filter_window", 3)
    ) >= 1
    print(
        "\n[MVP FINAL branch-aware] Phase 1/3 registration: capture "
        f"{int(settings['registration_frames'])} frames. "
        "Keep camera / AprilTag / tissue / Cup FIXED.\n"
        f"Causal filter window={int(settings['temporal_filter_window'])}. "
        "Single 95% cluster gate is DISABLED.\n",
        flush=True,
    )
    registration_capture = _capture_session_frames(
        config=config,
        frames=int(settings["registration_frames"]),
        registered_world_pose=None,
        reference_centroid=None,
        settings=settings,
        stationary_frames=int(settings["registration_frames"]),
        phase_name="registration",
        branch_registry=None,
        branch_centroids=None,
    )
    reg_rows, branch_info, branch_centroids = assign_april_tag_branches(
        registration_capture["rows"],
        settings,
    )
    reference_csv_rows = [
        {key: row.get(key, "") for key in REFERENCE_CLUSTER_FIELDS} for row in reg_rows
    ]
    _write_csv(
        output_root / "reference_cluster_frames.csv",
        reference_csv_rows,
        REFERENCE_CLUSTER_FIELDS,
    )
    _write_json(output_root / "branch_assignment_summary.json", branch_info)

    branch_results = register_branch_relative_poses(reg_rows, settings)
    sample_rows: list[dict[str, Any]] = []
    accepted_registry: dict[int, dict[str, Any]] = {}
    registration_dir = output_root / "registration"
    registration_dir.mkdir(parents=True, exist_ok=True)
    branch_summaries: dict[str, Any] = {}
    for branch_id, result in branch_results.items():
        for sample in result.get("samples") or []:
            sample_rows.append(
                {
                    "frame_idx": sample["frame_idx"],
                    "timestamp_utc": sample["timestamp_utc"],
                    "pose_cluster_id": branch_id,
                    "in_reference_cluster": True,
                    "anchor_operational_valid": True,
                    "reprojection_error_px": sample.get("reprojection_error_px", ""),
                    "T_world_camera_tag_json": sample.get("T_world_camera_tag_json", ""),
                    "T_camera_object_json": sample.get("T_camera_object_json", ""),
                    "T_world_object_json": sample.get("T_world_object_json", ""),
                    "translation_residual_m": sample.get("translation_residual_m", ""),
                    "rotation_residual_deg": sample.get("rotation_residual_deg", ""),
                    "outlier": sample.get("outlier", False),
                    "outlier_reason": sample.get("outlier_reason", ""),
                }
            )
        path = None
        if result.get("accepted") and result.get("T_tag_object") is not None:
            path = save_world_pose_registration(
                registration_dir / f"branch_{branch_id}_tissue_box_relative_pose.yaml",
                object_id="tissue_box_01",
                T_world_object=result["T_tag_object"],
                source="mvp_final_branch_aware_relative",
                metadata={
                    "transform_name": "T_tag_object",
                    "convention": "inv(T_camera_tag) @ T_camera_object_filtered",
                    "pose_cluster_id": branch_id,
                    "april_tag_ids": [0],
                    "temporary_branch_reference_only": True,
                    "inlier_count": result.get("inlier_count"),
                },
            )
            accepted_registry[branch_id] = {
                "T_tag_object": result["T_tag_object"],
                "registration_file": _relative(path),
            }
        residual = result.get("residual_gate") or {}
        branch_summaries[str(branch_id)] = {
            "accepted": bool(result.get("accepted")),
            "ok": bool(result.get("ok")),
            "reason": result.get("reason"),
            "candidate_count": result.get("candidate_count"),
            "inlier_count": result.get("inlier_count"),
            "outlier_count": result.get("outlier_count"),
            "excluded_reasons": result.get("excluded_reasons") or {},
            "residual_gate": residual,
            "registration_file": _relative(path) if path is not None else None,
            "centroid_T_world_camera": branch_centroids.get(branch_id, np.eye(4)).tolist(),
        }

    registration_sample_fields = REGISTRATION_SAMPLE_FIELDS + ("pose_cluster_id",)
    _write_csv(
        output_root / "registration_samples.csv",
        sample_rows,
        registration_sample_fields,
    )
    registration_summary = {
        "mode": "branch_aware_relative",
        "single_reference_cluster_gate_used": False,
        "temporal_filter_window": int(settings["temporal_filter_window"]),
        "branch_info": branch_info,
        "branches": branch_summaries,
        "accepted_branch_ids": sorted(accepted_registry.keys()),
        "note": (
            "Each AprilTag SE(3) branch has its own T_tag_object. "
            "Branches are never averaged together. Not physical GT."
        ),
    }
    _write_json(output_root / "registration_summary.json", registration_summary)
    if not accepted_registry:
        decision = {
            "decision": "C",
            "label": "1st_MVP_incomplete",
            "rationale": (
                "No AprilTag branch produced an accepted relative Object Anchor "
                "registration under MVP residual thresholds."
            ),
            "registration_summary": registration_summary,
            "aborted_before_comparison": True,
        }
        _write_json(output_root / "mvp_final_decision.json", decision)
        _write_readme(output_root, decision, branch_info, registration_summary, None, None)
        return {"output": str(output_root), "decision": decision}

    # Match live frames against ALL registration prototypes. Calibration is used
    # only when that prototype's branch was accepted — never mix branches.
    all_prototypes = {
        branch_id: centroid for branch_id, centroid in branch_centroids.items()
    }
    prototypes_payload = {
        "prototypes": {
            str(branch_id): {
                "internal_branch_id": branch_id,
                "has_calibration": branch_id in accepted_registry,
                "T_world_camera_tag_prototype": centroid.tolist(),
                "registration_file": (
                    accepted_registry[branch_id]["registration_file"]
                    if branch_id in accepted_registry
                    else None
                ),
            }
            for branch_id, centroid in all_prototypes.items()
        },
        "note": (
            "Numeric IDs are registration-time clustering labels only. "
            "Live phases match by prototype pose, then look up calibration."
        ),
    }
    _write_json(registration_dir / "branch_prototypes.json", prototypes_payload)
    registration_summary["branch_prototypes_file"] = _relative(
        registration_dir / "branch_prototypes.json"
    )
    _write_json(output_root / "registration_summary.json", registration_summary)

    print(
        "\n[MVP FINAL branch-aware] Phase 2/3 preflight: capture "
        f"{int(settings['preflight_frames'])} frames. Keep Cup FIXED.\n",
        flush=True,
    )
    preflight = _capture_session_frames(
        config=config,
        frames=int(settings["preflight_frames"]),
        registered_world_pose=None,
        reference_centroid=None,
        settings=settings,
        stationary_frames=int(settings["preflight_frames"]),
        phase_name="preflight",
        branch_registry=accepted_registry,
        branch_centroids=all_prototypes,
    )
    preflight_rows = preflight["rows"]
    # Always persist preflight rows so offline remapping is possible on abort.
    _write_csv(
        output_root / "preflight_frames.csv",
        [{key: row.get(key, "") for key in FRAME_COMPARE_FIELDS} for row in preflight_rows],
        FRAME_COMPARE_FIELDS,
    )
    preflight_gate = evaluate_preflight_gate(
        tag_reference_valid=sum(bool(r["apriltag_reference_valid"]) for r in preflight_rows),
        object_world_pose=sum(bool(r["object_world_pose_valid"]) for r in preflight_rows),
        pose_comparable=sum(bool(r["pose_comparable"]) for r in preflight_rows),
        cup_camera=sum(r.get("P_camera_cup_x") != "" for r in preflight_rows),
        settings=settings,
    )
    # Branch-aware: comparable frames are the required gate (>=15).
    preflight_gate["branch_aware_required_comparable_ge_15"] = (
        int(preflight_gate["counts"]["pose_comparable"]) >= 15
    )
    preflight_gate["passed"] = bool(preflight_gate["branch_aware_required_comparable_ge_15"])
    preflight_gate["exclude_reason_counts"] = dict(
        Counter(str(r.get("apriltag_exclude_reason") or "ok") for r in preflight_rows)
    )
    _write_json(output_root / "preflight_summary.json", preflight_gate)
    if not preflight_gate["passed"]:
        decision = {
            "decision": "C",
            "label": "1st_MVP_incomplete",
            "rationale": "20-frame preflight gate failed; 300-frame comparison was not started.",
            "preflight_gate": preflight_gate,
            "registration_summary": registration_summary,
            "aborted_before_comparison": True,
        }
        _write_json(output_root / "mvp_final_decision.json", decision)
        _write_readme(
            output_root, decision, branch_info, registration_summary, preflight_gate, None
        )
        return {"output": str(output_root), "decision": decision}

    print(
        "\n[MVP FINAL branch-aware] Phase 3/3 comparison: capture "
        f"{int(settings['comparison_frames'])} frames.\n"
        f"Keep Cup FIXED for the first {int(settings['stationary_frames'])} frames.\n"
        "Exact console prompt will appear when Cup motion may begin.\n",
        flush=True,
    )
    comparison = _capture_session_frames(
        config=config,
        frames=int(settings["comparison_frames"]),
        registered_world_pose=None,
        reference_centroid=None,
        settings=settings,
        stationary_frames=int(settings["stationary_frames"]),
        phase_name="comparison",
        branch_registry=accepted_registry,
        branch_centroids=all_prototypes,
    )
    compare_rows = comparison["rows"]
    pose_rows = [{key: row.get(key, "") for key in FRAME_COMPARE_FIELDS} for row in compare_rows]
    cup_rows = [{key: row.get(key, "") for key in CUP_COMPARE_FIELDS} for row in compare_rows]
    _write_csv(output_root / "frame_pose_comparison.csv", pose_rows, FRAME_COMPARE_FIELDS)
    _write_csv(output_root / "cup_world_comparison.csv", cup_rows, CUP_COMPARE_FIELDS)
    pose_summary = _summarize_pose_rows_branch_aware(compare_rows)
    cup_summary = _summarize_cup_rows_branch_aware(compare_rows)
    _write_json(output_root / "pose_comparison_summary.json", pose_summary)
    _write_json(output_root / "cup_comparison_summary.json", cup_summary)
    _save_representative_overlays(output_root, compare_rows)

    decision = evaluate_mvp_decision_branch_aware(
        pose_summary=pose_summary,
        cup_summary=cup_summary,
        settings=settings,
    )
    decision.update(
        {
            "output": _relative(output_root),
            "branch_assignment": branch_info,
            "registration_summary": registration_summary,
            "preflight_gate": preflight_gate,
            "pose_summary": pose_summary,
            "cup_summary": cup_summary,
            "temporal_filter_window": int(settings["temporal_filter_window"]),
            "single_reference_cluster_gate_used": False,
            "production_code_modified": False,
            "production_config_modified": False,
            "production_calibration_modified": False,
            "automatic_world_source_switch": False,
            "camera_was_requested": True,
        }
    )
    _write_json(output_root / "mvp_final_decision.json", decision)
    _write_readme(
        output_root, decision, branch_info, registration_summary, preflight_gate, pose_summary
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return {"output": str(output_root), "decision": decision}


def _write_readme(
    output_root: Path,
    decision: dict[str, Any],
    cluster_gate: dict[str, Any] | None,
    registration_summary: dict[str, Any] | None,
    preflight_gate: dict[str, Any] | None,
    pose_summary: dict[str, Any] | None,
) -> None:
    lines = [
        "# Object Anchor 1st MVP final comparison",
        "",
        f"- Decision: `{decision.get('decision')}` ({decision.get('label')})",
        f"- Rationale: {decision.get('rationale')}",
        "- AprilTag majority cluster is a temporary MVP reference, not physical ground truth.",
        "- Production AprilTag / Object Anchor / Cup code and default configs were not modified.",
        "- Isolated registration only; no automatic world-source switch.",
        "",
    ]
    if cluster_gate is not None:
        lines.extend(
            [
                "## Reference cluster",
                f"- Detection rate: {cluster_gate.get('detection_rate')}",
                f"- Largest cluster: {cluster_gate.get('largest_cluster_size')} "
                f"({cluster_gate.get('largest_cluster_ratio')})",
                f"- Branch outliers: {cluster_gate.get('branch_outlier_count')}",
                "",
            ]
        )
    if registration_summary is not None:
        lines.extend(
            [
                "## Registration",
                f"- Candidates: {registration_summary.get('candidate_count')}",
                f"- Inliers: {registration_summary.get('inlier_count')}",
                f"- Outliers: {registration_summary.get('outlier_count')}",
                "",
            ]
        )
    if preflight_gate is not None:
        lines.extend(
            [
                "## Preflight",
                f"- Passed: {preflight_gate.get('passed')}",
                f"- Counts: {preflight_gate.get('counts')}",
                "",
            ]
        )
    if pose_summary is not None:
        lines.extend(
            [
                "## Comparison",
                f"- Pose comparable: {pose_summary.get('pose_comparable_count')}",
                f"- Object world pose: {pose_summary.get('object_anchor_world_pose_count')}",
                "",
            ]
        )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated Object Anchor 1st-MVP final comparison runner"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/orbbec_gemini_object_anchor_mvp_final.yaml",
    )
    parser.add_argument(
        "--i-confirm-scene-ready",
        action="store_true",
        help="Required to open the camera. Confirm fixed tag/tissue/cup scene first.",
    )
    parser.add_argument(
        "--check-config-only",
        action="store_true",
        help="Validate isolated config and exit without opening the camera.",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_single_apriltag_id0(config)
    settings = load_mvp_settings(config)
    if args.check_config_only:
        payload = {
            "ok": True,
            "config": _relative(config_path),
            "april_tag_ids": sorted(
                int(key) for key in ((config.get("apriltag_world") or {}).get("tags") or {})
            ),
            "output_root": settings["output_root"],
            "frames": {
                "registration": settings["registration_frames"],
                "preflight": settings["preflight_frames"],
                "comparison": settings["comparison_frames"],
            },
            "camera_opened": False,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    run_mvp_final(config_path=config_path, confirm_live=bool(args.i_confirm_scene_ready))


if __name__ == "__main__":
    main()
