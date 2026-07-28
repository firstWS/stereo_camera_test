#!/usr/bin/env python3
"""Offline AprilTag IPPE_SQUARE dual-branch temporal selection diagnosis.

Reads saved corners from an intrinsic diagnostic run. Never opens a camera and
never modifies production AprilTag / Object Anchor / Cup code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from apriltag_world import build_apriltag_world_config  # noqa: E402
from object_anchor_world import average_transforms, rotation_delta_deg  # noqa: E402

DEFAULT_INPUT = (
    ROOT
    / "out/object_anchor_full99/replacement_feasibility"
    / "apriltag_intrinsic_diagnostics/20260726_151502"
)
FOCUS_FRAMES = (2, 232, 233)
ROTATION_JUMP_DEG = 30.0
TRANSLATION_JUMP_M = 0.50
CLUSTER_TRANSLATION_M = 0.25
CLUSTER_ROTATION_DEG = 20.0
CLUSTER_MIN_SAMPLES = 3

# Fixed a priori weight sets. Do not tune against this dataset.
POLICY_D_WEIGHTS = (
    ("D_equal_1_1_1", 1.0, 1.0, 1.0),
    ("D_reproj_heavy_2_1_1", 2.0, 1.0, 1.0),
    ("D_continuity_heavy_1_2_2", 1.0, 2.0, 2.0),
)
REPROJ_SCALE_PX = 1.0
ROTATION_SCALE_DEG = 10.0
TRANSLATION_SCALE_M = 0.10


def _json_array(value: np.ndarray | list[Any]) -> str:
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


def _translation_distance_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(rotation_delta_deg(a[:3, :3], b[:3, :3]))


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _pose_cluster_summary(transforms: list[np.ndarray]) -> dict[str, Any]:
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
        current = [
            other
            for other, second in enumerate(transforms)
            if _translation_distance_m(first, second) <= CLUSTER_TRANSLATION_M
            and _rotation_distance_deg(first, second) <= CLUSTER_ROTATION_DEG
        ]
        neighbors.append(current)
    labels = np.full(count, -1, dtype=np.int32)
    visited = np.zeros(count, dtype=bool)
    cluster = 0
    for index in range(count):
        if visited[index]:
            continue
        visited[index] = True
        if len(neighbors[index]) < CLUSTER_MIN_SAMPLES:
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
                if len(neighbors[candidate]) >= CLUSTER_MIN_SAMPLES:
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
        "largest_cluster_ratio": sizes[0] / count if sizes else 0.0,
        "labels": [int(value) for value in labels],
    }


def solve_ippe_square_branches(
    corners_xy: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[dict[str, Any]]:
    image_points = np.ascontiguousarray(corners_xy, dtype=np.float64).reshape(4, 2)
    object_points = _tag_object_points(tag_size_m)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_IPPE)
    try:
        retval, rvecs, tvecs, _reproj = cv2.solvePnPGeneric(
            object_points, image_points, K, distortion, flags=flags
        )
    except cv2.error:
        return []
    if not retval:
        return []
    branches: list[dict[str, Any]] = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, distortion)
        error = float(
            np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1))
        )
        rotation, _ = cv2.Rodrigues(rvec)
        camera_tag = _homogeneous(rotation, tvec)
        camera_points = (rotation @ object_points.T).T + tvec.reshape(3)
        branches.append(
            {
                "candidate_index": index,
                "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
                "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
                "positive_depth": bool(np.all(camera_points[:, 2] > 0.0)),
                "reprojection_error_px": error,
                "T_camera_tag": camera_tag,
            }
        )
    return branches


def _same_as_production(
    branch: dict[str, Any], production_rvec: np.ndarray | None, production_tvec: np.ndarray | None
) -> bool:
    if production_rvec is None or production_tvec is None:
        return False
    return bool(
        np.allclose(branch["rvec"], production_rvec, atol=1e-5, rtol=1e-5)
        and np.allclose(branch["tvec"], production_tvec, atol=1e-5, rtol=1e-5)
    )


def _select_seed(branches: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in branches if item["positive_depth"]]
    pool = valid or branches
    if not pool:
        return None
    return min(pool, key=lambda item: item["reprojection_error_px"])


def _select_min_reproj(branches: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _select_seed(branches)


def _select_min_rotation(
    branches: list[dict[str, Any]], previous: np.ndarray | None
) -> dict[str, Any] | None:
    if previous is None:
        return _select_seed(branches)
    valid = [item for item in branches if item["positive_depth"]] or branches
    if not valid:
        return None
    return min(
        valid,
        key=lambda item: _rotation_distance_deg(item["T_world_camera"], previous),
    )


def _select_min_translation(
    branches: list[dict[str, Any]], previous: np.ndarray | None
) -> dict[str, Any] | None:
    if previous is None:
        return _select_seed(branches)
    valid = [item for item in branches if item["positive_depth"]] or branches
    if not valid:
        return None
    return min(
        valid,
        key=lambda item: _translation_distance_m(item["T_world_camera"], previous),
    )


def _select_combined(
    branches: list[dict[str, Any]],
    previous: np.ndarray | None,
    *,
    w_reproj: float,
    w_rot: float,
    w_trans: float,
) -> dict[str, Any] | None:
    if previous is None:
        return _select_seed(branches)
    valid = [item for item in branches if item["positive_depth"]] or branches
    if not valid:
        return None

    def score(item: dict[str, Any]) -> float:
        return (
            w_reproj * (item["reprojection_error_px"] / REPROJ_SCALE_PX)
            + w_rot
            * (
                _rotation_distance_deg(item["T_world_camera"], previous)
                / ROTATION_SCALE_DEG
            )
            + w_trans
            * (
                _translation_distance_m(item["T_world_camera"], previous)
                / TRANSLATION_SCALE_M
            )
        )

    return min(valid, key=score)


def _policy_names() -> list[str]:
    return [
        "A_min_reprojection",
        "B_min_rotation_continuity",
        "C_min_translation_continuity",
        *[name for name, *_ in POLICY_D_WEIGHTS],
    ]


def _evaluate_policy(
    selected: list[dict[str, Any] | None],
) -> dict[str, Any]:
    valid = [item for item in selected if item is not None]
    transforms = [item["T_world_camera"] for item in valid]
    clusters = _pose_cluster_summary(transforms)
    translation_deltas: list[float] = []
    rotation_deltas: list[float] = []
    rotation_jumps = 0
    translation_jumps = 0
    previous: np.ndarray | None = None
    previous_frame: int | None = None
    for item in valid:
        current = item["T_world_camera"]
        frame = int(item["frame_idx"])
        if previous is not None and previous_frame == frame - 1:
            td = _translation_distance_m(current, previous)
            rd = _rotation_distance_deg(current, previous)
            translation_deltas.append(td)
            rotation_deltas.append(rd)
            if td >= TRANSLATION_JUMP_M:
                translation_jumps += 1
            if rd >= ROTATION_JUMP_DEG:
                rotation_jumps += 1
        previous = current
        previous_frame = frame
    reprojection = [float(item["reprojection_error_px"]) for item in valid]
    different_from_production = sum(
        1 for item in valid if not item["same_as_production_solver"]
    )
    focus_selected = {
        str(frame): next(
            (
                {
                    "candidate_index": item["candidate_index"],
                    "reprojection_error_px": item["reprojection_error_px"],
                    "same_as_production_solver": item["same_as_production_solver"],
                }
                for item in valid
                if int(item["frame_idx"]) == frame
            ),
            None,
        )
        for frame in FOCUS_FRAMES
    }
    return {
        "valid_pose_count": len(valid),
        "pose_clusters": {
            "cluster_count": clusters["cluster_count"],
            "largest_cluster_size": (
                clusters["cluster_sizes"][0] if clusters["cluster_sizes"] else 0
            ),
            "largest_cluster_ratio": clusters["largest_cluster_ratio"],
            "noise_count": clusters["noise_count"],
            "cluster_sizes": clusters["cluster_sizes"],
        },
        "rotation_jump_ge_30deg": rotation_jumps,
        "translation_jump_ge_50cm": translation_jumps,
        "frame_to_frame_rotation_delta_deg": _distribution(rotation_deltas),
        "frame_to_frame_translation_delta_m": _distribution(translation_deltas),
        "reprojection_error_px": _distribution(reprojection),
        "different_from_production_count": different_from_production,
        "focus_frame_selection": focus_selected,
        "removed_existing_jumps": {
            str(frame): (
                focus_selected[str(frame)] is not None
                and not focus_selected[str(frame)]["same_as_production_solver"]
            )
            for frame in FOCUS_FRAMES
        },
    }


def _analyze_focus_frame(
    frame_idx: int,
    frames: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    by_idx = {int(item["frame_idx"]): item for item in frames}
    current = by_idx[frame_idx]
    branches = current[f"{method}_branches"]
    if len(branches) < 2:
        return {
            "frame_idx": frame_idx,
            "method": method,
            "branch_count": len(branches),
            "note": "fewer_than_two_branches",
        }
    first, second = branches[0], branches[1]
    previous = by_idx.get(frame_idx - 1)
    following = by_idx.get(frame_idx + 1)
    production = current[f"{method}_production_T_world_camera"]
    production_branch = next(
        (item for item in branches if item["same_as_production_solver"]),
        None,
    )

    def deltas_to(reference: dict[str, Any] | None) -> dict[str, Any]:
        if reference is None or not reference[f"{method}_branches"]:
            return {"available": False}
        ref_pose = reference[f"{method}_production_T_world_camera"]
        if ref_pose is None:
            return {"available": False}
        return {
            "available": True,
            "reference_frame": int(reference["frame_idx"]),
            "candidate_0_translation_m": _translation_distance_m(
                first["T_world_camera"], ref_pose
            ),
            "candidate_0_rotation_deg": _rotation_distance_deg(
                first["T_world_camera"], ref_pose
            ),
            "candidate_1_translation_m": _translation_distance_m(
                second["T_world_camera"], ref_pose
            ),
            "candidate_1_rotation_deg": _rotation_distance_deg(
                second["T_world_camera"], ref_pose
            ),
        }

    previous_pose = (
        previous[f"{method}_production_T_world_camera"] if previous else None
    )
    temporal_choice = (
        _select_min_rotation(branches, previous_pose)
        if previous_pose is not None
        else _select_seed(branches)
    )
    continuity_choice = (
        _select_combined(
            branches,
            previous_pose,
            w_reproj=1.0,
            w_rot=1.0,
            w_trans=1.0,
        )
        if previous_pose is not None
        else _select_seed(branches)
    )
    warmup_or_no_previous = previous is None or previous_pose is None
    return {
        "frame_idx": frame_idx,
        "method": method,
        "tag_id": current["tag_id"],
        "corners_xy": current["corners_xy"].tolist(),
        "polygon_area_px2": current["polygon_area_px2"],
        "mean_width_px": current["mean_width_px"],
        "mean_height_px": current["mean_height_px"],
        "shortest_edge_px": current["shortest_edge_px"],
        "branch_count": len(branches),
        "candidate_0": {
            "reprojection_error_px": first["reprojection_error_px"],
            "positive_depth": first["positive_depth"],
            "same_as_production_solver": first["same_as_production_solver"],
            "tvec": first["tvec"].tolist(),
        },
        "candidate_1": {
            "reprojection_error_px": second["reprojection_error_px"],
            "positive_depth": second["positive_depth"],
            "same_as_production_solver": second["same_as_production_solver"],
            "tvec": second["tvec"].tolist(),
        },
        "inter_branch_translation_m": _translation_distance_m(
            first["T_world_camera"], second["T_world_camera"]
        ),
        "inter_branch_rotation_deg": _rotation_distance_deg(
            first["T_world_camera"], second["T_world_camera"]
        ),
        "production_selected_candidate_index": (
            production_branch["candidate_index"] if production_branch else None
        ),
        "vs_previous_production_pose": deltas_to(previous),
        "vs_next_production_pose": deltas_to(following),
        "rotation_continuity_would_select": (
            temporal_choice["candidate_index"] if temporal_choice else None
        ),
        "combined_equal_would_select": (
            continuity_choice["candidate_index"] if continuity_choice else None
        ),
        "temporal_policy_selects_non_production_branch": bool(
            temporal_choice is not None
            and not temporal_choice["same_as_production_solver"]
        ),
        "warmup_or_missing_previous_pose": warmup_or_no_previous,
        "frame_2_assessment": (
            "initial_frame_with_no_previous_pose_or_cold_start"
            if frame_idx == 2 and warmup_or_no_previous
            else (
                "has_previous_pose_available_for_continuity"
                if frame_idx == 2
                else None
            )
        ),
    }


def run(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(
        (input_dir / "diagnostic_config_snapshot.yaml").read_text(encoding="utf-8")
    )
    apriltag_config = build_apriltag_world_config(config.get("apriltag_world") or {})
    obs_rows = list(
        csv.DictReader(
            (input_dir / "apriltag_observation_comparison.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    frame_rows = list(
        csv.DictReader(
            (input_dir / "apriltag_frame_summary.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    frame_by_idx = {int(row["frame_idx"]): row for row in frame_rows}
    if len(obs_rows) != 300:
        raise RuntimeError(f"expected 300 observation rows, found {len(obs_rows)}")

    frames: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for obs in obs_rows:
        frame_idx = int(obs["frame_idx"])
        frame = frame_by_idx[frame_idx]
        corners = np.asarray(json.loads(obs["corners_xy_json"]), dtype=np.float64)
        K = np.asarray(json.loads(frame["camera_matrix_json"]), dtype=np.float64)
        dist = np.asarray(
            json.loads(frame["orbbec_dist_coeffs_json"]), dtype=np.float64
        ).reshape(-1, 1)
        tag_id = int(obs["tag_id"])
        tag = apriltag_config.tags[tag_id]
        methods: dict[str, Any] = {}
        for method, distortion in (
            ("zero", np.zeros_like(dist)),
            ("orbbec", dist),
        ):
            production_rvec = (
                np.asarray(json.loads(obs[f"{method if method != 'orbbec' else 'distortion'}_rvec_json"]), dtype=np.float64)
                if obs[f"{method if method != 'orbbec' else 'distortion'}_rvec_json"]
                else None
            )
            production_tvec = (
                np.asarray(json.loads(obs[f"{method if method != 'orbbec' else 'distortion'}_tvec_json"]), dtype=np.float64)
                if obs[f"{method if method != 'orbbec' else 'distortion'}_tvec_json"]
                else None
            )
            production_world = (
                np.asarray(
                    json.loads(
                        obs[
                            f"{method if method != 'orbbec' else 'distortion'}_T_world_camera_json"
                        ]
                    ),
                    dtype=np.float64,
                )
                if obs[
                    f"{method if method != 'orbbec' else 'distortion'}_T_world_camera_json"
                ]
                else None
            )
            branches = solve_ippe_square_branches(
                corners, apriltag_config.tag_size_m, K, distortion
            )
            enriched: list[dict[str, Any]] = []
            for branch in branches:
                try:
                    world = tag.T_world_tag @ np.linalg.inv(branch["T_camera_tag"])
                except np.linalg.LinAlgError:
                    continue
                item = {
                    **branch,
                    "frame_idx": frame_idx,
                    "method": method,
                    "T_world_camera": world,
                    "same_as_production_solver": _same_as_production(
                        branch, production_rvec, production_tvec
                    ),
                }
                enriched.append(item)
            methods[method] = {
                "branches": enriched,
                "production_T_world_camera": production_world,
                "production_rvec": production_rvec,
                "production_tvec": production_tvec,
            }
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_utc": obs["timestamp_utc"],
                "tag_id": tag_id,
                "corners_xy": corners,
                "polygon_area_px2": float(obs["polygon_area_px2"]),
                "mean_width_px": float(obs["mean_width_px"]),
                "mean_height_px": float(obs["mean_height_px"]),
                "shortest_edge_px": float(obs["shortest_edge_px"]),
                "center_distance_px": float(obs["center_distance_px"]),
                "near_image_boundary": obs["near_image_boundary"].lower() == "true",
                "camera_matrix": K,
                "orbbec_dist_coeffs": dist,
                "zero_branches": methods["zero"]["branches"],
                "orbbec_branches": methods["orbbec"]["branches"],
                "zero_production_T_world_camera": methods["zero"][
                    "production_T_world_camera"
                ],
                "orbbec_production_T_world_camera": methods["orbbec"][
                    "production_T_world_camera"
                ],
            }
        )

    # Prefer Orbbec-distortion for primary branch policy analysis; also evaluate zero.
    primary_method = "orbbec"
    policy_selected: dict[str, list[dict[str, Any] | None]] = {
        name: [] for name in _policy_names()
    }
    previous_by_policy: dict[str, np.ndarray | None] = {
        name: None for name in _policy_names()
    }

    for frame in frames:
        branches = frame[f"{primary_method}_branches"]
        for branch in branches:
            previous_any = previous_by_policy["A_min_reprojection"]
            translation_from_prev = (
                _translation_distance_m(branch["T_world_camera"], previous_any)
                if previous_any is not None
                else ""
            )
            rotation_from_prev = (
                _rotation_distance_deg(branch["T_world_camera"], previous_any)
                if previous_any is not None
                else ""
            )
            candidate_rows.append(
                {
                    "frame_idx": frame["frame_idx"],
                    "timestamp_utc": frame["timestamp_utc"],
                    "distortion_method": primary_method,
                    "tag_id": frame["tag_id"],
                    "candidate_index": branch["candidate_index"],
                    "rvec_json": _json_array(branch["rvec"]),
                    "tvec_json": _json_array(branch["tvec"]),
                    "positive_depth": branch["positive_depth"],
                    "reprojection_error_px": branch["reprojection_error_px"],
                    "T_camera_tag_json": _json_array(branch["T_camera_tag"]),
                    "T_world_camera_json": _json_array(branch["T_world_camera"]),
                    "translation_difference_from_previous_selected_m": translation_from_prev,
                    "rotation_difference_from_previous_selected_deg": rotation_from_prev,
                    "same_as_production_solver": branch["same_as_production_solver"],
                    "corners_xy_json": _json_array(frame["corners_xy"]),
                    "polygon_area_px2": frame["polygon_area_px2"],
                    "mean_width_px": frame["mean_width_px"],
                    "mean_height_px": frame["mean_height_px"],
                    "shortest_edge_px": frame["shortest_edge_px"],
                }
            )

        choices = {
            "A_min_reprojection": _select_min_reproj(branches),
            "B_min_rotation_continuity": _select_min_rotation(
                branches, previous_by_policy["B_min_rotation_continuity"]
            ),
            "C_min_translation_continuity": _select_min_translation(
                branches, previous_by_policy["C_min_translation_continuity"]
            ),
        }
        for name, w_reproj, w_rot, w_trans in POLICY_D_WEIGHTS:
            choices[name] = _select_combined(
                branches,
                previous_by_policy[name],
                w_reproj=w_reproj,
                w_rot=w_rot,
                w_trans=w_trans,
            )
        for name, choice in choices.items():
            policy_selected[name].append(choice)
            previous_by_policy[name] = (
                choice["T_world_camera"] if choice is not None else None
            )

    # Fill continuity deltas relative to each policy's own previous selection.
    # Recompute candidate CSV previous-selected deltas against production for clarity,
    # then add policy-selected markers.
    selected_lookup = {
        name: {
            int(item["frame_idx"]): item
            for item in values
            if item is not None
        }
        for name, values in policy_selected.items()
    }
    for row in candidate_rows:
        frame_idx = int(row["frame_idx"])
        for name in _policy_names():
            selected = selected_lookup[name].get(frame_idx)
            row[f"selected_by_{name}"] = bool(
                selected is not None
                and int(selected["candidate_index"]) == int(row["candidate_index"])
            )

    policy_results = {
        name: _evaluate_policy(values) for name, values in policy_selected.items()
    }

    # Also evaluate production baseline on the same metric definitions.
    production_selected = []
    for frame in frames:
        production = frame[f"{primary_method}_production_T_world_camera"]
        if production is None:
            production_selected.append(None)
            continue
        matching = next(
            (
                item
                for item in frame[f"{primary_method}_branches"]
                if item["same_as_production_solver"]
            ),
            None,
        )
        if matching is None:
            # Fall back to logged production pose as a synthetic selected item.
            production_selected.append(
                {
                    "frame_idx": frame["frame_idx"],
                    "candidate_index": -1,
                    "reprojection_error_px": float("nan"),
                    "T_world_camera": production,
                    "same_as_production_solver": True,
                }
            )
        else:
            production_selected.append(matching)
    policy_results["production_baseline"] = _evaluate_policy(production_selected)

    jump_analysis = {
        "focus_frames": list(FOCUS_FRAMES),
        "primary_method": primary_method,
        "zero_distortion": {
            str(frame): _analyze_focus_frame(frame, frames, "zero")
            for frame in FOCUS_FRAMES
        },
        "orbbec_distortion": {
            str(frame): _analyze_focus_frame(frame, frames, "orbbec")
            for frame in FOCUS_FRAMES
        },
    }

    branch_counts = {
        "orbbec_frames_with_two_branches": sum(
            1 for frame in frames if len(frame["orbbec_branches"]) >= 2
        ),
        "zero_frames_with_two_branches": sum(
            1 for frame in frames if len(frame["zero_branches"]) >= 2
        ),
        "orbbec_mean_inter_branch_rotation_deg": float(
            np.mean(
                [
                    _rotation_distance_deg(
                        frame["orbbec_branches"][0]["T_world_camera"],
                        frame["orbbec_branches"][1]["T_world_camera"],
                    )
                    for frame in frames
                    if len(frame["orbbec_branches"]) >= 2
                ]
            )
        ),
        "orbbec_mean_inter_branch_translation_m": float(
            np.mean(
                [
                    _translation_distance_m(
                        frame["orbbec_branches"][0]["T_world_camera"],
                        frame["orbbec_branches"][1]["T_world_camera"],
                    )
                    for frame in frames
                    if len(frame["orbbec_branches"]) >= 2
                ]
            )
        ),
    }

    best_policy = min(
        (
            (name, result)
            for name, result in policy_results.items()
            if name != "production_baseline"
        ),
        key=lambda item: (
            item[1]["rotation_jump_ge_30deg"] + item[1]["translation_jump_ge_50cm"],
            -item[1]["pose_clusters"]["largest_cluster_ratio"],
            item[1]["pose_clusters"]["cluster_count"],
        ),
    )
    production_jumps = (
        policy_results["production_baseline"]["rotation_jump_ge_30deg"]
        + policy_results["production_baseline"]["translation_jump_ge_50cm"]
    )
    best_jumps = (
        best_policy[1]["rotation_jump_ge_30deg"]
        + best_policy[1]["translation_jump_ge_50cm"]
    )
    if (
        best_policy[1]["pose_clusters"]["cluster_count"] == 1
        and best_policy[1]["pose_clusters"]["largest_cluster_ratio"] >= 0.99
        and best_jumps == 0
    ):
        size_verdict = "A_temporal_branch_selection_alone_stabilizes"
        next_step = 1
    elif best_jumps < production_jumps or (
        best_policy[1]["pose_clusters"]["largest_cluster_ratio"]
        > policy_results["production_baseline"]["pose_clusters"][
            "largest_cluster_ratio"
        ]
    ):
        size_verdict = "B_improved_but_residual_instability"
        next_step = 2
    else:
        size_verdict = "C_both_branches_unstable_larger_tag_needed"
        next_step = 3

    summary = {
        "input": str(input_dir),
        "output": str(output_dir),
        "camera_reopened": False,
        "saved_corners_sufficient": True,
        "frames": len(frames),
        "solver": "cv2.solvePnPGeneric + SOLVEPNP_IPPE_SQUARE",
        "primary_method": primary_method,
        "branch_counts": branch_counts,
        "policy_weights_fixed_a_priori": {
            name: {
                "reprojection": w_reproj,
                "rotation_continuity": w_rot,
                "translation_continuity": w_trans,
                "normalization": {
                    "reprojection_px": REPROJ_SCALE_PX,
                    "rotation_deg": ROTATION_SCALE_DEG,
                    "translation_m": TRANSLATION_SCALE_M,
                },
            }
            for name, w_reproj, w_rot, w_trans in POLICY_D_WEIGHTS
        },
        "policy_results": policy_results,
        "best_offline_policy": best_policy[0],
        "size_and_branch_verdict": size_verdict,
        "next_step_decision": next_step,
        "object_anchor_registration_auto_run": False,
        "protection": {
            "production_apriltag_code_modified": False,
            "production_object_anchor_code_modified": False,
            "default_config_modified": False,
            "run_ps1_modified": False,
            "cup_code_modified": False,
            "full99_model_modified": False,
        },
    }

    candidate_fields = tuple(candidate_rows[0].keys()) if candidate_rows else ()
    with (output_dir / "candidate_pose_per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    comparison_rows = []
    for name, result in policy_results.items():
        comparison_rows.append(
            {
                "policy": name,
                "valid_pose_count": result["valid_pose_count"],
                "cluster_count": result["pose_clusters"]["cluster_count"],
                "largest_cluster_size": result["pose_clusters"][
                    "largest_cluster_size"
                ],
                "largest_cluster_ratio": result["pose_clusters"][
                    "largest_cluster_ratio"
                ],
                "rotation_jump_ge_30deg": result["rotation_jump_ge_30deg"],
                "translation_jump_ge_50cm": result["translation_jump_ge_50cm"],
                "rotation_delta_mean_deg": result[
                    "frame_to_frame_rotation_delta_deg"
                ]["mean"],
                "rotation_delta_median_deg": result[
                    "frame_to_frame_rotation_delta_deg"
                ]["median"],
                "rotation_delta_p90_deg": result[
                    "frame_to_frame_rotation_delta_deg"
                ]["p90"],
                "rotation_delta_max_deg": result[
                    "frame_to_frame_rotation_delta_deg"
                ]["max"],
                "translation_delta_mean_m": result[
                    "frame_to_frame_translation_delta_m"
                ]["mean"],
                "translation_delta_median_m": result[
                    "frame_to_frame_translation_delta_m"
                ]["median"],
                "translation_delta_p90_m": result[
                    "frame_to_frame_translation_delta_m"
                ]["p90"],
                "translation_delta_max_m": result[
                    "frame_to_frame_translation_delta_m"
                ]["max"],
                "reprojection_mean_px": result["reprojection_error_px"]["mean"],
                "reprojection_median_px": result["reprojection_error_px"]["median"],
                "reprojection_p90_px": result["reprojection_error_px"]["p90"],
                "different_from_production_count": result[
                    "different_from_production_count"
                ],
            }
        )
    with (output_dir / "branch_policy_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    (output_dir / "jump_frame_analysis.json").write_text(
        json.dumps(jump_analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "pose_cluster_summary.json").write_text(
        json.dumps(
            {
                name: result["pose_clusters"]
                for name, result in policy_results.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""# AprilTag IPPE_SQUARE branch diagnostics

- Input: `{input_dir.relative_to(ROOT)}`
- Camera was not opened. Analysis used saved corners only.
- Solver: `cv2.solvePnPGeneric` + `SOLVEPNP_IPPE_SQUARE`
- Primary distortion method for policy comparison: `{primary_method}`
- Production AprilTag / Object Anchor / Cup code and default config were not modified.

## Branch geometry

- Orbbec frames with two branches: {branch_counts['orbbec_frames_with_two_branches']}/300
- Mean inter-branch rotation: {branch_counts['orbbec_mean_inter_branch_rotation_deg']:.2f} deg
- Mean inter-branch translation: {branch_counts['orbbec_mean_inter_branch_translation_m']:.3f} m

## Best offline policy

- `{best_policy[0]}`
- Clusters: {best_policy[1]['pose_clusters']['cluster_count']}
- Largest cluster ratio: {best_policy[1]['pose_clusters']['largest_cluster_ratio']:.3f}
- Rotation jumps >=30deg: {best_policy[1]['rotation_jump_ge_30deg']}
- Translation jumps >=50cm: {best_policy[1]['translation_jump_ge_50cm']}

## Verdict

- Size/branch verdict: `{size_verdict}`
- Next-step decision code: `{next_step}`
- Object Anchor registration was not executed.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "out/object_anchor_full99/replacement_feasibility"
            / "apriltag_branch_diagnostics"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
    args = parser.parse_args()
    summary = run(args.input.resolve(), args.output.resolve())
    print(json.dumps(
        {
            "output": summary["output"],
            "best_offline_policy": summary["best_offline_policy"],
            "size_and_branch_verdict": summary["size_and_branch_verdict"],
            "next_step_decision": summary["next_step_decision"],
            "policy_results": {
                name: {
                    "clusters": result["pose_clusters"]["cluster_count"],
                    "largest_ratio": result["pose_clusters"][
                        "largest_cluster_ratio"
                    ],
                    "rot_jumps": result["rotation_jump_ge_30deg"],
                    "trans_jumps": result["translation_jump_ge_50cm"],
                }
                for name, result in summary["policy_results"].items()
            },
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
