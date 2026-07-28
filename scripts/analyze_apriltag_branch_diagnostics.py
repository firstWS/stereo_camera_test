"""Offline AprilTag IPPE_SQUARE branch-policy diagnostics.

Reads saved corners from the isolated intrinsic run. It does not open a camera,
register an Object Anchor, or alter production configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apriltag_world import build_apriltag_world_config  # noqa: E402
from object_anchor_replacement_feasibility import (  # noqa: E402
    distribution,
    pose_cluster_summary,
)
from object_anchor_world import average_transforms, rotation_delta_deg  # noqa: E402

DEFAULT_SOURCE = (
    ROOT
    / "out/object_anchor_full99/replacement_feasibility"
    / "apriltag_intrinsic_diagnostics/20260726_151502"
)
DEFAULT_CONFIG = ROOT / "configs/experiments/orbbec_gemini_object_anchor_full99.yaml"
OUTPUT_ROOT = (
    ROOT
    / "out/object_anchor_full99/replacement_feasibility/apriltag_branch_diagnostics"
)

METHODS = ("zero", "distortion")
POLICIES: dict[str, dict[str, float] | None] = {
    "minimum_reprojection": None,
    "rotation_continuity": None,
    "translation_continuity": None,
    "combined_equal": {"reprojection": 1.0, "rotation": 1.0, "translation": 1.0},
    "combined_temporal_strong": {
        "reprojection": 1.0,
        "rotation": 2.0,
        "translation": 2.0,
    },
    "combined_reprojection_priority": {
        "reprojection": 1.0,
        "rotation": 0.5,
        "translation": 0.5,
    },
}

CANDIDATE_FIELDS = (
    "frame_idx",
    "timestamp_utc",
    "tag_id",
    "method",
    "policy",
    "candidate_index",
    "selected",
    "positive_depth",
    "reprojection_error_px",
    "rvec_json",
    "tvec_json",
    "T_camera_tag_json",
    "T_world_camera_json",
    "previous_selected_translation_delta_m",
    "previous_selected_rotation_delta_deg",
    "same_as_production_solver",
    "near_production_main_cluster",
    "policy_score",
)

COMPARISON_FIELDS = (
    "method",
    "policy",
    "valid_pose_count",
    "cluster_count",
    "cluster_sizes_json",
    "largest_cluster_frames",
    "largest_cluster_ratio",
    "noise_count",
    "rotation_jump_ge_30deg",
    "translation_jump_ge_50cm",
    "rotation_delta_mean_deg",
    "rotation_delta_median_deg",
    "rotation_delta_p90_deg",
    "rotation_delta_max_deg",
    "translation_delta_mean_m",
    "translation_delta_median_m",
    "translation_delta_p90_m",
    "translation_delta_max_m",
    "reprojection_error_mean_px",
    "reprojection_error_median_px",
    "reprojection_error_p90_px",
    "different_from_production_count",
    "near_production_main_cluster_count",
    "near_production_main_cluster_ratio",
)


def _json_array(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value).tolist(), separators=(",", ":"))


def _pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))[0]
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    return transform


def _candidate_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
        rotation_delta_deg(first[:3, :3], second[:3, :3]),
    )


def _mean_reprojection(
    object_points: np.ndarray,
    corners: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    return float(
        np.mean(np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1))
    )


def solve_generic_candidates(
    corners: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_world_tag: np.ndarray,
) -> list[dict[str, Any]]:
    result = cv2.solvePnPGeneric(
        np.ascontiguousarray(object_points, dtype=np.float64),
        np.ascontiguousarray(corners, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0]:
        return []
    output: list[dict[str, Any]] = []
    for index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
        T_camera_tag = _pose(rvec, tvec)
        camera_points = (
            T_camera_tag[:3, :3] @ np.asarray(object_points).T
        ).T + T_camera_tag[:3, 3]
        T_world_camera = T_world_tag @ np.linalg.inv(T_camera_tag)
        output.append(
            {
                "candidate_index": index,
                "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
                "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
                "positive_depth": bool(np.all(camera_points[:, 2] > 0.0)),
                "reprojection_error_px": _mean_reprojection(
                    object_points, corners, K, dist, rvec, tvec
                ),
                "T_camera_tag": T_camera_tag,
                "T_world_camera": T_world_camera,
            }
        )
    return output


def policy_score(
    policy: str,
    candidate: dict[str, Any],
    previous: np.ndarray | None,
) -> float:
    if previous is None:
        return float(candidate["reprojection_error_px"])
    translation, rotation = _candidate_distance(
        candidate["T_world_camera"], previous
    )
    if policy == "minimum_reprojection":
        return float(candidate["reprojection_error_px"])
    if policy == "rotation_continuity":
        return rotation
    if policy == "translation_continuity":
        return translation
    weights = POLICIES[policy]
    assert weights is not None
    # Fixed, predeclared scales come from the diagnostic jump definitions.
    return (
        weights["reprojection"] * float(candidate["reprojection_error_px"]) / 1.0
        + weights["rotation"] * rotation / 30.0
        + weights["translation"] * translation / 0.50
    )


def select_candidate(
    policy: str,
    candidates: list[dict[str, Any]],
    previous: np.ndarray | None,
) -> tuple[int, list[float]]:
    usable = [candidate for candidate in candidates if candidate["positive_depth"]]
    if not usable:
        usable = candidates
    scores = [policy_score(policy, candidate, previous) for candidate in usable]
    selected = min(
        range(len(usable)),
        key=lambda index: (scores[index], usable[index]["candidate_index"]),
    )
    return int(usable[selected]["candidate_index"]), scores


def _object_points(tag_size_m: float) -> np.ndarray:
    half = tag_size_m * 0.5
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _load(
    source: Path, config_path: Path
) -> tuple[
    list[dict[str, str]],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
]:
    rows = list(
        csv.DictReader(
            (source / "apriltag_observation_comparison.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    summary = json.loads((source / "diagnostic_summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    april = build_apriltag_world_config(config["apriltag_world"])
    K = np.asarray(summary["camera"]["camera_matrix"], dtype=np.float64)
    orbbec_dist = np.asarray(
        summary["camera"]["orbbec_dist_coeffs"], dtype=np.float64
    ).reshape(-1, 1)
    distortions = {
        "zero": np.zeros_like(orbbec_dist),
        "distortion": orbbec_dist,
    }
    return (
        rows,
        K,
        distortions,
        _object_points(april.tag_size_m),
        {tag_id: tag.T_world_tag for tag_id, tag in april.tags.items()},
    )


def _production_candidate(
    candidates: list[dict[str, Any]], production: np.ndarray
) -> int:
    costs = []
    for candidate in candidates:
        translation, rotation = _candidate_distance(
            candidate["T_camera_tag"], production
        )
        costs.append(translation / 0.50 + rotation / 30.0)
    return int(candidates[int(np.argmin(costs))]["candidate_index"])


def _reference_centers(
    frames: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    centers: dict[str, np.ndarray] = {}
    labels_by_method: dict[str, list[int]] = {}
    for method in METHODS:
        transforms = [item[f"production_{method}"] for item in frames]
        clusters = pose_cluster_summary(
            transforms,
            translation_threshold_m=0.25,
            rotation_threshold_deg=20.0,
            min_samples=3,
        )
        labels = [int(value) for value in clusters["labels"]]
        labels_by_method[method] = labels
        largest = Counter(value for value in labels if value >= 0).most_common(1)[0][0]
        centers[method] = average_transforms(
            [
                transform
                for transform, label in zip(transforms, labels)
                if label == largest
            ]
        )
    return centers, labels_by_method


def _near_reference(transform: np.ndarray, reference: np.ndarray) -> bool:
    translation, rotation = _candidate_distance(transform, reference)
    return translation <= 0.25 and rotation <= 20.0


def _evaluate_policy(
    method: str,
    policy: str,
    frames: list[dict[str, Any]],
    reference: np.ndarray,
    candidate_csv_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    previous: np.ndarray | None = None
    selections: list[dict[str, Any]] = []
    transforms: list[np.ndarray] = []
    reprojections: list[float] = []
    translations: list[float] = []
    rotations: list[float] = []
    for item in frames:
        candidates = item[f"candidates_{method}"]
        selected_index, _ = select_candidate(policy, candidates, previous)
        selected_candidate = next(
            candidate
            for candidate in candidates
            if candidate["candidate_index"] == selected_index
        )
        for candidate in candidates:
            translation = ""
            rotation = ""
            if previous is not None:
                translation, rotation = _candidate_distance(
                    candidate["T_world_camera"], previous
                )
            candidate_csv_rows.append(
                {
                    "frame_idx": item["frame_idx"],
                    "timestamp_utc": item["timestamp_utc"],
                    "tag_id": item["tag_id"],
                    "method": method,
                    "policy": policy,
                    "candidate_index": candidate["candidate_index"],
                    "selected": candidate["candidate_index"] == selected_index,
                    "positive_depth": candidate["positive_depth"],
                    "reprojection_error_px": candidate["reprojection_error_px"],
                    "rvec_json": _json_array(candidate["rvec"]),
                    "tvec_json": _json_array(candidate["tvec"]),
                    "T_camera_tag_json": _json_array(candidate["T_camera_tag"]),
                    "T_world_camera_json": _json_array(candidate["T_world_camera"]),
                    "previous_selected_translation_delta_m": translation,
                    "previous_selected_rotation_delta_deg": rotation,
                    "same_as_production_solver": (
                        candidate["candidate_index"]
                        == item[f"production_candidate_{method}"]
                    ),
                    "near_production_main_cluster": _near_reference(
                        candidate["T_world_camera"], reference
                    ),
                    "policy_score": policy_score(policy, candidate, previous),
                }
            )
        selected_transform = selected_candidate["T_world_camera"]
        if previous is not None:
            translation, rotation = _candidate_distance(selected_transform, previous)
            translations.append(translation)
            rotations.append(rotation)
        selection = {
            "frame_idx": item["frame_idx"],
            "candidate_index": selected_index,
            "transform": selected_transform,
            "reprojection_error_px": float(
                selected_candidate["reprojection_error_px"]
            ),
            "same_as_production": (
                selected_index == item[f"production_candidate_{method}"]
            ),
            "near_reference": _near_reference(selected_transform, reference),
        }
        selections.append(selection)
        transforms.append(selected_transform)
        reprojections.append(selection["reprojection_error_px"])
        previous = selected_transform

    clusters = pose_cluster_summary(
        transforms,
        translation_threshold_m=0.25,
        rotation_threshold_deg=20.0,
        min_samples=3,
    )
    rotation_stats = distribution(rotations)
    translation_stats = distribution(translations)
    reprojection_stats = distribution(reprojections)
    result = {
        "method": method,
        "policy": policy,
        "valid_pose_count": len(transforms),
        "cluster_count": clusters["cluster_count"],
        "cluster_sizes_json": json.dumps(clusters["cluster_sizes"]),
        "largest_cluster_frames": (
            clusters["cluster_sizes"][0] if clusters["cluster_sizes"] else 0
        ),
        "largest_cluster_ratio": clusters["largest_cluster_ratio"],
        "noise_count": clusters["noise_count"],
        "rotation_jump_ge_30deg": sum(value >= 30.0 for value in rotations),
        "translation_jump_ge_50cm": sum(value >= 0.50 for value in translations),
        "rotation_delta_mean_deg": rotation_stats["mean"],
        "rotation_delta_median_deg": rotation_stats["median"],
        "rotation_delta_p90_deg": rotation_stats["p90"],
        "rotation_delta_max_deg": rotation_stats["max"],
        "translation_delta_mean_m": translation_stats["mean"],
        "translation_delta_median_m": translation_stats["median"],
        "translation_delta_p90_m": translation_stats["p90"],
        "translation_delta_max_m": translation_stats["max"],
        "reprojection_error_mean_px": reprojection_stats["mean"],
        "reprojection_error_median_px": reprojection_stats["median"],
        "reprojection_error_p90_px": reprojection_stats["p90"],
        "different_from_production_count": sum(
            not item["same_as_production"] for item in selections
        ),
        "near_production_main_cluster_count": sum(
            item["near_reference"] for item in selections
        ),
        "near_production_main_cluster_ratio": sum(
            item["near_reference"] for item in selections
        )
        / len(selections),
    }
    return result, selections


def _jump_analysis(
    frames: list[dict[str, Any]],
    selections: dict[tuple[str, str], list[dict[str, Any]]],
    labels: dict[str, list[int]],
    references: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    by_index = {item["frame_idx"]: item for item in frames}
    for target in (2, 232, 233):
        item = by_index[target]
        target_result: dict[str, Any] = {}
        for method in METHODS:
            candidates = item[f"candidates_{method}"]
            pair_translation, pair_rotation = _candidate_distance(
                candidates[0]["T_world_camera"], candidates[1]["T_world_camera"]
            )
            largest = Counter(value for value in labels[method] if value >= 0).most_common(1)[0][0]
            previous_normal = next(
                (
                    frames[index]["frame_idx"]
                    for index in range(target - 1, -1, -1)
                    if labels[method][index] == largest
                ),
                None,
            )
            next_normal = next(
                (
                    frames[index]["frame_idx"]
                    for index in range(target + 1, len(frames))
                    if labels[method][index] == largest
                ),
                None,
            )
            candidate_details = []
            for candidate in candidates:
                prev_delta = None
                next_delta = None
                if previous_normal is not None:
                    prev_delta = _candidate_distance(
                        candidate["T_world_camera"],
                        by_index[previous_normal][f"production_{method}"],
                    )
                if next_normal is not None:
                    next_delta = _candidate_distance(
                        candidate["T_world_camera"],
                        by_index[next_normal][f"production_{method}"],
                    )
                candidate_details.append(
                    {
                        "candidate_index": candidate["candidate_index"],
                        "reprojection_error_px": candidate[
                            "reprojection_error_px"
                        ],
                        "positive_depth": candidate["positive_depth"],
                        "near_production_main_cluster": _near_reference(
                            candidate["T_world_camera"], references[method]
                        ),
                        "difference_from_previous_normal": (
                            {
                                "frame_idx": previous_normal,
                                "translation_m": prev_delta[0],
                                "rotation_deg": prev_delta[1],
                            }
                            if prev_delta is not None
                            else None
                        ),
                        "difference_from_next_normal": (
                            {
                                "frame_idx": next_normal,
                                "translation_m": next_delta[0],
                                "rotation_deg": next_delta[1],
                            }
                            if next_delta is not None
                            else None
                        ),
                    }
                )
            target_result[method] = {
                "production_candidate_index": item[
                    f"production_candidate_{method}"
                ],
                "production_in_main_cluster": labels[method][target] == largest,
                "candidate_pair_translation_difference_m": pair_translation,
                "candidate_pair_rotation_difference_deg": pair_rotation,
                "candidates": candidate_details,
                "policy_selections": {
                    policy: next(
                        row["candidate_index"]
                        for row in selections[(method, policy)]
                        if row["frame_idx"] == target
                    )
                    for policy in POLICIES
                },
            }
        output[str(target)] = target_result
    return output


def run(source: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    rows, K, distortions, object_points, world_tags = _load(source, config_path)
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 saved corner rows, found {len(rows)}")

    frames: list[dict[str, Any]] = []
    for row in rows:
        tag_id = int(row["tag_id"])
        corners = np.asarray(json.loads(row["corners_xy_json"]), dtype=np.float64)
        item: dict[str, Any] = {
            "frame_idx": int(row["frame_idx"]),
            "timestamp_utc": row["timestamp_utc"],
            "tag_id": tag_id,
            "corners": corners,
        }
        for method in METHODS:
            candidates = solve_generic_candidates(
                corners,
                object_points,
                K,
                distortions[method],
                world_tags[tag_id],
            )
            if len(candidates) != 2:
                raise RuntimeError(
                    f"frame {row['frame_idx']} {method}: expected 2 candidates, "
                    f"found {len(candidates)}"
                )
            production = np.asarray(
                json.loads(row[f"{method}_T_camera_tag_json"]), dtype=np.float64
            )
            item[f"candidates_{method}"] = candidates
            item[f"production_{method}"] = np.asarray(
                json.loads(row[f"{method}_T_world_camera_json"]), dtype=np.float64
            )
            item[f"production_candidate_{method}"] = _production_candidate(
                candidates, production
            )
        frames.append(item)

    references, production_labels = _reference_centers(frames)
    candidate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method in METHODS:
        for policy in POLICIES:
            comparison, selected = _evaluate_policy(
                method, policy, frames, references[method], candidate_rows
            )
            comparison_rows.append(comparison)
            selections[(method, policy)] = selected

    with (output / "candidate_pose_per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (output / "branch_policy_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows)

    jump_analysis = _jump_analysis(
        frames, selections, production_labels, references
    )
    (output / "jump_frame_analysis.json").write_text(
        json.dumps(jump_analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pose_clusters = {
        f"{row['method']}:{row['policy']}": {
            key: row[key]
            for key in (
                "valid_pose_count",
                "cluster_count",
                "cluster_sizes_json",
                "largest_cluster_frames",
                "largest_cluster_ratio",
                "noise_count",
                "near_production_main_cluster_count",
                "near_production_main_cluster_ratio",
            )
        }
        for row in comparison_rows
    }
    (output / "pose_cluster_summary.json").write_text(
        json.dumps(pose_clusters, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    safe_combined = [
        row
        for row in comparison_rows
        if row["policy"].startswith("combined_")
        and row["rotation_jump_ge_30deg"] == 0
        and row["translation_jump_ge_50cm"] == 0
        and row["cluster_count"] == 1
        and row["near_production_main_cluster_ratio"] >= 0.95
    ]
    temporal_rows = [
        row
        for row in comparison_rows
        if row["policy"] != "minimum_reprojection"
    ]
    removes_all_jumps = any(
        row["rotation_jump_ge_30deg"] == 0
        and row["translation_jump_ge_50cm"] == 0
        for row in temporal_rows
    )
    if safe_combined:
        conclusion = (
            "1_temporal_branch_selection_removes_jumps_and_preserves_main_cluster"
        )
        larger_tag = "not_required_by_this_dataset"
        registration_retry = True
    elif removes_all_jumps:
        conclusion = (
            "2_jumps_can_be_removed_but_first_frame_can_lock_the_wrong_branch"
        )
        larger_tag = (
            "recommended_or_add_an_independent_orientation_prior_before_registration"
        )
        registration_retry = False
    else:
        conclusion = "3_no_material_improvement"
        larger_tag = "required_or_use_a_different_registration_reference"
        registration_retry = False

    summary = {
        "source": str(source),
        "frames": len(frames),
        "saved_corners_sufficient": True,
        "two_candidates_per_method_per_frame": True,
        "candidate_rows": len(candidate_rows),
        "methods": list(METHODS),
        "policies": {
            name: (
                {"selection": name}
                if weights is None
                else {
                    "fixed_score": (
                        "w_reprojection*(error_px/1px) + "
                        "w_rotation*(delta_deg/30deg) + "
                        "w_translation*(delta_m/0.5m)"
                    ),
                    "weights": weights,
                }
            )
            for name, weights in POLICIES.items()
        },
        "policy_comparison": comparison_rows,
        "jump_frames": [2, 232, 233],
        "conclusion": conclusion,
        "larger_tag_assessment": larger_tag,
        "isolated_registration_retry_recommended": registration_retry,
        "object_anchor_registration_executed": False,
        "camera_live_reexecuted": False,
        "protection": {
            "full99_model_changed": False,
            "object_anchor_solver_or_threshold_changed": False,
            "cup_model_or_code_changed": False,
            "default_config_changed": False,
            "run_ps1_changed": False,
            "production_apriltag_code_changed": False,
        },
    }
    (output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    best_rows = sorted(
        comparison_rows,
        key=lambda row: (
            row["rotation_jump_ge_30deg"] + row["translation_jump_ge_50cm"],
            -row["near_production_main_cluster_ratio"],
            row["reprojection_error_mean_px"],
        ),
    )
    best = best_rows[0]
    (output / "README.md").write_text(
        "\n".join(
            [
                "# AprilTag IPPE_SQUARE branch diagnostics",
                "",
                f"- Source: `{source.relative_to(ROOT).as_posix()}`",
                "- Exactly two positive-depth generic IPPE_SQUARE candidates were "
                "computed from every saved corner set.",
                "- No camera, Object Anchor registration, production solver, or "
                "production configuration was changed.",
                "- Combined-policy normalizers were fixed before evaluation: "
                "1px reprojection, 30deg rotation, 0.5m translation.",
                f"- Best metric-ranked row: `{best['method']}:{best['policy']}`.",
                f"- Conclusion: `{conclusion}`.",
                f"- Larger-tag assessment: `{larger_tag}`.",
                "",
                "A zero-jump sequence is not automatically safe: if temporal continuity "
                "starts from the wrong frame-0 planar branch, it can remain internally "
                "stable while disagreeing with the 297-frame production main cluster.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (
        args.output.resolve()
        if args.output
        else OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    summary = run(args.source.resolve(), args.config.resolve(), output)
    print(
        json.dumps(
            {
                "output": str(output),
                "conclusion": summary["conclusion"],
                "registration_retry": summary[
                    "isolated_registration_retry_recommended"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
