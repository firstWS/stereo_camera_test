#!/usr/bin/env python3
"""Offline translation-error diagnosis and causal temporal-filter comparison for MVP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_world import (  # noqa: E402
    average_transforms,
    rotation_delta_deg,
    rotation_matrix_to_quaternion,
)

DEFAULT_BRANCH_AWARE = (
    ROOT / "out/object_anchor_full99/mvp_branch_aware_comparison/20260726_164157"
)
DEFAULT_LIVE_SOURCE = (
    ROOT / "out/object_anchor_full99/mvp_final_comparison/20260726_163325"
)
DEFAULT_OUTPUT_ROOT = ROOT / "out/object_anchor_full99/mvp_translation_diagnostics"

FILTERS = {
    "A_raw": 1,
    "B_median3": 3,
    "C_median5": 5,
    "D_median7": 7,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _matrix(value: str | None) -> np.ndarray | None:
    if value is None or str(value).strip() == "":
        return None
    return np.asarray(json.loads(str(value)), dtype=np.float64).reshape(4, 4)


def _json_matrix(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    return json.dumps(np.asarray(value, dtype=np.float64).tolist(), separators=(",", ":"))


def t_camera_tag_from_rvec_tvec(rvec_json: str, tvec_json: str) -> np.ndarray:
    rvec = np.asarray(json.loads(rvec_json), dtype=np.float64).reshape(3)
    tvec = np.asarray(json.loads(tvec_json), dtype=np.float64).reshape(3)
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec
    return transform


def predict_camera_tag(
    T_camera_object: np.ndarray,
    T_tag_object_registered: np.ndarray,
) -> np.ndarray:
    return T_camera_object @ np.linalg.inv(T_tag_object_registered)


def causal_filter_pose(
    history_including_current: list[np.ndarray],
    window: int,
) -> np.ndarray:
    """Causal SE(3) filter: translation median + sign-aligned quaternion average.

    Uses only the trailing `window` poses from history that ends at the current frame.
    Never looks ahead. If history is shorter than window, uses all available past+current.
    """
    if not history_including_current:
        raise ValueError("empty history")
    if window < 1:
        raise ValueError("window must be >= 1")
    selected = history_including_current[-window:]
    if len(selected) == 1:
        return np.asarray(selected[0], dtype=np.float64).copy()
    return average_transforms(selected, position_median=True)


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_assignments(branch_aware_dir: Path) -> dict[int, str]:
    path = branch_aware_dir / "branch_frame_assignments.csv"
    mapping: dict[int, str] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        mapping[int(row["frame_idx"])] = str(row["split"])
    return mapping


def load_registrations(branch_aware_dir: Path) -> dict[int, np.ndarray]:
    payload = json.loads(
        (branch_aware_dir / "branch_registration_summary.json").read_text(encoding="utf-8")
    )
    output: dict[int, np.ndarray] = {}
    for key, value in payload["clusters"].items():
        matrix = value.get("T_tag_object_registered")
        if matrix is not None:
            output[int(key)] = np.asarray(matrix, dtype=np.float64)
    return output


def load_joint_frames(live_source: Path, assignments: dict[int, str]) -> list[dict[str, Any]]:
    rows = list(
        csv.DictReader((live_source / "reference_cluster_frames.csv").open(encoding="utf-8"))
    )
    frames: list[dict[str, Any]] = []
    previous_object: np.ndarray | None = None
    for row in sorted(rows, key=lambda item: int(item["frame_idx"])):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in assignments:
            continue
        if not _as_bool(row.get("pnp_operational_valid")):
            continue
        if not str(row.get("T_camera_object_json", "")).strip():
            continue
        T_camera_object = _matrix(row["T_camera_object_json"])
        T_camera_tag = t_camera_tag_from_rvec_tvec(
            row["apriltag_rvec_json"], row["apriltag_tvec_json"]
        )
        if T_camera_object is None:
            continue
        f2f = None
        if previous_object is not None:
            f2f = float(np.linalg.norm(T_camera_object[:3, 3] - previous_object[:3, 3]))
        previous_object = T_camera_object
        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_utc": row["timestamp_utc"],
                "pose_cluster_id": int(row["pose_cluster_id"]),
                "split": assignments[frame_idx],
                "reprojection_error_px": (
                    float(row["reprojection_error_px"])
                    if str(row.get("reprojection_error_px", "")).strip()
                    else None
                ),
                "T_camera_object": T_camera_object,
                "T_camera_tag": T_camera_tag,
                "object_f2f_translation_m": f2f,
                "bbox_area_px2": None,
                "polygon_area_px2": None,
                "object_center_u": None,
                "object_center_v": None,
            }
        )
    return frames


def axis_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "signed_mean": None,
            "median": None,
            "std": None,
            "p90_abs": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "signed_mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p90_abs": float(np.percentile(np.abs(array), 90)),
    }


def analyze_axes(
    frames: list[dict[str, Any]],
    registrations: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        if frame["split"] != "validation":
            continue
        registered = registrations.get(int(frame["pose_cluster_id"]))
        if registered is None:
            continue
        predicted = predict_camera_tag(frame["T_camera_object"], registered)
        gt = frame["T_camera_tag"]
        err = predicted[:3, 3] - gt[:3, 3]
        rot = rotation_delta_deg(predicted[:3, :3], gt[:3, :3])
        rows.append(
            {
                "frame_idx": frame["frame_idx"],
                "pose_cluster_id": frame["pose_cluster_id"],
                "split": frame["split"],
                "error_x_m": float(err[0]),
                "error_y_m": float(err[1]),
                "error_z_m": float(err[2]),
                "error_norm_m": float(np.linalg.norm(err)),
                "error_norm_cm": float(np.linalg.norm(err) * 100.0),
                "rotation_error_deg": float(rot),
                "reprojection_error_px": frame["reprojection_error_px"],
                "object_f2f_translation_m": frame["object_f2f_translation_m"],
                "bbox_area_px2": "",
                "polygon_area_px2": "",
                "object_center_u": "",
                "object_center_v": "",
                "bbox_polygon_available": False,
            }
        )
    by_cluster: dict[str, Any] = {}
    for cluster in sorted({int(row["pose_cluster_id"]) for row in rows}):
        subset = [row for row in rows if int(row["pose_cluster_id"]) == cluster]
        xs = [float(row["error_x_m"]) for row in subset]
        ys = [float(row["error_y_m"]) for row in subset]
        zs = [float(row["error_z_m"]) for row in subset]
        norms = [float(row["error_norm_cm"]) for row in subset]
        reproj = [
            float(row["reprojection_error_px"])
            for row in subset
            if row["reprojection_error_px"] is not None and row["reprojection_error_px"] != ""
        ]
        f2f = [
            float(row["object_f2f_translation_m"])
            for row in subset
            if row["object_f2f_translation_m"] is not None
            and row["object_f2f_translation_m"] != ""
        ]
        by_cluster[str(cluster)] = {
            "count": len(subset),
            "axis_x_m": axis_stats(xs),
            "axis_y_m": axis_stats(ys),
            "axis_z_m": axis_stats(zs),
            "norm_cm": _distribution(norms),
            "dominant_axis_by_abs_signed_mean": max(
                (
                    ("x", abs(axis_stats(xs)["signed_mean"] or 0.0)),
                    ("y", abs(axis_stats(ys)["signed_mean"] or 0.0)),
                    ("z", abs(axis_stats(zs)["signed_mean"] or 0.0)),
                ),
                key=lambda item: item[1],
            )[0],
            "bias_vs_random": {
                "x_abs_mean_over_std": (
                    abs(axis_stats(xs)["signed_mean"] or 0.0)
                    / max(axis_stats(xs)["std"] or 1e-12, 1e-12)
                ),
                "y_abs_mean_over_std": (
                    abs(axis_stats(ys)["signed_mean"] or 0.0)
                    / max(axis_stats(ys)["std"] or 1e-12, 1e-12)
                ),
                "z_abs_mean_over_std": (
                    abs(axis_stats(zs)["signed_mean"] or 0.0)
                    / max(axis_stats(zs)["std"] or 1e-12, 1e-12)
                ),
            },
            "correlations": {
                "norm_vs_reprojection": pearson(norms, reproj) if len(reproj) == len(norms) else None,
                "norm_vs_object_f2f": pearson(
                    [float(row["error_norm_cm"]) for row in subset if row["object_f2f_translation_m"] is not None],
                    [
                        float(row["object_f2f_translation_m"]) * 100.0
                        for row in subset
                        if row["object_f2f_translation_m"] is not None
                    ],
                ),
                "bbox_unavailable": True,
                "polygon_unavailable": True,
                "screen_position_unavailable": True,
            },
        }
    # Weighted / all validation
    xs = [float(row["error_x_m"]) for row in rows]
    ys = [float(row["error_y_m"]) for row in rows]
    zs = [float(row["error_z_m"]) for row in rows]
    norms = [float(row["error_norm_cm"]) for row in rows]
    reproj = [
        float(row["reprojection_error_px"])
        for row in rows
        if row["reprojection_error_px"] is not None and row["reprojection_error_px"] != ""
    ]
    correlation_summary = {
        "bbox_polygon_center_note": (
            "Object Anchor bbox/keypoint polygon/center were not stored in "
            "reference_cluster_frames.csv; those correlations are unavailable offline."
        ),
        "branches": by_cluster,
        "weighted": {
            "count": len(rows),
            "axis_x_m": axis_stats(xs),
            "axis_y_m": axis_stats(ys),
            "axis_z_m": axis_stats(zs),
            "norm_cm": _distribution(norms),
            "dominant_axis_by_abs_signed_mean": max(
                (
                    ("x", abs(axis_stats(xs)["signed_mean"] or 0.0)),
                    ("y", abs(axis_stats(ys)["signed_mean"] or 0.0)),
                    ("z", abs(axis_stats(zs)["signed_mean"] or 0.0)),
                ),
                key=lambda item: item[1],
            )[0],
            "correlations": {
                "norm_vs_reprojection": pearson(norms, reproj) if len(reproj) == len(norms) else None,
            },
        },
    }
    return rows, correlation_summary


def evaluate_filter_on_cluster(
    frames: list[dict[str, Any]],
    *,
    cluster_id: int,
    registered: np.ndarray,
    window: int,
    filter_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cluster_frames = sorted(
        [frame for frame in frames if int(frame["pose_cluster_id"]) == cluster_id],
        key=lambda item: item["frame_idx"],
    )
    history: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    prev_validation_filtered: np.ndarray | None = None
    jump_rot = 0
    jump_trans = 0
    for frame in cluster_frames:
        history.append(frame["T_camera_object"])
        filtered = causal_filter_pose(history, window)
        if frame["split"] != "validation":
            # Keep history for causality, but do not score registration frames.
            continue
        predicted = predict_camera_tag(filtered, registered)
        gt = frame["T_camera_tag"]
        translation_m = float(np.linalg.norm(predicted[:3, 3] - gt[:3, 3]))
        rotation_deg = rotation_delta_deg(predicted[:3, :3], gt[:3, :3])
        f2f_t = f2f_r = None
        if prev_validation_filtered is not None:
            f2f_t = float(
                np.linalg.norm(filtered[:3, 3] - prev_validation_filtered[:3, 3])
            )
            f2f_r = rotation_delta_deg(
                filtered[:3, :3], prev_validation_filtered[:3, :3]
            )
            if f2f_t >= 0.50:
                jump_trans += 1
            if f2f_r >= 30.0:
                jump_rot += 1
        prev_validation_filtered = filtered
        rows.append(
            {
                "frame_idx": frame["frame_idx"],
                "pose_cluster_id": cluster_id,
                "split": "validation",
                "filter": filter_name,
                "window": window,
                "translation_difference_cm": translation_m * 100.0,
                "rotation_difference_deg": rotation_deg,
                "filtered_f2f_translation_m": f2f_t if f2f_t is not None else "",
                "filtered_f2f_rotation_deg": f2f_r if f2f_r is not None else "",
                "reprojection_error_px": frame["reprojection_error_px"],
                "T_camera_object_filtered_json": _json_matrix(filtered),
            }
        )
    translations = [float(row["translation_difference_cm"]) for row in rows]
    rotations = [float(row["rotation_difference_deg"]) for row in rows]
    summary = {
        "filter": filter_name,
        "window": window,
        "pose_cluster_id": cluster_id,
        "validation_frames": len(rows),
        "translation_cm": _distribution(translations),
        "rotation_deg": _distribution(rotations),
        "translation_error_ge_10cm": sum(value >= 10.0 for value in translations),
        "rotation_error_ge_10deg": sum(value >= 10.0 for value in rotations),
        "temporal_rotation_jump_ge_30deg": jump_rot,
        "temporal_translation_jump_ge_50cm": jump_trans,
        "filter_latency_frames": max(window - 1, 0),
        "mvp_checks": {
            "translation_median_le_5cm": bool(translations)
            and float(np.median(translations)) <= 5.0,
            "translation_p90_le_10cm": bool(translations)
            and float(np.percentile(translations, 90)) <= 10.0,
            "rotation_median_le_5deg": bool(rotations)
            and float(np.median(rotations)) <= 5.0,
            "rotation_p90_le_10deg": bool(rotations)
            and float(np.percentile(rotations, 90)) <= 10.0,
            "temporal_jump_rate_lt_1pct": (jump_rot + jump_trans) / max(len(rows), 1) < 0.01,
        },
    }
    summary["mvp_pose_passed"] = all(summary["mvp_checks"].values())
    return rows, summary


def improvement_rate(raw: float | None, filtered: float | None) -> float | None:
    if raw is None or filtered is None or raw == 0:
        return None
    return float((raw - filtered) / raw)


def select_filter(filter_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select among A/B/C/D using the required priority rules."""
    candidates = []
    for name, payload in filter_summaries.items():
        weighted = payload["weighted"]
        branches = payload["branches"]
        raw_weighted = filter_summaries["A_raw"]["weighted"]
        branch_ok = all(bool(branch.get("mvp_pose_passed")) for branch in branches.values())
        weighted_ok = (
            weighted["translation_cm"]["median"] is not None
            and weighted["translation_cm"]["median"] <= 5.0
            and weighted["translation_cm"]["p90"] is not None
            and weighted["translation_cm"]["p90"] <= 10.0
            and weighted["rotation_deg"]["median"] is not None
            and weighted["rotation_deg"]["median"] <= 5.0
            and weighted["rotation_deg"]["p90"] is not None
            and weighted["rotation_deg"]["p90"] <= 10.0
            and (
                (
                    weighted["temporal_rotation_jump_ge_30deg"]
                    + weighted["temporal_translation_jump_ge_50cm"]
                )
                / max(weighted["validation_frames"], 1)
                < 0.01
            )
        )
        # Branch translation should not worsen a lot vs raw (>= +1cm median or +2cm p90).
        branch_not_worse = True
        for cluster, branch in branches.items():
            raw_branch = filter_summaries["A_raw"]["branches"][cluster]
            if branch["translation_cm"]["median"] is None or raw_branch["translation_cm"]["median"] is None:
                branch_not_worse = False
                break
            if branch["translation_cm"]["median"] > raw_branch["translation_cm"]["median"] + 1.0:
                branch_not_worse = False
                break
            if branch["translation_cm"]["p90"] > raw_branch["translation_cm"]["p90"] + 2.0:
                branch_not_worse = False
                break
        jumps_not_increased = (
            weighted["temporal_rotation_jump_ge_30deg"]
            + weighted["temporal_translation_jump_ge_50cm"]
        ) <= (
            raw_weighted["temporal_rotation_jump_ge_30deg"]
            + raw_weighted["temporal_translation_jump_ge_50cm"]
        )
        max_not_pathological = (
            weighted["translation_cm"]["max"] is not None
            and raw_weighted["translation_cm"]["max"] is not None
            and weighted["translation_cm"]["max"]
            <= raw_weighted["translation_cm"]["max"] + 5.0
        )
        if branch_ok and weighted_ok and branch_not_worse and jumps_not_increased and max_not_pathological:
            candidates.append(
                {
                    "filter": name,
                    "window": payload["window"],
                    "weighted": weighted,
                }
            )
    if not candidates:
        return {
            "selected": None,
            "verdict": "C",
            "reason": (
                "No causal filter among raw/3/5/7 satisfied both-branch and weighted "
                "MVP translation thresholds without worsening jumps/max error."
            ),
        }
    candidates.sort(key=lambda item: item["window"])
    chosen = candidates[0]
    return {
        "selected": chosen["filter"],
        "window": chosen["window"],
        "verdict": "A",
        "reason": (
            f"Filter {chosen['filter']} meets both-branch and weighted MVP thresholds "
            "with the smallest causal window among successful candidates."
        ),
        "apply_to_isolated_runner_only": True,
        "do_not_modify_production": True,
    }


def classify_bias(correlation_summary: dict[str, Any]) -> dict[str, Any]:
    weighted = correlation_summary["weighted"]
    ratios = {
        "x": abs(weighted["axis_x_m"]["signed_mean"] or 0.0)
        / max(weighted["axis_x_m"]["std"] or 1e-12, 1e-12),
        "y": abs(weighted["axis_y_m"]["signed_mean"] or 0.0)
        / max(weighted["axis_y_m"]["std"] or 1e-12, 1e-12),
        "z": abs(weighted["axis_z_m"]["signed_mean"] or 0.0)
        / max(weighted["axis_z_m"]["std"] or 1e-12, 1e-12),
    }
    dominant = max(ratios.items(), key=lambda item: item[1])
    # Heuristic: mean/std >= 1.0 on an axis => consistent bias; else random-dominated.
    if dominant[1] >= 1.0 and abs(weighted["axis_" + dominant[0] + "_m"]["signed_mean"] or 0) >= 0.02:
        return {
            "mode": "consistent_axis_bias",
            "dominant_axis": dominant[0],
            "mean_over_std": dominant[1],
            "verdict_hint": "B",
            "priority_checks": [
                "object_anchor physical size / keypoints_3d coordinates",
                "object origin definition (object_center vs face)",
                "camera intrinsic / depthless PnP scale coupling on Z",
                "RGB keypoint localization bias",
            ],
        }
    return {
        "mode": "random_dominated_or_mixed",
        "dominant_axis": dominant[0],
        "mean_over_std": dominant[1],
        "verdict_hint": "C",
        "priority_checks": [
            "keypoint jitter / detector stability",
            "RGB-D translation assist",
            "larger apparent object size in image",
        ],
    }


def inspect_cup_logging_readiness(runner_path: Path) -> dict[str, Any]:
    text = runner_path.read_text(encoding="utf-8")
    reference_has_cup = "P_camera_cup_x" in text and "REFERENCE_CLUSTER_FIELDS" in text
    # More precise: check whether REFERENCE_CLUSTER_FIELDS tuple includes cup fields.
    start = text.find("REFERENCE_CLUSTER_FIELDS")
    end = text.find("REGISTRATION_SAMPLE_FIELDS")
    block = text[start:end] if start >= 0 and end > start else ""
    cup_in_reference = "P_camera_cup_x" in block
    compare_has_filtered = "T_camera_object_filtered_json" in text
    return {
        "runner": str(runner_path),
        "root_cause_of_previous_miss": (
            "Registration-phase export used REFERENCE_CLUSTER_FIELDS, which omitted "
            "cup_* / P_camera_cup even though capture rows already computed Cup camera points. "
            "The run aborted before comparison CSVs were written."
        ),
        "cup_fields_in_reference_cluster_export": cup_in_reference,
        "filtered_pose_field_present": compare_has_filtered,
        "required_next_live_fields": [
            "cup_detected",
            "cup_valid_depth",
            "P_camera_cup_x/y/z",
            "pose_cluster_id / AprilTag cluster",
            "T_world_camera_tag",
            "T_camera_object raw",
            "T_camera_object filtered",
            "T_world_camera_object",
            "P_world_cup_tag",
            "P_world_cup_object",
            "cup_world_difference_cm",
        ],
        "production_code_changed": False,
    }


def run(
    branch_aware_dir: Path,
    live_source: Path,
    output_root: Path,
) -> dict[str, Any]:
    assignments = load_assignments(branch_aware_dir)
    registrations = load_registrations(branch_aware_dir)
    frames = load_joint_frames(live_source, assignments)
    # Ensure no validation frame was used as registration in assignments file.
    reg_ids = {f["frame_idx"] for f in frames if f["split"] == "registration"}
    val_ids = {f["frame_idx"] for f in frames if f["split"] == "validation"}
    assert reg_ids.isdisjoint(val_ids)

    axis_rows, correlation_summary = analyze_axes(frames, registrations)
    bias = classify_bias(correlation_summary)

    all_filter_rows: list[dict[str, Any]] = []
    filter_summaries: dict[str, Any] = {}
    for filter_name, window in FILTERS.items():
        branch_summaries: dict[str, Any] = {}
        branch_rows: list[dict[str, Any]] = []
        for cluster_id, registered in registrations.items():
            rows, summary = evaluate_filter_on_cluster(
                frames,
                cluster_id=cluster_id,
                registered=registered,
                window=window,
                filter_name=filter_name,
            )
            branch_rows.extend(rows)
            branch_summaries[str(cluster_id)] = summary
        translations = [float(r["translation_difference_cm"]) for r in branch_rows]
        rotations = [float(r["rotation_difference_deg"]) for r in branch_rows]
        jump_rot = sum(int(s["temporal_rotation_jump_ge_30deg"]) for s in branch_summaries.values())
        jump_trans = sum(
            int(s["temporal_translation_jump_ge_50cm"]) for s in branch_summaries.values()
        )
        weighted = {
            "validation_frames": len(branch_rows),
            "translation_cm": _distribution(translations),
            "rotation_deg": _distribution(rotations),
            "translation_error_ge_10cm": sum(v >= 10.0 for v in translations),
            "rotation_error_ge_10deg": sum(v >= 10.0 for v in rotations),
            "temporal_rotation_jump_ge_30deg": jump_rot,
            "temporal_translation_jump_ge_50cm": jump_trans,
            "filter_latency_frames": max(window - 1, 0),
            "mvp_pose_passed": all(s["mvp_pose_passed"] for s in branch_summaries.values())
            and bool(translations)
            and float(np.median(translations)) <= 5.0
            and float(np.percentile(translations, 90)) <= 10.0
            and float(np.median(rotations)) <= 5.0
            and float(np.percentile(rotations, 90)) <= 10.0
            and (jump_rot + jump_trans) / max(len(branch_rows), 1) < 0.01,
        }
        filter_summaries[filter_name] = {
            "window": window,
            "branches": branch_summaries,
            "weighted": weighted,
        }
        all_filter_rows.extend(branch_rows)

    # Add improvement vs raw
    raw_w = filter_summaries["A_raw"]["weighted"]["translation_cm"]
    for name, payload in filter_summaries.items():
        payload["improvement_vs_raw"] = {
            "translation_median": improvement_rate(
                raw_w["median"], payload["weighted"]["translation_cm"]["median"]
            ),
            "translation_p90": improvement_rate(
                raw_w["p90"], payload["weighted"]["translation_cm"]["p90"]
            ),
        }

    selected = select_filter(filter_summaries)
    # Final diagnostic verdict A/B/C combining filter success and bias analysis.
    if selected.get("selected") is not None:
        verdict = "A"
        verdict_reason = selected["reason"]
    elif bias["mode"] == "consistent_axis_bias":
        verdict = "B"
        verdict_reason = (
            "No short causal filter met MVP translation thresholds; "
            f"axis analysis indicates consistent bias on {bias['dominant_axis']}."
        )
    else:
        verdict = "C"
        verdict_reason = (
            "No short causal filter met MVP translation thresholds; "
            "errors look mixed/random-dominated rather than a clean fixed offset."
        )

    cup_ready = inspect_cup_logging_readiness(
        ROOT / "experiments/object_anchor_mvp_final_comparison.py"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    _write_csv(
        output_dir / "translation_axis_analysis.csv",
        axis_rows,
        [
            "frame_idx",
            "pose_cluster_id",
            "split",
            "error_x_m",
            "error_y_m",
            "error_z_m",
            "error_norm_m",
            "error_norm_cm",
            "rotation_error_deg",
            "reprojection_error_px",
            "object_f2f_translation_m",
            "bbox_area_px2",
            "polygon_area_px2",
            "object_center_u",
            "object_center_v",
            "bbox_polygon_available",
        ],
    )
    _write_json(output_dir / "translation_correlation_summary.json", {
        **correlation_summary,
        "bias_classification": bias,
    })
    _write_csv(
        output_dir / "temporal_filter_frame_comparison.csv",
        all_filter_rows,
        [
            "frame_idx",
            "pose_cluster_id",
            "split",
            "filter",
            "window",
            "translation_difference_cm",
            "rotation_difference_deg",
            "filtered_f2f_translation_m",
            "filtered_f2f_rotation_deg",
            "reprojection_error_px",
            "T_camera_object_filtered_json",
        ],
    )
    _write_json(output_dir / "temporal_filter_summary.json", filter_summaries)
    selected_payload = {
        **selected,
        "diagnostic_verdict": verdict,
        "diagnostic_verdict_reason": verdict_reason,
        "bias_classification": bias,
        "mvp_thresholds_unchanged": True,
        "camera_executed": False,
    }
    _write_json(output_dir / "selected_filter.json", selected_payload)
    _write_json(output_dir / "cup_logging_readiness.json", cup_ready)
    diagnostic_summary = {
        "source_branch_aware": str(branch_aware_dir),
        "source_live": str(live_source),
        "output": str(output_dir),
        "split_reuse_validated": True,
        "registration_validation_disjoint": True,
        "filters_compared": list(FILTERS.keys()),
        "axis_analysis": correlation_summary["weighted"],
        "bias_classification": bias,
        "filter_weighted": {
            name: payload["weighted"] for name, payload in filter_summaries.items()
        },
        "selected_filter": selected_payload,
        "cup_logging_readiness": cup_ready,
        "camera_executed": False,
        "production_unmodified": True,
    }
    _write_json(output_dir / "diagnostic_summary.json", diagnostic_summary)

    readme = [
        "# MVP translation diagnostics (offline)",
        "",
        f"- Verdict: `{verdict}`",
        f"- Reason: {verdict_reason}",
        f"- Selected filter: `{selected.get('selected')}`",
        "- Camera was not opened.",
        "- Production code/config/thresholds were not modified.",
        "",
        "## Notes",
        "",
        "- Object Anchor bbox/polygon/center were unavailable in the saved registration CSV.",
        "- Cup P_camera_cup was computed live but omitted from REFERENCE_CLUSTER_FIELDS export.",
        "",
    ]
    for name, payload in filter_summaries.items():
        w = payload["weighted"]["translation_cm"]
        r = payload["weighted"]["rotation_deg"]
        readme.append(
            f"- {name}: trans med/p90={w['median']:.3f}/{w['p90']:.3f} cm; "
            f"rot med/p90={r['median']:.3f}/{r['p90']:.3f} deg"
        )
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return diagnostic_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-aware", default=str(DEFAULT_BRANCH_AWARE))
    parser.add_argument("--live-source", default=str(DEFAULT_LIVE_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    summary = run(Path(args.branch_aware), Path(args.live_source), Path(args.output_root))
    print(json.dumps(summary["selected_filter"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
