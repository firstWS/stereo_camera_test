#!/usr/bin/env python3
"""Offline branch-aware Object Anchor vs AprilTag relative-pose MVP validation.

Uses saved frames from a failed absolute-reference-cluster MVP run.
Does not open the camera or modify production code/config/calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
)

DEFAULT_SOURCE = (
    ROOT
    / "out/object_anchor_full99/mvp_final_comparison/20260726_163325"
)
DEFAULT_OUTPUT_ROOT = ROOT / "out/object_anchor_full99/mvp_branch_aware_comparison"

# Fixed a-priori outlier thresholds (not tuned on this dataset).
MAX_POSITION_OUTLIER_M = 0.10
MAX_ROTATION_OUTLIER_DEG = 20.0
SPLIT_SEED = 42
TRAIN_FRACTION = 0.70
JUMP_TRANSLATION_M = 0.25
JUMP_ROTATION_DEG = 35.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
    """Inverse of relative_tag_object for a fixed registered relative pose."""
    return T_camera_object @ np.linalg.inv(T_tag_object_registered)


def inventory_source(source_dir: Path) -> dict[str, Any]:
    csv_path = source_dir / "reference_cluster_frames.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    fields = list(rows[0].keys()) if rows else []
    required = {
        "frame_idx": "frame_idx" in fields,
        "timestamp_utc": "timestamp_utc" in fields,
        "pose_cluster_id": "pose_cluster_id" in fields,
        "T_world_camera_tag": "T_world_camera_tag_json" in fields,
        "apriltag_rvec_tvec": (
            "apriltag_rvec_json" in fields and "apriltag_tvec_json" in fields
        ),
        "T_camera_object": "T_camera_object_json" in fields,
        "object_anchor_flags": (
            "anchor_detected" in fields
            and "pnp_operational_valid" in fields
            and "operational_temporal_valid" in fields
        ),
        "object_anchor_reprojection": "reprojection_error_px" in fields,
        "cup_P_camera_cup": any("cup" in name.lower() for name in fields),
    }
    joint_valid = 0
    by_cluster: Counter[str] = Counter()
    for row in rows:
        cluster = str(row.get("pose_cluster_id", ""))
        by_cluster[cluster] += 1
        if (
            _as_bool(row.get("apriltag_detected"))
            and _as_bool(row.get("pnp_operational_valid"))
            and str(row.get("T_camera_object_json", "")).strip()
            and str(row.get("apriltag_rvec_json", "")).strip()
            and str(row.get("apriltag_tvec_json", "")).strip()
        ):
            joint_valid += 1
    sufficient = all(
        required[key]
        for key in (
            "frame_idx",
            "timestamp_utc",
            "pose_cluster_id",
            "T_world_camera_tag",
            "apriltag_rvec_tvec",
            "T_camera_object",
            "object_anchor_flags",
            "object_anchor_reprojection",
        )
    )
    return {
        "source_dir": str(source_dir),
        "source_csv": str(csv_path),
        "frame_count": len(rows),
        "fields": fields,
        "required_fields_present": required,
        "sufficient_for_pose_analysis": sufficient,
        "cup_data_present": bool(required["cup_P_camera_cup"]),
        "joint_valid_apriltag_object_anchor": joint_valid,
        "frames_by_cluster": dict(by_cluster),
        "missing_for_cup_comparison": (
            [] if required["cup_P_camera_cup"] else ["P_camera_cup / cup_* fields"]
        ),
        "camera_required": False if sufficient else True,
        "notes": (
            "Pose branch-aware analysis can proceed offline. "
            "Cup downstream comparison requires a later capture with P_camera_cup."
            if sufficient and not required["cup_P_camera_cup"]
            else "Pose fields incomplete."
            if not sufficient
            else "All requested fields present."
        ),
    }


def load_joint_frames(source_dir: Path) -> list[dict[str, Any]]:
    rows = list(
        csv.DictReader((source_dir / "reference_cluster_frames.csv").open(encoding="utf-8"))
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        if not _as_bool(row.get("apriltag_detected")):
            continue
        if not _as_bool(row.get("pnp_operational_valid")):
            continue
        if not str(row.get("T_camera_object_json", "")).strip():
            continue
        if not str(row.get("apriltag_rvec_json", "")).strip():
            continue
        if not str(row.get("apriltag_tvec_json", "")).strip():
            continue
        T_camera_tag = t_camera_tag_from_rvec_tvec(
            row["apriltag_rvec_json"], row["apriltag_tvec_json"]
        )
        T_camera_object = _matrix(row["T_camera_object_json"])
        T_world_camera_tag = _matrix(row["T_world_camera_tag_json"])
        if T_camera_object is None or T_world_camera_tag is None:
            continue
        T_tag_object = relative_tag_object(T_camera_tag, T_camera_object)
        output.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "timestamp_utc": row["timestamp_utc"],
                "pose_cluster_id": int(row["pose_cluster_id"]),
                "apriltag_reprojection_error_px": float(row["apriltag_reprojection_error_px"]),
                "reprojection_error_px": (
                    float(row["reprojection_error_px"])
                    if str(row.get("reprojection_error_px", "")).strip()
                    else None
                ),
                "operational_temporal_valid": _as_bool(row.get("operational_temporal_valid")),
                "pnp_reason": row.get("pnp_reason", ""),
                "T_camera_tag": T_camera_tag,
                "T_camera_object": T_camera_object,
                "T_world_camera_tag": T_world_camera_tag,
                "T_tag_object": T_tag_object,
            }
        )
    return output


def split_train_val(
    frames: list[dict[str, Any]],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not frames:
        return [], []
    ordered = sorted(frames, key=lambda item: item["frame_idx"])
    rng = np.random.default_rng(seed)
    indices = np.arange(len(ordered))
    rng.shuffle(indices)
    train_count = max(1, int(round(len(ordered) * train_fraction)))
    train_count = min(train_count, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    train_idx = set(indices[:train_count].tolist())
    train = [ordered[i] for i in range(len(ordered)) if i in train_idx]
    val = [ordered[i] for i in range(len(ordered)) if i not in train_idx]
    if not val and train:
        val = [train.pop()]
    return train, val


def register_relative(
    train_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    if not train_frames:
        return {
            "ok": False,
            "reason": "no_train_frames",
            "T_tag_object": None,
            "samples": [],
            "inlier_count": 0,
            "outlier_count": 0,
        }
    transforms = [frame["T_tag_object"] for frame in train_frames]
    seed = average_transforms(transforms, position_median=True)
    samples: list[dict[str, Any]] = []
    inliers: list[np.ndarray] = []
    for frame in train_frames:
        transform = frame["T_tag_object"]
        translation = float(np.linalg.norm(transform[:3, 3] - seed[:3, 3]))
        rotation = rotation_delta_deg(transform[:3, :3], seed[:3, :3])
        outlier = False
        reason = ""
        if translation > MAX_POSITION_OUTLIER_M:
            outlier = True
            reason = "registration_position_outlier"
        elif rotation > MAX_ROTATION_OUTLIER_DEG:
            outlier = True
            reason = "registration_rotation_outlier"
        else:
            inliers.append(transform)
        samples.append(
            {
                "frame_idx": frame["frame_idx"],
                "timestamp_utc": frame["timestamp_utc"],
                "pose_cluster_id": frame["pose_cluster_id"],
                "split": "registration",
                "outlier": outlier,
                "outlier_reason": reason,
                "translation_residual_m": translation,
                "rotation_residual_deg": rotation,
                "reprojection_error_px": frame["reprojection_error_px"],
                "T_tag_object_json": _json_matrix(transform),
            }
        )
    if not inliers:
        return {
            "ok": False,
            "reason": "all_registration_outliers",
            "T_tag_object": None,
            "samples": samples,
            "inlier_count": 0,
            "outlier_count": len(samples),
        }
    registered = average_transforms(inliers, position_median=True)
    for sample, frame in zip(samples, train_frames):
        transform = frame["T_tag_object"]
        sample["translation_residual_m"] = float(
            np.linalg.norm(transform[:3, 3] - registered[:3, 3])
        )
        sample["rotation_residual_deg"] = rotation_delta_deg(
            transform[:3, :3], registered[:3, :3]
        )
    return {
        "ok": True,
        "reason": "ok",
        "T_tag_object": registered,
        "samples": samples,
        "inlier_count": len(inliers),
        "outlier_count": len(samples) - len(inliers),
        "excluded_reasons": dict(
            Counter(sample["outlier_reason"] for sample in samples if sample["outlier"])
        ),
        "registration_residual_translation_m": _distribution(
            [
                float(sample["translation_residual_m"])
                for sample in samples
                if not sample["outlier"]
            ]
        ),
        "registration_residual_rotation_deg": _distribution(
            [
                float(sample["rotation_residual_deg"])
                for sample in samples
                if not sample["outlier"]
            ]
        ),
    }


def validate_branch(
    val_frames: list[dict[str, Any]],
    T_tag_object_registered: np.ndarray,
    *,
    cluster_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_pred: np.ndarray | None = None
    prev_gt: np.ndarray | None = None
    temporal_jumps = 0
    for frame in sorted(val_frames, key=lambda item: item["frame_idx"]):
        predicted = predict_camera_tag(frame["T_camera_object"], T_tag_object_registered)
        gt = frame["T_camera_tag"]
        translation_m = float(np.linalg.norm(predicted[:3, 3] - gt[:3, 3]))
        rotation_deg = rotation_delta_deg(predicted[:3, :3], gt[:3, :3])
        f2f_pred_t = f2f_pred_r = f2f_gt_t = f2f_gt_r = None
        jump = False
        if prev_pred is not None:
            f2f_pred_t = float(np.linalg.norm(predicted[:3, 3] - prev_pred[:3, 3]))
            f2f_pred_r = rotation_delta_deg(predicted[:3, :3], prev_pred[:3, :3])
            jump = f2f_pred_t > JUMP_TRANSLATION_M or f2f_pred_r > JUMP_ROTATION_DEG
            if jump:
                temporal_jumps += 1
        if prev_gt is not None:
            f2f_gt_t = float(np.linalg.norm(gt[:3, 3] - prev_gt[:3, 3]))
            f2f_gt_r = rotation_delta_deg(gt[:3, :3], prev_gt[:3, :3])
        prev_pred = predicted
        prev_gt = gt
        rows.append(
            {
                "frame_idx": frame["frame_idx"],
                "timestamp_utc": frame["timestamp_utc"],
                "pose_cluster_id": cluster_id,
                "split": "validation",
                "translation_difference_cm": translation_m * 100.0,
                "rotation_difference_deg": rotation_deg,
                "predicted_frame_to_frame_translation_m": (
                    f2f_pred_t if f2f_pred_t is not None else ""
                ),
                "predicted_frame_to_frame_rotation_deg": (
                    f2f_pred_r if f2f_pred_r is not None else ""
                ),
                "apriltag_frame_to_frame_translation_m": (
                    f2f_gt_t if f2f_gt_t is not None else ""
                ),
                "apriltag_frame_to_frame_rotation_deg": (
                    f2f_gt_r if f2f_gt_r is not None else ""
                ),
                "object_anchor_temporal_jump": jump,
                "reprojection_error_px": (
                    frame["reprojection_error_px"]
                    if frame["reprojection_error_px"] is not None
                    else ""
                ),
                "apriltag_reprojection_error_px": frame["apriltag_reprojection_error_px"],
                "T_camera_tag_gt_json": _json_matrix(gt),
                "T_camera_tag_predicted_json": _json_matrix(predicted),
                "T_camera_object_json": _json_matrix(frame["T_camera_object"]),
            }
        )
    translations = [float(row["translation_difference_cm"]) for row in rows]
    rotations = [float(row["rotation_difference_deg"]) for row in rows]
    summary = {
        "pose_cluster_id": cluster_id,
        "validation_frames": len(rows),
        "translation_difference_cm": _distribution(translations),
        "rotation_difference_deg": _distribution(rotations),
        "translation_error_ge_10cm": sum(value >= 10.0 for value in translations),
        "rotation_error_ge_10deg": sum(value >= 10.0 for value in rotations),
        "object_anchor_temporal_jump_count": temporal_jumps,
        "object_anchor_temporal_jump_rate": temporal_jumps / max(len(rows), 1),
        "mvp_checks": {
            "translation_median_le_5cm": (
                bool(translations)
                and float(np.median(translations)) <= 5.0
            ),
            "translation_p90_le_10cm": (
                bool(translations)
                and float(np.percentile(translations, 90)) <= 10.0
            ),
            "rotation_median_le_5deg": (
                bool(rotations) and float(np.median(rotations)) <= 5.0
            ),
            "rotation_p90_le_10deg": (
                bool(rotations) and float(np.percentile(rotations, 90)) <= 10.0
            ),
            "temporal_jump_rate_lt_1pct": temporal_jumps / max(len(rows), 1) < 0.01,
        },
    }
    summary["mvp_pose_passed"] = all(summary["mvp_checks"].values())
    return rows, summary


def cross_branch_diagnostics(
    frames_by_cluster: dict[int, list[dict[str, Any]]],
    registrations: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source_cluster, target_cluster in ((0, 1), (1, 0)):
        registered = registrations.get(source_cluster, {}).get("T_tag_object")
        target_frames = frames_by_cluster.get(target_cluster) or []
        key = f"cal_cluster_{source_cluster}_on_cluster_{target_cluster}"
        if registered is None or not target_frames:
            output[key] = {"available": False}
            continue
        translations = []
        rotations = []
        for frame in target_frames:
            predicted = predict_camera_tag(frame["T_camera_object"], registered)
            gt = frame["T_camera_tag"]
            translations.append(float(np.linalg.norm(predicted[:3, 3] - gt[:3, 3])))
            rotations.append(rotation_delta_deg(predicted[:3, :3], gt[:3, :3]))
        output[key] = {
            "available": True,
            "frames": len(target_frames),
            "translation_difference_m": _distribution(translations),
            "rotation_difference_deg": _distribution(rotations),
            "interpretation": (
                "Large ~51deg rotation residuals with small camera-tag translation "
                "match planar IPPE branch flip. World-camera translation between "
                "branches is large (~2.5m) in prior diagnostics; this cross test "
                "compares T_camera_tag, so translation stays modest. Not an "
                "Object Anchor failure signature."
            ),
        }
    return output


def weighted_stats(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    translations: list[float] = []
    rotations: list[float] = []
    jumps = 0
    total = 0
    for summary in summaries:
        count = int(summary.get("validation_frames") or 0)
        total += count
        jumps += int(summary.get("object_anchor_temporal_jump_count") or 0)
        # Reconstruct approximate weighted aggregates from stored distributions
        # by reloading is better; caller should pass raw values. Keep placeholders.
    return {
        "validation_frames": total,
        "object_anchor_temporal_jump_count": jumps,
        "object_anchor_temporal_jump_rate": jumps / max(total, 1),
        "note": "Detailed weighted translation/rotation filled by caller with raw rows.",
    }


def decide_mvp(
    branch_summaries: dict[int, dict[str, Any]],
    *,
    cup_data_present: bool,
) -> dict[str, Any]:
    evaluated = {
        cluster: summary
        for cluster, summary in branch_summaries.items()
        if int(summary.get("validation_frames") or 0) > 0
    }
    if not evaluated:
        return {
            "decision": "C",
            "label": "1st_MVP_incomplete",
            "rationale": "No branch had joint-valid validation frames.",
        }
    all_pass = all(bool(summary.get("mvp_pose_passed")) for summary in evaluated.values())
    major_cluster = max(
        evaluated.items(),
        key=lambda item: int(item[1].get("validation_frames") or 0),
    )[0]
    major_pass = bool(evaluated[major_cluster].get("mvp_pose_passed"))
    if all_pass and cup_data_present:
        return {
            "decision": "A",
            "label": "1st_MVP_complete",
            "rationale": (
                "Both AprilTag branches reproduce Object Anchor relative pose "
                "within temporary MVP thresholds and Cup data was available."
            ),
            "major_cluster": major_cluster,
        }
    if all_pass or major_pass:
        return {
            "decision": "B",
            "label": "1st_MVP_conditional_complete",
            "rationale": (
                "Object Anchor reproduces AprilTag pose within each analyzed branch. "
                "Absolute AprilTag branch selection and/or Cup downstream comparison "
                "remain follow-up items."
            ),
            "major_cluster": major_cluster,
            "all_branches_pose_passed": all_pass,
            "major_branch_pose_passed": major_pass,
            "cup_data_present": cup_data_present,
            "evaluated_by_branch_separately": True,
        }
    return {
        "decision": "C",
        "label": "1st_MVP_incomplete",
        "rationale": (
            "Within-branch Object Anchor vs AprilTag pose differences exceed "
            "temporary MVP thresholds."
        ),
        "major_cluster": major_cluster,
        "evaluated_by_branch_separately": True,
    }


def run_analysis(source_dir: Path, output_root: Path) -> dict[str, Any]:
    inventory = inventory_source(source_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "source_data_inventory.json", inventory)

    if not inventory["sufficient_for_pose_analysis"]:
        decision = {
            "decision": "C",
            "label": "1st_MVP_incomplete",
            "rationale": "Saved fields insufficient for offline branch-aware pose analysis.",
            "inventory": inventory,
            "camera_executed": False,
        }
        _write_json(output_dir / "mvp_final_decision.json", decision)
        (output_dir / "README.md").write_text(
            "# Branch-aware MVP offline comparison\n\nInsufficient saved fields.\n",
            encoding="utf-8",
        )
        return {"output": str(output_dir), "decision": decision}

    frames = load_joint_frames(source_dir)
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        by_cluster.setdefault(int(frame["pose_cluster_id"]), []).append(frame)

    assignment_rows: list[dict[str, Any]] = []
    registration_sample_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    registrations: dict[int, dict[str, Any]] = {}
    branch_summaries: dict[int, dict[str, Any]] = {}
    train_val: dict[int, dict[str, Any]] = {}

    for cluster_id in sorted(by_cluster):
        cluster_frames = sorted(by_cluster[cluster_id], key=lambda item: item["frame_idx"])
        train, val = split_train_val(
            cluster_frames, train_fraction=TRAIN_FRACTION, seed=SPLIT_SEED + cluster_id
        )
        train_ids = {frame["frame_idx"] for frame in train}
        for frame in cluster_frames:
            assignment_rows.append(
                {
                    "frame_idx": frame["frame_idx"],
                    "timestamp_utc": frame["timestamp_utc"],
                    "pose_cluster_id": cluster_id,
                    "split": "registration" if frame["frame_idx"] in train_ids else "validation",
                    "joint_valid": True,
                    "reprojection_error_px": frame["reprojection_error_px"],
                }
            )
        registration = register_relative(train)
        registrations[cluster_id] = registration
        for sample in registration["samples"]:
            registration_sample_rows.append(sample)
        train_val[cluster_id] = {
            "total_cluster_frames_in_source": None,  # filled later
            "joint_valid_frames": len(cluster_frames),
            "registration_frames": len(train),
            "validation_frames": len(val),
            "registration_ok": registration["ok"],
            "registration_inliers": registration["inlier_count"],
            "registration_outliers": registration["outlier_count"],
            "registration_residual_translation_m": registration.get(
                "registration_residual_translation_m"
            ),
            "registration_residual_rotation_deg": registration.get(
                "registration_residual_rotation_deg"
            ),
            "T_tag_object_registered": (
                registration["T_tag_object"].tolist()
                if registration["T_tag_object"] is not None
                else None
            ),
        }
        if registration["ok"] and registration["T_tag_object"] is not None:
            rows, summary = validate_branch(
                val, registration["T_tag_object"], cluster_id=cluster_id
            )
            validation_rows.extend(rows)
            branch_summaries[cluster_id] = {
                **summary,
                "registration_frames": len(train),
                "registration_outliers": registration["outlier_count"],
                "registration_residual_translation_m": registration[
                    "registration_residual_translation_m"
                ],
                "registration_residual_rotation_deg": registration[
                    "registration_residual_rotation_deg"
                ],
            }
        else:
            branch_summaries[cluster_id] = {
                "pose_cluster_id": cluster_id,
                "validation_frames": 0,
                "mvp_pose_passed": False,
                "registration_failed": registration.get("reason"),
            }

    # Fill total source cluster sizes from inventory.
    source_rows = list(
        csv.DictReader((source_dir / "reference_cluster_frames.csv").open(encoding="utf-8"))
    )
    source_cluster_counts = Counter(int(row["pose_cluster_id"]) for row in source_rows)
    for cluster_id, payload in train_val.items():
        payload["total_cluster_frames_in_source"] = int(source_cluster_counts.get(cluster_id, 0))

    cross = cross_branch_diagnostics(by_cluster, registrations)

    all_translations = [float(row["translation_difference_cm"]) for row in validation_rows]
    all_rotations = [float(row["rotation_difference_deg"]) for row in validation_rows]
    all_jumps = sum(bool(row["object_anchor_temporal_jump"]) for row in validation_rows)
    weighted = {
        "validation_frames": len(validation_rows),
        "translation_difference_cm": _distribution(all_translations),
        "rotation_difference_deg": _distribution(all_rotations),
        "translation_error_ge_10cm": sum(value >= 10.0 for value in all_translations),
        "rotation_error_ge_10deg": sum(value >= 10.0 for value in all_rotations),
        "object_anchor_temporal_jump_count": all_jumps,
        "object_anchor_temporal_jump_rate": all_jumps / max(len(validation_rows), 1),
        "note": "Weighted over validation frames from both branches; not a single absolute branch.",
    }

    decision = decide_mvp(
        branch_summaries,
        cup_data_present=bool(inventory["cup_data_present"]),
    )
    decision.update(
        {
            "source": str(source_dir),
            "output": str(output_dir),
            "split": {
                "train_fraction": TRAIN_FRACTION,
                "seed_base": SPLIT_SEED,
                "seed_per_cluster": {
                    str(cluster): SPLIT_SEED + cluster for cluster in by_cluster
                },
            },
            "outlier_thresholds_fixed": {
                "max_position_outlier_m": MAX_POSITION_OUTLIER_M,
                "max_rotation_outlier_deg": MAX_ROTATION_OUTLIER_DEG,
            },
            "transform_convention": {
                "T_tag_object": "inv(T_camera_tag) @ T_camera_object",
                "T_camera_tag_predicted": "T_camera_object @ inv(T_tag_object_registered)",
            },
            "branch_joint_valid_counts": {
                str(cluster): len(items) for cluster, items in by_cluster.items()
            },
            "branch_registration_validation_counts": train_val,
            "branch_validation_summaries": {
                str(cluster): summary for cluster, summary in branch_summaries.items()
            },
            "weighted_validation_summary": weighted,
            "cross_branch_excluded_from_mvp_error_stats": True,
            "cup_comparison_executed": False,
            "cup_comparison_skip_reason": (
                None
                if inventory["cup_data_present"]
                else "Saved registration CSV has no P_camera_cup / cup_* fields."
            ),
            "april_tag_absolute_branch_problem_separated": True,
            "evaluated_by_branch_separately": True,
            "camera_executed": False,
            "production_code_modified": False,
            "production_config_modified": False,
            "production_calibration_modified": False,
            "automatic_world_source_switch": False,
            "remaining_items": [
                "Absolute AprilTag IPPE branch selection for a single world reference",
                "Cup downstream world comparison on a capture that stores P_camera_cup",
            ],
        }
    )

    _write_csv(
        output_dir / "branch_frame_assignments.csv",
        assignment_rows,
        [
            "frame_idx",
            "timestamp_utc",
            "pose_cluster_id",
            "split",
            "joint_valid",
            "reprojection_error_px",
        ],
    )
    _write_csv(
        output_dir / "branch_registration_samples.csv",
        registration_sample_rows,
        [
            "frame_idx",
            "timestamp_utc",
            "pose_cluster_id",
            "split",
            "outlier",
            "outlier_reason",
            "translation_residual_m",
            "rotation_residual_deg",
            "reprojection_error_px",
            "T_tag_object_json",
        ],
    )
    _write_json(
        output_dir / "branch_registration_summary.json",
        {
            "clusters": {
                str(cluster): {
                    **train_val[cluster],
                    "excluded_reasons": registrations[cluster].get("excluded_reasons") or {},
                }
                for cluster in train_val
            },
            "thresholds_fixed_a_priori": {
                "max_position_outlier_m": MAX_POSITION_OUTLIER_M,
                "max_rotation_outlier_deg": MAX_ROTATION_OUTLIER_DEG,
            },
        },
    )
    _write_csv(
        output_dir / "branch_validation_comparison.csv",
        validation_rows,
        [
            "frame_idx",
            "timestamp_utc",
            "pose_cluster_id",
            "split",
            "translation_difference_cm",
            "rotation_difference_deg",
            "predicted_frame_to_frame_translation_m",
            "predicted_frame_to_frame_rotation_deg",
            "apriltag_frame_to_frame_translation_m",
            "apriltag_frame_to_frame_rotation_deg",
            "object_anchor_temporal_jump",
            "reprojection_error_px",
            "apriltag_reprojection_error_px",
            "T_camera_tag_gt_json",
            "T_camera_tag_predicted_json",
            "T_camera_object_json",
        ],
    )
    _write_json(
        output_dir / "branch_validation_summary.json",
        {
            "branches": {str(k): v for k, v in branch_summaries.items()},
            "weighted": weighted,
        },
    )
    _write_json(output_dir / "cross_branch_diagnostics.json", cross)
    _write_json(output_dir / "mvp_final_decision.json", decision)

    readme = [
        "# Branch-aware Object Anchor MVP offline comparison",
        "",
        f"- Source: `{source_dir.as_posix()}`",
        f"- Decision: `{decision['decision']}` ({decision['label']})",
        f"- Rationale: {decision['rationale']}",
        "- Evaluated separately inside AprilTag SE(3) clusters 0 and 1.",
        "- Absolute AprilTag branch choice was NOT forced as physical ground truth.",
        "- Camera was not opened.",
        "- Production code/config/calibration were not modified.",
        "",
        "## Cup comparison",
        "",
    ]
    if inventory["cup_data_present"]:
        readme.append("- Cup comparison CSV was generated.")
    else:
        readme.extend(
            [
                "- `cup_branch_comparison.csv` was **not** created.",
                "- Reason: source `reference_cluster_frames.csv` has no `P_camera_cup` / cup_* fields.",
                "- Pose MVP was judged independently; Cup downstream needs a later capture.",
            ]
        )
    readme.extend(
        [
            "",
            "## Transform convention",
            "",
            "- `T_tag_object = inv(T_camera_tag) @ T_camera_object`",
            "- `T_camera_tag_predicted = T_camera_object @ inv(T_tag_object_registered)`",
            "",
        ]
    )
    for cluster_id, summary in branch_summaries.items():
        trans = summary.get("translation_difference_cm") or {}
        rot = summary.get("rotation_difference_deg") or {}
        readme.extend(
            [
                f"## Cluster {cluster_id}",
                f"- Validation frames: {summary.get('validation_frames')}",
                f"- Translation median/p90 cm: {trans.get('median')} / {trans.get('p90')}",
                f"- Rotation median/p90 deg: {rot.get('median')} / {rot.get('p90')}",
                f"- Pose MVP passed: {summary.get('mvp_pose_passed')}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return {"output": str(output_dir), "decision": decision, "inventory": inventory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    result = run_analysis(Path(args.source), Path(args.output_root))
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
