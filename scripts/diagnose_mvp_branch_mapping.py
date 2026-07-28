#!/usr/bin/env python3
"""Offline diagnostics for MVP registration↔preflight branch identity mapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from object_anchor_mvp_final_comparison import (  # noqa: E402
    average_transforms,
    load_mvp_settings,
    match_pose_to_prototypes,
    t_camera_tag_from_row,
)

DEFAULT_SOURCE = (
    ROOT / "out/object_anchor_full99/mvp_final_comparison/20260726_172325"
)
DEFAULT_OUTPUT_ROOT = ROOT / "out/object_anchor_full99/mvp_branch_mapping_diagnostics"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _matrix(value: str | None) -> np.ndarray | None:
    if value is None or str(value).strip() == "":
        return None
    return np.asarray(json.loads(str(value)), dtype=np.float64).reshape(4, 4)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_registration_prototypes(
    source_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    rows = list(
        csv.DictReader((source_dir / "reference_cluster_frames.csv").open(encoding="utf-8"))
    )
    summary = json.loads((source_dir / "registration_summary.json").read_text(encoding="utf-8"))
    by_branch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("pose_cluster_id") in ("", None):
            continue
        branch_id = int(row["pose_cluster_id"])
        T_world = _matrix(row.get("T_world_camera_tag_json"))
        T_camera_tag = t_camera_tag_from_row(row)
        if T_world is None:
            continue
        by_branch.setdefault(branch_id, []).append(
            {
                "frame_idx": int(row["frame_idx"]),
                "T_world_camera_tag": T_world,
                "T_camera_tag": T_camera_tag,
            }
        )
    prototypes: dict[str, Any] = {}
    for branch_id, members in sorted(by_branch.items()):
        world_centroid = average_transforms(
            [item["T_world_camera_tag"] for item in members], position_median=True
        )
        camera_members = [
            item["T_camera_tag"] for item in members if item["T_camera_tag"] is not None
        ]
        camera_tag_centroid = (
            average_transforms(camera_members, position_median=True)
            if camera_members
            else None
        )
        branch_meta = (summary.get("branches") or {}).get(str(branch_id), {})
        residual = branch_meta.get("residual_gate") or {}
        prototypes[str(branch_id)] = {
            "internal_branch_id": branch_id,
            "frame_count": len(members),
            "has_calibration": bool(branch_meta.get("accepted")),
            "registration_file": branch_meta.get("registration_file"),
            "T_world_camera_tag_prototype": world_centroid.tolist(),
            "T_camera_tag_prototype": (
                camera_tag_centroid.tolist() if camera_tag_centroid is not None else None
            ),
            "translation_center_m": world_centroid[:3, 3].tolist(),
            "registration_residual_translation_m": residual.get("translation_residual_m"),
            "registration_residual_rotation_deg": residual.get("rotation_residual_deg"),
            "match_thresholds": {
                "translation_m": float(settings["cluster_translation_threshold_m"]),
                "rotation_deg": float(settings["cluster_rotation_threshold_deg"]),
                "ambiguous_margin_translation_m": 0.05,
                "ambiguous_margin_rotation_deg": 5.0,
            },
            "note": (
                "Internal numeric IDs are clustering labels only; matching must use "
                "prototype poses, never bare filename numbers across phases."
            ),
        }
    return {
        "source": str(source_dir),
        "prototypes": prototypes,
        "created_for": "mvp_branch_mapping_diagnostics",
    }


def audit_registration_counts(source_dir: Path) -> dict[str, Any]:
    rows = list(
        csv.DictReader((source_dir / "reference_cluster_frames.csv").open(encoding="utf-8"))
    )
    samples = list(
        csv.DictReader((source_dir / "registration_samples.csv").open(encoding="utf-8"))
    )
    summary = json.loads((source_dir / "registration_summary.json").read_text(encoding="utf-8"))
    cluster_sizes_sorted = list(
        ((summary.get("branch_info") or {}).get("pose_clusters") or {}).get("cluster_sizes")
        or []
    )
    frames_by_label = Counter(
        int(r["pose_cluster_id"])
        for r in rows
        if str(r.get("pose_cluster_id", "")).strip() != ""
    )
    candidates_by_label = Counter(
        int(r["pose_cluster_id"])
        for r in samples
        if str(r.get("pose_cluster_id", "")).strip() != ""
    )
    joint_by_label = Counter()
    for row in rows:
        if str(row.get("pose_cluster_id", "")).strip() == "":
            continue
        if not _as_bool(row.get("pnp_operational_valid")):
            continue
        if int(row.get("valid_keypoints") or 0) < 4:
            continue
        if not (
            str(row.get("T_camera_object_filtered_json") or "").strip()
            or str(row.get("T_camera_object_json") or "").strip()
        ):
            continue
        joint_by_label[int(row["pose_cluster_id"])] += 1

    sample_frame_ids = [r["frame_idx"] for r in samples]
    duplicate_frame_ids = [
        frame_id for frame_id, count in Counter(sample_frame_ids).items() if count > 1
    ]

    mistaken_mapping = {
        "reported_branch_0_frames_from_sorted_sizes": (
            cluster_sizes_sorted[0] if cluster_sizes_sorted else None
        ),
        "reported_branch_1_frames_from_sorted_sizes": (
            cluster_sizes_sorted[1] if len(cluster_sizes_sorted) > 1 else None
        ),
        "actual_branch_0_frames": frames_by_label.get(0),
        "actual_branch_1_frames": frames_by_label.get(1),
        "sorted_sizes_equal_label_order": (
            cluster_sizes_sorted == [frames_by_label.get(0), frames_by_label.get(1)]
            if len(cluster_sizes_sorted) == 2
            else None
        ),
    }

    per_branch = {}
    for branch_id in sorted(set(frames_by_label) | set(candidates_by_label)):
        frames = int(frames_by_label.get(branch_id, 0))
        joint = int(joint_by_label.get(branch_id, 0))
        candidates = int(candidates_by_label.get(branch_id, 0))
        per_branch[str(branch_id)] = {
            "registration_cluster_frames": frames,
            "joint_valid_candidate_frames": joint,
            "registration_sample_rows": candidates,
            "candidate_equals_joint_valid": candidates == joint,
            "candidates_le_cluster_frames": candidates <= frames,
        }

    if duplicate_frame_ids:
        verdict = "B"
        reason = "Duplicate frame_idx found in registration samples."
    elif any(not item["candidates_le_cluster_frames"] for item in per_branch.values()):
        if mistaken_mapping["sorted_sizes_equal_label_order"] is False and all(
            item["candidate_equals_joint_valid"] and item["candidates_le_cluster_frames"]
            for item in per_branch.values()
        ):
            verdict = "D"
            reason = (
                "Report mapped sorted cluster_sizes onto labels 0/1. Actual label 1 has "
                f"{frames_by_label.get(1)} frames and {candidates_by_label.get(1)} candidates."
            )
        else:
            verdict = "C"
            reason = "Candidate count exceeds cluster frames after assignment."
    elif all(item["candidate_equals_joint_valid"] for item in per_branch.values()):
        if mistaken_mapping["sorted_sizes_equal_label_order"] is False:
            verdict = "D"
            reason = (
                "Counts are internally consistent (candidates == joint-valid subset of "
                "cluster frames). The apparent 118 vs 152 paradox came from connecting "
                "sorted cluster_sizes[0/1] to branch labels 0/1. "
                f"Actual: label0={frames_by_label.get(0)} frames/"
                f"{candidates_by_label.get(0)} candidates; "
                f"label1={frames_by_label.get(1)} frames/"
                f"{candidates_by_label.get(1)} candidates."
            )
        else:
            verdict = "A"
            reason = (
                "Cluster frame counts and candidate counts are different populations "
                "(all AprilTag frames vs joint-valid OA frames)."
            )
    else:
        verdict = "C"
        reason = "Candidate counts do not match reconstructed joint-valid population."

    # Secondary population note: even with correct labels, candidates < frames (A).
    secondary = None
    if verdict == "D" and all(
        item["candidate_equals_joint_valid"] and item["candidates_le_cluster_frames"]
        for item in per_branch.values()
    ):
        secondary = "A"
        reason += (
            " Secondary: A also applies — candidates are the joint-valid subset, "
            "not equal to cluster frame counts."
        )

    return {
        "verdict": verdict,
        "secondary_verdict": secondary,
        "reason": reason,
        "cluster_sizes_sorted_desc": cluster_sizes_sorted,
        "mistaken_label_size_mapping": mistaken_mapping,
        "frames_by_label": {str(k): int(v) for k, v in sorted(frames_by_label.items())},
        "joint_valid_by_label": {str(k): int(v) for k, v in sorted(joint_by_label.items())},
        "candidates_by_label": {
            str(k): int(v) for k, v in sorted(candidates_by_label.items())
        },
        "per_branch": per_branch,
        "duplicate_sample_frame_ids": duplicate_frame_ids,
        "duplicate_append_detected": bool(duplicate_frame_ids),
        "residual_recompute_needed": bool(duplicate_frame_ids),
    }


def analyze_runner_branch_flow() -> dict[str, Any]:
    return {
        "registration_branch_ids_created_at": (
            "assign_april_tag_branches() after registration capture, clustering "
            "T_world_camera_tag over the 300 registration frames."
        ),
        "preflight_branch_ids_created_at": (
            "assign_branch_id() inside _capture_session_frames during preflight, "
            "using provided branch_centroids (not a fresh clustering of preflight)."
        ),
        "preflight_independent_clustering": False,
        "preflight_used_numeric_ids_against_calibration_files": True,
        "critical_bug_in_aborted_run": (
            "active_centroids kept ONLY accepted branches (internal id 1). "
            "Rejected registration branch 0 prototype was removed from matching, so "
            "poses on that physical branch became apriltag_branch_unassigned "
            "(branch_id=None), not branch_calibration_missing. "
            "The live report's 'branch_calibration_missing' explanation was therefore "
            "likely a misattribution; with accepted-only centroids that exclude reason "
            "cannot fire for the rejected branch."
        ),
        "label_swap_across_independent_clustering": (
            "Not applicable for preflight in this runner version: preflight did not "
            "re-cluster. Numeric IDs remain brittle if future code re-clusters, so "
            "prototype-pose matching is required."
        ),
        "preflight_frame_poses_persisted": False,
        "preflight_frame_poses_missing_reason": (
            "On preflight gate failure the runner wrote only preflight_summary.json "
            "and discarded in-memory preflight rows. No AprilTag pose CSV exists for "
            "the 20 preflight frames."
        ),
    }


def simulate_old_vs_new_on_registration(
    source_dir: Path,
    prototypes_payload: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Proxy analysis: registration frames under old vs fixed matching rules."""
    rows = list(
        csv.DictReader((source_dir / "reference_cluster_frames.csv").open(encoding="utf-8"))
    )
    summary = json.loads((source_dir / "registration_summary.json").read_text(encoding="utf-8"))
    proto_mats = {
        int(branch_id): np.asarray(meta["T_world_camera_tag_prototype"], dtype=np.float64)
        for branch_id, meta in (prototypes_payload.get("prototypes") or {}).items()
    }
    accepted = {
        int(branch_id)
        for branch_id, meta in (summary.get("branches") or {}).items()
        if meta.get("accepted")
    }
    accepted_only = {bid: proto_mats[bid] for bid in accepted if bid in proto_mats}
    t_thresh = float(settings["cluster_translation_threshold_m"])
    r_thresh = float(settings["cluster_rotation_threshold_deg"])

    old_reasons: Counter[str] = Counter()
    new_reasons: Counter[str] = Counter()
    remap_confusion: Counter[str] = Counter()
    identity_ok = 0
    identity_total = 0

    for row in rows:
        if str(row.get("pose_cluster_id", "")).strip() == "":
            continue
        true_id = int(row["pose_cluster_id"])
        T_world = _matrix(row.get("T_world_camera_tag_json"))
        if T_world is None:
            continue
        identity_total += 1
        full_match = match_pose_to_prototypes(
            T_world,
            proto_mats,
            translation_threshold_m=t_thresh,
            rotation_threshold_deg=r_thresh,
        )
        nearest = full_match.get("branch_id")
        if nearest == true_id:
            identity_ok += 1
        remap_confusion[f"true_{true_id}->nearest_{nearest}"] += 1

        # OLD live logic: match only accepted centroids.
        old_match = match_pose_to_prototypes(
            T_world,
            accepted_only,
            translation_threshold_m=t_thresh,
            rotation_threshold_deg=r_thresh,
        )
        if old_match.get("branch_id") is None:
            old_reasons["apriltag_branch_unassigned_or_unknown"] += 1
        elif old_match["branch_id"] not in accepted:
            old_reasons[f"branch_calibration_missing:{old_match['branch_id']}"] += 1
        else:
            old_reasons["would_attempt_compare"] += 1

        # NEW logic: match all prototypes; exclude if no calibration.
        if nearest is None:
            new_reasons[f"unknown:{full_match.get('status')}"] += 1
        elif nearest not in accepted:
            new_reasons[f"branch_calibration_missing:{nearest}"] += 1
        else:
            new_reasons["would_attempt_compare"] += 1

    return {
        "note": (
            "Proxy on registration frames only. Real preflight poses were not saved; "
            "this estimates how often accepted-only matching hides the rejected branch."
        ),
        "prototype_identity_recovery_rate": (
            identity_ok / identity_total if identity_total else None
        ),
        "identity_ok": identity_ok,
        "identity_total": identity_total,
        "true_to_nearest_confusion": dict(remap_confusion),
        "old_accepted_only_exclude_reasons": dict(old_reasons),
        "new_all_prototypes_exclude_reasons": dict(new_reasons),
        "implication_for_preflight_5_of_20": (
            "Under old accepted-only matching, frames on rejected physical branch "
            "cannot become branch_calibration_missing; they become unassigned. "
            "5/20 comparable is consistent with majority of preflight poses landing "
            "on the rejected physical branch (no usable calibration), not with a "
            "label-swap that would rescue them by remapping to branch 1."
        ),
    }


def rematch_unavailable_preflight(
    prototypes_payload: dict[str, Any],
    audit: dict[str, Any],
    flow: dict[str, Any],
    proxy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "preflight_frames_available": False,
        "remapping_executed": False,
        "original_comparable_count": 5,
        "original_total": 20,
        "remapped_comparable_count": None,
        "gate_15_of_20": None,
        "existing_preflight_branch0_rematched_to_registration_branch1": None,
        "false_branch_calibration_missing_count": None,
        "likely_true_exclude_for_rejected_branch_under_old_runner": (
            "apriltag_branch_unassigned (not branch_calibration_missing)"
        ),
        "camera_rerun_verdict": "B",
        "camera_rerun_reason": (
            "Cannot prove a label-swap rescue offline because preflight frame poses "
            "were not saved. Live runner matched only the accepted branch-1 prototype; "
            "5/20 comparable implies most frames were outside that prototype "
            "(physically the rejected branch). Remapping cannot turn rejected-branch "
            "poses into comparable frames without a calibration for that prototype. "
            "Fix runner persistence/prototype matching before any rerun; "
            "do not rerun immediately."
        ),
        "expected_physical_interpretation": {
            "registration_accepted_internal_id": 1,
            "registration_rejected_internal_id": 0,
            "actual_frames_label_0": audit["frames_by_label"].get("0"),
            "actual_frames_label_1": audit["frames_by_label"].get("1"),
            "note": (
                "Sorted cluster_sizes were [182,118], but labels are "
                "0→118 frames and 1→182 frames."
            ),
        },
        "registration_proxy_simulation": proxy,
        "runner_flow": flow,
        "prototypes_available": sorted((prototypes_payload.get("prototypes") or {}).keys()),
    }


def production_hashes() -> dict[str, str]:
    paths = [
        ROOT / "run.ps1",
        ROOT / "src/apriltag_world.py",
        ROOT / "src/object_anchor_pose.py",
        ROOT / "src/object_anchor_runtime.py",
    ]
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def run(source_dir: Path, output_root: Path) -> dict[str, Any]:
    config = yaml.safe_load((source_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    settings = load_mvp_settings(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    prototypes = build_registration_prototypes(source_dir, settings)
    # Local artifact under the aborted run's registration/ (not production calibration).
    _write_json(source_dir / "registration" / "branch_prototypes.json", prototypes)
    _write_json(output_dir / "registration_branch_prototypes.json", prototypes)

    audit = audit_registration_counts(source_dir)
    _write_json(output_dir / "registration_count_audit.json", audit)

    flow = analyze_runner_branch_flow()
    proxy = simulate_old_vs_new_on_registration(source_dir, prototypes, settings)
    rematch = rematch_unavailable_preflight(prototypes, audit, flow, proxy)
    _write_json(output_dir / "preflight_original_vs_remapped_summary.json", rematch)

    fields = [
        "frame_idx",
        "original_preflight_branch_id",
        "nearest_registration_branch_id",
        "translation_distance_to_branch_0_m",
        "rotation_distance_to_branch_0_deg",
        "translation_distance_to_branch_1_m",
        "rotation_distance_to_branch_1_deg",
        "calibration_exists_for_nearest",
        "object_anchor_filtered_pose_present",
        "pose_comparable_remapped",
        "original_exclude_reason",
        "remapped_exclude_reason",
        "note",
    ]
    _write_csv(
        output_dir / "preflight_branch_remapping.csv",
        [
            {
                "frame_idx": "",
                "original_preflight_branch_id": "",
                "nearest_registration_branch_id": "",
                "translation_distance_to_branch_0_m": "",
                "rotation_distance_to_branch_0_deg": "",
                "translation_distance_to_branch_1_m": "",
                "rotation_distance_to_branch_1_deg": "",
                "calibration_exists_for_nearest": "",
                "object_anchor_filtered_pose_present": "",
                "pose_comparable_remapped": "",
                "original_exclude_reason": "",
                "remapped_exclude_reason": "preflight_frame_poses_not_persisted",
                "note": (
                    "No per-frame preflight AprilTag poses were saved in "
                    f"{source_dir.as_posix()}"
                ),
            }
        ],
        fields,
    )

    summary = {
        "source": str(source_dir),
        "output": str(output_dir),
        "registration_independent_clustering": True,
        "preflight_independent_clustering": False,
        "branch_id_swap_possible_via_independent_clustering": False,
        "branch_id_numeric_brittleness": True,
        "count_audit_verdict": audit["verdict"],
        "count_audit_secondary_verdict": audit.get("secondary_verdict"),
        "count_audit_reason": audit["reason"],
        "actual_frames_by_label": audit["frames_by_label"],
        "actual_candidates_by_label": audit["candidates_by_label"],
        "preflight_poses_available": False,
        "remapped_comparable_frames": None,
        "preflight_gate_15_of_20_after_remap": None,
        "camera_rerun_verdict": rematch["camera_rerun_verdict"],
        "registration_proxy_identity_recovery_rate": proxy.get(
            "prototype_identity_recovery_rate"
        ),
        "runner_fix_required": [
            "Match preflight/final frames to ALL registration prototypes by pose",
            "Use calibration only if that prototype's branch was accepted",
            "Persist branch_prototypes.json",
            "Persist preflight_frames.csv even on gate failure",
        ],
        "production_hashes": production_hashes(),
        "camera_executed": False,
    }
    _write_json(output_dir / "diagnostic_summary.json", summary)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# MVP branch mapping diagnostics",
                "",
                f"- Source: `{source_dir.as_posix()}`",
                f"- Count audit verdict: `{audit['verdict']}`"
                + (
                    f" (secondary `{audit.get('secondary_verdict')}`)"
                    if audit.get("secondary_verdict")
                    else ""
                ),
                f"- {audit['reason']}",
                f"- Actual frames by label: `{audit['frames_by_label']}`",
                f"- Actual candidates by label: `{audit['candidates_by_label']}`",
                "- Preflight did **not** re-cluster; it matched against "
                "`active_centroids` = accepted branches only.",
                "- Preflight frame poses were not persisted; remapping CSV is schema-only.",
                f"- Registration proxy prototype identity recovery: "
                f"`{proxy.get('prototype_identity_recovery_rate')}`",
                "- Camera was not opened.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    summary = run(Path(args.source), Path(args.output_root))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
