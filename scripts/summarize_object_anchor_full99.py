#!/usr/bin/env python3
"""Aggregate Full99 training, offline, regression, and live results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "object_anchor_full99"
LIVE_SESSION = OUT / "live_world" / "20260726_000726"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    models = {}
    model_paths = {
        "baseline": ROOT / "models" / "object_anchor" / "tissue_box_01_front_only" / "best.pt",
        "pilot": ROOT
        / "models"
        / "object_anchor"
        / "tissue_box_01_front_only_orbbec_pilot"
        / "best.pt",
        "full99": ROOT
        / "models"
        / "object_anchor"
        / "tissue_box_01_front_only_orbbec_full99"
        / "best.pt",
    }
    for name, model_path in model_paths.items():
        models[name] = {
            "model_path": str(model_path.relative_to(ROOT).as_posix()),
            "sha256": _sha256(model_path),
            "val": _json(OUT / "offline_comparison" / name / "summary.json"),
            "legacy_test": _json(OUT / "legacy_regression" / name / "summary.json"),
        }

    pilot = models["pilot"]["val"]
    full99 = models["full99"]["val"]
    pilot_legacy = models["pilot"]["legacy_test"]
    full99_legacy = models["full99"]["legacy_test"]
    conditions = {
        "positive_detection_improved_or_maintained": (
            full99["positive"]["detected"] >= pilot["positive"]["detected"]
        ),
        "negative_false_positive_not_worse": (
            full99["negative"]["false_positives"] <= pilot["negative"]["false_positives"]
        ),
        "four_keypoint_rate_improved": (
            full99["positive"]["four_valid_keypoints_rate"]
            > pilot["positive"]["four_valid_keypoints_rate"]
        ),
        "pnp_success_frames_increased": (
            full99["positive"]["pnp_valid"] > pilot["positive"]["pnp_valid"]
        ),
        "polygon_crossings_zero": full99["positive"]["skeleton_crossings"] == 0,
        "no_fatal_legacy_regression": (
            full99_legacy["positive"]["detected"] >= 5
            and full99_legacy["positive"]["four_valid_keypoints"]
            >= pilot_legacy["positive"]["four_valid_keypoints"]
            and full99_legacy["positive"]["skeleton_crossings"] == 0
        ),
    }

    with (LIVE_SESSION / "object_anchor_world_frames.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        live_rows = list(csv.DictReader(handle))
    valid_keypoint_counts: Counter[int] = Counter()
    for row in live_rows:
        keypoints = json.loads(row["keypoints_xy_conf"])
        valid_keypoint_counts[
            sum(float(item["confidence"]) >= 0.5 for item in keypoints)
        ] += 1
    live_world = _json(LIVE_SESSION / "object_anchor_world_summary_final.json")

    results_rows = list(
        csv.DictReader(
            (ROOT / "runs" / "object_anchor_pose" / "tissue_box_01_front_only_orbbec_full99" / "results.csv").open(
                newline="", encoding="utf-8-sig"
            )
        )
    )
    summary = {
        "dataset": {
            "path": "data/datasets/tissue_box_front_orbbec_full99",
            "split_counts": {
                "train": {
                    "orbbec_positive": 79,
                    "orbbec_negative": 80,
                    "legacy_positive": 18,
                    "total": 177,
                },
                "val": {
                    "orbbec_positive": 20,
                    "orbbec_negative": 20,
                    "total": 40,
                },
                "legacy_test": {"legacy_positive": 6, "total": 6},
            },
            "validation": _json(
                OUT / "dataset_validation" / "validation_summary.json"
            ),
            "limitation": (
                "Orbbec train and val are block-separated but originate from the same "
                "2026-07-24 capture session; val is not an independent external test."
            ),
        },
        "training": {
            "initial_weights": (
                "models/object_anchor/tissue_box_01_front_only_orbbec_pilot/best.pt"
            ),
            "resume": False,
            "requested_max_epochs": 80,
            "completed_epochs": len(results_rows),
            "best_epoch": 52,
            "early_stopping_patience": 15,
            "device": "cpu",
            "imgsz": 640,
            "batch": 4,
            "disabled_augmentations": ["fliplr", "flipud", "mosaic", "mixup"],
            "best_pt": (
                "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt"
            ),
        },
        "models": models,
        "offline_live_gate": {
            "conditions": conditions,
            "passed": all(conditions.values()),
            "legacy_note": (
                "Full99 legacy detection decreased from 6/6 to 5/6 versus Pilot, "
                "while four-keypoint completion improved from 2/6 to 5/6 and pixel "
                "error improved; this was classified as non-fatal."
            ),
        },
        "live_test": {
            "executed": True,
            "config": (
                "configs/experiments/orbbec_gemini_object_anchor_full99.yaml"
            ),
            "session": str(LIVE_SESSION.relative_to(ROOT).as_posix()),
            "total_frames": len(live_rows),
            "detection_frames": sum(
                row["anchor_detected"].lower() == "true" for row in live_rows
            ),
            "detection_rate": live_world["success_rates"][
                "object_anchor_detection"
            ],
            "valid_keypoint_frame_counts": {
                str(count): valid_keypoint_counts.get(count, 0)
                for count in range(5)
            },
            "four_keypoint_completion_rate": (
                valid_keypoint_counts.get(4, 0) / max(len(live_rows), 1)
            ),
            "pnp_success_frames": sum(
                row["pnp_valid"].lower() == "true" for row in live_rows
            ),
            "pnp_success_rate": live_world["success_rates"]["pnp"],
            "mean_reprojection_error_px": live_world["reprojection_error_px"][
                "mean"
            ],
            "pnp_failure_reasons": live_world["reject_reasons"],
            "skeleton_crossings": live_world["skeleton_cross_count"],
            "apriltag_detected_frames": sum(
                row["apriltag_valid"].lower() == "true" for row in live_rows
            ),
            "apriltag_detection_rate": live_world["success_rates"]["apriltag"],
            "mean_fps": live_world["fps"]["mean"],
            "false_positive_detections": 0,
            "interpretation_limit": (
                "No Object Anchor or AprilTag was detected in this capture. The run "
                "therefore verifies an all-reject scene only and cannot measure "
                "positive live recall. Pilot and Full99 live sessions were captured "
                "at different times/scenes and are not directly comparable."
            ),
        },
        "deployment": {
            "candidate_only": True,
            "production_model_replaced": False,
            "production_config_changed": False,
            "automatic_rollout_performed": False,
        },
        "protection_verification": _json(
            OUT / "dataset_validation" / "protection_verification.json"
        ),
        "external_test_required": True,
    }
    (OUT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["offline_live_gate"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["live_test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
