#!/usr/bin/env python3
"""Verify Full99 copies and protected model inputs after the experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "datasets" / "tissue_box_front_orbbec_full99"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_paths(row: dict[str, str]) -> tuple[Path, Path]:
    stem = Path(row["filename"]).stem
    if row["source_type"] == "orbbec_positive":
        base = ROOT / "data" / "datasets" / "tissue_box_front_orbbec_valid99_source"
        return base / "images" / "all" / row["filename"], base / "labels" / "all" / f"{stem}.txt"
    if row["source_type"] == "orbbec_negative":
        base = ROOT / "data" / "object_anchor_capture" / "negative"
        return base / "images" / row["filename"], base / "labels" / f"{stem}.txt"
    source_split = "val" if row["split"] == "legacy_test" else "train"
    base = ROOT / "data" / "tissue_box_front_only_pose"
    return (
        base / "images" / source_split / row["filename"],
        base / "labels" / source_split / f"{stem}.txt",
    )


def main() -> None:
    with (DATASET / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mismatches: list[str] = []
    decode_failures: list[str] = []
    label_errors: list[str] = []
    split_counts: Counter[str] = Counter()
    stem_splits: defaultdict[str, set[str]] = defaultdict(set)
    hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_image, source_label = _source_paths(row)
        copied_image = ROOT / row["image_path"]
        copied_label = ROOT / row["label_path"]
        expected = (
            (source_image, copied_image, row["image_sha256"]),
            (source_label, copied_label, row["label_sha256"]),
        )
        for source, copied, recorded_hash in expected:
            if not source.is_file() or not copied.is_file():
                mismatches.append(f"missing:{source}:{copied}")
                continue
            if _sha256(source) != recorded_hash or _sha256(copied) != recorded_hash:
                mismatches.append(f"hash:{source}:{copied}")
        split_counts[row["split"]] += 1
        stem_splits[Path(row["filename"]).stem].add(row["split"])
        hash_splits[row["image_sha256"]].add(row["split"])
        if cv2.imread(str(copied_image), cv2.IMREAD_COLOR) is None:
            decode_failures.append(str(copied_image))
        label_text = copied_label.read_text(encoding="utf-8").strip()
        if row["is_positive"] == "false":
            if copied_label.stat().st_size != 0:
                label_errors.append(f"nonempty_negative:{copied_label}")
            continue
        values = label_text.split()
        if len(values) != 17 or values[0] != "0":
            label_errors.append(f"positive_format:{copied_label}")
            continue
        points = np.asarray([float(value) for value in values[5:]], dtype=np.float64).reshape(4, 3)

        def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return float(
                (b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0])
            )

        def crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
            return (
                orientation(a, b, c) * orientation(a, b, d) < 0
                and orientation(c, d, a) * orientation(c, d, b) < 0
            )

        xy = points[:, :2]
        if crosses(xy[0], xy[1], xy[2], xy[3]) or crosses(
            xy[1], xy[2], xy[3], xy[0]
        ):
            label_errors.append(f"polygon_crossing:{copied_label}")

    cross_split_stems = {
        stem: sorted(splits) for stem, splits in stem_splits.items() if len(splits) > 1
    }
    cross_split_hashes = {
        digest: sorted(splits) for digest, splits in hash_splits.items() if len(splits) > 1
    }

    baseline = (
        ROOT / "models" / "object_anchor" / "tissue_box_01_front_only" / "best.pt"
    )
    pilot = (
        ROOT
        / "models"
        / "object_anchor"
        / "tissue_box_01_front_only_orbbec_pilot"
        / "best.pt"
    )
    pilot_run = (
        ROOT
        / "runs"
        / "object_anchor_pose"
        / "tissue_box_01_front_only_orbbec_pilot"
        / "weights"
        / "best.pt"
    )
    full99 = (
        ROOT
        / "models"
        / "object_anchor"
        / "tissue_box_01_front_only_orbbec_full99"
        / "best.pt"
    )
    full99_run = (
        ROOT
        / "runs"
        / "object_anchor_pose"
        / "tissue_box_01_front_only_orbbec_full99"
        / "weights"
        / "best.pt"
    )
    hashes = {
        "baseline": _sha256(baseline),
        "pilot": _sha256(pilot),
        "pilot_training_best": _sha256(pilot_run),
        "full99": _sha256(full99),
        "full99_training_best": _sha256(full99_run),
        "production_config": _sha256(ROOT / "configs" / "orbbec_gemini.yaml"),
        "pilot_experiment_config": _sha256(
            ROOT / "configs" / "experiments" / "orbbec_gemini_object_anchor_pilot.yaml"
        ),
    }
    checks = {
        "all_source_and_copy_hashes_match": not mismatches,
        "baseline_matches_pre_experiment_fingerprint": (
            hashes["baseline"].startswith("d37fd4")
            and hashes["baseline"].endswith("f6db4")
        ),
        "pilot_model_matches_original_training_artifact": (
            hashes["pilot"] == hashes["pilot_training_best"]
        ),
        "full99_model_matches_new_training_artifact": (
            hashes["full99"] == hashes["full99_training_best"]
        ),
        "all_dataset_images_decode": not decode_failures,
        "all_dataset_labels_valid": not label_errors,
        "no_cross_split_stems": not cross_split_stems,
        "no_cross_split_image_hashes": not cross_split_hashes,
        "split_totals_match": split_counts
        == Counter({"train": 177, "val": 40, "legacy_test": 6}),
    }
    report = {
        "checks": checks,
        "mismatches": mismatches,
        "decode_failures": decode_failures,
        "label_errors": label_errors,
        "cross_split_stems": cross_split_stems,
        "cross_split_image_hashes": cross_split_hashes,
        "split_totals": dict(split_counts),
        "verified_manifest_records": len(rows),
        "hashes": hashes,
    }
    destination = (
        ROOT
        / "out"
        / "object_anchor_full99"
        / "dataset_validation"
        / "protection_verification.json"
    )
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not all(checks.values()):
        raise SystemExit(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
