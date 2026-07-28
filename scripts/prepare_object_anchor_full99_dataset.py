#!/usr/bin/env python3
"""Build the block-split Full99 training dataset from protected source data."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
POSITIVE_ROOT = ROOT / "data" / "datasets" / "tissue_box_front_orbbec_valid99_source"
NEGATIVE_ROOT = ROOT / "data" / "object_anchor_capture" / "negative"
LEGACY_ROOT = ROOT / "data" / "tissue_box_front_only_pose"
DATASET_ROOT = ROOT / "data" / "datasets" / "tissue_box_front_orbbec_full99"
OUTPUT_ROOT = ROOT / "out" / "object_anchor_full99"
VAL_BLOCKS = {3, 9}  # Sequence 021-030 and 081-090.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse sequence number: {path.name}") from exc


def _validate_positive(path: Path) -> None:
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one non-empty row")
    values = np.asarray([float(item) for item in rows[0].split()], dtype=np.float64)
    if values.size != 17 or values[0] != 0:
        raise ValueError(f"{path}: expected class 0 and 17 values")
    points = values[5:].reshape(4, 3)
    if np.any((points[:, :2] < 0.0) | (points[:, :2] > 1.0)):
        raise ValueError(f"{path}: keypoint outside normalized range")
    if np.any(~np.isin(points[:, 2], [0.0, 1.0, 2.0])):
        raise ValueError(f"{path}: invalid visibility")

    xy = points[:, :2]

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        return (
            orientation(a, b, c) * orientation(a, b, d) < 0
            and orientation(c, d, a) * orientation(c, d, b) < 0
        )

    if crosses(xy[0], xy[1], xy[2], xy[3]) or crosses(xy[1], xy[2], xy[3], xy[0]):
        raise ValueError(f"{path}: polygon crossing")
    area = 0.5 * sum(
        xy[index, 0] * xy[(index + 1) % 4, 1]
        - xy[(index + 1) % 4, 0] * xy[index, 1]
        for index in range(4)
    )
    if area <= 0:
        raise ValueError(f"{path}: non-canonical polygon direction")


def _copy_record(
    *,
    image: Path,
    label: Path,
    source_type: str,
    source_dataset: str,
    capture_session: str,
    temporal_block: str,
    split: str,
    is_positive: bool,
    records: list[dict[str, str]],
) -> None:
    image_destination = DATASET_ROOT / "images" / split / image.name
    label_destination = DATASET_ROOT / "labels" / split / label.name
    if image_destination.exists() or label_destination.exists():
        raise FileExistsError(f"destination collision: {image.stem}")
    shutil.copy2(image, image_destination)
    shutil.copy2(label, label_destination)
    image_hash = _sha256(image)
    label_hash = _sha256(label)
    if _sha256(image_destination) != image_hash or _sha256(label_destination) != label_hash:
        raise OSError(f"copy hash mismatch: {image.name}")
    records.append(
        {
            "filename": image.name,
            "source_type": source_type,
            "source_dataset": source_dataset,
            "capture_session": capture_session,
            "temporal_block": temporal_block,
            "split": split,
            "is_positive": str(is_positive).lower(),
            "has_label": str(bool(label.read_text(encoding="utf-8").strip())).lower(),
            "image_sha256": image_hash,
            "label_sha256": label_hash,
            "image_path": image_destination.relative_to(ROOT).as_posix(),
            "label_path": label_destination.relative_to(ROOT).as_posix(),
            "capture_type": "positive" if is_positive else "negative",
            "source": source_type,
        }
    )


def main() -> None:
    if DATASET_ROOT.exists():
        raise SystemExit(f"refusing to overwrite dataset: {DATASET_ROOT}")
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"refusing to overwrite output: {OUTPUT_ROOT}")

    positive_images = sorted((POSITIVE_ROOT / "images" / "all").glob("*.jpg"), key=_sequence)
    positive_labels = {path.stem: path for path in (POSITIVE_ROOT / "labels" / "all").glob("*.txt")}
    negative_images = sorted((NEGATIVE_ROOT / "images").glob("*.jpg"), key=_sequence)
    negative_labels = {path.stem: path for path in (NEGATIVE_ROOT / "labels").glob("*.txt")}
    if len(positive_images) != 99 or len(positive_labels) != 99:
        raise ValueError("valid99 source must contain 99 image/label pairs")
    if len(negative_images) != 100 or len(negative_labels) != 100:
        raise ValueError("negative source must contain 100 image/label pairs")

    for image in positive_images:
        label = positive_labels.get(image.stem)
        if label is None:
            raise ValueError(f"missing positive label: {image.name}")
        _validate_positive(label)
    for image in negative_images:
        label = negative_labels.get(image.stem)
        if label is None:
            raise ValueError(f"missing negative label: {image.name}")
        if label.read_bytes():
            raise ValueError(f"negative label is not zero-byte: {label}")

    for split in ("train", "val", "legacy_test"):
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=False)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "dataset_validation").mkdir(parents=True, exist_ok=False)

    records: list[dict[str, str]] = []
    for image in positive_images:
        sequence = _sequence(image)
        block = (sequence - 1) // 10 + 1
        split = "val" if block in VAL_BLOCKS else "train"
        _copy_record(
            image=image,
            label=positive_labels[image.stem],
            source_type="orbbec_positive",
            source_dataset="tissue_box_front_orbbec_valid99_source",
            capture_session="orbbec_20260724",
            temporal_block=f"positive_{block:02d}_{(block - 1) * 10 + 1:03d}-{block * 10:03d}",
            split=split,
            is_positive=True,
            records=records,
        )
    for image in negative_images:
        sequence = _sequence(image)
        block = (sequence - 1) // 10 + 1
        split = "val" if block in VAL_BLOCKS else "train"
        _copy_record(
            image=image,
            label=negative_labels[image.stem],
            source_type="orbbec_negative",
            source_dataset="object_anchor_capture_negative",
            capture_session="orbbec_20260724",
            temporal_block=f"negative_{block:02d}_{(block - 1) * 10 + 1:03d}-{block * 10:03d}",
            split=split,
            is_positive=False,
            records=records,
        )

    legacy_manifest = json.loads(
        (LEGACY_ROOT / "split_manifest.json").read_text(encoding="utf-8-sig")
    )
    for source_split, destination_split in (("train", "train"), ("val", "legacy_test")):
        expected_stems = set(legacy_manifest[source_split])
        images = {path.stem: path for path in (LEGACY_ROOT / "images" / source_split).glob("*.jpg")}
        labels = {path.stem: path for path in (LEGACY_ROOT / "labels" / source_split).glob("*.txt")}
        if set(images) != expected_stems or set(labels) != expected_stems:
            raise ValueError(f"legacy {source_split} differs from its protected split manifest")
        for stem in sorted(expected_stems):
            _validate_positive(labels[stem])
            _copy_record(
                image=images[stem],
                label=labels[stem],
                source_type="legacy_positive",
                source_dataset="tissue_box_front_only_pose",
                capture_session="legacy_white_background",
                temporal_block=f"legacy_{source_split}",
                split=destination_split,
                is_positive=True,
                records=records,
            )

    dataset_yaml = {
        "path": "data/datasets/tissue_box_front_orbbec_full99",
        "train": "images/train",
        "val": "images/val",
        "test": "images/legacy_test",
        "kpt_shape": [4, 3],
        "flip_idx": [0, 1, 2, 3],
        "names": {0: "tissue_box"},
    }
    (DATASET_ROOT / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = DATASET_ROOT / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    shutil.copy2(manifest_path, OUTPUT_ROOT / "split_manifest.csv")

    split_counts = Counter((record["split"], record["source_type"]) for record in records)
    stem_splits: defaultdict[str, set[str]] = defaultdict(set)
    hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        stem_splits[Path(record["filename"]).stem].add(record["split"])
        hash_splits[record["image_sha256"]].add(record["split"])
    cross_split_stems = {stem: sorted(splits) for stem, splits in stem_splits.items() if len(splits) > 1}
    cross_split_hashes = {digest: sorted(splits) for digest, splits in hash_splits.items() if len(splits) > 1}
    if cross_split_stems or cross_split_hashes:
        raise ValueError(
            f"cross-split leakage: stems={cross_split_stems}, hashes={cross_split_hashes}"
        )

    validation: dict[str, object] = {
        "split_counts": {
            f"{split}:{source}": count for (split, source), count in sorted(split_counts.items())
        },
        "total": len(records),
        "cross_split_stems": cross_split_stems,
        "cross_split_image_hashes": cross_split_hashes,
        "val_blocks": sorted(VAL_BLOCKS),
        "val_is_same_capture_session": True,
        "dataset_yaml": dataset_yaml,
    }
    expected_counts = {
        ("train", "orbbec_positive"): 79,
        ("train", "orbbec_negative"): 80,
        ("train", "legacy_positive"): 18,
        ("val", "orbbec_positive"): 20,
        ("val", "orbbec_negative"): 20,
        ("legacy_test", "legacy_positive"): 6,
    }
    if split_counts != Counter(expected_counts):
        raise ValueError(f"unexpected split counts: {split_counts}")
    (OUTPUT_ROOT / "dataset_validation" / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
