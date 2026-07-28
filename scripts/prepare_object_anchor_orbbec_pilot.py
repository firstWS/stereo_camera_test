#!/usr/bin/env python3
"""Build the isolated FRONT-only Orbbec pilot dataset without changing sources."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / "data" / "tissue_box_front_only_pose"
POSITIVE_ROOT = ROOT / "data" / "cvat_exports" / "tissue_box_front_positive_v2_dataset"
NEGATIVE_ROOT = ROOT / "data" / "object_anchor_capture" / "negative"
PILOT_ROOT = ROOT / "data" / "datasets" / "tissue_box_front_orbbec_pilot"
OVERLAY_ROOT = ROOT / "out" / "object_anchor_pilot" / "label_overlays"

POSITIVE_VAL_INDICES = (1, 31, 66, 96)
NEGATIVE_SELECTED_INDICES = tuple(range(1, 97, 5))
NEGATIVE_VAL_INDICES = (1, 31, 66, 96)
SKELETON = ((0, 1), (1, 2), (2, 3), (3, 0))


def _label_values(path: Path, *, allow_empty: bool) -> list[float]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        if allow_empty:
            return []
        raise ValueError(f"positive label is empty: {path}")
    rows = text.splitlines()
    if len(rows) != 1:
        raise ValueError(f"expected exactly one label row: {path}")
    values = [float(value) for value in rows[0].split()]
    if len(values) != 17:
        raise ValueError(f"expected 17 label values: {path}")
    if values[0] != 0:
        raise ValueError(f"expected class 0: {path}")
    if any(value < 0.0 or value > 1.0 for value in values[1:5]):
        raise ValueError(f"bbox outside normalized range: {path}")
    for index in range(4):
        x, y, visibility = values[5 + index * 3 : 8 + index * 3]
        if visibility not in (0.0, 1.0, 2.0):
            raise ValueError(f"invalid visibility at keypoint {index}: {path}")
        if visibility > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"visible keypoint outside normalized range: {path}")
    return values


def _crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _validate_geometry(path: Path, values: list[float]) -> None:
    points = np.asarray(values[5:], dtype=np.float64).reshape(4, 3)[:, :2]
    if _crosses(points[0], points[1], points[2], points[3]) or _crosses(
        points[1], points[2], points[3], points[0]
    ):
        raise ValueError(f"crossed keypoint polygon: {path}")
    area = 0.5 * sum(
        points[index, 0] * points[(index + 1) % 4, 1]
        - points[(index + 1) % 4, 0] * points[index, 1]
        for index in range(4)
    )
    if area <= 0:
        raise ValueError(f"keypoint order differs from canonical TL,TR,BR,BL: {path}")


def _draw_overlay(image_path: Path, label_path: Path, source: str) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    values = _label_values(label_path, allow_empty=False)
    height, width = image.shape[:2]
    cx, cy, bw, bh = values[1:5]
    x1, y1 = int(round((cx - bw / 2) * width)), int(round((cy - bh / 2) * height))
    x2, y2 = int(round((cx + bw / 2) * width)), int(round((cy + bh / 2) * height))
    points = np.asarray(values[5:], dtype=np.float64).reshape(4, 3)
    points[:, 0] *= width
    points[:, 1] *= height
    points_xy = np.rint(points[:, :2]).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
    for start, end in SKELETON:
        cv2.line(image, tuple(points_xy[start]), tuple(points_xy[end]), (0, 165, 255), 3)
    for index, point in enumerate(points_xy):
        cv2.circle(image, tuple(point), 7, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        f"{source} | {image_path.name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    destination = OVERLAY_ROOT / f"{source}__{image_path.name}"
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"failed to write overlay: {destination}")


def _paired_files(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(images_dir.glob("*.jpg"))
    labels = sorted(labels_dir.glob("*.txt"))
    image_by_stem = {path.stem: path for path in images}
    label_by_stem = {path.stem: path for path in labels}
    if set(image_by_stem) != set(label_by_stem):
        raise ValueError(
            f"image/label stem mismatch: missing labels={sorted(set(image_by_stem) - set(label_by_stem))}, "
            f"missing images={sorted(set(label_by_stem) - set(image_by_stem))}"
        )
    return [(image_by_stem[stem], label_by_stem[stem]) for stem in sorted(image_by_stem)]


def _copy_sample(
    image_path: Path,
    label_path: Path,
    *,
    source: str,
    capture_type: str,
    split: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    destination_image = PILOT_ROOT / "images" / split / image_path.name
    destination_label = PILOT_ROOT / "labels" / split / label_path.name
    if destination_image.exists() or destination_label.exists():
        raise FileExistsError(f"duplicate destination stem: {image_path.stem}")
    shutil.copy2(image_path, destination_image)
    shutil.copy2(label_path, destination_label)
    manifest_rows.append(
        {
            "filename": image_path.name,
            "source": source,
            "capture_type": capture_type,
            "split": split,
            "image_path": destination_image.relative_to(ROOT).as_posix(),
            "label_path": destination_label.relative_to(ROOT).as_posix(),
            "original_image_path": image_path.relative_to(ROOT).as_posix(),
            "original_label_path": label_path.relative_to(ROOT).as_posix(),
        }
    )


def main() -> None:
    if PILOT_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing pilot dataset: {PILOT_ROOT}")
    if OVERLAY_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing overlays: {OVERLAY_ROOT}")

    legacy_yaml = yaml.safe_load(
        (ROOT / "configs" / "datasets" / "tissue_box_front_only_pose.yaml").read_text(
            encoding="utf-8"
        )
    )
    cvat_yaml = yaml.safe_load((POSITIVE_ROOT / "data.yaml").read_text(encoding="utf-8"))
    if legacy_yaml.get("kpt_shape") != [4, 3] or cvat_yaml.get("kpt_shape") != [4, 3]:
        raise ValueError("legacy and CVAT exports must both use kpt_shape [4, 3]")
    if legacy_yaml.get("names", {}).get(0) != "tissue_box":
        raise ValueError("unexpected legacy class name")

    legacy_train = _paired_files(LEGACY_ROOT / "images" / "train", LEGACY_ROOT / "labels" / "train")
    legacy_val = _paired_files(LEGACY_ROOT / "images" / "val", LEGACY_ROOT / "labels" / "val")
    positives = _paired_files(POSITIVE_ROOT / "images", POSITIVE_ROOT / "labels" / "train")
    negatives = _paired_files(NEGATIVE_ROOT / "images", NEGATIVE_ROOT / "labels")
    if (len(legacy_train), len(legacy_val), len(positives), len(negatives)) != (18, 6, 20, 100):
        raise ValueError("unexpected source dataset counts")

    for _, label in legacy_train + legacy_val + positives:
        values = _label_values(label, allow_empty=False)
        _validate_geometry(label, values)
    for _, label in negatives:
        if _label_values(label, allow_empty=True):
            raise ValueError(f"negative label contains object data: {label}")

    for split in ("train", "val", "test"):
        (PILOT_ROOT / "images" / split).mkdir(parents=True, exist_ok=False)
        (PILOT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=False)
    OVERLAY_ROOT.mkdir(parents=True, exist_ok=False)

    positive_val_stems = {
        f"tissue_positive_20260724_131652_{index:03d}" for index in POSITIVE_VAL_INDICES
    }
    negative_selected_stems = {
        f"tissue_negative_20260724_131450_{index:03d}" for index in NEGATIVE_SELECTED_INDICES
    }
    negative_val_stems = {
        f"tissue_negative_20260724_131450_{index:03d}" for index in NEGATIVE_VAL_INDICES
    }

    manifest_rows: list[dict[str, str]] = []
    for image, label in legacy_train:
        _copy_sample(
            image,
            label,
            source="legacy_white_background",
            capture_type="positive",
            split="train",
            manifest_rows=manifest_rows,
        )
    for image, label in positives:
        split = "val" if image.stem in positive_val_stems else "train"
        _copy_sample(
            image,
            label,
            source="orbbec_positive",
            capture_type="positive",
            split=split,
            manifest_rows=manifest_rows,
        )
    for image, label in negatives:
        if image.stem not in negative_selected_stems:
            continue
        split = "val" if image.stem in negative_val_stems else "train"
        _copy_sample(
            image,
            label,
            source="orbbec_negative",
            capture_type="negative",
            split=split,
            manifest_rows=manifest_rows,
        )
    for image, label in legacy_val:
        _copy_sample(
            image,
            label,
            source="legacy_white_background",
            capture_type="positive",
            split="test",
            manifest_rows=manifest_rows,
        )

    dataset_yaml = {
        "path": "data/datasets/tissue_box_front_orbbec_pilot",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [4, 3],
        "flip_idx": [0, 1, 2, 3],
        "names": {0: "tissue_box"},
    }
    (PILOT_ROOT / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )
    with (PILOT_ROOT / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (PILOT_ROOT / "selected_negative_files.txt").write_text(
        "\n".join(f"{stem}.jpg" for stem in sorted(negative_selected_stems)) + "\n",
        encoding="utf-8",
    )

    for image, label in positives:
        _draw_overlay(image, label, "orbbec_positive")
    for image, label in legacy_train + legacy_val:
        _draw_overlay(image, label, "legacy_white_background")

    counts: dict[str, dict[str, int]] = {}
    for row in manifest_rows:
        split_counts = counts.setdefault(row["split"], {})
        key = f"{row['source']}:{row['capture_type']}"
        split_counts[key] = split_counts.get(key, 0) + 1
    summary = {
        "pilot_root": str(PILOT_ROOT),
        "overlay_root": str(OVERLAY_ROOT),
        "counts": counts,
        "selected_negative_files": [
            f"{stem}.jpg" for stem in sorted(negative_selected_stems)
        ],
        "positive_validation_files": [
            f"{stem}.jpg" for stem in sorted(positive_val_stems)
        ],
        "negative_validation_files": [
            f"{stem}.jpg" for stem in sorted(negative_val_stems)
        ],
    }
    (PILOT_ROOT / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
