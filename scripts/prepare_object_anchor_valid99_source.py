#!/usr/bin/env python3
"""Validate and merge the 99 usable Orbbec FRONT-only positive samples."""

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
EXPORTS = {
    "v2_existing20": ROOT / "data" / "cvat_exports" / "tissue_box_front_positive_v2_dataset",
    "v3_remaining80": ROOT / "data" / "cvat_exports" / "tissue_box_front_positive_v3_dataset",
}
EXCLUDED_STEM = "tissue_positive_20260724_131652_067"
DATASET_ROOT = ROOT / "data" / "datasets" / "tissue_box_front_orbbec_valid99_source"
OUTPUT_ROOT = ROOT / "out" / "object_anchor_valid99"
SKELETON = ((0, 1), (1, 2), (2, 3), (3, 0))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float(
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    return (
        orientation(a, b, c) * orientation(a, b, d) < 0
        and orientation(c, d, a) * orientation(c, d, b) < 0
    )


def _read_label(path: Path) -> tuple[np.ndarray, list[str]]:
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty row, got {len(rows)}")
    parts = rows[0].split()
    if len(parts) != 17:
        raise ValueError(f"{path}: expected 17 values, got {len(parts)}")
    try:
        values = np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric label value") from exc
    if values[0] != 0:
        raise ValueError(f"{path}: expected class 0, got {values[0]}")
    if np.any((values[1:5] < 0.0) | (values[1:5] > 1.0)):
        raise ValueError(f"{path}: bbox value outside [0, 1]")

    keypoints = values[5:].reshape(4, 3)
    for index, (x, y, visibility) in enumerate(keypoints):
        if visibility not in (0.0, 1.0, 2.0):
            raise ValueError(f"{path}: keypoint {index} has invalid visibility {visibility}")
        if visibility > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{path}: keypoint {index} outside [0, 1]")

    xy = keypoints[:, :2]
    if _crosses(xy[0], xy[1], xy[2], xy[3]) or _crosses(
        xy[1], xy[2], xy[3], xy[0]
    ):
        raise ValueError(f"{path}: skeleton crossing")
    signed_area = 0.5 * sum(
        xy[index, 0] * xy[(index + 1) % 4, 1]
        - xy[(index + 1) % 4, 0] * xy[index, 1]
        for index in range(4)
    )
    if signed_area <= 0.0:
        raise ValueError(f"{path}: polygon order differs from TL_TR_BR_BL")

    suspicious: list[str] = []
    cx, cy, width, height = values[1:5]
    x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
    outside = [
        index
        for index, (x, y, visibility) in enumerate(keypoints)
        if visibility > 0 and not (x1 - 1e-4 <= x <= x2 + 1e-4 and y1 - 1e-4 <= y <= y2 + 1e-4)
    ]
    if outside:
        suspicious.append(f"keypoints_outside_bbox={outside}")
    if abs(signed_area) < 0.05 * width * height:
        suspicious.append(
            f"small_polygon_to_bbox_ratio={abs(signed_area) / max(width * height, 1e-12):.4f}"
        )
    return values, suspicious


def _draw_overlay(
    image_path: Path,
    values: np.ndarray,
    source_export: str,
    destination: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    height, width = image.shape[:2]
    cx, cy, box_width, box_height = values[1:5]
    bbox = np.rint(
        [
            (cx - box_width / 2) * width,
            (cy - box_height / 2) * height,
            (cx + box_width / 2) * width,
            (cy + box_height / 2) * height,
        ]
    ).astype(int)
    keypoints = values[5:].reshape(4, 3).copy()
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    points = np.rint(keypoints[:, :2]).astype(int)

    cv2.rectangle(
        image,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0, 220, 0),
        2,
    )
    for start, end in SKELETON:
        cv2.line(image, tuple(points[start]), tuple(points[end]), (0, 165, 255), 2)
    for index, point in enumerate(points):
        cv2.circle(image, tuple(point), 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 7, int(point[1]) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(index),
            (int(point[0]) + 7, int(point[1]) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        f"{source_export} | {image_path.name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"failed to write overlay: {destination}")


def _create_review_sheets(overlays: list[Path], destination_dir: Path) -> list[Path]:
    sheets: list[Path] = []
    chunk_size = 24
    columns = 4
    for sheet_index, start in enumerate(range(0, len(overlays), chunk_size), 1):
        thumbs: list[np.ndarray] = []
        for path in overlays[start : start + chunk_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode overlay: {path}")
            thumbs.append(cv2.resize(image, (384, 240), interpolation=cv2.INTER_AREA))
        rows: list[np.ndarray] = []
        for index in range(0, len(thumbs), columns):
            row = thumbs[index : index + columns]
            while len(row) < columns:
                row.append(np.zeros_like(thumbs[0]))
            rows.append(np.hstack(row))
        destination = destination_dir / f"valid99_review_{sheet_index:02d}.jpg"
        if not cv2.imwrite(str(destination), np.vstack(rows)):
            raise OSError(f"failed to write review sheet: {destination}")
        sheets.append(destination)
    return sheets


def main() -> None:
    if DATASET_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {DATASET_ROOT}")
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT_ROOT}")

    source_records: list[dict[str, object]] = []
    errors: list[str] = []
    suspicious_labels: list[str] = []
    image_hashes: defaultdict[str, list[str]] = defaultdict(list)
    label_hashes: defaultdict[str, list[str]] = defaultdict(list)
    all_stems: defaultdict[str, list[str]] = defaultdict(list)
    source_counts: dict[str, dict[str, int]] = {}

    for source_export, export_root in EXPORTS.items():
        config = yaml.safe_load((export_root / "data.yaml").read_text(encoding="utf-8"))
        if config.get("kpt_shape") != [4, 3]:
            errors.append(f"{source_export}: kpt_shape={config.get('kpt_shape')}")
        images = sorted((export_root / "images").glob("*.jpg"))
        labels = sorted((export_root / "labels" / "train").glob("*.txt"))
        image_map = {path.stem: path for path in images}
        label_map = {path.stem: path for path in labels}
        source_counts[source_export] = {"images": len(images), "labels": len(labels)}

        expected = (20, 20) if source_export == "v2_existing20" else (80, 79)
        if (len(images), len(labels)) != expected:
            errors.append(
                f"{source_export}: expected images/labels={expected}, got {(len(images), len(labels))}"
            )
        missing_labels = set(image_map) - set(label_map)
        missing_images = set(label_map) - set(image_map)
        expected_missing = set() if source_export == "v2_existing20" else {EXCLUDED_STEM}
        if missing_labels != expected_missing:
            errors.append(
                f"{source_export}: missing labels={sorted(missing_labels)}, "
                f"expected={sorted(expected_missing)}"
            )
        if missing_images:
            errors.append(f"{source_export}: labels without images={sorted(missing_images)}")

        for stem, image_path in image_map.items():
            all_stems[stem].append(source_export)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"{image_path}: decode failed")
                continue
            if image.shape[:2] != (800, 1280):
                errors.append(f"{image_path}: unexpected shape={image.shape}")
            image_hash = _sha256(image_path)
            image_hashes[image_hash].append(f"{source_export}/{image_path.name}")
            label_path = label_map.get(stem)
            if label_path is None:
                source_records.append(
                    {
                        "filename": image_path.name,
                        "source_export": source_export,
                        "image_path": image_path,
                        "label_path": None,
                        "values": None,
                        "image_sha256": image_hash,
                        "label_sha256": "",
                    }
                )
                continue
            try:
                values, suspicious = _read_label(label_path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            for reason in suspicious:
                suspicious_labels.append(f"{label_path.name}: {reason}")
            label_hash = _sha256(label_path)
            label_hashes[label_hash].append(f"{source_export}/{label_path.name}")
            source_records.append(
                {
                    "filename": image_path.name,
                    "source_export": source_export,
                    "image_path": image_path,
                    "label_path": label_path,
                    "values": values,
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                }
            )

    duplicate_stems = {stem: sources for stem, sources in all_stems.items() if len(sources) > 1}
    duplicate_images = [paths for paths in image_hashes.values() if len(paths) > 1]
    duplicate_labels = [paths for paths in label_hashes.values() if len(paths) > 1]
    if duplicate_stems:
        errors.append(f"duplicate stems={duplicate_stems}")
    if duplicate_images:
        errors.append(f"duplicate image hashes={duplicate_images}")
    if duplicate_labels:
        errors.append(f"duplicate label hashes={duplicate_labels}")

    valid_records = [record for record in source_records if record["label_path"] is not None]
    excluded_records = [record for record in source_records if record["label_path"] is None]
    if len(valid_records) != 99:
        errors.append(f"expected 99 valid records, got {len(valid_records)}")
    if len(excluded_records) != 1 or Path(str(excluded_records[0]["image_path"])).stem != EXCLUDED_STEM:
        errors.append("intentional exclusion set is not exactly the 067 frame")
    if errors:
        raise SystemExit("validation failed:\n- " + "\n- ".join(errors))

    image_destination = DATASET_ROOT / "images" / "all"
    label_destination = DATASET_ROOT / "labels" / "all"
    overlay_destination = OUTPUT_ROOT / "label_overlays"
    excluded_destination = OUTPUT_ROOT / "excluded"
    review_destination = OUTPUT_ROOT / "review_sheets"
    image_destination.mkdir(parents=True, exist_ok=False)
    label_destination.mkdir(parents=True, exist_ok=False)
    overlay_destination.mkdir(parents=True, exist_ok=False)
    excluded_destination.mkdir(parents=True, exist_ok=False)
    review_destination.mkdir(parents=True, exist_ok=False)

    manifest_rows: list[dict[str, object]] = []
    overlays: list[Path] = []
    for record in sorted(source_records, key=lambda item: str(item["filename"])):
        image_path = Path(str(record["image_path"]))
        label_value = record["label_path"]
        if label_value is None:
            shutil.copy2(image_path, excluded_destination / image_path.name)
            manifest_rows.append(
                {
                    "filename": image_path.name,
                    "source_export": record["source_export"],
                    "included_in_dataset": "false",
                    "exclusion_reason": "front_face_clipped_missing_required_keypoints",
                    "class_id": "",
                    "object_count": "",
                    "label_value_count": "",
                    "keypoint_order": "",
                    "image_sha256": record["image_sha256"],
                    "label_sha256": "",
                    "validation_status": "excluded_intentionally",
                }
            )
            continue

        label_path = Path(str(label_value))
        copied_image = image_destination / image_path.name
        copied_label = label_destination / label_path.name
        shutil.copy2(image_path, copied_image)
        shutil.copy2(label_path, copied_label)
        if _sha256(copied_image) != record["image_sha256"] or _sha256(copied_label) != record["label_sha256"]:
            raise OSError(f"copy hash mismatch: {image_path.name}")
        overlay_path = overlay_destination / image_path.name
        _draw_overlay(
            image_path,
            np.asarray(record["values"], dtype=np.float64),
            str(record["source_export"]),
            overlay_path,
        )
        overlays.append(overlay_path)
        manifest_rows.append(
            {
                "filename": image_path.name,
                "source_export": record["source_export"],
                "included_in_dataset": "true",
                "exclusion_reason": "",
                "class_id": 0,
                "object_count": 1,
                "label_value_count": 17,
                "keypoint_order": "TL_TR_BR_BL",
                "image_sha256": record["image_sha256"],
                "label_sha256": record["label_sha256"],
                "validation_status": "valid",
            }
        )

    manifest_path = DATASET_ROOT / "merge_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    sheets = _create_review_sheets(overlays, review_destination)
    (OUTPUT_ROOT / "suspected_labels.txt").write_text(
        "\n".join(suspicious_labels) + ("\n" if suspicious_labels else ""),
        encoding="utf-8",
    )
    summary = {
        "source_counts": source_counts,
        "valid_records": len(valid_records),
        "excluded_records": [str(record["filename"]) for record in excluded_records],
        "dataset_images": len(list(image_destination.glob("*.jpg"))),
        "dataset_labels": len(list(label_destination.glob("*.txt"))),
        "overlays": len(overlays),
        "review_sheets": [str(path) for path in sheets],
        "suspicious_labels": suspicious_labels,
        "duplicate_stems": duplicate_stems,
        "duplicate_image_hashes": duplicate_images,
        "duplicate_label_hashes": duplicate_labels,
    }
    (OUTPUT_ROOT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
