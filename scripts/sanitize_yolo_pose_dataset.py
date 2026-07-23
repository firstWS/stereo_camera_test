#!/usr/bin/env python3
"""Sanitize CVAT Ultralytics YOLO Pose labels and render review overlays."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_config import ObjectAnchorConfig, load_object_anchor_config  # noqa: E402
from yolo_pose_sanitize import (  # noqa: E402
    PoseLabelValidationError,
    SanitizedPoseLabel,
    cuboid_face_keypoint_ids,
    find_skeleton_crossings,
    infer_named_view,
    sanitize_yolo_pose_line,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class LabelFileResult:
    source_path: Path
    relative_label_path: Path
    image_path: Path
    rows: list[SanitizedPoseLabel]


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            target = (destination / Path(*name.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"ZIP member escapes output directory: {member.filename}")
        archive.extractall(destination)


def _find_labels_root(root: Path) -> Path:
    if root.name.lower() == "labels" and root.is_dir():
        return root
    candidates = sorted(path for path in root.rglob("labels") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one labels directory, found {len(candidates)}")
    return candidates[0]


def _image_index(root: Path) -> tuple[dict[str, Path], list[str]]:
    by_stem: dict[str, Path] = {}
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.lower()
        if key in by_stem:
            errors.append(f"duplicate image stem: {path.stem} ({by_stem[key]}, {path})")
        else:
            by_stem[key] = path
    return by_stem, errors


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _draw_verification(
    image: np.ndarray,
    rows: list[SanitizedPoseLabel],
    config: ObjectAnchorConfig,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    colors = {1: (0, 190, 255), 2: (40, 220, 40)}
    for row_index, row in enumerate(rows):
        cx, cy, bw, bh = row.bbox_cxcywh
        x1 = int(round((cx - bw * 0.5) * width))
        y1 = int(round((cy - bh * 0.5) * height))
        x2 = int(round((cx + bw * 0.5) * width))
        y2 = int(round((cy + bh * 0.5) * height))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 20), 3)
        points = row.keypoints_xyv
        pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height))
        for start, end in config.skeleton:
            if points[start, 2] <= 0 or points[end, 2] <= 0:
                continue
            p1 = tuple(np.rint(pixels[start]).astype(int))
            p2 = tuple(np.rint(pixels[end]).astype(int))
            cv2.line(canvas, p1, p2, (255, 120, 20), 3, cv2.LINE_AA)
        for index, ((x, y), (_, _, visibility)) in enumerate(zip(pixels, points)):
            visible = int(visibility)
            if visible <= 0:
                continue
            center = (int(round(x)), int(round(y)))
            color = colors[visible]
            cv2.circle(canvas, center, 8, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 10, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{index}:{config.keypoints[index].name}(v{visible})",
                (center[0] + 11, center[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"row={row_index} class={row.class_id} visible={list(row.visible_ids)}",
            (max(5, x1), max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 20),
            2,
            cv2.LINE_AA,
        )

    legend_x = 20
    legend_y = max(28, height - (len(config.keypoints) + 1) * 25)
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (10, legend_y - 24),
        (min(width - 10, 590), height - 10),
        (255, 255, 255),
        -1,
    )
    canvas = cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0)
    cv2.putText(
        canvas,
        "tissue_box_01 keypoint order",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for row_index, keypoint in enumerate(config.keypoints, start=1):
        cv2.putText(
            canvas,
            f"{keypoint.keypoint_id}: {keypoint.name}",
            (legend_x, legend_y + row_index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _metadata_order_check(data_yaml: dict[str, Any], config: ObjectAnchorConfig) -> dict[str, Any]:
    expected_names = list(config.keypoint_names)
    exported = data_yaml.get("keypoint_names", data_yaml.get("kpt_names"))
    if exported is None:
        return {
            "status": "visual_review_required",
            "reason": "CVAT data.yaml has kpt_shape but no keypoint name metadata",
            "expected": expected_names,
        }
    if isinstance(exported, dict):
        exported_names = [str(exported[key]) for key in sorted(exported, key=lambda x: int(x))]
    elif isinstance(exported, list):
        exported_names = [str(value) for value in exported]
    else:
        return {"status": "error", "reason": "Unsupported keypoint name metadata type"}
    return {
        "status": "match" if exported_names == expected_names else "mismatch",
        "expected": expected_names,
        "exported": exported_names,
    }


def _write_order_file(path: Path, config: ObjectAnchorConfig) -> None:
    lines = ["ID\tname\tX\tY\tZ"]
    for point in config.keypoints:
        lines.append(
            f"{point.keypoint_id}\t{point.name}\t"
            f"{point.xyz[0]:+.4f}\t{point.xyz[1]:+.4f}\t{point.xyz[2]:+.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CVAT export ZIP or dataset/labels folder")
    parser.add_argument("--images-dir", help="External image folder when ZIP has no images")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--anchor-config",
        default="configs/object_anchors/tissue_box_01.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-zip", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    anchor_path = Path(args.anchor_config)
    if not anchor_path.is_absolute():
        anchor_path = ROOT / anchor_path
    config = load_object_anchor_config(anchor_path)

    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists; use --overwrite: {output_path}")
        if output_path == output_path.anchor or output_path == input_path:
            raise SystemExit(f"Refusing unsafe output removal: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="pose_sanitize_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            extracted = temporary_root / "export"
            extracted.mkdir()
            _safe_extract_zip(input_path, extracted)
            dataset_root = extracted
        elif input_path.is_dir():
            dataset_root = input_path
        else:
            raise SystemExit(f"Input must be a ZIP or directory: {input_path}")

        labels_root = _find_labels_root(dataset_root)
        label_paths = sorted(labels_root.rglob("*.txt"))
        errors: list[str] = []
        warnings: list[str] = []
        if not label_paths:
            errors.append("No label files found")

        data_yaml_paths = sorted(dataset_root.rglob("data.yaml"))
        data_yaml = (
            yaml.safe_load(data_yaml_paths[0].read_text(encoding="utf-8")) or {}
            if len(data_yaml_paths) == 1
            else {}
        )
        if len(data_yaml_paths) != 1:
            errors.append(f"Expected one data.yaml, found {len(data_yaml_paths)}")
        kpt_shape = data_yaml.get("kpt_shape")
        if kpt_shape != [len(config.keypoints), 3]:
            errors.append(
                f"data.yaml kpt_shape={kpt_shape!r}; expected [{len(config.keypoints)}, 3]"
            )
        metadata_order = _metadata_order_check(data_yaml, config)
        if metadata_order["status"] in {"error", "mismatch"}:
            errors.append(f"keypoint metadata order: {metadata_order}")
        elif metadata_order["status"] == "visual_review_required":
            warnings.append(metadata_order["reason"])

        internal_images = dataset_root / "images"
        if internal_images.is_dir():
            images_root = internal_images
            external_images = False
        elif args.images_dir:
            images_root = Path(args.images_dir).resolve()
            external_images = True
        else:
            raise SystemExit("ZIP/folder has no images; pass --images-dir")
        image_by_stem, image_errors = _image_index(images_root)
        errors.extend(image_errors)

        label_stems = {path.stem.lower() for path in label_paths}
        manifest_paths = sorted(dataset_root.rglob("train.txt"))
        if manifest_paths:
            manifest_stems = {
                Path(line.strip()).stem.lower()
                for line in manifest_paths[0].read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if manifest_stems != label_stems:
                errors.append(
                    f"train.txt/label filename mismatch: manifest={sorted(manifest_stems)} "
                    f"labels={sorted(label_stems)}"
                )
        else:
            warnings.append("train.txt not found; filename check uses labels and images only")

        missing_images = sorted(stem for stem in label_stems if stem not in image_by_stem)
        if missing_images:
            errors.append(f"labels without matching images: {missing_images}")
        if not external_images:
            extra_images = sorted(set(image_by_stem) - label_stems)
            if extra_images:
                errors.append(f"images without matching labels: {extra_images}")

        file_results: list[LabelFileResult] = []
        total_rows = 0
        total_hidden_rewritten = 0
        face_ids = (
            cuboid_face_keypoint_ids(config.object_points)
            if config.anchor_mode == "cuboid_8point"
            else {"front": tuple(range(len(config.keypoints)))}
        )
        topology_checks: list[dict[str, Any]] = []
        for label_path in label_paths:
            image_path = image_by_stem.get(label_path.stem.lower())
            if image_path is None:
                continue
            rows: list[SanitizedPoseLabel] = []
            raw_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not raw_lines:
                errors.append(f"{label_path.name}: empty label file")
                continue
            for line_number, line in enumerate(raw_lines, start=1):
                source = f"{label_path.name}:{line_number}"
                try:
                    row = sanitize_yolo_pose_line(
                        line,
                        expected_keypoints=len(config.keypoints),
                        source=source,
                    )
                except PoseLabelValidationError as exc:
                    errors.append(str(exc))
                    continue
                rows.append(row)
                total_rows += 1
                total_hidden_rewritten += row.hidden_keypoints_rewritten

                named_view = infer_named_view(label_path.stem)
                crossings = find_skeleton_crossings(row.keypoints_xyv, config.skeleton)
                if (
                    config.anchor_mode == "front_only"
                    and named_view is not None
                    and named_view != "front"
                ):
                    errors.append(
                        f"{label_path.name}: {named_view} view is outside front_only scope"
                    )
                    topology_checks.append(
                        {
                            "label": label_path.name,
                            "view": named_view,
                            "expected_visible_ids": list(range(len(config.keypoints))),
                            "actual_visible_ids": list(row.visible_ids),
                            "crossing_edges": [
                                [list(first), list(second)] for first, second in crossings
                            ],
                            "status": "unsupported_view",
                        }
                    )
                    continue
                if named_view is not None:
                    expected_visible = face_ids[named_view]
                    ids_matched = (
                        set(row.visible_ids).issubset(expected_visible)
                        if config.anchor_mode == "front_only"
                        else row.visible_ids == expected_visible
                    )
                    matched = ids_matched and not crossings
                    check = {
                        "label": label_path.name,
                        "view": named_view,
                        "expected_visible_ids": list(expected_visible),
                        "actual_visible_ids": list(row.visible_ids),
                        "crossing_edges": [
                            [list(first), list(second)] for first, second in crossings
                        ],
                        "status": "match" if matched else "mismatch",
                    }
                    topology_checks.append(check)
                    if not ids_matched:
                        errors.append(
                            f"{label_path.name}: {named_view} view visible IDs "
                            f"{list(row.visible_ids)} do not match {config.object_id} "
                            f"{list(expected_visible)}"
                        )
                    if crossings:
                        errors.append(
                            f"{label_path.name}: visible skeleton edges cross: {crossings}; "
                            "keypoint order is inconsistent"
                        )
                elif crossings:
                    errors.append(
                        f"{label_path.name}: visible skeleton edges cross: {crossings}; "
                        "keypoint order is inconsistent"
                    )
            if rows:
                relative = label_path.relative_to(labels_root)
                file_results.append(LabelFileResult(label_path, relative, image_path, rows))

        dataset_output = output_path / "dataset"
        verification_output = output_path / "verification"
        for result in file_results:
            label_destination = dataset_output / "labels" / result.relative_label_path
            image_destination = (
                dataset_output
                / "images"
                / result.relative_label_path.parent
                / result.image_path.name
            )
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.write_text(
                "\n".join(row.to_yolo_line() for row in result.rows) + "\n",
                encoding="utf-8",
            )
            shutil.copy2(result.image_path, image_destination)

            image = _read_image(result.image_path)
            verification = _draw_verification(image, result.rows, config)
            verification_output.mkdir(parents=True, exist_ok=True)
            verification_path = verification_output / f"{result.image_path.stem}_verified.jpg"
            if not cv2.imwrite(str(verification_path), verification):
                errors.append(f"Could not write verification image: {verification_path}")

        sanitized_data_yaml = {
            "path": ".",
            "train": "images/train",
            "kpt_shape": [len(config.keypoints), 3],
            "flip_idx": list(range(len(config.keypoints))),
            "keypoint_names": list(config.keypoint_names),
            "anchor_mode": config.anchor_mode,
            "names": {0: "tissue_box"},
        }
        dataset_output.mkdir(parents=True, exist_ok=True)
        (dataset_output / "data.yaml").write_text(
            yaml.safe_dump(sanitized_data_yaml, sort_keys=False), encoding="utf-8"
        )
        _write_order_file(output_path / "keypoint_order.tsv", config)

        report = {
            "input": str(input_path),
            "images_dir": str(images_root),
            "object_anchor_config": str(anchor_path),
            "anchor_mode": config.anchor_mode,
            "training_ready": not errors,
            "label_files": len(label_paths),
            "matched_files": len(file_results),
            "label_rows": total_rows,
            "expected_values_per_row": 1 + 4 + len(config.keypoints) * 3,
            "hidden_keypoints_rewritten": total_hidden_rewritten,
            "metadata_order_check": metadata_order,
            "view_topology_checks": topology_checks,
            "warnings": warnings,
            "errors": errors,
        }
        report_path = output_path / "sanitation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"output={output_path}")
        if args.output_zip and not errors:
            zip_path = shutil.make_archive(str(output_path), "zip", root_dir=output_path)
            print(f"zip={zip_path}")
        if errors:
            raise SystemExit(f"Sanitation completed with {len(errors)} error(s); not training-ready")


if __name__ == "__main__":
    main()
