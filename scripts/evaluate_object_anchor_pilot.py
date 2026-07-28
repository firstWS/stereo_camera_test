#!/usr/bin/env python3
"""Evaluate a FRONT-only Object Anchor model on a fixed image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_config import load_object_anchor_config  # noqa: E402
from object_anchor_geometry import find_skeleton_crossings  # noqa: E402
from object_anchor_pose import estimate_object_pose  # noqa: E402


def _parse_vector(value: str, expected: int) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.size != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated values")
    return values


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _load_label(path: Path, width: int, height: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None, None
    lines = text.splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty label row")
    values = [float(item) for item in lines[0].split()]
    if len(values) != 17 or int(values[0]) != 0:
        raise ValueError(f"{path}: expected class 0 and 17 values")
    cx, cy, bw, bh = values[1:5]
    bbox = np.asarray(
        [(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height],
        dtype=np.float64,
    )
    keypoints = np.asarray(values[5:], dtype=np.float64).reshape(4, 3)
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return bbox, keypoints


def _draw_polygon(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    prefix: str,
    confidences: np.ndarray | None = None,
) -> None:
    for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
        cv2.line(image, tuple(np.rint(points[start]).astype(int)), tuple(np.rint(points[end]).astype(int)), color, 2)
    for index, point in enumerate(points):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 5, color, -1)
        cv2.putText(
            image,
            (
                f"{prefix}{index}"
                if confidences is None
                else f"{prefix}{index}:{float(confidences[index]):.2f}"
            ),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--anchor-config",
        default="configs/object_anchors/tissue_box_01_front_only.yaml",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-matrix", required=True)
    parser.add_argument("--dist-coeffs", required=True)
    parser.add_argument(
        "--skip-pnp",
        action="store_true",
        help="Skip PnP for legacy images whose camera calibration is unavailable.",
    )
    args = parser.parse_args()

    model_path = _resolve(args.model, ROOT)
    manifest_path = _resolve(args.manifest, ROOT)
    output_dir = _resolve(args.output, ROOT)
    anchor_path = _resolve(args.anchor_config, ROOT)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    anchor = load_object_anchor_config(anchor_path)
    if anchor.anchor_mode != "front_only" or len(anchor.keypoints) != 4:
        raise SystemExit("evaluation requires a FRONT-only four-keypoint anchor")
    K = _parse_vector(args.camera_matrix, 9).reshape(3, 3)
    dist = _parse_vector(args.dist_coeffs, 8).reshape(-1, 1)

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if not rows:
        raise SystemExit(f"no manifest rows for split={args.split}")

    output_dir.mkdir(parents=True, exist_ok=False)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir()
    model = YOLO(str(model_path))
    if model.task != "pose" or list(getattr(model.model, "kpt_shape", [])) != [4, 3]:
        raise SystemExit("checkpoint is not a four-keypoint YOLO-Pose model")

    frame_rows: list[dict[str, Any]] = []
    pnp_reasons: Counter[str] = Counter()
    for item in rows:
        image_path = _resolve(item["image_path"], ROOT)
        label_path = _resolve(item["label_path"], ROOT)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        gt_bbox, gt_keypoints = _load_label(label_path, width, height)
        is_positive = item["capture_type"] == "positive"
        if is_positive != (gt_keypoints is not None):
            raise ValueError(f"{item['filename']}: capture_type/label content mismatch")

        prediction = model.predict(
            image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=1,
            verbose=False,
        )[0]
        detected = bool(prediction.boxes is not None and len(prediction.boxes) > 0)
        bbox_confidence: float | None = None
        keypoint_confidences: np.ndarray | None = None
        predicted_keypoints: np.ndarray | None = None
        valid_keypoints = 0
        pixel_error: float | None = None
        skeleton_crossed = False
        pnp_valid = False
        pnp_reason = "no_object_anchor_detection"
        pnp_inliers = 0
        reprojection_error: float | None = None
        pnp_callable = False

        overlay = image.copy()
        if gt_bbox is not None and gt_keypoints is not None:
            x1, y1, x2, y2 = np.rint(gt_bbox).astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), 2)
            _draw_polygon(overlay, gt_keypoints[:, :2], (0, 220, 0), "G")

        if detected and prediction.keypoints is not None:
            bbox_confidence = float(prediction.boxes.conf[0].cpu().item())
            predicted_keypoints = prediction.keypoints.xy[0].cpu().numpy().astype(np.float64)
            confidence_tensor = prediction.keypoints.conf
            keypoint_confidences = (
                np.ones(4, dtype=np.float64)
                if confidence_tensor is None
                else confidence_tensor[0].cpu().numpy().astype(np.float64)
            )
            if predicted_keypoints.shape != (4, 2) or keypoint_confidences.shape != (4,):
                raise ValueError(f"{item['filename']}: model returned unexpected keypoint shape")
            visibility = np.where(
                keypoint_confidences >= anchor.pose_settings.confidence_threshold, 2, 0
            ).astype(np.int32)
            valid_keypoints = int(np.count_nonzero(visibility >= 1))
            crossings = find_skeleton_crossings(
                predicted_keypoints,
                anchor.skeleton,
                valid_mask=visibility >= 1,
            )
            skeleton_crossed = bool(crossings)
            pnp_callable = valid_keypoints >= anchor.pose_settings.min_correspondences and not skeleton_crossed
            if args.skip_pnp:
                pnp_reason = "not_evaluated_legacy_intrinsics"
            else:
                pose = estimate_object_pose(
                    predicted_keypoints,
                    anchor.object_points,
                    K,
                    dist_coeffs=dist,
                    confidences=keypoint_confidences,
                    visibility=visibility,
                    settings=anchor.pose_settings,
                )
                if skeleton_crossed and pose.valid:
                    pose.valid = False
                    pose.reason = "skeleton_crossing"
                pnp_valid = bool(pose.valid)
                pnp_reason = pose.reason
                pnp_inliers = len(pose.inlier_indices)
                reprojection_error = pose.mean_reprojection_error_px
            if gt_keypoints is not None:
                visible = gt_keypoints[:, 2] > 0
                pixel_error = float(
                    np.mean(np.linalg.norm(predicted_keypoints[visible] - gt_keypoints[visible, :2], axis=1))
                )
            px1, py1, px2, py2 = np.rint(prediction.boxes.xyxy[0].cpu().numpy()).astype(int)
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 165, 255), 2)
            _draw_polygon(
                overlay,
                predicted_keypoints,
                (0, 165, 255),
                "P",
                confidences=keypoint_confidences,
            )

        pnp_reasons[pnp_reason] += 1
        text_lines = [
            item["filename"],
            f"det={detected} box_conf={bbox_confidence if bbox_confidence is not None else 'n/a'}",
            f"valid_kpt={valid_keypoints}/4 crossing={skeleton_crossed}",
            f"pixel_err={pixel_error if pixel_error is not None else 'n/a'}",
            f"pnp={pnp_valid} reason={pnp_reason} reproj={reprojection_error if reprojection_error is not None else 'n/a'}",
        ]
        for index, text in enumerate(text_lines):
            cv2.putText(
                overlay,
                text,
                (12, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                text,
                (12, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        if not cv2.imwrite(str(overlay_dir / item["filename"]), overlay):
            raise OSError(f"failed to write overlay for {item['filename']}")

        frame_rows.append(
            {
                "filename": item["filename"],
                "source": item["source"],
                "capture_type": item["capture_type"],
                "detected": detected,
                "bbox_confidence": bbox_confidence,
                "keypoint_confidences": (
                    json.dumps(keypoint_confidences.tolist()) if keypoint_confidences is not None else ""
                ),
                "valid_keypoints": valid_keypoints,
                "keypoint_pixel_error": pixel_error,
                "skeleton_crossed": skeleton_crossed,
                "pnp_callable": pnp_callable,
                "pnp_valid": pnp_valid,
                "pnp_inliers": pnp_inliers,
                "pnp_reason": pnp_reason,
                "reprojection_error_px": reprojection_error,
            }
        )

    fieldnames = list(frame_rows[0])
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    positives = [row for row in frame_rows if row["capture_type"] == "positive"]
    negatives = [row for row in frame_rows if row["capture_type"] == "negative"]
    detected_positive = [row for row in positives if row["detected"]]
    false_positives = [row for row in negatives if row["detected"]]
    positive_keypoint_confidences: list[list[float]] = [[], [], [], []]
    for row in detected_positive:
        for index, confidence in enumerate(json.loads(str(row["keypoint_confidences"]))):
            positive_keypoint_confidences[index].append(float(confidence))
    positive_pnp_reasons = Counter(
        str(row["pnp_reason"]) for row in positives if not bool(row["pnp_valid"])
    )
    summary = {
        "model": str(model_path),
        "manifest": str(manifest_path),
        "split": args.split,
        "pnp_evaluated": not args.skip_pnp,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "positive": {
            "images": len(positives),
            "detected": len(detected_positive),
            "detection_rate": len(detected_positive) / max(len(positives), 1),
            "mean_bbox_confidence": _safe_mean(
                [float(row["bbox_confidence"]) for row in detected_positive]
            ),
            "four_valid_keypoints": sum(row["valid_keypoints"] == 4 for row in positives),
            "four_valid_keypoints_rate": sum(row["valid_keypoints"] == 4 for row in positives)
            / max(len(positives), 1),
            "mean_keypoint_confidence_by_id": [
                _safe_mean(values) for values in positive_keypoint_confidences
            ],
            "mean_keypoint_pixel_error": _safe_mean(
                [float(row["keypoint_pixel_error"]) for row in positives if row["keypoint_pixel_error"] is not None]
            ),
            "skeleton_crossings": sum(bool(row["skeleton_crossed"]) for row in positives),
            "pnp_callable_frames": sum(bool(row["pnp_callable"]) for row in positives),
            "pnp_valid": sum(bool(row["pnp_valid"]) for row in positives),
            "pnp_valid_rate": sum(bool(row["pnp_valid"]) for row in positives)
            / max(len(positives), 1),
            "mean_reprojection_error_px": _safe_mean(
                [float(row["reprojection_error_px"]) for row in positives if row["reprojection_error_px"] is not None]
            ),
            "reprojection_threshold_exceeded": sum(
                row["reprojection_error_px"] is not None
                and float(row["reprojection_error_px"])
                > anchor.pose_settings.max_mean_reprojection_error_px
                for row in positives
            ),
            "pnp_failure_reasons": dict(positive_pnp_reasons),
        },
        "negative": {
            "images": len(negatives),
            "false_positives": len(false_positives),
            "false_positive_rate": len(false_positives) / max(len(negatives), 1),
            "mean_false_positive_confidence": _safe_mean(
                [float(row["bbox_confidence"]) for row in false_positives]
            ),
            "max_false_positive_confidence": (
                max(float(row["bbox_confidence"]) for row in false_positives)
                if false_positives
                else None
            ),
        },
        "pnp_failure_reasons": dict(pnp_reasons),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
