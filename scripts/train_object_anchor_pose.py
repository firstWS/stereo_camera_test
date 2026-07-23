#!/usr/bin/env python3
"""Train a tissue-box YOLO-Pose pilot with geometry-safe augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="configs/datasets/tissue_box_front_only_pose.yaml"
    )
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--name", default="tissue_box_01_front_only_pilot")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        raise SystemExit(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=max(0, args.workers),
        patience=max(0, args.patience),
        project=str(ROOT / "runs" / "object_anchor_pose"),
        name=args.name,
        exist_ok=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=8.0,
        translate=0.05,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.25,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
