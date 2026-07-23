from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_front_only_dataset_sanitizer_accepts_17_value_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "export"
    labels = dataset / "labels"
    images = tmp_path / "images"
    output = tmp_path / "sanitized"
    labels.mkdir(parents=True)
    images.mkdir()

    stem = "tissue_box_front_001"
    image = np.full((120, 200, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(images / f"{stem}.jpg"), image)
    points = [
        "0.20 0.20 2",
        "0.80 0.20 2",
        "0.80 0.80 2",
        "1.50 -0.20 0",
    ]
    labels.joinpath(f"{stem}.txt").write_text(
        "0 0.5 0.5 0.7 0.7 " + " ".join(points) + "\n",
        encoding="utf-8",
    )
    dataset.joinpath("data.yaml").write_text(
        "kpt_shape: [4, 3]\n"
        "keypoint_names: [front_top_left, front_top_right, "
        "front_bottom_right, front_bottom_left]\n"
        "names: {0: tissue_box}\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sanitize_yolo_pose_dataset.py"),
            "--input",
            str(dataset),
            "--images-dir",
            str(images),
            "--anchor-config",
            str(
                ROOT
                / "configs"
                / "object_anchors"
                / "tissue_box_01_front_only.yaml"
            ),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    sanitized = output / "dataset" / "labels" / f"{stem}.txt"
    tokens = sanitized.read_text(encoding="utf-8").split()
    assert len(tokens) == 17
    assert tokens[-3:] == ["0.000000", "0.000000", "0"]
    report = json.loads(
        output.joinpath("sanitation_report.json").read_text(encoding="utf-8")
    )
    assert report["training_ready"] is True
    assert report["anchor_mode"] == "front_only"
    assert report["expected_values_per_row"] == 17
