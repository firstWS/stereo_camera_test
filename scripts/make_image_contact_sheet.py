#!/usr/bin/env python3
"""Create a labeled contact sheet from an image directory."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=320)
    args = parser.parse_args()

    input_dir = Path(args.input)
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"No images found: {input_dir}")

    columns = max(1, args.columns)
    cell_width = max(160, args.cell_width)
    cell_height = max(120, args.cell_height)
    label_height = 30
    rows = math.ceil(len(paths) / columns)
    canvas = np.full(
        (rows * (cell_height + label_height), columns * cell_width, 3),
        245,
        dtype=np.uint8,
    )

    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {path}")
        scale = min(cell_width / image.shape[1], cell_height / image.shape[0])
        resized = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        x0 = column * cell_width + (cell_width - resized.shape[1]) // 2
        y0 = row * (cell_height + label_height) + (cell_height - resized.shape[0]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            path.stem,
            (column * cell_width + 8, row * (cell_height + label_height) + cell_height + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise SystemExit(f"Could not write image: {output_path}")
    print(f"images={len(paths)} output={output_path}")


if __name__ == "__main__":
    main()
