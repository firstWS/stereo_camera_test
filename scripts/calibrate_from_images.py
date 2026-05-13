#!/usr/bin/env python3
"""Run stereo calibration from saved left/right image pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from calib_pipeline import (  # noqa: E402
    calibrate_stereo_from_pairs,
    mean_epipolar_y_shift_rectified,
    rectify_pair,
    save_calib_bundle,
)


def load_pairs(left_dir: Path, right_dir: Path | None, pattern: str) -> tuple[list, list]:
    left_dir = Path(left_dir)
    lefts = sorted(left_dir.glob(pattern))
    L, R = [], []
    for lp in lefts:
        if right_dir is not None:
            rp = Path(right_dir) / lp.name
        else:
            rp = lp.parent / lp.name.replace("left", "right")
            if not rp.exists() and "left" in lp.stem:
                rp = lp.with_name(lp.stem.replace("left", "right") + lp.suffix)
        if not rp.exists():
            continue
        li = cv2.imread(str(lp))
        ri = cv2.imread(str(rp))
        if li is None or ri is None:
            continue
        L.append(li)
        R.append(ri)
    return L, R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_dir", type=Path, required=True)
    ap.add_argument("--right_dir", type=Path, default=None)
    ap.add_argument("--pattern", type=str, default="*.png")
    ap.add_argument("--board", type=str, default="9,6", help="Inner corners cols,rows")
    ap.add_argument("--square_mm", type=float, default=25.0)
    ap.add_argument("--out_yaml", type=Path, default=Path("calibration/stereo_calib.yaml"))
    ap.add_argument("--maps_prefix", type=Path, default=Path("calibration/rectify_maps"))
    ap.add_argument("--epipolar_sample", type=int, default=0, help="Index of pair to check")
    args = ap.parse_args()

    cols, rows = map(int, args.board.split(","))
    board_size = (cols, rows)

    L, R = load_pairs(args.left_dir, args.right_dir, args.pattern)
    if len(L) < 3:
        raise SystemExit(f"Need >=3 pairs; found {len(L)}")

    calib, _ = calibrate_stereo_from_pairs(L, R, board_size, args.square_mm / 1000.0)
    maps = calib.ensure_maps()

    mean_dy: float | None = None
    if 0 <= args.epipolar_sample < len(L):
        lr, rr = rectify_pair(L[args.epipolar_sample], R[args.epipolar_sample], maps)
        mean_dy = mean_epipolar_y_shift_rectified(lr, rr, board_size)
        print(f"Mean |dv| on rectified corners (sample {args.epipolar_sample}): {mean_dy}")

    save_calib_bundle(args.out_yaml, calib, args.maps_prefix)
    print(f"Wrote {args.out_yaml} and maps {args.maps_prefix}*.npy")


if __name__ == "__main__":
    main()
