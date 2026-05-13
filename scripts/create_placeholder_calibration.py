#!/usr/bin/env python3
"""Write synthetic stereo calibration for local smoke tests (not metric-accurate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from calib_pipeline import save_calib_bundle  # noqa: E402
from calibration_repository import StereoCalibration  # noqa: E402


def build_synthetic_stereo_calibration(
    width: int,
    height: int,
    baseline_m: float,
    fx: float | None,
) -> StereoCalibration:
    if fx is None:
        fx = 0.75 * float(width)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K1 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K2 = K1.copy()
    D1 = np.zeros((5, 1), dtype=np.float64)
    D2 = np.zeros((5, 1), dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    T = np.array([[baseline_m], [0.0], [0.0]], dtype=np.float64)

    tx = T.reshape(3)
    skew = np.array(
        [[0.0, -tx[2], tx[1]], [tx[2], 0.0, -tx[0]], [-tx[1], tx[0], 0.0]],
        dtype=np.float64,
    )
    E = skew @ R
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    F = K2_inv.T @ E @ K1_inv

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (width, height), R, T)

    return StereoCalibration(
        image_size=(width, height),
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        maps=None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic stereo calibration for local runs.")
    ap.add_argument("--width", type=int, default=1600, help="Single-eye width after SBS split")
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--baseline_m", type=float, default=0.06)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument(
        "--out_yaml",
        type=Path,
        default=_ROOT / "calibration" / "stereo_calib.yaml",
    )
    ap.add_argument(
        "--maps_prefix",
        type=Path,
        default=_ROOT / "calibration" / "rectify_maps",
    )
    args = ap.parse_args()

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.maps_prefix.parent.mkdir(parents=True, exist_ok=True)

    calib = build_synthetic_stereo_calibration(
        args.width, args.height, args.baseline_m, args.fx
    )
    save_calib_bundle(args.out_yaml, calib, args.maps_prefix)
    print(f"Wrote {args.out_yaml} and maps {args.maps_prefix}*.npy")


if __name__ == "__main__":
    main()
