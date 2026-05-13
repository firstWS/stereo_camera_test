#!/usr/bin/env python3
"""Quick pipeline smoke test (no camera)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from calibration_repository import load_calibration  # noqa: E402
from depth_dense import SGBMConfig, compute_disparity_map, depth_dense_track_a, make_sgbm  # noqa: E402
from depth_sparse import depth_sparse_track_b  # noqa: E402
from detect import DummyCenterDetector, pick_primary_box  # noqa: E402
from split_rectify import crop_to_calib_size, rectify_stereo_frame  # noqa: E402
from stereo_types import StereoFrame  # noqa: E402


def main() -> None:
    calib_path = ROOT / "calibration" / "stereo_calib.yaml"
    if not calib_path.is_file():
        print("Missing calibration; run: python scripts/create_placeholder_calibration.py")
        sys.exit(1)

    calib = load_calibration(calib_path)
    maps = calib.ensure_maps()
    Q = calib.Q

    rng = np.random.default_rng(0)
    h, w = 480, 640
    left = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    shift = 12
    right = np.roll(left, shift, axis=1)

    frame = StereoFrame(left_bgr=left, right_bgr=right)
    frame = crop_to_calib_size(frame, calib)
    rect = rectify_stereo_frame(frame, maps)
    gray_l = rect.gray_left()
    gray_r = rect.gray_right()

    sgbm = make_sgbm(SGBMConfig())
    disp = compute_disparity_map(gray_l, gray_r, sgbm, scale_down=2)

    det = DummyCenterDetector(frac=0.3)
    dets = det.predict(rect.left_bgr)
    prim = pick_primary_box(dets)
    assert prim is not None

    est_a = depth_dense_track_a(disp, prim, Q, min_disp=1.0)
    ref_u, ref_v = prim.bottom_center
    est_b = depth_sparse_track_b(gray_l, gray_r, ref_u, ref_v, Q, 7, 128, 1)

    print("smoke_test_ok", {"A_valid": est_a.valid, "B_valid": est_b.valid})
    sys.exit(0)


if __name__ == "__main__":
    main()
