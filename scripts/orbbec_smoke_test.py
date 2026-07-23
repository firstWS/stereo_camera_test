#!/usr/bin/env python3
"""Open the configured Orbbec camera and verify synchronized RGB-D frames."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbbec_rgbd_capture import OrbbecRGBDCapture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orbbec_gemini.yaml")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    root = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    orbbec_config = root.get("orbbec")
    if not isinstance(orbbec_config, dict):
        raise SystemExit("orbbec config mapping is required")

    capture = OrbbecRGBDCapture(orbbec_config)
    deadline = time.monotonic() + max(1.0, float(args.timeout_s))
    received = 0
    try:
        capture.start()
        print("orbbec_pipeline_started")
        while received < max(1, int(args.frames)) and time.monotonic() < deadline:
            ok, frame = capture.read_rgbd()
            if not ok or frame is None:
                continue
            h, w = frame.bgr.shape[:2]
            dh, dw = frame.depth_m.shape[:2]
            valid_depth = frame.depth_m[
                np.isfinite(frame.depth_m) & (frame.depth_m > 0.0)
            ]
            median_depth = float(np.median(valid_depth)) if valid_depth.size else float("nan")
            if (h, w) != (dh, dw):
                raise SystemExit(f"RGB/depth shape mismatch: RGB={w}x{h}, depth={dw}x{dh}")
            print(
                f"frame={received} rgb={w}x{h} depth={dw}x{dh} "
                f"median_depth_m={median_depth:.3f}"
            )
            print("K=", np.array2string(frame.K, precision=4, suppress_small=True))
            print(
                "dist_coeffs=",
                np.array2string(frame.dist_coeffs.reshape(-1), precision=6, suppress_small=True),
            )
            received += 1
    finally:
        capture.release()

    if received < max(1, int(args.frames)):
        raise SystemExit(
            f"Timed out: received {received}/{args.frames} synchronized RGB-D frames "
            f"within {args.timeout_s:.1f}s"
        )
    print("orbbec_smoke_ok")


if __name__ == "__main__":
    main()
