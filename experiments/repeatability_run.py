#!/usr/bin/env python3
"""
Live / video repeatability logger: detection -> Track A (dense SGBM ROI) + Track B (sparse NCC).
Writes CSV for KPI summarization. Optional OpenCV preview windows (see YAML `preview`).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()

from calibration_repository import load_calibration  # noqa: E402
from capture import CaptureAdapter, SBSSplitConfig, split_sbs_frame  # noqa: E402
from depth_dense import (  # noqa: E402
    SGBMConfig,
    compute_disparity_map,
    depth_dense_track_a,
    make_sgbm,
)
from depth_sparse import depth_sparse_track_b  # noqa: E402
from detect import (  # noqa: E402
    DummyCenterDetector,
    UltralyticsYOLODetector,
    pick_primary_box,
)
from split_rectify import crop_to_calib_size, rectify_stereo_frame  # noqa: E402

_PREVIEW_REGISTERED: set[str] = set()


def _preview_reset_registered() -> None:
    _PREVIEW_REGISTERED.clear()


def _preview_ensure_named(win_title: str) -> None:
    if win_title not in _PREVIEW_REGISTERED:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        _PREVIEW_REGISTERED.add(win_title)


def _preview_update(
    preview_cfg: dict[str, Any],
    left_bgr: np.ndarray,
    prim: Any,
    disparity: np.ndarray | None,
    est_a: Any,
    est_b: Any,
) -> bool:
    """
    Show OpenCV windows. Returns True if user pressed 'q' or 'Q' to quit.
    Requires GUI-enabled OpenCV (not headless-only wheels on some systems).
    """
    scale = float(preview_cfg.get("scale", 0.45))
    win_left = str(preview_cfg.get("window_left", "stereo-3d-poc | left (rectified)"))
    win_disp = str(preview_cfg.get("window_disparity", "stereo-3d-poc | disparity"))
    wait_ms = max(1, int(preview_cfg.get("wait_key_ms", 1)))

    vis = left_bgr.copy()
    if prim is not None:
        x1, y1, x2, y2 = map(int, prim.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bcx = int((x1 + x2) * 0.5)
        bcy = int(y2)
        cv2.circle(vis, (bcx, bcy), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    lines: list[str] = []
    if prim is None:
        lines.append("no detection")
    else:
        if est_a is not None and getattr(est_a, "valid", False):
            lines.append(f"A Z={est_a.Z:.3f}  disp={est_a.disparity:.2f}px")
        else:
            notes_a = getattr(est_a, "notes", "") if est_a is not None else ""
            lines.append(f"A invalid {notes_a}"[:80])
        if est_b is not None and getattr(est_b, "valid", False):
            lines.append(f"B Z={est_b.Z:.3f}  disp={est_b.disparity:.2f}px")
        else:
            notes_b = getattr(est_b, "notes", "") if est_b is not None else ""
            lines.append(f"B: {notes_b}"[:80])

    overlay_bgr = (0, 0, 255)  # red text (BGR)
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (8, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            overlay_bgr,
            1,
            cv2.LINE_AA,
        )

    if 0 < scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    _preview_ensure_named(win_left)
    cv2.imshow(win_left, vis)

    if preview_cfg.get("show_disparity") and disparity is not None:
        d = disparity.astype(np.float32)
        mask = d > 0
        color = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        if np.any(mask):
            lo = float(np.percentile(d[mask], 5))
            hi = float(np.percentile(d[mask], 95))
            if hi <= lo:
                hi = lo + 1e-3
            u = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
            u8 = (u * 255).astype(np.uint8)
            u8[~mask] = 0
            color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        if 0 < scale != 1.0:
            color = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _preview_ensure_named(win_disp)
        cv2.imshow(win_disp, color)

    key = cv2.waitKey(wait_ms) & 0xFF
    return key in (ord("q"), ord("Q"))


def build_detector(cfg: dict):
    d = cfg.get("detector", {})
    kind = (d.get("kind") or "yolo").lower()
    if kind == "dummy":
        return DummyCenterDetector(frac=float(d.get("frac", 0.2)))
    return UltralyticsYOLODetector(
        model_path=d.get("model_path", "yolov8n.pt"),
        conf_threshold=float(d.get("conf", 0.25)),
        iou_threshold=float(d.get("iou", 0.45)),
        imgsz=d.get("imgsz", 640),
        device=d.get("device"),
    )


def run_session(cfg_path: Path) -> Path:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    calib_yaml = cfg.get("calibration", {}).get("yaml")
    if not calib_yaml:
        raise SystemExit("calibration.yaml missing in config")
    calib_path = _resolve_repo_path(calib_yaml)
    assert calib_path is not None
    calib = load_calibration(calib_path)

    cam_cfg = cfg["camera"]
    split_cfg = SBSSplitConfig(
        left_width=cfg.get("sbs", {}).get("left_width"),
        swap_eyes=bool(cfg.get("sbs", {}).get("swap_eyes", False)),
    )

    inp = cfg.get("input") or {}
    in_type = (inp.get("type") or "camera").lower()
    cap = None
    if in_type == "camera":
        cap = CaptureAdapter(
            device_index=int(cam_cfg.get("device_index", 0)),
            width=cam_cfg.get("width"),
            height=cam_cfg.get("height"),
            fps=cam_cfg.get("fps"),
            backend=cam_cfg.get("backend"),
        )
        cap.open()
    elif in_type == "video":
        p = inp.get("video_path")
        if not p:
            raise SystemExit("input.video_path required for video mode")
        vp = _resolve_repo_path(p)
        assert vp is not None
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video {vp}")
    else:
        raise SystemExit(f"Unknown input.type {in_type}")

    detector = build_detector(cfg)
    sgbm_yaml = cfg.get("sgbm", {})
    sgbm = make_sgbm(
        SGBMConfig(
            min_disparity=int(sgbm_yaml.get("min_disparity", 0)),
            num_disparities=int(sgbm_yaml.get("num_disparities", 128)),
            block_size=int(sgbm_yaml.get("block_size", 5)),
            disp12_max_diff=int(sgbm_yaml.get("disp12_max_diff", 1)),
            uniqueness_ratio=int(sgbm_yaml.get("uniqueness_ratio", 10)),
            speckle_window_size=int(sgbm_yaml.get("speckle_window_size", 100)),
            speckle_range=int(sgbm_yaml.get("speckle_range", 32)),
        )
    )
    scale_down = int(sgbm_yaml.get("scale_down", 1))

    spcfg = cfg.get("sparse", {})
    tpl_r = int(spcfg.get("template_radius", 7))
    max_d = int(spcfg.get("max_disparity", 128))
    min_d = int(spcfg.get("min_disparity", 1))

    rep = cfg.get("repeatability", {})
    warmup = int(rep.get("warmup_frames", 10))
    preview_cfg = dict(cfg.get("preview") or {})
    preview_enabled = bool(preview_cfg.get("enabled", False))
    max_frames = int(rep.get("max_frames", 300))
    out_csv = _resolve_repo_path(rep.get("output_csv", "out/repeatability.csv"))
    assert out_csv is not None
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    maps = calib.ensure_maps()
    Q = calib.Q

    if preview_enabled:
        print(
            "Preview windows enabled — click the OpenCV window (taskbar / Alt+Tab), then press Q to quit."
        )
        print(
            f"  wait_key_ms={preview_cfg.get('wait_key_ms', 1)}  "
            f"max_frames={max_frames}"
        )

    fieldnames = [
        "frame_idx",
        "t_wall",
        "capture_ms",
        "disp_ms",
        "det_ms",
        "det_conf",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "ref_u",
        "ref_v",
        "A_valid",
        "A_X",
        "A_Y",
        "A_Z",
        "A_disp",
        "A_valid_ratio",
        "B_valid",
        "B_X",
        "B_Y",
        "B_Z",
        "B_disp",
        "B_notes",
    ]

    t0 = time.perf_counter()
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            idx = 0
            while idx < max_frames:
                t_cap0 = time.perf_counter()
                if in_type == "camera":
                    ok, frame_stereo = cap.read_stereo(split_cfg)
                    if not ok:
                        continue
                else:
                    ok, raw = cap.read()
                    if not ok or raw is None:
                        break
                    frame_stereo = split_sbs_frame(raw, split_cfg)

                frame_stereo = crop_to_calib_size(frame_stereo, calib)
                rect = rectify_stereo_frame(frame_stereo, maps)
                capture_ms = (time.perf_counter() - t_cap0) * 1000.0

                left = rect.left_bgr
                gray_l = rect.gray_left()
                gray_r = rect.gray_right()

                t_det0 = time.perf_counter()
                dets = detector.predict(left)
                det_ms = (time.perf_counter() - t_det0) * 1000.0

                prim = pick_primary_box(dets)
                if prim is None:
                    if preview_enabled:
                        if _preview_update(preview_cfg, left, None, None, None, None):
                            break
                    if idx >= warmup:
                        w.writerow(
                            {
                                "frame_idx": idx,
                                "t_wall": time.perf_counter() - t0,
                                "capture_ms": capture_ms,
                                "disp_ms": 0,
                                "det_ms": det_ms,
                                "det_conf": "",
                                "box_x1": "",
                                "box_y1": "",
                                "box_x2": "",
                                "box_y2": "",
                                "ref_u": "",
                                "ref_v": "",
                                "A_valid": False,
                                "A_X": "",
                                "A_Y": "",
                                "A_Z": "",
                                "A_disp": "",
                                "A_valid_ratio": "",
                                "B_valid": False,
                                "B_X": "",
                                "B_Y": "",
                                "B_Z": "",
                                "B_disp": "",
                                "B_notes": "no_detection",
                            }
                        )
                    idx += 1
                    continue

                ref_u, ref_v = prim.bottom_center

                t_disp0 = time.perf_counter()
                disp = compute_disparity_map(gray_l, gray_r, sgbm, scale_down=scale_down)
                disp_ms = (time.perf_counter() - t_disp0) * 1000.0

                est_a = depth_dense_track_a(disp, prim, Q, min_disp=1.0)
                est_b = depth_sparse_track_b(
                    gray_l, gray_r, ref_u, ref_v, Q, tpl_r, max_d, min_d
                )

                if preview_enabled:
                    if _preview_update(preview_cfg, left, prim, disp, est_a, est_b):
                        break

                if idx >= warmup:
                    w.writerow(
                        {
                            "frame_idx": idx,
                            "t_wall": time.perf_counter() - t0,
                            "capture_ms": capture_ms,
                            "disp_ms": disp_ms,
                            "det_ms": det_ms,
                            "det_conf": prim.confidence,
                            "box_x1": prim.xyxy[0],
                            "box_y1": prim.xyxy[1],
                            "box_x2": prim.xyxy[2],
                            "box_y2": prim.xyxy[3],
                            "ref_u": ref_u,
                            "ref_v": ref_v,
                            "A_valid": est_a.valid,
                            "A_X": est_a.X if est_a.valid else "",
                            "A_Y": est_a.Y if est_a.valid else "",
                            "A_Z": est_a.Z if est_a.valid else "",
                            "A_disp": est_a.disparity if est_a.disparity is not None else "",
                            "A_valid_ratio": est_a.valid_pixel_ratio
                            if est_a.valid_pixel_ratio is not None
                            else "",
                            "B_valid": est_b.valid,
                            "B_X": est_b.X if est_b.valid else "",
                            "B_Y": est_b.Y if est_b.valid else "",
                            "B_Z": est_b.Z if est_b.valid else "",
                            "B_disp": est_b.disparity if est_b.disparity is not None else "",
                            "B_notes": est_b.notes,
                        }
                    )
                idx += 1
    finally:
        if preview_enabled:
            cv2.destroyAllWindows()
            _preview_reset_registered()
        if cap is not None:
            cap.release()

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    cfg_path = args.config.resolve() if args.config.is_absolute() else (_REPO_ROOT / args.config).resolve()
    out = run_session(cfg_path)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
