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
from datetime import datetime
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
from capture import (
    CaptureAdapter,
    SBSSplitConfig,
    StereoImageFolderReader,
    split_sbs_frame,
)
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
from stereo_types import StereoFrame  # noqa: E402
from split_rectify import crop_to_calib_size, rectify_stereo_frame  # noqa: E402

_PREVIEW_REGISTERED: set[str] = set()


def _preview_reset_registered() -> None:
    _PREVIEW_REGISTERED.clear()


def _preview_ensure_named(win_title: str) -> None:
    if win_title not in _PREVIEW_REGISTERED:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        _PREVIEW_REGISTERED.add(win_title)


def _disparity_colormap_bgr(
    disparity: np.ndarray | None, fallback_hw: tuple[int, int]
) -> np.ndarray:
    h, w = fallback_hw
    color = np.zeros((h, w, 3), dtype=np.uint8)
    if disparity is None:
        return color
    d = disparity.astype(np.float32)
    mask = d > 0
    if not np.any(mask):
        return color
    lo = float(np.percentile(d[mask], 5))
    hi = float(np.percentile(d[mask], 95))
    if hi <= lo:
        hi = lo + 1e-3
    u = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    u8 = (u * 255).astype(np.uint8)
    u8[~mask] = 0
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _annotate_left(left_bgr: np.ndarray, prim: Any, est_a: Any, est_b: Any) -> np.ndarray:
    vis = left_bgr.copy()
    if prim is not None:
        x1, y1, x2, y2 = map(int, prim.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bcx = int((x1 + x2) * 0.5)
        bcy = int(y2)
        cv2.circle(vis, (bcx, bcy), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    overlay_bgr = (0, 0, 255)
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
    return vis


def _preview_scale_visual(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _snapshot_image_folder_pair(
    dest_dir: Path,
    seq: list[int],
    frame_loop_idx: int,
    stereo_pre_rectify: StereoFrame,
) -> None:
    """
    Saves cropped (calib-sized) left/right BGR for ``input.type: image_folder`` replay.
    Same basename under dest_dir/left and dest_dir/right.
    """
    left_dir = dest_dir / "left"
    right_dir = dest_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    seq[0] += 1
    n = seq[0]
    name = f"{n:04d}_f{frame_loop_idx:05d}.png"
    lp = left_dir / name
    rp = right_dir / name
    cv2.imwrite(str(lp), stereo_pre_rectify.left_bgr)
    cv2.imwrite(str(rp), stereo_pre_rectify.right_bgr)
    print(f"Saved image_folder pair #{n}: {lp.name} → {left_dir} / {right_dir}")


def _preview_image_folder_hold_spin(
    preview_cfg: dict[str, Any],
    preview_state: dict[str, Any],
    loop_idx: int,
    stereo_pre_rectify: StereoFrame,
) -> bool:
    """
    After ``_preview_tick``, keep windows open until user chooses next action.

    Returns True if user pressed Q (quit entire session).
    Returns False if Space or n/N (advance to next stereo pair).
    S saves another PNG pair without advancing.
    """
    poll_ms = max(1, int(preview_cfg.get("image_folder_hold_poll_ms", 50)))
    print(
        "[hold] Space or n = next pair  |  Q = quit session  |  S = save left/right PNG again",
        flush=True,
    )
    while True:
        key = cv2.waitKey(poll_ms) & 0xFF
        if key in (ord("q"), ord("Q")):
            return True
        if key in (ord(" "), ord("n"), ord("N")):
            return False
        if key in (ord("s"), ord("S")):
            sess = preview_state.get("snapshot_session_dir")
            sq = preview_state.get("snap_seq")
            if isinstance(sess, Path) and isinstance(sq, list):
                _snapshot_image_folder_pair(sess, sq, loop_idx, stereo_pre_rectify)
            else:
                print("Snapshots disabled (no snapshot session directory).")


def _preview_tick(
    preview_cfg: dict[str, Any],
    rect: StereoFrame,
    prim: Any,
    disparity: np.ndarray | None,
    est_a: Any,
    est_b: Any,
    loop_idx: int,
    preview_state: dict[str, Any],
    stereo_pre_rectify: StereoFrame,
) -> bool:
    """
    Multi-window preview at native resolution unless preview.scale != 1.
    Keys: Q quit; S saves ``left/`` + ``right/`` PNG pair (calib-cropped, pre-rectify)
    for ``image_folder`` analysis replay.

    Returns True if user pressed Q (quit session). After return, callers may invoke
    ``_preview_image_folder_hold_spin`` when ``preview.image_folder_hold_until_quit`` is set.
    """
    scale = float(preview_cfg.get("scale", 1.0))

    wins = preview_cfg.get("windows")
    if isinstance(wins, dict):
        show_left = bool(wins.get("left", True))
        show_right = bool(wins.get("right", True))
        show_combined = bool(wins.get("combined", True))
        show_disp = bool(wins.get("disparity", True))
    else:
        show_left = True
        show_right = True
        show_combined = True
        show_disp = bool(preview_cfg.get("show_disparity", True))

    ttl = preview_cfg.get("titles") if isinstance(preview_cfg.get("titles"), dict) else {}
    wt_left = str(ttl.get("left", preview_cfg.get("window_left", "stereo-3d-poc | left (rectified)")))
    wt_right = str(ttl.get("right", "stereo-3d-poc | right (rectified)"))
    wt_combo = str(ttl.get("combined", "stereo-3d-poc | LR rectified concat"))
    wt_disp = str(ttl.get("disparity", preview_cfg.get("window_disparity", "stereo-3d-poc | disparity")))

    hl, wl = rect.left_bgr.shape[:2]
    overlay_left_full = _annotate_left(rect.left_bgr, prim, est_a, est_b)
    disp_color_full = _disparity_colormap_bgr(disparity, (hl, wl))
    lr_concat = np.hstack([overlay_left_full, rect.right_bgr])

    panels: list[tuple[np.ndarray, str]] = []
    if show_left:
        panels.append((_preview_scale_visual(overlay_left_full, scale), wt_left))
    if show_right:
        panels.append((_preview_scale_visual(rect.right_bgr.copy(), scale), wt_right))
    if show_combined:
        panels.append((_preview_scale_visual(lr_concat, scale), wt_combo))
    if show_disp:
        panels.append((_preview_scale_visual(disp_color_full, scale), wt_disp))

    for img, title in panels:
        _preview_ensure_named(title)
        cv2.imshow(title, img)

    wait_ms = max(1, int(preview_cfg.get("wait_key_ms", 1)))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (ord("q"), ord("Q")):
        return True
    if key in (ord("s"), ord("S")):
        sess = preview_state.get("snapshot_session_dir")
        sq = preview_state.get("snap_seq")
        if isinstance(sess, Path) and isinstance(sq, list):
            _snapshot_image_folder_pair(sess, sq, loop_idx, stereo_pre_rectify)
        else:
            print("Snapshots disabled (no snapshot session directory).")
    return False


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
    elif in_type in ("image_folder", "images"):
        img_cfg = inp.get("image_folder") or inp.get("images")
        if not isinstance(img_cfg, dict):
            raise SystemExit(
                "input.image_folder (or images) dict required "
                "(keys: left_dir, right_dir [, patterns])."
            )
        ld_raw = img_cfg.get("left_dir")
        rd_raw = img_cfg.get("right_dir")
        if not ld_raw or not rd_raw:
            raise SystemExit("image_folder.left_dir and image_folder.right_dir are required.")
        ld = _resolve_repo_path(ld_raw)
        rd = _resolve_repo_path(rd_raw)
        assert ld is not None and rd is not None
        if not ld.is_dir() or not rd.is_dir():
            raise SystemExit(f"Stereo image dirs must exist: {ld} , {rd}")
        patterns_raw = img_cfg.get("patterns") or ["*.png", "*.jpg", "*.jpeg"]
        patt = [str(x) for x in patterns_raw]
        cap = StereoImageFolderReader.from_dirs(ld, rd, patt)
        print(f"[image_folder] Stereo pairs queued: {cap.pair_count} (resolution from files).")
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

    preview_state: dict[str, Any] = {}

    img_folder_hold = bool(preview_cfg.get("image_folder_hold_until_quit"))

    if preview_enabled:
        print(
            "Preview: focus an OpenCV window (Alt+Tab) — "
            "[Q]=quit session  |  [S]=save left/right PNG pair (image_folder replay) under snapshots session."
        )
        if img_folder_hold and in_type in ("image_folder", "images"):
            print(
                "  image_folder hold until quit: after each pair, Space/n = next  |  Q = exit run."
            )
        snap_root = _resolve_repo_path(preview_cfg.get("snapshot_dir", "out/snapshots"))
        assert snap_root is not None
        sess_dir = snap_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        sess_dir.mkdir(parents=True, exist_ok=True)
        preview_state["snapshot_session_dir"] = sess_dir
        preview_state["snap_seq"] = [0]
        print(f"  wait_key_ms={preview_cfg.get('wait_key_ms', 1)}  max_frames={max_frames}")
        print(f"  snapshot session: {sess_dir}")

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
                elif in_type in ("image_folder", "images"):
                    ok, frame_stereo = cap.read_stereo_pair()
                    if not ok:
                        break
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
                        if _preview_tick(
                            preview_cfg,
                            rect,
                            None,
                            None,
                            None,
                            None,
                            idx,
                            preview_state,
                            frame_stereo,
                        ):
                            break
                        if img_folder_hold and in_type in ("image_folder", "images"):
                            if _preview_image_folder_hold_spin(
                                preview_cfg, preview_state, idx, frame_stereo
                            ):
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
                    if _preview_tick(
                        preview_cfg,
                        rect,
                        prim,
                        disp,
                        est_a,
                        est_b,
                        idx,
                        preview_state,
                        frame_stereo,
                    ):
                        break
                    if img_folder_hold and in_type in ("image_folder", "images"):
                        if _preview_image_folder_hold_spin(
                            preview_cfg, preview_state, idx, frame_stereo
                        ):
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
