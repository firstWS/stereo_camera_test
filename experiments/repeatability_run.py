#!/usr/bin/env python3
"""
Live repeatability logger: stereo uses SGBM + ``Q`` and sparse NCC; Orbbec RGB-D uses
aligned depth + intrinsics when ``input.type: orbbec``. CSV + optional OpenCV preview.
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


def _resolve_writable_csv(path: Path) -> Path:
    """
    Pick a CSV path we can open for writing.

    If the configured file is locked (Excel, another run), fall back to a timestamped sibling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _probe(target: Path) -> bool:
        try:
            with target.open("w", newline="", encoding="utf-8"):
                pass
            return True
        except PermissionError:
            return False

    if _probe(path):
        return path

    alt = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    print(
        f"Cannot write {path} (locked by another app or python.exe?). "
        f"Using fallback: {alt}",
        flush=True,
    )
    if _probe(alt):
        return alt

    raise SystemExit(
        f"Cannot write CSV: {path}\n"
        "Close Excel/editors viewing the CSV and end other repeatability_run sessions, then retry."
    )

from apriltag_rgbd_validate import (  # noqa: E402
    compute_apriltag_distance_validation_rgbd,
)
from apriltag_scale import (  # noqa: E402
    AprilTagScaleOutcome,
    compute_apriltag_metric_scale,
    scale_depth_estimate,
)
from apriltag_world import (  # noqa: E402
    AprilTagWorldConfig,
    AprilTagWorldResult,
    WorldPointEstimate,
    build_apriltag_world_config,
    estimate_apriltag_world,
    world_point_from_camera_estimate,
)
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
from object_anchor_runtime import (  # noqa: E402
    ObjectAnchorFrameResult,
    build_optional_object_anchor_runtime,
)
from object_anchor_capture import (  # noqa: E402
    ObjectAnchorCaptureSession,
    ObjectAnchorCaptureSettings,
)
from object_anchor_world import (  # noqa: E402
    TRANSFORM_FORMULA,
    ObjectAnchorWorldTracker,
    build_world_settings,
)
from object_anchor_preview import (  # noqa: E402
    ObjectAnchorPreviewSession,
    PreviewFrameView,
    build_preview_session,
    draw_object_preview_axes,
    draw_preview_banner,
    load_preview_settings,
)
from stereo_types import BBox, StereoFrame  # noqa: E402
from rgbd_geometry import depth_estimate_rgbd_bbox, orbbec_sparse_stub  # noqa: E402

_PREVIEW_REGISTERED: set[str] = set()


def _preview_reset_registered() -> None:
    _PREVIEW_REGISTERED.clear()


def _preview_ensure_named(win_title: str, preview_cfg: dict[str, Any], scale: float) -> None:
    if win_title not in _PREVIEW_REGISTERED:
        autosize = bool(preview_cfg.get("window_autosize", True))
        flags = cv2.WINDOW_AUTOSIZE if autosize else cv2.WINDOW_NORMAL
        cv2.namedWindow(win_title, flags)
        _PREVIEW_REGISTERED.add(win_title)


def _disparity_colormap_bgr(
    disparity: np.ndarray | None,
    fallback_hw: tuple[int, int],
    preview_cfg: dict[str, Any],
    preview_state: dict[str, Any],
) -> np.ndarray:
    """
    Turbo colormap for valid disparity (>0).

    Stabilizers (YAML ``preview``):

    - ``disparity_vis_min`` / ``disparity_vis_max``: fixed pixel range (strongest anti-flicker).
    - ``disparity_percentile_smooth_alpha`` in (0, 1]: smooth percentile bounds over time (default 0).
    """
    h, w = fallback_hw
    color = np.zeros((h, w, 3), dtype=np.uint8)
    if disparity is None:
        return color
    d = disparity.astype(np.float32)
    mask = d > 0
    if not np.any(mask):
        return color

    vmin_key = preview_cfg.get("disparity_vis_min")
    vmax_key = preview_cfg.get("disparity_vis_max")
    use_fixed = vmin_key is not None and vmax_key is not None

    if use_fixed:
        lo = float(vmin_key)
        hi = float(vmax_key)
        preview_state.pop("_disp_vis_lo", None)
        preview_state.pop("_disp_vis_hi", None)
    else:
        p_lo = float(preview_cfg.get("disparity_percentile_low", 5))
        p_hi = float(preview_cfg.get("disparity_percentile_high", 95))
        lo = float(np.percentile(d[mask], p_lo))
        hi = float(np.percentile(d[mask], p_hi))
        smooth_a = float(preview_cfg.get("disparity_percentile_smooth_alpha", 0.0))
        if smooth_a > 0.0:
            prev_lo = preview_state.get("_disp_vis_lo")
            prev_hi = preview_state.get("_disp_vis_hi")
            if prev_lo is not None and prev_hi is not None:
                lo = smooth_a * lo + (1.0 - smooth_a) * float(prev_lo)
                hi = smooth_a * hi + (1.0 - smooth_a) * float(prev_hi)
            preview_state["_disp_vis_lo"] = lo
            preview_state["_disp_vis_hi"] = hi

    if hi <= lo:
        hi = lo + 1e-3
    u = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    u8 = (u * 255).astype(np.uint8)
    u8[~mask] = 0
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _depth_m_colormap_bgr(
    depth_m: np.ndarray | None,
    fallback_hw: tuple[int, int],
    preview_cfg: dict[str, Any],
    preview_state: dict[str, Any],
) -> np.ndarray:
    """Colormap for aligned depth in meters. Same stabilizer pattern as disparity preview."""
    h, w = fallback_hw
    color = np.zeros((h, w, 3), dtype=np.uint8)
    if depth_m is None:
        return color
    d = depth_m.astype(np.float32)
    mask = np.isfinite(d) & (d > float(preview_cfg.get("depth_vis_min_m", 0.05)))
    if not np.any(mask):
        return color

    vmin_key = preview_cfg.get("depth_vis_min_m_clip")
    vmax_key = preview_cfg.get("depth_vis_max_m_clip")
    use_fixed = vmin_key is not None and vmax_key is not None

    if use_fixed:
        lo = float(vmin_key)
        hi = float(vmax_key)
        preview_state.pop("_depth_vis_lo", None)
        preview_state.pop("_depth_vis_hi", None)
    else:
        p_lo = float(preview_cfg.get("depth_percentile_low", 5))
        p_hi = float(preview_cfg.get("depth_percentile_high", 95))
        lo = float(np.percentile(d[mask], p_lo))
        hi = float(np.percentile(d[mask], p_hi))
        smooth_a = float(preview_cfg.get("depth_percentile_smooth_alpha", 0.0))
        if smooth_a > 0.0:
            prev_lo = preview_state.get("_depth_vis_lo")
            prev_hi = preview_state.get("_depth_vis_hi")
            if prev_lo is not None and prev_hi is not None:
                lo = smooth_a * lo + (1.0 - smooth_a) * float(prev_lo)
                hi = smooth_a * hi + (1.0 - smooth_a) * float(prev_hi)
            preview_state["_depth_vis_lo"] = lo
            preview_state["_depth_vis_hi"] = hi

    if hi <= lo:
        hi = lo + 1e-3
    u = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    u8 = (u * 255).astype(np.uint8)
    u8[~mask] = 0
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


_BOX_COLORS_BGR = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 165, 255),
    (255, 0, 255),
    (147, 20, 255),
]


def _annotate_left(
    left_bgr: np.ndarray,
    boxes_depths: list[tuple[Any, ...]],
    extra_lines: list[str] | None = None,
    *,
    debug_overlay_enabled: bool = True,
) -> np.ndarray:
    """Draw boxes plus camera/world coordinates, when available."""
    vis = left_bgr.copy()
    overlay_bgr = (0, 0, 255)
    lines: list[str] = []

    if not boxes_depths:
        lines.append("no detection")
    else:
        for i, item in enumerate(boxes_depths):
            bbox, est_a, est_b = item[:3]
            world_est = item[3] if len(item) > 3 else None
            x1, y1, x2, y2 = map(int, bbox.xyxy)
            color = _BOX_COLORS_BGR[i % len(_BOX_COLORS_BGR)]
            cx_f = (x1 + x2) * 0.5
            cy_f = (y1 + y2) * 0.5
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            bcx = int((x1 + x2) * 0.5)
            bcy = int((y1 + y2) * 0.5)
            cv2.circle(vis, (bcx, bcy), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            lbl = (bbox.label or "?").replace("\n", " ")[:20]
            tip = [f"#{i}", lbl]
            if est_a is not None and getattr(est_a, "valid", False):
                tip.append(f"A_Z={est_a.Z:.2f}")
            if est_b is not None and getattr(est_b, "valid", False):
                tip.append(f"B_Z={est_b.Z:.2f}")
            if world_est is not None and getattr(world_est, "valid", False):
                tip.append(f"WZ={world_est.Z:.2f}")
            ty = max(14, y1 - 8)
            cv2.putText(
                vis,
                " ".join(tip)[:90],
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                1,
                cv2.LINE_AA,
            )
            fs, tk = 1.35, 3
            font_face = cv2.FONT_HERSHEY_DUPLEX
            if world_est is not None and getattr(world_est, "valid", False):
                xyz_line = (
                    f"W X={world_est.X:.2f}  Y={world_est.Y:.2f}  Z={world_est.Z:.2f} m"
                )
            elif est_a is not None and getattr(est_a, "valid", False):
                xyz_line = (
                    f"X={est_a.X:.2f}  Y={est_a.Y:.2f}  Z={est_a.Z:.2f} m"
                )
            elif est_b is not None and getattr(est_b, "valid", False):
                xyz_line = (
                    f"X={est_b.X:.2f}  Y={est_b.Y:.2f}  Z={est_b.Z:.2f} m"
                )
            else:
                xyz_line = "X=--- Y=--- Z=---"
            xyz_line = xyz_line[:80]
            if debug_overlay_enabled:
                (tw, th), bl = cv2.getTextSize(xyz_line, font_face, fs, tk)
                tx = int(round(cx_f - tw * 0.5))
                ty = int(round(cy_f + th * 0.5 - bl))
                for ox2, oy2 in (
                    (-3, -3),
                    (-3, 3),
                    (3, -3),
                    (3, 3),
                    (-4, 0),
                    (4, 0),
                    (0, -4),
                    (0, 4),
                ):
                    cv2.putText(
                        vis,
                        xyz_line,
                        (tx + ox2, ty + oy2),
                        font_face,
                        fs,
                        (0, 0, 0),
                        tk + 1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    vis,
                    xyz_line,
                    (tx, ty),
                    font_face,
                    fs,
                    color,
                    tk,
                    cv2.LINE_AA,
                )

        est_a0, est_b0 = boxes_depths[0][1], boxes_depths[0][2]
        world0 = boxes_depths[0][3] if len(boxes_depths[0]) > 3 else None
        if world0 is not None and getattr(world0, "valid", False):
            src = ",".join(str(x) for x in getattr(world0, "source_tag_ids", ()))
            lines.append(f"[#0] W=({world0.X:.3f},{world0.Y:.3f},{world0.Z:.3f}) tags={src}"[:100])
        elif world0 is not None:
            lines.append(f"[#0] W invalid {getattr(world0, 'notes', '')}"[:100])
        if est_a0 is not None and getattr(est_a0, "valid", False):
            disp_a = (
                f"{est_a0.disparity:.2f}px"
                if est_a0.disparity is not None
                else "---"
            )
            lines.append(f"[#0] A Z={est_a0.Z:.3f}  disp={disp_a}")
        else:
            notes_a = getattr(est_a0, "notes", "") if est_a0 is not None else ""
            lines.append(f"[#0] A invalid {notes_a}"[:80])
        if est_b0 is not None and getattr(est_b0, "valid", False):
            disp_b = (
                f"{est_b0.disparity:.2f}px"
                if est_b0.disparity is not None
                else "---"
            )
            lines.append(f"[#0] B Z={est_b0.Z:.3f}  disp={disp_b}")
        else:
            notes_b = getattr(est_b0, "notes", "") if est_b0 is not None else ""
            lines.append(f"[#0] B: {notes_b}"[:80])
        if len(boxes_depths) > 1:
            lines.append(f"{len(boxes_depths)} objects (see labels on boxes)")
    if extra_lines and debug_overlay_enabled:
        lines.extend(extra_lines)
    for i, line in enumerate(lines if debug_overlay_enabled else []):
        cv2.putText(
            vis,
            line,
            (8, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            overlay_bgr,
            1,
            cv2.LINE_AA,
        )
    return vis


def _preview_scale_visual(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _vstack_match_width(top_bgr: np.ndarray, bottom_bgr: np.ndarray) -> np.ndarray:
    """Stack ``bottom`` under ``top``, resizing bottom width to match top."""
    tw = top_bgr.shape[1]
    h2, w2 = bottom_bgr.shape[:2]
    if w2 != tw:
        nh = max(1, int(round(h2 * tw / float(w2))))
        bottom_bgr = cv2.resize(bottom_bgr, (tw, nh), interpolation=cv2.INTER_AREA)
    return np.vstack([top_bgr, bottom_bgr])


def _snapshot_session(
    dest_dir: Path,
    seq: list[int],
    frame_loop_idx: int,
    stereo_pre_rectify: StereoFrame,
    *,
    overlay_rgb: np.ndarray | None = None,
    overlay_depth: np.ndarray | None = None,
    overlay_merged: np.ndarray | None = None,
) -> None:
    """
    Saves left/right BGR (replay / raw) plus optional preview overlays (annotated RGB, depth map).

    Same basename ``{seq:04d}_f{frame:05d}.png`` under each subfolder.
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
    saved: list[str] = [f"left/{name}", f"right/{name}"]

    if overlay_rgb is not None:
        rgb_dir = dest_dir / "overlay_rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rgb_dir / name), overlay_rgb)
        saved.append(f"overlay_rgb/{name}")
    if overlay_depth is not None:
        depth_dir = dest_dir / "overlay_depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(depth_dir / name), overlay_depth)
        saved.append(f"overlay_depth/{name}")
    if overlay_merged is not None:
        merged_dir = dest_dir / "overlay_merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(merged_dir / name), overlay_merged)
        saved.append(f"overlay_merged/{name}")

    print(f"Saved snapshot #{n}: {', '.join(saved)}")


def _snapshot_image_folder_pair(
    dest_dir: Path,
    seq: list[int],
    frame_loop_idx: int,
    stereo_pre_rectify: StereoFrame,
) -> None:
    """Legacy alias: left/right only (no overlay)."""
    _snapshot_session(dest_dir, seq, frame_loop_idx, stereo_pre_rectify)


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
    S saves another snapshot (left/right + preview overlays) without advancing.
    """
    poll_ms = max(1, int(preview_cfg.get("image_folder_hold_poll_ms", 50)))
    print(
        "[hold] Space or n = next pair  |  Q = quit session  |  S = save snapshot again",
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
                _snapshot_session(
                    sess,
                    sq,
                    loop_idx,
                    stereo_pre_rectify,
                    overlay_rgb=preview_state.get("last_overlay_rgb"),
                    overlay_depth=preview_state.get("last_overlay_depth"),
                    overlay_merged=preview_state.get("last_overlay_merged"),
                )
            else:
                print("Snapshots disabled (no snapshot session directory).")


def _preview_needs_disparity_map(preview_cfg: dict[str, Any]) -> bool:
    """True when ``_preview_tick`` will call ``_disparity_colormap_bgr`` (needs disparity array)."""
    wins = preview_cfg.get("windows")
    show_stack_disp = bool(preview_cfg.get("stack_disparity_below", False))
    if isinstance(wins, dict):
        show_combined = bool(wins.get("combined", True))
        show_disp = bool(wins.get("disparity", True))
    else:
        show_combined = True
        show_disp = bool(preview_cfg.get("show_disparity", True))
    return bool(show_disp or (show_stack_disp and show_combined))


def _preview_tick(
    preview_cfg: dict[str, Any],
    rect: StereoFrame,
    disparity: np.ndarray | None,
    boxes_depths: list[tuple[Any, ...]],
    loop_idx: int,
    preview_state: dict[str, Any],
    stereo_pre_rectify: StereoFrame,
    overlay_left_bgr: np.ndarray | None = None,
    extra_lines: list[str] | None = None,
    scalar_depth_m: np.ndarray | None = None,
    post_annotate: Any | None = None,
    key_handler: Any | None = None,
    debug_overlay_enabled: bool = True,
) -> bool:
    """
    미리보기: **창 2개** — (1) ``preview.combined_side_by_side_stereo`` 가 참이면 정류 ``좌|우`` 가로 합성,
    거짓이면 정류 **좌안 RGB 한 장**(YOLO와 동일 뷰). (2) disparity 컬러.

    ``preview.stack_disparity_below: true`` 이면 (1)(2)를 세로 한 장으로 합쳐 단일 창(호환용).

    Keys: Q quit; S saves ``left/``, ``right/``, and preview overlays (``overlay_rgb/``, etc.).
    Optional ``key_handler(key)`` receives other keys (MVP DEMO preview uses C/R/O/P).

    Returns True if user pressed Q (quit session). After return, callers may invoke
    ``_preview_image_folder_hold_spin`` when ``preview.image_folder_hold_until_quit`` is set.
    """
    scale = float(preview_cfg.get("scale", 1.0))

    wins = preview_cfg.get("windows")
    show_stack_disp = bool(preview_cfg.get("stack_disparity_below", False))
    if isinstance(wins, dict):
        show_combined = bool(wins.get("combined", True))
        show_disp = bool(wins.get("disparity", True))
    else:
        show_combined = True
        show_disp = bool(preview_cfg.get("show_disparity", True))

    ttl = preview_cfg.get("titles") if isinstance(preview_cfg.get("titles"), dict) else {}
    side_by_side = bool(preview_cfg.get("combined_side_by_side_stereo", False))
    wt_combo_default = "stereo-3d-poc | LR rectified" if side_by_side else "stereo-3d-poc | left RGB"
    wt_combo = str(ttl.get("combined", wt_combo_default))
    wt_disp = str(ttl.get("disparity", preview_cfg.get("window_disparity", "stereo-3d-poc | disparity")))
    wt_merged_default = (
        "stereo-3d-poc | LR + disparity" if side_by_side else "stereo-3d-poc | left RGB + disparity"
    )
    wt_merged = str(ttl.get("merged", wt_merged_default))

    hl, wl = rect.left_bgr.shape[:2]
    lb = overlay_left_bgr if overlay_left_bgr is not None else rect.left_bgr
    overlay_left_full = _annotate_left(
        lb,
        boxes_depths,
        extra_lines=extra_lines,
        debug_overlay_enabled=debug_overlay_enabled,
    )
    if callable(post_annotate):
        overlay_left_full = post_annotate(overlay_left_full)
    combined_visual = (
        np.hstack([overlay_left_full, rect.right_bgr]) if side_by_side else overlay_left_full
    )

    disp_color_full: np.ndarray | None = None
    if show_disp or (show_stack_disp and show_combined):
        if scalar_depth_m is not None:
            disp_color_full = _depth_m_colormap_bgr(scalar_depth_m, (hl, wl), preview_cfg, preview_state)
        else:
            disp_color_full = _disparity_colormap_bgr(disparity, (hl, wl), preview_cfg, preview_state)

    merged_full: np.ndarray | None = None
    panels: list[tuple[np.ndarray, str]] = []
    # --- 예전 4창 구성 중 좌·우 단독 창 (비표시). LR 합성은 wt_combo 창에서만.
    # wt_left = "stereo-3d-poc | left (rectified)"
    # wt_right = "stereo-3d-poc | right (rectified)"
    # if show_left:
    #     panels.append((_preview_scale_visual(overlay_left_full, scale), wt_left))
    # if show_right:
    #     panels.append((_preview_scale_visual(rect.right_bgr.copy(), scale), wt_right))
    if show_combined and show_disp and show_stack_disp and disp_color_full is not None:
        merged_full = _vstack_match_width(combined_visual, disp_color_full)
        panels.append((_preview_scale_visual(merged_full, scale), wt_merged))
    elif show_combined and show_disp and disp_color_full is not None:
        panels.append((_preview_scale_visual(combined_visual, scale), wt_combo))
        panels.append((_preview_scale_visual(disp_color_full, scale), wt_disp))
    elif show_combined:
        panels.append((_preview_scale_visual(combined_visual, scale), wt_combo))
    elif show_disp and disp_color_full is not None:
        panels.append((_preview_scale_visual(disp_color_full, scale), wt_disp))

    preview_state["last_overlay_rgb"] = combined_visual if show_combined else None
    preview_state["last_overlay_depth"] = disp_color_full if show_disp else None
    preview_state["last_overlay_merged"] = merged_full

    for img, title in panels:
        _preview_ensure_named(title, preview_cfg, scale)
        cv2.imshow(title, img)

    wait_ms = max(1, int(preview_cfg.get("wait_key_ms", 1)))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (ord("q"), ord("Q")):
        return True
    if key in (ord("s"), ord("S")):
        sess = preview_state.get("snapshot_session_dir")
        sq = preview_state.get("snap_seq")
        if isinstance(sess, Path) and isinstance(sq, list):
            _snapshot_session(
                sess,
                sq,
                loop_idx,
                stereo_pre_rectify,
                overlay_rgb=preview_state.get("last_overlay_rgb"),
                overlay_depth=preview_state.get("last_overlay_depth"),
                overlay_merged=preview_state.get("last_overlay_merged"),
            )
        else:
            print("Snapshots disabled (no snapshot session directory).")
    elif callable(key_handler):
        key_handler(key)
    return False


def _apriltag_csv_fields(at: AprilTagScaleOutcome | None, enabled: bool) -> dict[str, Any]:
    if not enabled or at is None:
        return {
            "apriltag_scale": "",
            "apriltag_meas_m": "",
            "apriltag_tag_ids": "",
            "apriltag_note": "",
        }
    return {
        "apriltag_scale": at.scale if at.scale is not None else "",
        "apriltag_meas_m": at.measured_distance_m if at.measured_distance_m is not None else "",
        "apriltag_tag_ids": f"{at.tag_ids[0]},{at.tag_ids[1]}" if at.tag_ids else "",
        "apriltag_note": (at.notes or "")[:200],
    }


def _apriltag_extra_overlay(at: AprilTagScaleOutcome | None, enabled: bool) -> list[str]:
    if not enabled or at is None:
        return []
    if at.scale is not None:
        return [
            f"APRIL ids={at.tag_ids} meas={at.measured_distance_m:.3f}m "
            f"tgt={at.known_spacing_m:.3f}m scale={at.scale:.4f}",
        ]
    if at.measured_distance_m is not None:
        return [
            f"APRIL(rgbd) ids={at.tag_ids} meas={at.measured_distance_m:.3f}m "
            f"tgt={at.known_spacing_m:.3f}m {at.notes}"[:140],
        ]
    return [f"APRIL: {at.notes}"[:120]]


_WORLD_CSV_FIELDNAMES = [
    "world_valid",
    "world_X",
    "world_Y",
    "world_Z",
    "world_source_tag_ids",
    "world_tag_count",
    "world_note",
]


def _world_csv_fields(
    world: WorldPointEstimate | None,
    pose: AprilTagWorldResult | None,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {name: "" for name in _WORLD_CSV_FIELDNAMES}

    tag_count: int | str = len(pose.observations) if pose is not None else 0
    if world is not None and world.valid:
        return {
            "world_valid": True,
            "world_X": world.X,
            "world_Y": world.Y,
            "world_Z": world.Z,
            "world_source_tag_ids": ",".join(str(x) for x in world.source_tag_ids),
            "world_tag_count": tag_count,
            "world_note": world.notes[:200],
        }

    note = ""
    if world is not None:
        note = world.notes
    elif pose is not None:
        note = pose.notes
    return {
        "world_valid": False,
        "world_X": "",
        "world_Y": "",
        "world_Z": "",
        "world_source_tag_ids": "",
        "world_tag_count": tag_count,
        "world_note": note[:200],
    }


def _apriltag_world_extra_overlay(
    pose: AprilTagWorldResult | None,
    enabled: bool,
) -> list[str]:
    if not enabled:
        return []
    if pose is None:
        return ["WORLD: not computed"]
    if not pose.observations:
        return [f"WORLD: {pose.notes}"[:120]]
    ids = ",".join(str(x) for x in pose.visible_tag_ids)
    err = min(obs.reprojection_error_px for obs in pose.observations)
    return [f"WORLD tags={ids} reproj={err:.2f}px"[:120]]


def _preview_display_world(
    operational_world: WorldPointEstimate | None,
    view: PreviewFrameView | None,
) -> WorldPointEstimate | None:
    """Visual-only representative world point. Operational CSV source is unchanged."""
    if view is None or view.display_source != "OBJECT_ANCHOR_PREVIEW":
        return operational_world
    point = view.p_world_cup_object_m
    if point is None:
        return WorldPointEstimate(valid=False, notes="object_anchor_preview_unavailable")
    return WorldPointEstimate(
        X=float(point[0]),
        Y=float(point[1]),
        Z=float(point[2]),
        valid=True,
        source_tag_ids=(),
        notes="mvp_demo_object_anchor_preview_display_only",
    )


def _oa_preview_hooks(
    oa_preview: ObjectAnchorPreviewSession | None,
    view: PreviewFrameView | None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    axis_length_m: float = 0.08,
) -> tuple[Any | None, Any | None]:
    if oa_preview is None:
        return None, None

    def _post(image: np.ndarray) -> np.ndarray:
        if view is None:
            return image
        if camera_matrix is not None:
            image = draw_object_preview_axes(
                image,
                view,
                camera_matrix,
                dist_coeffs,
                oa_preview.settings,
                axis_length_m=axis_length_m,
            )
        return draw_preview_banner(image, view)

    def _keys(key: int) -> None:
        action = oa_preview.handle_key(key)
        if action:
            print(f"[MVP DEMO PREVIEW] {action}")

    return _post, _keys


def build_detector(cfg: dict):
    d = cfg.get("detector", {})
    kind = (d.get("kind") or "yolo").lower()
    if kind == "dummy":
        return DummyCenterDetector(frac=float(d.get("frac", 0.2)))
    class_ids_raw = d.get("class_ids", d.get("classes"))
    class_ids = [int(x) for x in class_ids_raw] if class_ids_raw else None
    return UltralyticsYOLODetector(
        model_path=d.get("model_path", "yolo11s.pt"),
        conf_threshold=float(d.get("conf", 0.25)),
        iou_threshold=float(d.get("iou", 0.45)),
        imgsz=d.get("imgsz", 640),
        device=d.get("device"),
        class_ids=class_ids,
    )


def _describe_rgbd_calibration(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[str]:
    h, w = rgb.shape[:2]
    dh, dw = depth_m.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    issues: list[str] = []
    if (dh, dw) != (h, w):
        issues.append(f"rgb_depth_shape_mismatch rgb={w}x{h} depth={dw}x{dh}")
    if not np.all(np.isfinite(K)) or fx <= 0.0 or fy <= 0.0:
        issues.append("invalid_rgb_intrinsic")
    if not (0.0 <= cx < w and 0.0 <= cy < h):
        issues.append(f"principal_point_outside_rgb cx={cx:.2f} cy={cy:.2f}")
    valid_depth = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    median_depth = float(np.median(valid_depth)) if valid_depth.size else float("nan")
    if not valid_depth.size:
        issues.append("no_valid_depth")
    elif median_depth > 100.0:
        issues.append(f"depth_probably_not_meters median={median_depth:.3f}")
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    status = "OK" if not issues else "WARN " + "; ".join(issues)
    return [
        f"RGB calibration: {w}x{h} fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}",
        f"RGB distortion ({len(distortion)}): "
        + np.array2string(distortion, precision=5, suppress_small=True),
        f"RGB-depth alignment: rgb={w}x{h} depth={dw}x{dh}; depth median={median_depth:.3f}m",
        f"RGB-D calibration check: {status}",
    ]


def run_session(
    cfg_path: Path,
    *,
    register_object_anchor: bool = False,
    capture_type: str | None = None,
    capture_count: int = 100,
    capture_interval: float = 1.0,
) -> Path:
    if capture_type is not None and register_object_anchor:
        raise SystemExit("Capture mode cannot be combined with Object Anchor registration")
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    inp = cfg.get("input") or {}
    in_type = (inp.get("type") or "camera").lower()
    if in_type == "orbbec":
        if capture_type is not None:
            return _run_orbbec_capture_session(
                cfg,
                capture_type=capture_type,
                capture_count=capture_count,
                capture_interval=capture_interval,
            )
        return _run_session_orbbec(
            cfg_path, cfg, register_object_anchor=register_object_anchor
        )
    if capture_type is not None:
        raise SystemExit("Object Anchor capture mode requires input.type: orbbec")
    return _run_session_stereo(cfg_path, cfg)


def _draw_capture_status(
    bgr: np.ndarray,
    *,
    capture_type: str,
    saved_count: int,
    target_count: int,
    interval_seconds: float,
    last_filename: str,
) -> np.ndarray:
    overlay = bgr.copy()
    lines = (
        f"CAPTURE MODE: {capture_type.upper()}",
        f"Saved: {saved_count} / {target_count}",
        f"Interval: {interval_seconds:g} sec",
        f"Last file: {last_filename}",
        "Q / ESC: stop",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    line_height = 34
    panel_width = min(bgr.shape[1] - 20, 780)
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 30 + line_height * len(lines)), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        color = (80, 255, 80) if index < 2 else (255, 255, 255)
        cv2.putText(
            overlay,
            line[:100],
            (24, 42 + index * line_height),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return overlay


def _capture_window_closed(title: str) -> bool:
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1.0
    except cv2.error:
        return False


def _run_orbbec_capture_session(
    cfg: dict[str, Any],
    *,
    capture_type: str,
    capture_count: int,
    capture_interval: float,
) -> Path:
    """Dedicated timed raw-RGB collection path; it does not enter the 300-frame run."""
    from orbbec_rgbd_capture import OrbbecRGBDCapture  # noqa: PLC0415

    ob_cfg = cfg.get("orbbec")
    if not isinstance(ob_cfg, dict):
        raise SystemExit("orbbec: configuration block is required for capture mode")
    try:
        settings = ObjectAnchorCaptureSettings(
            capture_type=capture_type,
            target_count=int(capture_count),
            interval_seconds=float(capture_interval),
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid capture settings: {exc}") from exc

    object_anchor_raw = cfg.get("object_anchor") or {}
    object_anchor_runtime, object_anchor_status = build_optional_object_anchor_runtime(
        object_anchor_raw, repo_root=_REPO_ROOT
    )
    model_path = ""
    if object_anchor_runtime is not None:
        resolved_model = _resolve_repo_path(object_anchor_raw.get("model_path"))
        model_path = str(resolved_model) if resolved_model is not None else ""
    atw_cfg: AprilTagWorldConfig = build_apriltag_world_config(cfg.get("apriltag_world") or {})

    cap = OrbbecRGBDCapture(ob_cfg)
    cap.start()
    capture_root = _resolve_repo_path("data/object_anchor_capture")
    assert capture_root is not None
    session = ObjectAnchorCaptureSession(
        capture_root,
        settings,
        camera_serial=cap.serial_number,
        loaded_model_path=model_path,
    )
    title = f"stereo-3d-poc | capture {capture_type}"
    preview_cfg = dict(cfg.get("preview") or {})
    preview_scale = float(preview_cfg.get("scale", 1.0))
    startup_timeout_s = max(0.0, float(ob_cfg.get("startup_timeout_s", 20.0)))
    start_time = time.perf_counter()
    read_failures = 0
    print(
        f"Object Anchor capture started: type={capture_type} target={capture_count} "
        f"interval={capture_interval:g}s"
    )
    print(f"  images: {session.image_dir}")
    if capture_type == "negative":
        print(f"  empty labels: {session.label_dir}")
    print(f"  manifest: {session.manifest_path}")
    print(f"  Object Anchor metadata: {object_anchor_status}")

    try:
        _preview_ensure_named(title, preview_cfg, preview_scale)
        while not session.complete:
            ok, frame = cap.read_rgbd()
            if not ok or frame is None:
                read_failures += 1
                if (
                    session.saved_count == 0
                    and startup_timeout_s > 0.0
                    and time.perf_counter() - start_time >= startup_timeout_s
                ):
                    raise SystemExit(
                        "Orbbec: no synchronized RGB-D frame within "
                        f"{startup_timeout_s:.1f}s during capture."
                    )
                continue

            raw_bgr = frame.bgr
            gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
            apriltag_result = (
                estimate_apriltag_world(gray, frame.K, atw_cfg)
                if atw_cfg.enabled
                else None
            )
            anchor_result = (
                object_anchor_runtime.process(raw_bgr, frame.K, frame.dist_coeffs)
                if object_anchor_runtime is not None
                else None
            )
            detection = anchor_result.detection if anchor_result is not None else None
            now = time.monotonic()
            saved = session.save(
                raw_bgr,
                now_monotonic=now,
                object_anchor_detected=detection is not None,
                object_anchor_confidence=detection.score if detection is not None else None,
                apriltag_detected=bool(apriltag_result and apriltag_result.observations),
            )
            if saved is not None:
                print(f"  saved {session.saved_count}/{settings.target_count}: {saved.name}")

            display = _draw_capture_status(
                raw_bgr,
                capture_type=capture_type,
                saved_count=session.saved_count,
                target_count=settings.target_count,
                interval_seconds=settings.interval_seconds,
                last_filename=session.last_filename,
            )
            cv2.imshow(title, _preview_scale_visual(display, preview_scale))
            key = cv2.waitKey(max(1, int(preview_cfg.get("wait_key_ms", 1)))) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or _capture_window_closed(title):
                print(f"Capture stopped early at {session.saved_count}/{settings.target_count}.")
                break
    except KeyboardInterrupt:
        print(f"Capture interrupted at {session.saved_count}/{settings.target_count}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        _preview_reset_registered()

    if session.complete:
        print(f"Capture complete: exactly {session.saved_count} image(s) saved.")
    print(f"Manifest preserved at {session.manifest_path}")
    return session.manifest_path


def _run_session_orbbec(
    cfg_path: Path,
    cfg: dict,
    *,
    register_object_anchor: bool = False,
) -> Path:
    from orbbec_rgbd_capture import OrbbecRGBDCapture, placeholder_stereo_frames  # noqa: PLC0415

    ob_cfg = cfg.get("orbbec")
    if not isinstance(ob_cfg, dict):
        raise SystemExit("orbbec: configuration block (mapping) is required for input.type: orbbec")

    detector = build_detector(cfg)
    rep = cfg.get("repeatability", {})
    warmup = int(rep.get("warmup_frames", 10))
    log_all_boxes = bool(rep.get("log_all_boxes", True))
    max_boxes_per_frame = max(1, int(rep.get("max_boxes_per_frame", 64)))
    preview_cfg = dict(cfg.get("preview") or {})
    preview_enabled = bool(preview_cfg.get("enabled", False))
    max_frames = int(rep.get("max_frames", 300))
    out_csv = _resolve_repo_path(rep.get("output_csv", "out/repeatability_orbbec.csv"))
    assert out_csv is not None
    out_csv = _resolve_writable_csv(out_csv)

    at_cfg = cfg.get("apriltag_scale") or {}
    apriltag_enabled = bool(at_cfg.get("enabled"))
    apply_at_scale = apriltag_enabled and bool(at_cfg.get("apply_scale_to_depth", True))
    atw_cfg: AprilTagWorldConfig = build_apriltag_world_config(cfg.get("apriltag_world") or {})
    atw_enabled = atw_cfg.enabled
    object_anchor_runtime, object_anchor_status = build_optional_object_anchor_runtime(
        cfg.get("object_anchor"), repo_root=_REPO_ROOT
    )
    object_anchor_raw = cfg.get("object_anchor") or {}
    world_tracker: ObjectAnchorWorldTracker | None = None
    if object_anchor_runtime is not None and atw_enabled:
        world_raw = object_anchor_raw.get("world_validation") or {}
        if bool(world_raw.get("enabled", True)):
            registration_path = _resolve_repo_path(
                object_anchor_raw.get("registration_file")
                or "out/object_anchor_calibration/tissue_box_01_world_pose.yaml"
            )
            session_root = _resolve_repo_path(
                world_raw.get("session_dir", "out/object_anchor_world")
            )
            assert registration_path is not None and session_root is not None
            world_session = session_root / datetime.now().strftime("%Y%m%d_%H%M%S")
            world_tracker = ObjectAnchorWorldTracker(
                object_id=object_anchor_runtime.config.object_id,
                keypoint_names=tuple(
                    keypoint.name for keypoint in object_anchor_runtime.config.keypoints
                ),
                settings=build_world_settings(world_raw),
                registration_file=registration_path,
                session_dir=world_session,
                start_registration=register_object_anchor,
            )

    preview_settings = load_preview_settings(cfg.get("object_anchor_preview"))
    oa_preview: ObjectAnchorPreviewSession | None = None
    if (
        preview_settings.enabled
        and object_anchor_runtime is not None
        and atw_enabled
    ):
        model_path_str = str(
            _resolve_repo_path(object_anchor_raw.get("model_path"))
            or object_anchor_raw.get("model_path")
            or ""
        )
        config_path_str = str(
            _resolve_repo_path(object_anchor_raw.get("config_path"))
            or object_anchor_raw.get("config_path")
            or ""
        )
        oa_preview = build_preview_session(
            preview_settings,
            repo_root=_REPO_ROOT,
            model_path=model_path_str,
            config_path=config_path_str,
        )
        if preview_settings.max_frames_override is not None:
            max_frames = int(preview_settings.max_frames_override)

    z_min_roi = float(ob_cfg.get("roi_z_min_m", 0.05))
    z_max_roi = float(ob_cfg.get("roi_z_max_m", 40.0))
    min_valid_ratio = float(ob_cfg.get("min_valid_depth_ratio", 0.03))
    startup_timeout_s = max(0.0, float(ob_cfg.get("startup_timeout_s", 20.0)))

    cap = OrbbecRGBDCapture(ob_cfg)
    cap.start()
    print("Orbbec pipeline started (preview opens after the first synchronized RGB+depth frame).")

    preview_state: dict[str, Any] = {}
    print(
        f"Orbbec RGB-D CSV rows: {'all detections per frame' if log_all_boxes else 'primary (max-conf) only'} "
        f"(max {max_boxes_per_frame} boxes/frame)."
    )
    if atw_enabled:
        print(
            f"AprilTag world coordinates enabled: tags={sorted(atw_cfg.tags)} "
            f"tag_size={atw_cfg.tag_size_m:.3f}m"
        )
    if object_anchor_runtime is None:
        print(f"Object Anchor camera-pose debug disabled: {object_anchor_status}")
    else:
        print(
            f"Object Anchor camera-pose debug enabled: "
            f"{object_anchor_runtime.config.object_id} camera_pose_only="
            f"{object_anchor_runtime.settings.camera_pose_only}"
        )
    if world_tracker is not None:
        print("Object Anchor AprilTag world validation enabled (comparison only).")
        print(f"  transform: {TRANSFORM_FORMULA}")
        print(f"  frame log: {world_tracker.csv_path}")
        print(f"  registration file: {world_tracker.registration_file}")
        print(f"  registration active: {world_tracker.registration is not None}")
    if oa_preview is not None:
        print("Object Anchor MVP DEMO / PREVIEW enabled (session calibration only).")
        print("  Does NOT replace operational AprilTag world source.")
        print(
            f"  Keys: [{preview_settings.start_calibration_key.upper()}]=session calibrate  "
            f"[{preview_settings.reset_calibration_key.upper()}]=reset  "
            f"[{preview_settings.switch_display_source_key.upper()}]=display source  "
            f"[{preview_settings.toggle_panel_key.upper()}]=panel"
        )
        if oa_preview.output_dir is not None:
            print(f"  preview logs: {oa_preview.output_dir}")

    if preview_enabled:
        print(
            "Preview: focus an OpenCV window — [Q]=quit session  |  "
            "[S]=save snapshot (left/right + overlay_rgb/overlay_depth)."
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
        "det_idx",
        "class_id",
        "label",
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
        *_WORLD_CSV_FIELDNAMES,
        "apriltag_scale",
        "apriltag_meas_m",
        "apriltag_tag_ids",
        "apriltag_note",
    ]

    t0 = time.perf_counter()
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            idx = 0
            read_fail_streak = 0
            first_frame_logged = False
            calibration_logged = False
            previous_frame_time: float | None = None
            fps_ema = 0.0
            while idx < max_frames:
                t_cap0 = time.perf_counter()
                ok, fr = cap.read_rgbd()
                capture_ms = (time.perf_counter() - t_cap0) * 1000.0
                if not ok or fr is None:
                    read_fail_streak += 1
                    startup_elapsed = time.perf_counter() - t0
                    if read_fail_streak == 1:
                        print(
                            "Waiting for synchronized RGB+depth frame "
                            "(no OpenCV window until the first good frame)..."
                        )
                    elif read_fail_streak in (50, 200, 500) or (
                        read_fail_streak > 500 and read_fail_streak % 1000 == 0
                    ):
                        print(
                            f"  still waiting ({read_fail_streak} read attempts, "
                            f"{time.perf_counter() - t0:.0f}s elapsed)"
                        )
                    if (
                        not first_frame_logged
                        and startup_timeout_s > 0.0
                        and startup_elapsed >= startup_timeout_s
                    ):
                        raise SystemExit(
                            "Orbbec: no synchronized RGB-D frame within "
                            f"{startup_timeout_s:.1f}s. Close Orbbec Viewer/other camera "
                            "processes, reconnect USB power, and run "
                            "scripts/orbbec_smoke_test.py."
                        )
                    continue
                if not first_frame_logged:
                    if read_fail_streak:
                        print(f"First RGB-D frame after {read_fail_streak} failed read(s).")
                    first_frame_logged = True
                read_fail_streak = 0
                frame_now = time.perf_counter()
                if previous_frame_time is not None:
                    instantaneous_fps = 1.0 / max(frame_now - previous_frame_time, 1e-6)
                    fps_ema = instantaneous_fps if fps_ema <= 0.0 else 0.90 * fps_ema + 0.10 * instantaneous_fps
                previous_frame_time = frame_now

                rgb = fr.bgr
                depth_m = fr.depth_m
                K = fr.K
                if not calibration_logged:
                    for line in _describe_rgbd_calibration(
                        rgb, depth_m, K, fr.dist_coeffs
                    ):
                        print(line)
                    calibration_logged = True
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                placeholder_l, placeholder_r = placeholder_stereo_frames(rgb)
                frame_stereo = StereoFrame(left_bgr=placeholder_l, right_bgr=placeholder_r)
                rect = frame_stereo

                t_det0 = time.perf_counter()
                dets = detector.predict(rgb)
                det_ms = (time.perf_counter() - t_det0) * 1000.0

                t_depth0 = time.perf_counter()
                at_outcome: AprilTagScaleOutcome | None = None
                atw_result: AprilTagWorldResult | None = None
                object_anchor_result: ObjectAnchorFrameResult | None = None
                object_anchor_lines: list[str] = []
                left_vis_bgr: np.ndarray | None = None
                if apriltag_enabled:
                    draw_tags = bool(at_cfg.get("draw", True))
                    if draw_tags:
                        left_vis_bgr = rgb.copy()
                    at_outcome = compute_apriltag_distance_validation_rgbd(
                        gray,
                        depth_m,
                        K,
                        dictionary=str(at_cfg.get("dictionary", "APRILTAG_36H11")),
                        known_spacing_m=float(at_cfg.get("known_spacing_m", 1.0)),
                        tag_id_a=at_cfg.get("tag_id_a"),
                        tag_id_b=at_cfg.get("tag_id_b"),
                        sample_radius=int(at_cfg.get("sample_radius", 2)),
                        z_min_m=float(at_cfg.get("min_depth_m", z_min_roi)),
                        z_max_m=float(at_cfg.get("max_depth_m", z_max_roi)),
                        draw_on_bgr=left_vis_bgr if draw_tags else None,
                    )
                if atw_enabled:
                    if atw_cfg.draw and left_vis_bgr is None:
                        left_vis_bgr = rgb.copy()
                    atw_result = estimate_apriltag_world(
                        gray,
                        K,
                        atw_cfg,
                        draw_on_bgr=left_vis_bgr if atw_cfg.draw else None,
                    )
                if object_anchor_runtime is not None:
                    preview_debug = (
                        oa_preview.debug_overlay_enabled
                        if oa_preview is not None
                        else True
                    )
                    object_anchor_result = object_anchor_runtime.process(
                        rgb,
                        K,
                        fr.dist_coeffs,
                        draw_on_bgr=left_vis_bgr,
                        debug_overlay=preview_debug,
                        draw_pose_axis=oa_preview is None,
                    )
                    left_vis_bgr = object_anchor_result.overlay_bgr
                    object_anchor_lines = object_anchor_runtime.overlay_lines(
                        object_anchor_result
                    )
                if world_tracker is not None:
                    if left_vis_bgr is None:
                        left_vis_bgr = rgb.copy()
                    _, world_lines = world_tracker.process(
                        frame_idx=idx,
                        fps=fps_ema,
                        apriltag_result=atw_result,
                        anchor_result=object_anchor_result,
                        raw_bgr=rgb,
                        overlay_bgr=left_vis_bgr,
                    )
                    object_anchor_lines.extend(world_lines)

                prim = pick_primary_box(dets)
                preview_view: PreviewFrameView | None = None
                if prim is None:
                    if oa_preview is not None:
                        preview_view = oa_preview.update(
                            frame_idx=idx,
                            timestamp=time.perf_counter() - t0,
                            fps=fps_ema,
                            apriltag_result=atw_result,
                            anchor_result=object_anchor_result,
                            cup_estimate=None,
                            cup_detected=False,
                            overlay_bgr=left_vis_bgr,
                        )
                    disp_ms_track = (time.perf_counter() - t_depth0) * 1000.0
                    post_annotate, key_handler = _oa_preview_hooks(
                        oa_preview,
                        preview_view,
                        K,
                        fr.dist_coeffs,
                        axis_length_m=min(
                            object_anchor_runtime.config.size.values()
                        )
                        * 0.7
                        if object_anchor_runtime is not None
                        else 0.08,
                    )
                    if preview_enabled:
                        if _preview_tick(
                            preview_cfg,
                            rect,
                            None,
                            [],
                            idx,
                            preview_state,
                            frame_stereo,
                            overlay_left_bgr=left_vis_bgr,
                            extra_lines=(
                                _apriltag_extra_overlay(at_outcome, apriltag_enabled)
                                + _apriltag_world_extra_overlay(atw_result, atw_enabled)
                                + object_anchor_lines
                            ),
                            scalar_depth_m=depth_m
                            if (_preview_needs_disparity_map(preview_cfg))
                            else None,
                            post_annotate=post_annotate,
                            key_handler=key_handler,
                            debug_overlay_enabled=(
                                oa_preview.debug_overlay_enabled
                                if oa_preview is not None
                                else True
                            ),
                        ):
                            break
                    if idx >= warmup:
                        w.writerow(
                            {
                                "frame_idx": idx,
                                "det_idx": "",
                                "class_id": "",
                                "label": "",
                                "t_wall": time.perf_counter() - t0,
                                "capture_ms": capture_ms,
                                "disp_ms": disp_ms_track,
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
                                **_world_csv_fields(None, atw_result, atw_enabled),
                                **_apriltag_csv_fields(at_outcome, apriltag_enabled),
                            }
                        )
                    idx += 1
                    continue

                if log_all_boxes:
                    boxes = sorted(dets.boxes, key=lambda b: -b.confidence)[:max_boxes_per_frame]
                else:
                    boxes = [prim]

                rows_payload: list[tuple[BBox, int, Any, Any, WorldPointEstimate | None]] = []
                for bi, bbox in enumerate(boxes):
                    est_a_i = depth_estimate_rgbd_bbox(
                        depth_m,
                        bbox,
                        K,
                        min_valid_ratio=min_valid_ratio,
                        z_min_m=z_min_roi,
                        z_max_m=z_max_roi,
                    )
                    est_b_i = orbbec_sparse_stub()
                    if apply_at_scale and at_outcome is not None and at_outcome.scale is not None:
                        est_a_i = scale_depth_estimate(est_a_i, at_outcome.scale)
                        est_b_i = scale_depth_estimate(est_b_i, at_outcome.scale)
                    world_i = (
                        world_point_from_camera_estimate(est_a_i, atw_result)
                        if atw_enabled
                        else None
                    )
                    rows_payload.append((bbox, bi, est_a_i, est_b_i, world_i))

                primary_est = rows_payload[0][2]
                if oa_preview is not None:
                    preview_view = oa_preview.update(
                        frame_idx=idx,
                        timestamp=time.perf_counter() - t0,
                        fps=fps_ema,
                        apriltag_result=atw_result,
                        anchor_result=object_anchor_result,
                        cup_estimate=primary_est,
                        cup_detected=True,
                        overlay_bgr=left_vis_bgr,
                    )

                disp_ms_track = (time.perf_counter() - t_depth0) * 1000.0
                apr_lines = (
                    _apriltag_extra_overlay(at_outcome, apriltag_enabled)
                    + _apriltag_world_extra_overlay(atw_result, atw_enabled)
                    + object_anchor_lines
                )
                # Operational world stays AprilTag; on-box display may follow display_source.
                boxes_depths = [
                    (
                        b,
                        ea,
                        eb,
                        _preview_display_world(wp, preview_view)
                        if bi == 0
                        else wp,
                    )
                    for bi, (b, _, ea, eb, wp) in enumerate(rows_payload)
                ]
                post_annotate, key_handler = _oa_preview_hooks(
                    oa_preview,
                    preview_view,
                    K,
                    fr.dist_coeffs,
                    axis_length_m=min(object_anchor_runtime.config.size.values()) * 0.7
                    if object_anchor_runtime is not None
                    else 0.08,
                )

                if preview_enabled:
                    if _preview_tick(
                        preview_cfg,
                        rect,
                        None,
                        boxes_depths,
                        idx,
                        preview_state,
                        frame_stereo,
                        overlay_left_bgr=left_vis_bgr,
                        extra_lines=apr_lines,
                        scalar_depth_m=depth_m,
                        post_annotate=post_annotate,
                        key_handler=key_handler,
                        debug_overlay_enabled=(
                            oa_preview.debug_overlay_enabled
                            if oa_preview is not None
                            else True
                        ),
                    ):
                        break

                if idx >= warmup:
                    at_csv = _apriltag_csv_fields(at_outcome, apriltag_enabled)
                    for bbox, bi, est_a_i, est_b_i, world_i in rows_payload:
                        ru, rv = bbox.center
                        w.writerow(
                            {
                                "frame_idx": idx,
                                "det_idx": bi,
                                "class_id": bbox.class_id,
                                "label": bbox.label or "",
                                "t_wall": time.perf_counter() - t0,
                                "capture_ms": capture_ms,
                                "disp_ms": disp_ms_track,
                                "det_ms": det_ms,
                                "det_conf": bbox.confidence,
                                "box_x1": bbox.xyxy[0],
                                "box_y1": bbox.xyxy[1],
                                "box_x2": bbox.xyxy[2],
                                "box_y2": bbox.xyxy[3],
                                "ref_u": ru,
                                "ref_v": rv,
                                "A_valid": est_a_i.valid,
                                "A_X": est_a_i.X if est_a_i.valid else "",
                                "A_Y": est_a_i.Y if est_a_i.valid else "",
                                "A_Z": est_a_i.Z if est_a_i.valid else "",
                                "A_disp": "",
                                "A_valid_ratio": est_a_i.valid_pixel_ratio
                                if est_a_i.valid_pixel_ratio is not None
                                else "",
                                "B_valid": est_b_i.valid,
                                "B_X": est_b_i.X if est_b_i.valid else "",
                                "B_Y": est_b_i.Y if est_b_i.valid else "",
                                "B_Z": est_b_i.Z if est_b_i.valid else "",
                                "B_disp": est_b_i.disparity if est_b_i.disparity is not None else "",
                                "B_notes": est_b_i.notes,
                                **_world_csv_fields(world_i, atw_result, atw_enabled),
                                **at_csv,
                            }
                        )
                idx += 1
    finally:
        if preview_enabled:
            cv2.destroyAllWindows()
            _preview_reset_registered()
        cap.release()
        if world_tracker is not None:
            print("Object Anchor world statistics:")
            print(yaml.safe_dump(world_tracker.summary(), sort_keys=False))
            summary_path = world_tracker.close()
            print(f"Object Anchor world summary: {summary_path}")
        if oa_preview is not None:
            preview_summary = oa_preview.close()
            print("Object Anchor MVP DEMO / PREVIEW summary:")
            print(yaml.safe_dump(oa_preview.summary(), sort_keys=False))
            if preview_summary is not None:
                print(f"Object Anchor MVP DEMO preview summary: {preview_summary}")

    return out_csv


def _run_session_stereo(cfg_path: Path, cfg: dict) -> Path:
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
    log_all_boxes = bool(rep.get("log_all_boxes", True))
    max_boxes_per_frame = max(1, int(rep.get("max_boxes_per_frame", 64)))
    preview_cfg = dict(cfg.get("preview") or {})
    preview_enabled = bool(preview_cfg.get("enabled", False))
    max_frames = int(rep.get("max_frames", 300))
    out_csv = _resolve_repo_path(rep.get("output_csv", "out/repeatability.csv"))
    assert out_csv is not None
    out_csv = _resolve_writable_csv(out_csv)

    maps = calib.ensure_maps()
    Q = calib.Q

    at_cfg = cfg.get("apriltag_scale") or {}
    apriltag_enabled = bool(at_cfg.get("enabled"))
    atw_cfg: AprilTagWorldConfig = build_apriltag_world_config(cfg.get("apriltag_world") or {})
    atw_enabled = atw_cfg.enabled

    preview_state: dict[str, Any] = {}

    img_folder_hold = bool(preview_cfg.get("image_folder_hold_until_quit"))

    print(
        f"Depth CSV rows: {'all detections per frame' if log_all_boxes else 'primary (max-conf) only'} "
        f"(max {max_boxes_per_frame} boxes/frame)."
    )
    if atw_enabled:
        print(
            f"AprilTag world coordinates enabled: tags={sorted(atw_cfg.tags)} "
            f"tag_size={atw_cfg.tag_size_m:.3f}m"
        )

    if preview_enabled:
        print(
            "Preview: focus an OpenCV window (Alt+Tab) — "
            "[Q]=quit session  |  [S]=save snapshot (left/right + overlay previews) under snapshots session."
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
        "det_idx",
        "class_id",
        "label",
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
        *_WORLD_CSV_FIELDNAMES,
        "apriltag_scale",
        "apriltag_meas_m",
        "apriltag_tag_ids",
        "apriltag_note",
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
                    disp_preview: np.ndarray | None = None
                    disp_ms_no_box = 0.0
                    if preview_enabled and _preview_needs_disparity_map(preview_cfg):
                        t_pd = time.perf_counter()
                        disp_preview = compute_disparity_map(
                            gray_l, gray_r, sgbm, scale_down=scale_down
                        )
                        disp_ms_no_box = (time.perf_counter() - t_pd) * 1000.0
                    if preview_enabled:
                        if _preview_tick(
                            preview_cfg,
                            rect,
                            disp_preview,
                            [],
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
                                "det_idx": "",
                                "class_id": "",
                                "label": "",
                                "t_wall": time.perf_counter() - t0,
                                "capture_ms": capture_ms,
                                "disp_ms": disp_ms_no_box,
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
                                **_world_csv_fields(None, None, atw_enabled),
                                **_apriltag_csv_fields(None, apriltag_enabled),
                            }
                        )
                    idx += 1
                    continue

                if log_all_boxes:
                    boxes = sorted(dets.boxes, key=lambda b: -b.confidence)[:max_boxes_per_frame]
                else:
                    boxes = [prim]

                t_disp0 = time.perf_counter()
                disp = compute_disparity_map(gray_l, gray_r, sgbm, scale_down=scale_down)
                disp_ms = (time.perf_counter() - t_disp0) * 1000.0

                at_outcome: AprilTagScaleOutcome | None = None
                atw_result: AprilTagWorldResult | None = None
                left_vis_bgr: np.ndarray | None = None
                if apriltag_enabled:
                    draw_tags = bool(at_cfg.get("draw", True))
                    if draw_tags:
                        left_vis_bgr = left.copy()
                    at_outcome = compute_apriltag_metric_scale(
                        gray_l,
                        disp,
                        Q,
                        dictionary=str(at_cfg.get("dictionary", "APRILTAG_36H11")),
                        known_spacing_m=float(at_cfg.get("known_spacing_m", 1.0)),
                        tag_id_a=at_cfg.get("tag_id_a"),
                        tag_id_b=at_cfg.get("tag_id_b"),
                        sample_radius=int(at_cfg.get("sample_radius", 2)),
                        min_disp=float(at_cfg.get("min_disp", 1.0)),
                        draw_on_bgr=left_vis_bgr if draw_tags else None,
                    )
                if atw_enabled:
                    if atw_cfg.draw and left_vis_bgr is None:
                        left_vis_bgr = left.copy()
                    K_rect_left = calib.P1[:3, :3].astype(np.float64)
                    atw_result = estimate_apriltag_world(
                        gray_l,
                        K_rect_left,
                        atw_cfg,
                        draw_on_bgr=left_vis_bgr if atw_cfg.draw else None,
                    )

                apply_at_scale = apriltag_enabled and bool(at_cfg.get("apply_scale_to_depth", True))

                rows_payload: list[tuple[BBox, int, Any, Any, WorldPointEstimate | None]] = []
                for bi, bbox in enumerate(boxes):
                    est_a_i = depth_dense_track_a(disp, bbox, Q, min_disp=1.0)
                    ru, rv = bbox.center
                    est_b_i = depth_sparse_track_b(
                        gray_l, gray_r, ru, rv, Q, tpl_r, max_d, min_d
                    )
                    if apply_at_scale and at_outcome is not None and at_outcome.scale is not None:
                        est_a_i = scale_depth_estimate(est_a_i, at_outcome.scale)
                        est_b_i = scale_depth_estimate(est_b_i, at_outcome.scale)
                    world_i = (
                        world_point_from_camera_estimate(est_a_i, atw_result)
                        if atw_enabled
                        else None
                    )
                    rows_payload.append((bbox, bi, est_a_i, est_b_i, world_i))

                apr_lines = (
                    _apriltag_extra_overlay(at_outcome, apriltag_enabled)
                    + _apriltag_world_extra_overlay(atw_result, atw_enabled)
                )
                boxes_depths = [(b, ea, eb, wp) for b, _, ea, eb, wp in rows_payload]

                if preview_enabled:
                    if _preview_tick(
                        preview_cfg,
                        rect,
                        disp,
                        boxes_depths,
                        idx,
                        preview_state,
                        frame_stereo,
                        overlay_left_bgr=left_vis_bgr,
                        extra_lines=apr_lines,
                    ):
                        break
                    if img_folder_hold and in_type in ("image_folder", "images"):
                        if _preview_image_folder_hold_spin(
                            preview_cfg, preview_state, idx, frame_stereo
                        ):
                            break

                if idx >= warmup:
                    at_csv = _apriltag_csv_fields(at_outcome, apriltag_enabled)
                    for bbox, bi, est_a_i, est_b_i, world_i in rows_payload:
                        ru, rv = bbox.center
                        w.writerow(
                            {
                                "frame_idx": idx,
                                "det_idx": bi,
                                "class_id": bbox.class_id,
                                "label": bbox.label or "",
                                "t_wall": time.perf_counter() - t0,
                                "capture_ms": capture_ms,
                                "disp_ms": disp_ms,
                                "det_ms": det_ms,
                                "det_conf": bbox.confidence,
                                "box_x1": bbox.xyxy[0],
                                "box_y1": bbox.xyxy[1],
                                "box_x2": bbox.xyxy[2],
                                "box_y2": bbox.xyxy[3],
                                "ref_u": ru,
                                "ref_v": rv,
                                "A_valid": est_a_i.valid,
                                "A_X": est_a_i.X if est_a_i.valid else "",
                                "A_Y": est_a_i.Y if est_a_i.valid else "",
                                "A_Z": est_a_i.Z if est_a_i.valid else "",
                                "A_disp": est_a_i.disparity if est_a_i.disparity is not None else "",
                                "A_valid_ratio": est_a_i.valid_pixel_ratio
                                if est_a_i.valid_pixel_ratio is not None
                                else "",
                                "B_valid": est_b_i.valid,
                                "B_X": est_b_i.X if est_b_i.valid else "",
                                "B_Y": est_b_i.Y if est_b_i.valid else "",
                                "B_Z": est_b_i.Z if est_b_i.valid else "",
                                "B_disp": est_b_i.disparity if est_b_i.disparity is not None else "",
                                "B_notes": est_b_i.notes,
                                **_world_csv_fields(world_i, atw_result, atw_enabled),
                                **at_csv,
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
    ap.add_argument(
        "--register-object-anchor",
        action="store_true",
        help="Collect and save a robust AprilTag-referenced Object Anchor world pose.",
    )
    ap.add_argument("--capture-type", choices=("positive", "negative"))
    ap.add_argument("--capture-count", type=int, default=100)
    ap.add_argument("--capture-interval", type=float, default=1.0)
    args = ap.parse_args()
    cfg_path = args.config.resolve() if args.config.is_absolute() else (_REPO_ROOT / args.config).resolve()
    out = run_session(
        cfg_path,
        register_object_anchor=args.register_object_anchor,
        capture_type=args.capture_type,
        capture_count=args.capture_count,
        capture_interval=args.capture_interval,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
