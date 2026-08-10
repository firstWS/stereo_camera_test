"""Phase 4.5-M2 tag-masked VIO demo video helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from dataset_recorder.reader import DatasetReader

from .ir_tag_mask import (
    MASK_INTERVAL_RECOVERY_FRAME,
    MASK_INTERVAL_START_FRAME,
    FrameTagMaskDiagnostics,
    apply_tag_roi_mask,
    is_frame_tag_mask_active,
)
from .phase45_vio_demo import (
    DEMO_WINDOW_ID,
    CupBbox,
    build_demo_replay_states,
    cup2_world_position,
    load_cup_bboxes_by_frame,
    load_demo_window,
    load_rgb_frame_paths,
    yaw_deg_from_transform,
)
from .dropout_protocol import load_frame_timestamps_from_rgb_index
from .hold_last_pose_runner import load_dropout_windows_from_manifest
from .stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID


def load_mask_diagnostics(path: Path) -> dict[int, FrameTagMaskDiagnostics]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_frame: dict[int, FrameTagMaskDiagnostics] = {}
    for row in payload.get("frames", []):
        left = row["left"]
        right = row["right"]
        by_frame[int(row["frame_number"])] = FrameTagMaskDiagnostics(
            frame_number=int(row["frame_number"]),
            tag_mask_active=bool(row["tag_mask_active"]),
            fill_value=int(row.get("fill_value", 0)),
            left_detection_failed=bool(row.get("left_detection_failed", False)),
            right_detection_failed=bool(row.get("right_detection_failed", False)),
            left=_roi_from_json(left),
            right=_roi_from_json(right),
        )
    return by_frame


def _roi_from_json(payload: Mapping[str, Any]):
    from .ir_tag_mask import TagMaskRoi

    corners = payload.get("corners_xy")
    corner_tuple = tuple((float(x), float(y)) for x, y in corners) if corners else None
    bbox = tuple(int(v) for v in payload["bbox_xyxy"]) if payload.get("bbox_xyxy") else None
    expanded = (
        tuple(int(v) for v in payload["expanded_bbox_xyxy"]) if payload.get("expanded_bbox_xyxy") else None
    )
    return TagMaskRoi(
        detected=bool(payload.get("detected", False)),
        corners_xy=corner_tuple,
        mask_corners_xy=tuple((float(x), float(y)) for x, y in payload["mask_corners_xy"])
        if payload.get("mask_corners_xy")
        else corner_tuple,
        bbox_xyxy=bbox,
        expanded_bbox_xyxy=expanded,
        area_ratio=float(payload.get("area_ratio", 0.0)),
        holdover=bool(payload.get("holdover", False)),
        roi_source=str(payload.get("roi_source", "unknown")),
    )


def load_left_ir_frame_paths(session_dir: Path) -> dict[int, Path]:
    reader = DatasetReader(session_dir)
    from .canonical_frames import load_canonical_frames_from_rgb_index

    canonical_numbers = [frame.frame_number for frame in load_canonical_frames_from_rgb_index(session_dir)]
    paths: dict[int, Path] = {}
    for canonical_frame, record in zip(canonical_numbers, reader.iterate_left_ir()):
        if record.file_path is not None and record.file_path.is_file():
            paths[canonical_frame] = record.file_path
    return paths


def build_masked_left_ir_preview(
    *,
    left_gray: np.ndarray,
    diag: FrameTagMaskDiagnostics | None,
) -> np.ndarray:
    if diag is None or not diag.tag_mask_active:
        preview = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    else:
        masked = apply_tag_roi_mask(left_gray, diag.left, fill_value=diag.fill_value)
        preview = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        if diag.left.expanded_bbox_xyxy is not None:
            x1, y1, x2, y2 = diag.left.expanded_bbox_xyxy
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return preview


def render_tag_mask_demo_frame(
    *,
    image_bgr: np.ndarray,
    frame_number: int,
    relative_time_sec: float,
    replay,
    cups: Sequence[CupBbox],
    mask_diag: FrameTagMaskDiagnostics | None,
    left_ir_preview_bgr: np.ndarray | None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    masked = is_frame_tag_mask_active(frame_number)

    for cup in cups:
        color = (80, 200, 80) if cup.semantic_id == "cup1" else (0, 165, 255)
        label = "Cup1" if cup.semantic_id == "cup1" else "Cup2"
        cv2.rectangle(canvas, (cup.x1, cup.y1), (cup.x2, cup.y2), color, 2)
        cv2.putText(
            canvas,
            label,
            (cup.x1, max(20, cup.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    header = "Phase 4.5 - Stereo+IMU VIO-Lite"
    subtitle = "Tag Visual Dependency Test"
    cv2.putText(canvas, header, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Frame: {frame_number}   Time: {relative_time_sec:.2f}s",
        (16, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    if frame_number < MASK_INTERVAL_START_FRAME:
        status_lines = [
            "APRILTAG INITIAL ANCHOR / NORMAL VIO",
            "Camera Tracking: VIO VALID",
        ]
    elif masked:
        status_lines = [
            "APRILTAG REMOVED FROM VIO VISUAL INPUT",
            "LEFT + RIGHT IR TAG REGION MASKED",
            "LOCAL POSE: STEREO + IMU VIO",
        ]
    elif frame_number >= MASK_INTERVAL_RECOVERY_FRAME:
        status_lines = [
            "TAG MASK RELEASED",
            "APRILTAG RE-ANCHORED",
            "Camera Tracking: VIO VALID",
        ]
    else:
        status_lines = ["Camera Tracking: VIO VALID"]

    y = 112
    for line in status_lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    if replay is not None and replay.world_valid and replay.T_world_camera is not None:
        tx, ty, tz = replay.T_world_camera[:3, 3]
        yaw = yaw_deg_from_transform(replay.T_world_camera)
        cv2.putText(
            canvas,
            f"Camera World  X:{tx:.2f}  Y:{ty:.2f}  Z:{tz:.2f}  Yaw:{yaw:.1f}deg",
            (16, canvas.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cup2_world = cup2_world_position(replay, cups)
    if cup2_world is not None:
        cv2.putText(
            canvas,
            f"Cup2 World  X:{cup2_world[0]:.2f}  Y:{cup2_world[1]:.2f}  Z:{cup2_world[2]:.2f}",
            (16, canvas.shape[0] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

    if masked:
        cv2.putText(
            canvas,
            "Visualization mask - actual VIO mask applied to L/R IR",
            (16, canvas.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 255),
            1,
            cv2.LINE_AA,
        )

    if left_ir_preview_bgr is not None:
        inset_w = 280
        inset_h = int(left_ir_preview_bgr.shape[0] * (inset_w / left_ir_preview_bgr.shape[1]))
        inset = cv2.resize(left_ir_preview_bgr, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        x0 = canvas.shape[1] - inset_w - 16
        y0 = 110
        canvas[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + inset_w + 2, y0 + inset_h + 2), (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "VIO LEFT IR INPUT",
            (x0, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if masked:
            cv2.putText(
                canvas,
                "TAG REGION MASKED",
                (x0, y0 + inset_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas


def generate_tag_mask_demo_video(
    *,
    session_dir: Path,
    trajectory_csv: Path,
    manifest_path: Path,
    mask_diagnostics_json: Path,
    output_mp4: Path,
    fps: float = 30.0,
    window_id: str = DEMO_WINDOW_ID,
) -> dict[str, Any]:
    load_demo_window(manifest_path, window_id=window_id)
    replay_by_frame = build_demo_replay_states(
        session_dir=session_dir,
        trajectory_csv=trajectory_csv,
        manifest_path=manifest_path,
        window_id=window_id,
    )
    rgb_paths = load_rgb_frame_paths(session_dir)
    left_ir_paths = load_left_ir_frame_paths(session_dir)
    cups_by_frame = load_cup_bboxes_by_frame(session_dir / "derived/cups/observations.csv")
    mask_by_frame = load_mask_diagnostics(mask_diagnostics_json)
    timestamps = load_frame_timestamps_from_rgb_index(session_dir)
    if not timestamps:
        raise ValueError("No RGB frame timestamps found")

    first_ts = timestamps[0].device_timestamp_us
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(rgb_paths[timestamps[0].frame_number]))
    if first_frame is None:
        raise RuntimeError("Failed to read first RGB frame")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_mp4}")

    written = 0
    for frame in timestamps:
        frame_number = frame.frame_number
        image_path = rgb_paths.get(frame_number)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        relative_time_sec = (frame.device_timestamp_us - first_ts) / 1_000_000.0
        diag = mask_by_frame.get(frame_number)
        left_preview = None
        left_path = left_ir_paths.get(frame_number)
        if left_path is not None:
            left_gray = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            if left_gray is not None:
                left_preview = build_masked_left_ir_preview(left_gray=left_gray, diag=diag)
        rendered = render_tag_mask_demo_frame(
            image_bgr=image,
            frame_number=frame_number,
            relative_time_sec=relative_time_sec,
            replay=replay_by_frame.get(frame_number),
            cups=cups_by_frame.get(frame_number, []),
            mask_diag=diag,
            left_ir_preview_bgr=left_preview,
        )
        writer.write(rendered)
        written += 1

    writer.release()
    return {
        "dataset": str(session_dir),
        "algorithm_id": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
        "ablation": "tag_masked_visual_input",
        "dropout_window_id": window_id,
        "masked_interval": [MASK_INTERVAL_START_FRAME, MASK_INTERVAL_RECOVERY_FRAME],
        "video_frame_count": written,
        "fps": fps,
        "resolution": [width, height],
        "output_path": str(output_mp4),
        "left_ir_inset": True,
    }
