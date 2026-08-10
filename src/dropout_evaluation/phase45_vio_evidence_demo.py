"""Phase 4.5-M3 evidence-focused VIO demo (visualization only, frozen M2 artifacts)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from apriltag_scale import APRILTAG_DICTIONARIES

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
    cup_world_position_tag0,
    load_cup_bboxes_by_frame,
    load_demo_window,
    load_rgb_frame_paths,
    yaw_deg_from_transform,
)
from .phase45_vio_tag_mask_demo import (
    build_masked_left_ir_preview,
    load_left_ir_frame_paths,
    load_mask_diagnostics,
)
from .dropout_protocol import load_frame_timestamps_from_rgb_index
from .rgbd_odometry_adapter import FrameReplayState
from .stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID

try:
    from application_world import (
        ApplicationWorldContract,
        load_application_world_contract,
        tag0_world_point_to_application_world,
        tag0_world_pose_to_application_world,
    )
except ImportError:  # pragma: no cover - src layout in tests
    ApplicationWorldContract = Any  # type: ignore[misc,assignment]
    load_application_world_contract = None  # type: ignore[assignment]
    tag0_world_point_to_application_world = None  # type: ignore[assignment]
    tag0_world_pose_to_application_world = None  # type: ignore[assignment]

HIGHLIGHT_FRAMES_AFTER_EVENT = 5
CUP2_TRAIL_MAX_POINTS = 40
TOP_VIEW_SIZE = (320, 250)
IR_INSET_WIDTH = 320
IR_INSET_HEIGHT = 125
LEFT_BOTTOM_X = 16
LEFT_BOTTOM_Y_START = 640
COLOR_RED = (60, 60, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_POSITIVE = (120, 255, 180)
COLOR_VIO_TRACKING = (255, 255, 255)
COLOR_CUP2_EVENT = (0, 215, 255)


@dataclass(frozen=True)
class WorldTopViewBounds:
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    def to_plot(self, x: float, z: float, width: int, height: int, *, margin: int = 24) -> tuple[int, int]:
        plot_w = max(width - 2 * margin, 1)
        plot_h = max(height - 2 * margin, 1)
        x_span = max(self.x_max - self.x_min, 1e-6)
        z_span = max(self.z_max - self.z_min, 1e-6)
        px = int(margin + (x - self.x_min) / x_span * plot_w)
        py = int(height - margin - (z - self.z_min) / z_span * plot_h)
        return px, py


def find_cup2_first_valid_frame(
    cups_by_frame: Mapping[int, Sequence[CupBbox]],
    replay_by_frame: Mapping[int, FrameReplayState],
) -> int | None:
    for frame_number in sorted(cups_by_frame):
        cups = cups_by_frame[frame_number]
        replay = replay_by_frame.get(frame_number)
        if cup2_world_position(replay, cups) is not None:
            return frame_number
    return None


def compute_world_top_view_bounds(
    replay_by_frame: Mapping[int, FrameReplayState],
    cup2_trail: Sequence[np.ndarray],
) -> WorldTopViewBounds:
    xs: list[float] = []
    zs: list[float] = []
    for replay in replay_by_frame.values():
        if replay.world_valid and replay.T_world_camera is not None:
            xs.append(float(replay.T_world_camera[0, 3]))
            zs.append(float(replay.T_world_camera[2, 3]))
    for point in cup2_trail:
        xs.append(float(point[0]))
        zs.append(float(point[2]))
    if not xs:
        return WorldTopViewBounds(-1.0, 1.0, 0.0, 3.0)
    pad_x = max(0.15, (max(xs) - min(xs)) * 0.12)
    pad_z = max(0.15, (max(zs) - min(zs)) * 0.12)
    return WorldTopViewBounds(
        x_min=min(xs) - pad_x,
        x_max=max(xs) + pad_x,
        z_min=min(zs) - pad_z,
        z_max=max(zs) + pad_z,
    )


def camera_heading_xz(T_world_camera: np.ndarray) -> tuple[float, float]:
    forward = T_world_camera[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.hypot(forward[0], forward[1]))
    if norm < 1e-9:
        return 1.0, 0.0
    return float(forward[0] / norm), float(forward[2] / norm)


def detect_rgb_tag_polygon(rgb_gray: np.ndarray, *, preferred_tag_id: int = 0) -> np.ndarray | None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARIES["APRILTAG_36H11"])
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners_list, ids, _ = detector.detectMarkers(rgb_gray)
    if ids is None or len(ids) == 0:
        return None
    selected = 0
    flat_ids = [int(v) for v in ids.reshape(-1)]
    if preferred_tag_id in flat_ids:
        selected = flat_ids.index(preferred_tag_id)
    return np.round(corners_list[selected].reshape(-1, 2)).astype(np.int32)


def draw_rgb_recorded_tag_overlay(
    canvas: np.ndarray,
    rgb_gray: np.ndarray,
    *,
    masked_active: bool,
) -> None:
    if not masked_active:
        return
    polygon = detect_rgb_tag_polygon(rgb_gray)
    if polygon is None:
        return
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [polygon], (0, 0, 180))
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    cx = int((x1 + x2) / 2)
    cy = int(max(20, y1 - 10))
    cv2.putText(
        canvas,
        "RECORDED TAG",
        (max(8, cx - 70), cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "NOT USED BY VIO",
        (max(8, cx - 80), cy + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_RED,
        2,
        cv2.LINE_AA,
    )


def status_line_color(line: str, *, frame_number: int, highlight_tag_dropout_red: bool) -> tuple[int, int, int]:
    if line == "STEREO + IMU VIO TRACKING":
        return COLOR_VIO_TRACKING
    if line in {"NEW OBJECT: CUP2", "CUP2 WORLD POSITION AVAILABLE"}:
        return COLOR_CUP2_EVENT
    if line == "APRILTAG RE-ANCHORED":
        return COLOR_POSITIVE
    if highlight_tag_dropout_red and is_tag_dropout_red_state(frame_number):
        if line in {"TAG INPUT REMOVED", "Tag Pose: BLOCKED", "Tag Visual Feature: MASKED"}:
            return COLOR_RED
    return COLOR_WHITE


@dataclass(frozen=True)
class PhaseStatusContent:
    primary: tuple[str, ...]
    secondary: tuple[str, ...]


def is_tag_dropout_red_state(frame_number: int) -> bool:
    return is_frame_tag_mask_active(frame_number)


def phase_status_content(frame_number: int, *, cup2_highlight: bool) -> PhaseStatusContent:
    if frame_number < MASK_INTERVAL_START_FRAME:
        return PhaseStatusContent(
            primary=("WORLD INITIALIZED BY APRILTAG", "STEREO + IMU VIO TRACKING"),
            secondary=(),
        )
    if is_frame_tag_mask_active(frame_number):
        primary = ["TAG INPUT REMOVED", "STEREO + IMU VIO TRACKING"]
        if cup2_highlight:
            primary.extend(["NEW OBJECT: CUP2", "CUP2 WORLD POSITION AVAILABLE"])
        return PhaseStatusContent(
            primary=tuple(primary),
            secondary=("Tag Pose: BLOCKED", "Tag Visual Feature: MASKED"),
        )
    if frame_number >= MASK_INTERVAL_RECOVERY_FRAME:
        return PhaseStatusContent(
            primary=("APRILTAG RE-ANCHORED", "STEREO + IMU VIO TRACKING"),
            secondary=(),
        )
    return PhaseStatusContent(primary=("STEREO + IMU VIO TRACKING",), secondary=())


def phase_status_lines(frame_number: int, *, cup2_highlight: bool) -> list[str]:
    content = phase_status_content(frame_number, cup2_highlight=cup2_highlight)
    return list(content.primary) + list(content.secondary)


def render_world_top_view(
    *,
    bounds: WorldTopViewBounds,
    camera_path: Sequence[tuple[float, float]],
    current_camera: tuple[float, float] | None,
    heading_xz: tuple[float, float] | None,
    cup2_trail_xz: Sequence[tuple[float, float]],
    current_cup2: tuple[float, float] | None,
) -> np.ndarray:
    width, height = TOP_VIEW_SIZE
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    margin = 24
    x0, y0 = bounds.to_plot(bounds.x_min, bounds.z_min, width, height)
    x1, y1 = bounds.to_plot(bounds.x_max, bounds.z_max, width, height)
    cv2.rectangle(panel, (margin, margin), (width - margin, height - margin), (70, 70, 70), 1)
    origin_px = bounds.to_plot(0.0, 0.0, width, height)
    cv2.drawMarker(panel, origin_px, (180, 180, 180), cv2.MARKER_CROSS, 10, 1)

    if len(camera_path) >= 2:
        pts = np.array([bounds.to_plot(x, z, width, height) for x, z in camera_path], dtype=np.int32)
        cv2.polylines(panel, [pts], False, (90, 170, 255), 2, cv2.LINE_AA)
        mid = pts[len(pts) // 2]
        cv2.putText(
            panel,
            "CAMERA PATH",
            (min(mid[0] + 6, width - 110), max(mid[1] - 6, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (90, 170, 255),
            1,
            cv2.LINE_AA,
        )

    for point in cup2_trail_xz[-CUP2_TRAIL_MAX_POINTS:]:
        cv2.circle(panel, bounds.to_plot(point[0], point[1], width, height), 2, (120, 220, 120), -1, cv2.LINE_AA)

    if current_cup2 is not None:
        cpx = bounds.to_plot(current_cup2[0], current_cup2[1], width, height)
        cv2.circle(panel, cpx, 7, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, cpx, 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            "CUP2 WORLD",
            (min(cpx[0] + 10, width - 90), max(cpx[1] - 8, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

    if current_camera is not None:
        cpx = bounds.to_plot(current_camera[0], current_camera[1], width, height)
        cv2.circle(panel, cpx, 6, (255, 255, 0), -1, cv2.LINE_AA)
        if heading_xz is not None:
            tip = (
                int(cpx[0] + heading_xz[0] * 22),
                int(cpx[1] - heading_xz[1] * 22),
            )
            cv2.arrowedLine(panel, cpx, tip, (255, 255, 0), 2, tipLength=0.35)
        cv2.putText(
            panel,
            "CAMERA",
            (min(cpx[0] + 10, width - 70), max(cpx[1] - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(panel, "WORLD TOP VIEW", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "APRILTAG ORIGIN",
        (min(origin_px[0] + 8, width - 120), max(origin_px[1] - 8, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(panel, "X", (width - margin - 10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, "Z", (8, margin + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return panel


def _paste_inset(
    canvas: np.ndarray,
    inset: np.ndarray,
    x: int,
    y: int,
    title: str,
    subtitle: str = "",
    *,
    border_color: tuple[int, int, int] = (220, 220, 220),
    subtitle_color: tuple[int, int, int] = (200, 200, 255),
) -> None:
    h, w = inset.shape[:2]
    canvas[y : y + h, x : x + w] = inset
    cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), border_color, 2 if border_color == COLOR_RED else 1)
    cv2.putText(canvas, title, (x, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, subtitle_color, 1, cv2.LINE_AA)


def left_bottom_layout_y_positions(*, has_cup2: bool, use_application_world: bool = False) -> tuple[int, ...]:
    if use_application_world:
        return (548, 608, 668)
    camera_y = LEFT_BOTTOM_Y_START
    cup2_y = camera_y + 88
    return (camera_y, cup2_y)


def application_world_contract_lines(contract: ApplicationWorldContract) -> tuple[str, str]:
    tx, ty, tz = contract.tag0_position_application_world_m
    return (
        "APPLICATION WORLD",
        f"Tag0 = ({tx:.2f}, {ty:.2f}, {tz:.2f}) m  |  +X right | +Y up | +Z front",
    )


def tag0_world_point_to_display(
    point_tag0_world: np.ndarray | None,
    *,
    use_application_world: bool,
    T_application_tag0: np.ndarray | None = None,
) -> np.ndarray | None:
    if point_tag0_world is None:
        return None
    if not use_application_world:
        return np.asarray(point_tag0_world, dtype=np.float64)
    if tag0_world_point_to_application_world is None:
        raise RuntimeError("application_world module unavailable")
    return tag0_world_point_to_application_world(
        point_tag0_world,
        T_application_tag0=T_application_tag0,
    )


def tag0_world_pose_to_display(
    T_tag0_world: np.ndarray | None,
    *,
    use_application_world: bool,
    T_application_tag0: np.ndarray | None = None,
) -> np.ndarray | None:
    if T_tag0_world is None:
        return None
    if not use_application_world:
        return np.asarray(T_tag0_world, dtype=np.float64)
    if tag0_world_pose_to_application_world is None:
        raise RuntimeError("application_world module unavailable")
    return tag0_world_pose_to_application_world(
        T_tag0_world,
        T_application_tag0=T_application_tag0,
    )


def _draw_world_xyz_block(
    canvas: np.ndarray,
    *,
    title: str,
    point: np.ndarray | None,
    y: int,
    title_color: tuple[int, int, int],
    waiting_text: str | None = None,
    include_yaw: float | None = None,
) -> None:
    cv2.putText(
        canvas,
        title,
        (LEFT_BOTTOM_X, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        title_color,
        2,
        cv2.LINE_AA,
    )
    if point is not None:
        text = f"X: {point[0]:+.2f} m   Y: {point[1]:+.2f} m   Z: {point[2]:+.2f} m"
        if include_yaw is not None:
            text += f"   Yaw: {include_yaw:.1f} deg"
        cv2.putText(
            canvas,
            text,
            (LEFT_BOTTOM_X, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            title_color,
            1,
            cv2.LINE_AA,
        )
    elif waiting_text is not None:
        cv2.putText(
            canvas,
            waiting_text,
            (LEFT_BOTTOM_X, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )


def render_evidence_frame(
    *,
    image_bgr: np.ndarray,
    rgb_gray: np.ndarray,
    frame_number: int,
    relative_time_sec: float,
    replay: FrameReplayState | None,
    cups: Sequence[CupBbox],
    mask_diag: FrameTagMaskDiagnostics | None,
    original_left_ir_bgr: np.ndarray | None,
    masked_left_ir_bgr: np.ndarray | None,
    top_view_panel: np.ndarray | None = None,
    cup2_first_frame: int | None,
    show_world_top_view: bool = True,
    highlight_tag_dropout_red: bool = False,
    use_application_world: bool = False,
    T_application_tag0: np.ndarray | None = None,
    application_world_contract: ApplicationWorldContract | None = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    masked_active = is_frame_tag_mask_active(frame_number)
    cup2_world_tag0 = cup2_world_position(replay, cups)
    cup2_world = tag0_world_point_to_display(
        cup2_world_tag0,
        use_application_world=use_application_world,
        T_application_tag0=T_application_tag0,
    )
    cup1_world = tag0_world_point_to_display(
        cup_world_position_tag0(replay, cups, "cup1"),
        use_application_world=use_application_world,
        T_application_tag0=T_application_tag0,
    )
    cup2_highlight = (
        cup2_first_frame is not None
        and cup2_first_frame <= frame_number <= cup2_first_frame + HIGHLIGHT_FRAMES_AFTER_EVENT
    )

    draw_rgb_recorded_tag_overlay(canvas, rgb_gray, masked_active=masked_active)

    for cup in cups:
        color = (80, 200, 80) if cup.semantic_id == "cup1" else (0, 165, 255)
        label = "Cup1" if cup.semantic_id == "cup1" else "Cup2"
        thickness = 3 if cup.semantic_id == "cup2" and cup2_highlight else 2
        cv2.rectangle(canvas, (cup.x1, cup.y1), (cup.x2, cup.y2), color, thickness)
        cv2.putText(
            canvas,
            label,
            (cup.x1, max(20, cup.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    header = "Phase 4.5 - Tag-Independent VIO Evidence"
    cv2.putText(canvas, header, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"F{frame_number}  {relative_time_sec:.1f}s",
        (canvas.shape[1] - 150, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    status = phase_status_content(frame_number, cup2_highlight=cup2_highlight)
    y = 52
    for line in status.primary:
        color = status_line_color(
            line,
            frame_number=frame_number,
            highlight_tag_dropout_red=highlight_tag_dropout_red,
        )
        thickness = 2 if color == COLOR_RED else 2
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, thickness, cv2.LINE_AA)
        y += 24
    for line in status.secondary:
        color = status_line_color(
            line,
            frame_number=frame_number,
            highlight_tag_dropout_red=highlight_tag_dropout_red,
        )
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        y += 20

    layout = left_bottom_layout_y_positions(
        has_cup2=cup2_world is not None,
        use_application_world=use_application_world,
    )
    camera_title = "CAMERA WORLD (APP)" if use_application_world else "CAMERA WORLD"
    cup1_title = "CUP1 WORLD (APP)" if use_application_world else "CUP1 WORLD"
    cup2_title = "CUP2 WORLD (APP)" if use_application_world else "CUP2 WORLD"
    camera_color = (255, 255, 0)
    cup1_color = (80, 220, 80)
    cup2_color = (0, 215, 255)

    T_display_camera = tag0_world_pose_to_display(
        replay.T_world_camera if replay is not None and replay.world_valid else None,
        use_application_world=use_application_world,
        T_application_tag0=T_application_tag0,
    )
    camera_yaw = yaw_deg_from_transform(T_display_camera) if T_display_camera is not None else None

    if use_application_world:
        camera_y, cup1_y, cup2_y = layout
        _draw_world_xyz_block(
            canvas,
            title=camera_title,
            point=T_display_camera[:3, 3] if T_display_camera is not None else None,
            y=camera_y,
            title_color=camera_color,
            include_yaw=camera_yaw,
        )
        _draw_world_xyz_block(
            canvas,
            title=cup1_title,
            point=cup1_world,
            y=cup1_y,
            title_color=cup1_color,
            waiting_text="Waiting for Cup1..." if cup1_world is None else None,
        )
        _draw_world_xyz_block(
            canvas,
            title=cup2_title,
            point=cup2_world,
            y=cup2_y,
            title_color=cup2_color,
            waiting_text="Waiting for Cup2..." if cup2_world is None else None,
        )
    else:
        camera_y = layout[0]
        cup2_y = layout[1] if len(layout) > 1 else layout[0] + 88
        if T_display_camera is not None:
            tx, ty, tz = T_display_camera[:3, 3]
            cv2.putText(
                canvas,
                camera_title,
                (LEFT_BOTTOM_X, camera_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                camera_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"X: {tx:+.2f} m   Y: {ty:+.2f} m   Z: {tz:+.2f} m   Yaw: {camera_yaw:.1f} deg",
                (LEFT_BOTTOM_X, camera_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                camera_color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            cup2_title,
            (LEFT_BOTTOM_X, cup2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cup2_color,
            2,
            cv2.LINE_AA,
        )
        if cup2_world is not None:
            cv2.putText(
                canvas,
                f"X: {cup2_world[0]:+.2f} m   Y: {cup2_world[1]:+.2f} m   Z: {cup2_world[2]:+.2f} m",
                (LEFT_BOTTOM_X, cup2_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                cup2_color,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "Waiting for Cup2...",
                (LEFT_BOTTOM_X, cup2_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

    if use_application_world and application_world_contract is not None:
        contract_title, contract_detail = application_world_contract_lines(application_world_contract)
        cv2.putText(
            canvas,
            contract_title,
            (420, canvas.shape[0] - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            contract_detail,
            (420, canvas.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Offline tag-mask ablation",
        (16, canvas.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Not physical tag exit",
        (16, canvas.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    inset_x = canvas.shape[1] - IR_INSET_WIDTH - 12
    if original_left_ir_bgr is not None:
        orig = cv2.resize(original_left_ir_bgr, (IR_INSET_WIDTH, IR_INSET_HEIGHT), interpolation=cv2.INTER_AREA)
        _paste_inset(canvas, orig, inset_x, 72, "ORIGINAL LEFT IR", "TAG VISIBLE")
    if masked_left_ir_bgr is not None:
        masked = cv2.resize(masked_left_ir_bgr, (IR_INSET_WIDTH, IR_INSET_HEIGHT), interpolation=cv2.INTER_AREA)
        subtitle = "TAG REGION MASKED" if masked_active else "SAME AS ORIGINAL"
        masked_border = COLOR_RED if masked_active and highlight_tag_dropout_red else (220, 220, 220)
        masked_subtitle_color = COLOR_RED if masked_active and highlight_tag_dropout_red else (200, 200, 255)
        _paste_inset(
            canvas,
            masked,
            inset_x,
            72 + IR_INSET_HEIGHT + 30,
            "ACTUAL VIO LEFT IR INPUT",
            subtitle,
            border_color=masked_border,
            subtitle_color=masked_subtitle_color,
        )

    if show_world_top_view and top_view_panel is not None:
        tv_x = canvas.shape[1] - TOP_VIEW_SIZE[0] - 12
        tv_y = canvas.shape[0] - TOP_VIEW_SIZE[1] - 16
        canvas[tv_y : tv_y + TOP_VIEW_SIZE[1], tv_x : tv_x + TOP_VIEW_SIZE[0]] = top_view_panel

    return canvas


def visual_update_metadata_from_trajectory(trajectory_csv: Path | None) -> dict[str, Any]:
    if trajectory_csv is None or not trajectory_csv.is_file():
        return {}
    import csv

    with trajectory_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    init_success = int(any(str(row.get("state")) == "init" and str(row.get("visual_update_success")).lower() in {"1", "true"} for row in rows))
    update_rows = rows[1:] if total > 1 else []
    success_count = sum(1 for row in update_rows if str(row.get("visual_update_success")).lower() in {"1", "true"})
    eligible = len(update_rows)
    return {
        "initialization_success": init_success,
        "visual_update_success_count": success_count,
        "visual_update_eligible_count": eligible,
        "visual_update_success_ratio_eligible_only": float(success_count / eligible) if eligible > 0 else None,
        "note": "Eligible frames exclude init; legacy summary ratio may include init in numerator.",
    }


def build_demo_summary_metadata(
    *,
    session_dir: Path,
    trajectory_csv: Path,
    evaluation_dir: Path,
    validation_summary_json: Path | None,
    output_mp4: Path,
    cup2_first_frame: int | None,
    video_frame_count: int,
    width: int,
    height: int,
    fps: float,
    presentation_version: str = "m3.1",
    show_world_top_view: bool = True,
    highlight_tag_dropout_red: bool = False,
    use_application_world: bool = False,
    application_world_contract: ApplicationWorldContract | None = None,
) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    if validation_summary_json is not None and validation_summary_json.is_file():
        validation = json.loads(validation_summary_json.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "dataset": str(session_dir),
        "algorithm_id": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
        "window_id": DEMO_WINDOW_ID,
        "canonical_frames": 436,
        "masked_interval": [MASK_INTERVAL_START_FRAME, MASK_INTERVAL_RECOVERY_FRAME],
        "cup2_first_valid_frame": cup2_first_frame,
        "trajectory_source": str(trajectory_csv),
        "evaluation_source": str(evaluation_dir),
        "vio_valid_frames": validation.get("trajectory_valid_frames"),
        "c5_pose_availability": validation.get("c5_pose_availability"),
        "cup2_availability": validation.get("cup2_availability"),
        "cup2_median_error_m": validation.get("cup2_median"),
        "catastrophic_jumps": validation.get("catastrophic_jump_count"),
        "original_ir_shown": True,
        "actual_masked_vio_input_shown": True,
        "world_top_view_shown": show_world_top_view,
        "highlight_tag_dropout_red": highlight_tag_dropout_red,
        "physical_tag_removed": False,
        "visual_tag_masked_from_vio_input": True,
        "presentation_version": presentation_version,
        "world_source_panel_removed": True,
        "visual_update_metadata": visual_update_metadata_from_trajectory(trajectory_csv),
        "output_path": str(output_mp4),
        "video_frame_count": video_frame_count,
        "fps": fps,
        "resolution": [width, height],
    }
    if use_application_world:
        contract = application_world_contract
        if contract is None and load_application_world_contract is not None:
            contract = load_application_world_contract()
        if contract is not None:
            summary.update(
                {
                    "coordinate_frame": "application_world",
                    "application_world_tag0_xyz_m": list(contract.tag0_position_application_world_m),
                    "axis_convention": {
                        "+X": "right",
                        "+Y": "up",
                        "+Z": "front",
                    },
                    "camera_world_display": "application_world",
                    "cup1_world_display": "application_world",
                    "cup2_world_display": "application_world",
                    "cup1_used_as_pose_source": False,
                    "cup2_used_as_pose_source": False,
                    "internal_evaluation_world_unchanged": True,
                    "application_world_config": contract.config_path,
                }
            )
    return summary


def generate_evidence_demo_video(
    *,
    session_dir: Path,
    trajectory_csv: Path,
    manifest_path: Path,
    mask_diagnostics_json: Path,
    evaluation_dir: Path,
    validation_summary_json: Path | None,
    output_mp4: Path,
    presentation_version: str = "m3.1",
    show_world_top_view: bool = True,
    highlight_tag_dropout_red: bool = False,
    use_application_world: bool = False,
    fps: float = 30.0,
) -> dict[str, Any]:
    application_world_contract = None
    T_application_tag0 = None
    if use_application_world:
        if load_application_world_contract is None:
            raise RuntimeError("application_world module unavailable")
        application_world_contract = load_application_world_contract()
        T_application_tag0 = application_world_contract.T_application_tag0
    load_demo_window(manifest_path)
    replay_by_frame = build_demo_replay_states(
        session_dir=session_dir,
        trajectory_csv=trajectory_csv,
        manifest_path=manifest_path,
    )
    rgb_paths = load_rgb_frame_paths(session_dir)
    left_ir_paths = load_left_ir_frame_paths(session_dir)
    cups_by_frame = load_cup_bboxes_by_frame(session_dir / "derived/cups/observations.csv")
    mask_by_frame = load_mask_diagnostics(mask_diagnostics_json)
    timestamps = load_frame_timestamps_from_rgb_index(session_dir)
    if not timestamps:
        raise ValueError("No RGB frame timestamps found")

    cup2_first_frame = find_cup2_first_valid_frame(cups_by_frame, replay_by_frame)
    bounds = None
    if show_world_top_view:
        cup2_trail: list[np.ndarray] = []
        for frame_number in sorted(replay_by_frame):
            world = cup2_world_position(replay_by_frame[frame_number], cups_by_frame.get(frame_number, []))
            if world is not None:
                cup2_trail.append(world)
        bounds = compute_world_top_view_bounds(replay_by_frame, cup2_trail)
    first_ts = timestamps[0].device_timestamp_us
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(rgb_paths[timestamps[0].frame_number]))
    if first_frame is None:
        raise RuntimeError("Failed to read first RGB frame")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_mp4}")

    written = 0
    camera_path: list[tuple[float, float]] = []
    cup2_trail_xz: list[tuple[float, float]] = []

    for frame in timestamps:
        frame_number = frame.frame_number
        image_path = rgb_paths.get(frame_number)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rgb_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        replay = replay_by_frame.get(frame_number)
        current_camera = None
        heading = None
        if replay is not None and replay.world_valid and replay.T_world_camera is not None:
            tx = float(replay.T_world_camera[0, 3])
            tz = float(replay.T_world_camera[2, 3])
            current_camera = (tx, tz)
            camera_path.append((tx, tz))
            heading = camera_heading_xz(replay.T_world_camera)

        cup2_world = cup2_world_position(replay, cups_by_frame.get(frame_number, []))
        current_cup2 = None
        if cup2_world is not None:
            current_cup2 = (float(cup2_world[0]), float(cup2_world[2]))
            cup2_trail_xz.append(current_cup2)

        top_view = None
        if show_world_top_view and bounds is not None:
            top_view = render_world_top_view(
                bounds=bounds,
                camera_path=camera_path,
                current_camera=current_camera,
                heading_xz=heading,
                cup2_trail_xz=cup2_trail_xz,
                current_cup2=current_cup2,
            )

        diag = mask_by_frame.get(frame_number)
        original_left_ir_bgr = None
        masked_left_ir_bgr = None
        left_path = left_ir_paths.get(frame_number)
        if left_path is not None:
            left_gray = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            if left_gray is not None:
                original_left_ir_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
                masked_left_ir_bgr = build_masked_left_ir_preview(left_gray=left_gray, diag=diag)

        rendered = render_evidence_frame(
            image_bgr=image,
            rgb_gray=rgb_gray,
            frame_number=frame_number,
            relative_time_sec=(frame.device_timestamp_us - first_ts) / 1_000_000.0,
            replay=replay,
            cups=cups_by_frame.get(frame_number, []),
            mask_diag=diag,
            original_left_ir_bgr=original_left_ir_bgr,
            masked_left_ir_bgr=masked_left_ir_bgr,
            top_view_panel=top_view,
            cup2_first_frame=cup2_first_frame,
            show_world_top_view=show_world_top_view,
            highlight_tag_dropout_red=highlight_tag_dropout_red,
            use_application_world=use_application_world,
            T_application_tag0=T_application_tag0,
            application_world_contract=application_world_contract,
        )
        writer.write(rendered)
        written += 1

    writer.release()
    return build_demo_summary_metadata(
        session_dir=session_dir,
        trajectory_csv=trajectory_csv,
        evaluation_dir=evaluation_dir,
        validation_summary_json=validation_summary_json,
        output_mp4=output_mp4,
        cup2_first_frame=cup2_first_frame,
        video_frame_count=written,
        width=width,
        height=height,
        fps=fps,
        presentation_version=presentation_version,
        show_world_top_view=show_world_top_view,
        highlight_tag_dropout_red=highlight_tag_dropout_red,
        use_application_world=use_application_world,
        application_world_contract=application_world_contract,
    )
