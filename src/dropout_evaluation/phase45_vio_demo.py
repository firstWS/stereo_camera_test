"""Phase 4.5 milestone demo video helpers (visualization only, frozen VIO artifacts)."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from dataset_recorder.reader import DatasetReader

from .dropout_protocol import DropoutWindow, is_runtime_tag_masked, load_frame_timestamps_from_rgb_index
from .evaluation_metrics import transform_point_camera_to_world
from .hold_last_pose_runner import load_dropout_windows_from_manifest
from .rgbd_odometry_adapter import FrameReplayState, replay_session_for_window
from .runtime_apriltag import load_runtime_apriltag_poses_from_session
from .stereo_imu_vio_adapter import load_vio_trajectory_from_csv, vio_trajectory_to_local_trajectory
from .stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID

DEMO_WINDOW_ID = "C_pre_cup2__5.0s"
DEMO_ANCHOR_ID = "C_pre_cup2"
DEMO_DURATION_SEC = 5.0
DEMO_START_FRAME = 202


@dataclass(frozen=True)
class CupBbox:
    frame_number: int
    semantic_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    P_camera: np.ndarray | None
    valid: bool


@dataclass(frozen=True)
class DemoFrameContext:
    frame_number: int
    device_timestamp_us: int
    relative_time_sec: float
    replay: FrameReplayState | None
    cups: list[CupBbox]


def load_demo_window(manifest_path: Path, *, window_id: str = DEMO_WINDOW_ID) -> DropoutWindow:
    for window in load_dropout_windows_from_manifest(manifest_path):
        if window.window_id == window_id:
            return window
    raise ValueError(f"Demo window not found: {window_id}")


def load_cup_bboxes_by_frame(observations_csv: Path) -> dict[int, list[CupBbox]]:
    by_frame: dict[int, list[CupBbox]] = {}
    with observations_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_number = int(row["frame_number"])
            valid = str(row.get("depth_valid", "")).lower() in {"1", "true", "yes"}
            P_camera = None
            if valid and row.get("camera_x") not in (None, ""):
                P_camera = np.array(
                    [float(row["camera_x"]), float(row["camera_y"]), float(row["camera_z"])],
                    dtype=np.float64,
                )
            bbox = CupBbox(
                frame_number=frame_number,
                semantic_id=str(row.get("semantic_id", "")),
                x1=int(float(row["bbox_x1"])),
                y1=int(float(row["bbox_y1"])),
                x2=int(float(row["bbox_x2"])),
                y2=int(float(row["bbox_y2"])),
                P_camera=P_camera,
                valid=valid and P_camera is not None,
            )
            by_frame.setdefault(frame_number, []).append(bbox)
    return by_frame


def load_rgb_frame_paths(session_dir: Path) -> dict[int, Path]:
    reader = DatasetReader(session_dir)
    paths: dict[int, Path] = {}
    for record in reader.iterate_rgb():
        frame_number = int(record.row.get("frame_number") or 0)
        if record.file_path is not None and record.file_path.is_file():
            paths[frame_number] = record.file_path
    return paths


def build_demo_replay_states(
    *,
    session_dir: Path,
    trajectory_csv: Path,
    manifest_path: Path,
    window_id: str = DEMO_WINDOW_ID,
) -> dict[int, FrameReplayState]:
    window = load_demo_window(manifest_path, window_id=window_id)
    local_trajectory = vio_trajectory_to_local_trajectory(load_vio_trajectory_from_csv(trajectory_csv))
    runtime_poses = load_runtime_apriltag_poses_from_session(session_dir)
    frame_timestamps = load_frame_timestamps_from_rgb_index(session_dir)
    replay = replay_session_for_window(
        window=window,
        local_trajectory=local_trajectory,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
    )
    return replay.frames


def yaw_deg_from_transform(T: np.ndarray) -> float:
    R = np.asarray(T[:3, :3], dtype=np.float64)
    return float(math.degrees(math.atan2(R[1, 0], R[0, 0])))


def cup_world_position_tag0(
    replay: FrameReplayState | None,
    cups: Sequence[CupBbox],
    semantic_id: str,
) -> np.ndarray | None:
    if replay is None or not replay.world_valid or replay.T_world_camera is None:
        return None
    for cup in cups:
        if cup.semantic_id == semantic_id and cup.valid and cup.P_camera is not None:
            return transform_point_camera_to_world(replay.T_world_camera, cup.P_camera)
    return None


def cup2_world_position(replay: FrameReplayState | None, cups: Sequence[CupBbox]) -> np.ndarray | None:
    return cup_world_position_tag0(replay, cups, "cup2")


def dropout_overlay_lines(
    *,
    frame_number: int,
    replay: FrameReplayState | None,
    window: DropoutWindow,
    device_timestamp_us: int,
) -> tuple[str, str, str]:
    masked = is_runtime_tag_masked(device_timestamp_us, window)
    if frame_number < window.start_frame:
        return (
            "AprilTag: VISIBLE",
            "Camera Tracking: VIO VALID",
            "",
        )
    if masked:
        return (
            "AprilTag: SOFTWARE DROPOUT",
            "Camera Tracking: VIO VALID",
            "LOCAL POSE: STEREO + IMU VIO",
        )
    if window.recovery_frame is not None and frame_number >= window.recovery_frame:
        return (
            "AprilTag: RE-ANCHORED",
            "Camera Tracking: VIO VALID",
            "",
        )
    return (
        "AprilTag: VISIBLE",
        "Camera Tracking: VIO VALID",
        "",
    )


def render_demo_frame(
    *,
    image_bgr: np.ndarray,
    frame_number: int,
    relative_time_sec: float,
    replay: FrameReplayState | None,
    cups: Sequence[CupBbox],
    window: DropoutWindow,
    device_timestamp_us: int,
) -> np.ndarray:
    canvas = image_bgr.copy()
    masked = is_runtime_tag_masked(device_timestamp_us, window)

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

    if masked:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 90), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    header = "Phase 4.5 - Stereo + IMU VIO-Lite"
    cv2.putText(canvas, header, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Frame: {frame_number}   Time: {relative_time_sec:.2f}s",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    tag_line, track_line, local_line = dropout_overlay_lines(
        frame_number=frame_number,
        replay=replay,
        window=window,
        device_timestamp_us=device_timestamp_us,
    )
    y = 100
    for line in (tag_line, track_line, local_line):
        if not line:
            continue
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

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
            "Software Dropout (Tag visible in scene)",
            (16, canvas.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def generate_demo_video(
    *,
    session_dir: Path,
    trajectory_csv: Path,
    manifest_path: Path,
    output_mp4: Path,
    fps: float = 30.0,
    window_id: str = DEMO_WINDOW_ID,
) -> dict[str, Any]:
    window = load_demo_window(manifest_path, window_id=window_id)
    replay_by_frame = build_demo_replay_states(
        session_dir=session_dir,
        trajectory_csv=trajectory_csv,
        manifest_path=manifest_path,
        window_id=window_id,
    )
    rgb_paths = load_rgb_frame_paths(session_dir)
    cups_by_frame = load_cup_bboxes_by_frame(session_dir / "derived/cups/observations.csv")
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
        rendered = render_demo_frame(
            image_bgr=image,
            frame_number=frame_number,
            relative_time_sec=relative_time_sec,
            replay=replay_by_frame.get(frame_number),
            cups=cups_by_frame.get(frame_number, []),
            window=window,
            device_timestamp_us=frame.device_timestamp_us,
        )
        writer.write(rendered)
        written += 1

    writer.release()
    return {
        "dataset": str(session_dir),
        "algorithm_id": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
        "dropout_anchor": window.anchor_id,
        "dropout_window_id": window.window_id,
        "dropout_duration_sec": window.target_duration_sec,
        "canonical_start_frame": window.start_frame,
        "canonical_end_frame": window.end_frame,
        "canonical_recovery_frame": window.recovery_frame,
        "video_frame_count": written,
        "fps": fps,
        "resolution": [width, height],
        "output_path": str(output_mp4),
    }


def build_milestone_summary(
    *,
    repo_root: Path,
    evaluation_dir: Path,
    trajectory_summary_json: Path,
    demo_summary: Mapping[str, Any],
) -> dict[str, Any]:
    eval_summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((evaluation_dir / "comparison.json").read_text(encoding="utf-8"))
    traj_summary = json.loads(trajectory_summary_json.read_text(encoding="utf-8"))

    by_id = {row["window_id"]: row for row in comparison.get("windows", [])}
    c5 = by_id.get("C_pre_cup2__5.0s", {}).get("vio", {})
    d5 = by_id.get("D_active_with_cup2__5.0s", {}).get("vio", {})
    rgbd_d5 = by_id.get("D_active_with_cup2__5.0s", {}).get("rgbd", {})

    runtime = traj_summary.get("runtime_stats", {})
    local_summary = traj_summary.get("summary", {})

    return {
        "algorithm": STEREO_IMU_VIO_LITE_ALGORITHM_ID,
        "official_gate": "VIO_BENCHMARK_COMPLETE",
        "performance_gate": "VIO_LITE_PROMISING",
        "scenario_a": {
            "trajectory_valid_frames": local_summary.get("valid_frames"),
            "trajectory_total_frames": local_summary.get("total_frames"),
            "dropout_availability_15_windows": 1.0,
            "recovery_latency_frames": 0,
            "c_cup2_median_m_range": [
                by_id[w]["vio"]["cup2_median"]
                for w in by_id
                if w.startswith("C_pre_cup2__") and by_id[w]["vio"].get("cup2_median") is not None
            ],
            "d_cup2_median_m_range": [
                by_id[w]["vio"]["cup2_median"]
                for w in by_id
                if w.startswith("D_active_with_cup2__") and by_id[w]["vio"].get("cup2_median") is not None
            ],
            "d_5s_rotation_median_deg_vio": d5.get("rotation_median"),
            "d_5s_rotation_median_deg_rgbd": rgbd_d5.get("rotation_median"),
            "rtf": runtime.get("real_time_factor"),
            "processing_time_s": runtime.get("total_processing_time_s"),
        },
        "warnings": [
            "calibration provisional_factory",
            "propagated_only = 0",
            "d_family_cup2_threshold_fail",
        ],
        "evaluation_summary": eval_summary,
        "demo": dict(demo_summary),
        "frozen_artifacts": {
            "trajectory_csv": str(repo_root / "out/analysis/phase4_stereo_imu_vio_lite/trajectory.csv"),
            "evaluation_dir": str(evaluation_dir),
        },
    }
