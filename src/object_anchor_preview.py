"""Session-only Object Anchor MVP DEMO / PREVIEW (not production world source)."""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from apriltag_world import AprilTagWorldResult
from object_anchor_runtime import ObjectAnchorFrameResult
from object_anchor_world import average_transforms, rotation_delta_deg
from stereo_types import DepthEstimate

PreviewState = Literal[
    "DISABLED",
    "CALIBRATION_REQUIRED",
    "CALIBRATING",
    "READY",
    "ACTIVE",
    "LOST",
    "WAITING",
    "INVALID",
]
DisplaySource = Literal["APRILTAG", "OBJECT_ANCHOR_PREVIEW"]

PREVIEW_CSV_FIELDS = (
    "frame_index",
    "timestamp",
    "display_source",
    "april_tag_detected",
    "april_tag_pose_valid",
    "object_anchor_detected",
    "object_anchor_keypoints_valid",
    "object_anchor_pnp_valid",
    "object_anchor_reprojection_px",
    "object_anchor_filter_window",
    "T_camera_object_raw_json",
    "T_camera_object_filtered_raw_json",
    "T_camera_object_aligned_json",
    "preview_state",
    "preview_calibration_count",
    "cup_detected",
    "cup_depth_valid",
    "p_camera_cup_x",
    "p_camera_cup_y",
    "p_camera_cup_z",
    "p_world_cup_tag_x",
    "p_world_cup_tag_y",
    "p_world_cup_tag_z",
    "p_world_cup_object_x",
    "p_world_cup_object_y",
    "p_world_cup_object_z",
    "cup_diff_x_cm",
    "cup_diff_y_cm",
    "cup_diff_z_cm",
    "cup_diff_distance_cm",
    "camera_translation_diff_cm",
    "camera_rotation_diff_deg",
)


@dataclass(frozen=True)
class ObjectAnchorPreviewSettings:
    enabled: bool = False
    model_name: str = "Full99"
    calibration_samples: int = 30
    temporal_filter_window: int = 3
    max_apriltag_translation_jump_m: float = 0.5
    max_apriltag_rotation_jump_deg: float = 30.0
    max_mean_reprojection_error_px: float = 5.0
    preferred_tag_id: int = 0
    start_calibration_key: str = "c"
    reset_calibration_key: str = "r"
    switch_display_source_key: str = "o"
    toggle_panel_key: str = "p"
    debug_overlay_enabled: bool = False
    debug_overlay_toggle_key: str = "d"
    align_object_frame_to_apriltag: bool = True
    object_frame_alignment_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
    draw_raw_object_axis_in_debug: bool = True
    draw_aligned_object_axis: bool = True
    save_preview_logs: bool = True
    auto_switch_world_source: bool = False
    persist_calibration: bool = False
    output_root: str = "out/object_anchor_preview"
    max_frames_override: int | None = None
    show_panel: bool = True


def load_preview_settings(raw: dict[str, Any] | None) -> ObjectAnchorPreviewSettings:
    cfg = raw if isinstance(raw, dict) else {}
    override = cfg.get("max_frames_override")
    alignment_raw = cfg.get("object_frame_alignment") or {}
    rotation = np.asarray(
        alignment_raw.get(
            "rotation_matrix",
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ),
        dtype=np.float64,
    ).reshape(3, 3)
    translation = np.asarray(
        alignment_raw.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(3)
    if not np.allclose(translation, 0.0, atol=1e-12):
        raise ValueError("object frame alignment must be rotation-only")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("object frame alignment rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("object frame alignment rotation determinant must be +1")
    return ObjectAnchorPreviewSettings(
        enabled=bool(cfg.get("enabled", False)),
        model_name=str(cfg.get("model_name", "Full99")),
        calibration_samples=max(1, int(cfg.get("calibration_samples", 30))),
        temporal_filter_window=max(1, int(cfg.get("temporal_filter_window", 3))),
        max_apriltag_translation_jump_m=float(
            cfg.get("max_apriltag_translation_jump_m", 0.5)
        ),
        max_apriltag_rotation_jump_deg=float(
            cfg.get("max_apriltag_rotation_jump_deg", 30.0)
        ),
        max_mean_reprojection_error_px=float(
            cfg.get("max_mean_reprojection_error_px", 5.0)
        ),
        preferred_tag_id=int(cfg.get("preferred_tag_id", 0)),
        start_calibration_key=str(cfg.get("start_calibration_key", "c")).lower()[:1],
        reset_calibration_key=str(cfg.get("reset_calibration_key", "r")).lower()[:1],
        switch_display_source_key=str(cfg.get("switch_display_source_key", "o")).lower()[
            :1
        ],
        toggle_panel_key=str(cfg.get("toggle_panel_key", "p")).lower()[:1],
        debug_overlay_enabled=bool(cfg.get("debug_overlay_enabled", False)),
        debug_overlay_toggle_key=str(
            cfg.get("debug_overlay_toggle_key", "d")
        ).lower()[:1],
        align_object_frame_to_apriltag=bool(
            cfg.get("align_object_frame_to_apriltag", True)
        ),
        object_frame_alignment_rotation=tuple(
            tuple(float(value) for value in row) for row in rotation
        ),
        draw_raw_object_axis_in_debug=bool(
            cfg.get("draw_raw_object_axis_in_debug", True)
        ),
        draw_aligned_object_axis=bool(cfg.get("draw_aligned_object_axis", True)),
        save_preview_logs=bool(cfg.get("save_preview_logs", True)),
        auto_switch_world_source=bool(cfg.get("auto_switch_world_source", False)),
        persist_calibration=bool(cfg.get("persist_calibration", False)),
        output_root=str(cfg.get("output_root", "out/object_anchor_preview")),
        max_frames_override=int(override) if override is not None else None,
        show_panel=bool(cfg.get("show_panel", True)),
    )


def causal_filter_pose(
    history_including_current: list[np.ndarray],
    window: int,
) -> np.ndarray:
    """Causal SE(3) filter: translation median + sign-aligned quaternion average."""
    if not history_including_current:
        raise ValueError("empty history")
    selected = history_including_current[-max(1, int(window)) :]
    if len(selected) == 1:
        return np.asarray(selected[0], dtype=np.float64).copy()
    return average_transforms(selected, position_median=True)


def object_frame_alignment_transform(
    settings: ObjectAnchorPreviewSettings,
) -> np.ndarray:
    """Return T_object_raw_to_aligned (rotation only).

    Raw tissue frame: +X right, +Y front-to-back, +Z up.
    AprilTag/aligned frame: +X right, +Y up, +Z out of the front face.
    Therefore aligned basis vectors expressed in raw coordinates are
    (+Xraw, +Zraw, -Yraw), i.e. Rx(+90 deg).
    """
    transform = np.eye(4, dtype=np.float64)
    if settings.align_object_frame_to_apriltag:
        transform[:3, :3] = np.asarray(
            settings.object_frame_alignment_rotation, dtype=np.float64
        )
    return transform


def align_object_pose(
    T_camera_object_raw: np.ndarray,
    settings: ObjectAnchorPreviewSettings,
) -> np.ndarray:
    return (
        np.asarray(T_camera_object_raw, dtype=np.float64)
        @ object_frame_alignment_transform(settings)
    )


def transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    homogeneous = np.array(
        [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0],
        dtype=np.float64,
    )
    return (np.asarray(transform, dtype=np.float64) @ homogeneous)[:3]


def cup_difference_cm(
    tag_xyz_m: np.ndarray,
    object_xyz_m: np.ndarray,
) -> dict[str, float]:
    delta_m = np.asarray(object_xyz_m, dtype=np.float64) - np.asarray(
        tag_xyz_m, dtype=np.float64
    )
    delta_cm = delta_m * 100.0
    return {
        "dx_cm": float(delta_cm[0]),
        "dy_cm": float(delta_cm[1]),
        "dz_cm": float(delta_cm[2]),
        "distance_cm": float(np.linalg.norm(delta_cm)),
    }


def select_tag0_observation(
    apriltag_result: AprilTagWorldResult | None,
    *,
    preferred_tag_id: int = 0,
) -> Any | None:
    if apriltag_result is None or not apriltag_result.observations:
        return None
    for obs in apriltag_result.observations:
        if int(obs.tag_id) == int(preferred_tag_id):
            return obs
    return None


def _fmt_xyz_cm(point_m: np.ndarray | None) -> str:
    if point_m is None:
        return "N/A"
    cm = np.asarray(point_m, dtype=np.float64) * 100.0
    return f"X={cm[0]:.1f}  Y={cm[1]:.1f}  Z={cm[2]:.1f} cm"


def draw_object_preview_axes(
    image_bgr: np.ndarray,
    view: PreviewFrameView,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    settings: ObjectAnchorPreviewSettings,
    *,
    axis_length_m: float = 0.08,
) -> np.ndarray:
    """Draw aligned OA axis by default; add raw axis and labels in debug mode."""
    canvas = image_bgr
    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    )

    def draw(transform: np.ndarray, label: str, y_offset: int, thickness: int) -> None:
        rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        translation = np.asarray(transform[:3, 3], dtype=np.float64).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(rotation)
        cv2.drawFrameAxes(
            canvas,
            np.asarray(camera_matrix, dtype=np.float64),
            distortion,
            rvec,
            translation,
            float(axis_length_m),
            thickness,
        )
        origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),
            rvec,
            translation,
            np.asarray(camera_matrix, dtype=np.float64),
            distortion,
        )
        x, y = np.rint(origin.reshape(2)).astype(int)
        cv2.putText(
            canvas,
            label,
            (x + 8, y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x + 8, y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

    if settings.draw_aligned_object_axis and view.T_camera_object_aligned is not None:
        draw(view.T_camera_object_aligned, "OA ALIGNED AXIS", -10, 3)
    if (
        view.debug_overlay_enabled
        and settings.draw_raw_object_axis_in_debug
        and (
            view.T_camera_object_filtered_raw is not None
            or view.T_camera_object_raw is not None
        )
    ):
        raw_axis_pose = (
            view.T_camera_object_filtered_raw
            if view.T_camera_object_filtered_raw is not None
            else view.T_camera_object_raw
        )
        assert raw_axis_pose is not None
        draw(raw_axis_pose, "OA RAW AXIS", 18, 1)
    return canvas


@dataclass
class PreviewFrameView:
    preview_state: PreviewState
    april_status: str
    object_status: str
    difference_status: str
    p_world_cup_tag_m: np.ndarray | None = None
    p_world_cup_object_m: np.ndarray | None = None
    last_p_world_cup_tag_m: np.ndarray | None = None
    cup_diff: dict[str, float] | None = None
    calibration_count: int = 0
    calibration_target: int = 30
    display_source: DisplaySource = "APRILTAG"
    show_panel: bool = True
    fps: float | None = None
    debug_overlay_enabled: bool = False
    T_camera_object_raw: np.ndarray | None = None
    T_camera_object_filtered_raw: np.ndarray | None = None
    T_camera_object_aligned: np.ndarray | None = None
    banner_title: str = "MVP DEMO / PREVIEW (not production)"
    note: str = ""
    camera_translation_diff_cm: float | None = None
    camera_rotation_diff_deg: float | None = None


@dataclass
class ObjectAnchorPreviewSession:
    settings: ObjectAnchorPreviewSettings
    model_path: str
    config_path: str
    output_dir: Path | None = None
    display_source: DisplaySource = "APRILTAG"
    show_panel: bool = True
    debug_overlay_enabled: bool = False
    calibrating: bool = False
    T_world_object_preview: np.ndarray | None = None
    calibration_samples: list[np.ndarray] = field(default_factory=list)
    pose_history: deque[np.ndarray] = field(default_factory=deque)
    previous_T_world_camera_tag: np.ndarray | None = None
    last_p_world_cup_tag_m: np.ndarray | None = None
    session_start_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    rows: list[dict[str, Any]] = field(default_factory=list)
    frame_count: int = 0
    joint_valid_frames: int = 0
    cup_compare_frames: int = 0
    tag_lost_oa_preview_frames: int = 0
    tag_lost_oa_cup_frames: int = 0
    fps_samples: list[float] = field(default_factory=list)
    _calibration_written: bool = False
    _csv_path: Path | None = None
    _rep_dir: Path | None = None
    _rep_saved: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.pose_history = deque(maxlen=max(1, int(self.settings.temporal_filter_window)))
        self.show_panel = bool(self.settings.show_panel)
        self.debug_overlay_enabled = bool(self.settings.debug_overlay_enabled)
        if self.settings.save_preview_logs:
            assert self.output_dir is not None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.output_dir / "preview_frames.csv"
            self._rep_dir = self.output_dir / "representative_frames"
            self._rep_dir.mkdir(parents=True, exist_ok=True)
            with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREVIEW_CSV_FIELDS)
                writer.writeheader()

    @property
    def is_ready(self) -> bool:
        return self.T_world_object_preview is not None

    def reset_calibration(self) -> None:
        self.calibrating = False
        self.calibration_samples.clear()
        self.T_world_object_preview = None
        self.previous_T_world_camera_tag = None
        self.pose_history.clear()
        self._calibration_written = False

    def start_calibration(self) -> None:
        self.calibrating = True
        self.calibration_samples.clear()
        self.T_world_object_preview = None
        self.previous_T_world_camera_tag = None
        self.pose_history.clear()
        self._calibration_written = False

    def handle_key(self, key: int) -> str | None:
        if key == 255 or key < 0:
            return None
        char = chr(key).lower() if 32 <= key < 127 else ""
        if char == self.settings.start_calibration_key:
            self.start_calibration()
            return "start_calibration"
        if char == self.settings.reset_calibration_key:
            self.reset_calibration()
            return "reset_calibration"
        if char == self.settings.switch_display_source_key:
            # Display-only. Never changes operational AprilTag world source.
            if self.settings.auto_switch_world_source:
                return "auto_switch_blocked"
            self.display_source = (
                "OBJECT_ANCHOR_PREVIEW"
                if self.display_source == "APRILTAG"
                else "APRILTAG"
            )
            return f"display_source:{self.display_source}"
        if char == self.settings.toggle_panel_key:
            self.show_panel = not self.show_panel
            return f"panel:{'on' if self.show_panel else 'off'}"
        if char == self.settings.debug_overlay_toggle_key:
            self.debug_overlay_enabled = not self.debug_overlay_enabled
            return f"debug_overlay:{'on' if self.debug_overlay_enabled else 'off'}"
        return None

    def _object_anchor_sample_ok(
        self, anchor_result: ObjectAnchorFrameResult | None
    ) -> tuple[
        bool,
        str,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float | None,
        bool,
        bool,
    ]:
        if anchor_result is None or anchor_result.detection is None:
            return (
                False,
                "object_anchor_not_detected",
                None,
                None,
                None,
                None,
                False,
                False,
            )
        detected = True
        visibility = anchor_result.effective_visibility
        keypoints_valid = (
            visibility is not None and int(np.count_nonzero(visibility >= 1)) >= 4
        )
        pose = anchor_result.pose
        reproj = pose.mean_reprojection_error_px
        pnp_valid = bool(
            pose.valid
            and pose.T_camera_object is not None
            and reproj is not None
            and float(reproj) <= float(self.settings.max_mean_reprojection_error_px)
        )
        if not keypoints_valid:
            return (
                False,
                "object_anchor_keypoints_invalid",
                None,
                None,
                None,
                reproj,
                detected,
                False,
            )
        if not pnp_valid:
            return (
                False,
                pose.reason or "object_anchor_pnp_invalid",
                None,
                None,
                None,
                reproj,
                detected,
                False,
            )
        raw = np.asarray(pose.T_camera_object, dtype=np.float64)
        self.pose_history.append(raw.copy())
        filtered_raw = causal_filter_pose(
            list(self.pose_history), self.settings.temporal_filter_window
        )
        filtered_aligned = align_object_pose(filtered_raw, self.settings)
        return (
            True,
            "ok",
            raw,
            filtered_raw,
            filtered_aligned,
            float(reproj),
            detected,
            True,
        )

    def _maybe_reset_on_tag_jump(self, T_world_camera_tag: np.ndarray) -> bool:
        previous = self.previous_T_world_camera_tag
        self.previous_T_world_camera_tag = T_world_camera_tag.copy()
        if previous is None or not self.calibrating:
            return False
        translation = float(
            np.linalg.norm(T_world_camera_tag[:3, 3] - previous[:3, 3])
        )
        rotation = rotation_delta_deg(T_world_camera_tag[:3, :3], previous[:3, :3])
        if (
            translation >= self.settings.max_apriltag_translation_jump_m
            or rotation >= self.settings.max_apriltag_rotation_jump_deg
        ):
            self.calibration_samples.clear()
            return True
        return False

    def _finalize_calibration(self) -> None:
        if len(self.calibration_samples) < self.settings.calibration_samples:
            return
        self.T_world_object_preview = average_transforms(
            self.calibration_samples, position_median=True
        )
        self.calibrating = False
        if self.settings.save_preview_logs and self.output_dir is not None:
            self._write_session_calibration()

    def _registration_spreads(self) -> tuple[float | None, float | None]:
        if len(self.calibration_samples) < 2 or self.T_world_object_preview is None:
            if not self.calibration_samples:
                return None, None
            return 0.0, 0.0
        center = self.T_world_object_preview
        translations = [
            float(np.linalg.norm(sample[:3, 3] - center[:3, 3]))
            for sample in self.calibration_samples
        ]
        rotations = [
            rotation_delta_deg(sample[:3, :3], center[:3, :3])
            for sample in self.calibration_samples
        ]
        return float(np.max(translations)), float(np.max(rotations))

    def _write_session_calibration(self) -> None:
        if self.output_dir is None or self.T_world_object_preview is None:
            return
        if self.settings.persist_calibration:
            raise RuntimeError("persist_calibration must remain false for MVP DEMO")
        t_spread, r_spread = self._registration_spreads()
        payload = {
            "mode": "MVP_DEMO_SESSION_PREVIEW",
            "not_production_calibration": True,
            "auto_load_on_next_run": False,
            "session_start_time": self.session_start_utc,
            "sample_count": len(self.calibration_samples),
            "T_world_object_preview": self.T_world_object_preview.tolist(),
            "registration_translation_spread_m": t_spread,
            "registration_rotation_spread_deg": r_spread,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "model_name": self.settings.model_name,
            "temporal_filter_window": self.settings.temporal_filter_window,
            "pose_convention": "aligned_to_apriltag",
            "T_object_raw_to_aligned": object_frame_alignment_transform(
                self.settings
            ).tolist(),
        }
        path = self.output_dir / "session_calibration.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._calibration_written = True

    def update(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        fps: float | None,
        apriltag_result: AprilTagWorldResult | None,
        anchor_result: ObjectAnchorFrameResult | None,
        cup_estimate: DepthEstimate | None,
        cup_detected: bool,
        overlay_bgr: np.ndarray | None = None,
    ) -> PreviewFrameView:
        self.frame_count += 1
        if fps is not None and fps > 0:
            self.fps_samples.append(float(fps))

        tag_obs = select_tag0_observation(
            apriltag_result, preferred_tag_id=self.settings.preferred_tag_id
        )
        april_detected = tag_obs is not None
        april_pose_valid = april_detected
        T_world_camera_tag = (
            np.asarray(tag_obs.T_world_camera, dtype=np.float64)
            if tag_obs is not None
            else None
        )

        (
            oa_ok,
            oa_reason,
            T_camera_object_raw,
            T_camera_object_filtered_raw,
            T_camera_object_aligned,
            reproj,
            oa_detected,
            kpts_ok,
        ) = self._object_anchor_sample_ok(anchor_result)
        pnp_valid = oa_ok

        cup_depth_valid = bool(
            cup_estimate is not None and getattr(cup_estimate, "valid", False)
        )
        p_camera_cup = None
        if cup_depth_valid and cup_estimate is not None:
            p_camera_cup = np.array(
                [cup_estimate.X, cup_estimate.Y, cup_estimate.Z], dtype=np.float64
            )

        jumped = False
        if T_world_camera_tag is not None:
            jumped = self._maybe_reset_on_tag_jump(T_world_camera_tag)

        if (
            self.calibrating
            and T_world_camera_tag is not None
            and T_camera_object_aligned is not None
            and oa_ok
        ):
            sample = T_world_camera_tag @ T_camera_object_aligned
            self.calibration_samples.append(sample)
            if len(self.calibration_samples) >= self.settings.calibration_samples:
                self._finalize_calibration()

        T_world_camera_object = None
        if self.T_world_object_preview is not None and T_camera_object_aligned is not None:
            T_world_camera_object = self.T_world_object_preview @ np.linalg.inv(
                T_camera_object_aligned
            )

        p_world_cup_tag = None
        if T_world_camera_tag is not None and p_camera_cup is not None:
            p_world_cup_tag = transform_point(T_world_camera_tag, p_camera_cup)
            self.last_p_world_cup_tag_m = p_world_cup_tag.copy()

        p_world_cup_object = None
        if (
            self.T_world_object_preview is not None
            and T_world_camera_object is not None
            and p_camera_cup is not None
            and oa_ok
        ):
            # Same-frame P_camera_cup for both transforms.
            p_world_cup_object = transform_point(T_world_camera_object, p_camera_cup)

        cup_diff = None
        camera_translation_diff_cm = None
        camera_rotation_diff_deg = None
        if p_world_cup_tag is not None and p_world_cup_object is not None:
            cup_diff = cup_difference_cm(p_world_cup_tag, p_world_cup_object)
            self.cup_compare_frames += 1
        if (
            T_world_camera_tag is not None
            and T_world_camera_object is not None
            and self.T_world_object_preview is not None
        ):
            camera_translation_diff_cm = float(
                np.linalg.norm(
                    T_world_camera_tag[:3, 3] - T_world_camera_object[:3, 3]
                )
                * 100.0
            )
            camera_rotation_diff_deg = rotation_delta_deg(
                T_world_camera_tag[:3, :3], T_world_camera_object[:3, :3]
            )

        if april_pose_valid and oa_ok:
            self.joint_valid_frames += 1

        if not april_pose_valid and self.T_world_object_preview is not None and oa_ok:
            self.tag_lost_oa_preview_frames += 1
            if p_world_cup_object is not None:
                self.tag_lost_oa_cup_frames += 1

        # Status text for overlay (never show 0,0,0 placeholders).
        if not april_detected:
            april_status = "LOST"
        elif not april_pose_valid:
            april_status = "INVALID"
        else:
            april_status = "ACTIVE"

        if self.T_world_object_preview is None:
            if self.calibrating:
                object_status = (
                    f"CALIBRATING {len(self.calibration_samples)}/"
                    f"{self.settings.calibration_samples}"
                )
                preview_state: PreviewState = "CALIBRATING"
            else:
                object_status = "CALIBRATION REQUIRED"
                preview_state = "CALIBRATION_REQUIRED"
        elif not oa_detected:
            object_status = "LOST OBJECT"
            preview_state = "LOST"
        elif not oa_ok:
            object_status = "INVALID" if cup_detected else "WAITING"
            preview_state = "INVALID"
        elif p_camera_cup is None:
            if cup_detected and not cup_depth_valid:
                object_status = "INVALID DEPTH"
            else:
                object_status = "READY"
            preview_state = "ACTIVE"
        else:
            object_status = "ACTIVE"
            preview_state = "ACTIVE"

        if cup_diff is not None:
            difference_status = "ACTIVE"
        elif not april_pose_valid and self.T_world_object_preview is not None:
            difference_status = "N/A"
        elif self.T_world_object_preview is None:
            difference_status = "WAITING"
        elif p_camera_cup is None and cup_detected and not cup_depth_valid:
            difference_status = "INVALID DEPTH"
        else:
            difference_status = "N/A"

        note = ""
        if jumped:
            note = "calibration reset: AprilTag pose jump"
        if self.is_ready:
            note = (note + " | " if note else "") + "OBJECT ANCHOR PREVIEW: READY"

        view = PreviewFrameView(
            preview_state=preview_state,
            april_status=april_status,
            object_status=object_status,
            difference_status=difference_status,
            p_world_cup_tag_m=p_world_cup_tag,
            p_world_cup_object_m=p_world_cup_object,
            last_p_world_cup_tag_m=self.last_p_world_cup_tag_m,
            cup_diff=cup_diff,
            calibration_count=len(self.calibration_samples),
            calibration_target=self.settings.calibration_samples,
            display_source=self.display_source,
            show_panel=self.show_panel,
            fps=fps,
            debug_overlay_enabled=self.debug_overlay_enabled,
            T_camera_object_raw=T_camera_object_raw,
            T_camera_object_filtered_raw=T_camera_object_filtered_raw,
            T_camera_object_aligned=T_camera_object_aligned,
            note=note,
            camera_translation_diff_cm=camera_translation_diff_cm,
            camera_rotation_diff_deg=camera_rotation_diff_deg,
        )

        row = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "display_source": self.display_source,
            "april_tag_detected": april_detected,
            "april_tag_pose_valid": april_pose_valid,
            "object_anchor_detected": oa_detected,
            "object_anchor_keypoints_valid": kpts_ok,
            "object_anchor_pnp_valid": pnp_valid,
            "object_anchor_reprojection_px": (
                "" if reproj is None else f"{reproj:.4f}"
            ),
            "object_anchor_filter_window": self.settings.temporal_filter_window,
            "T_camera_object_raw_json": (
                ""
                if T_camera_object_raw is None
                else json.dumps(T_camera_object_raw.tolist(), separators=(",", ":"))
            ),
            "T_camera_object_filtered_raw_json": (
                ""
                if T_camera_object_filtered_raw is None
                else json.dumps(
                    T_camera_object_filtered_raw.tolist(), separators=(",", ":")
                )
            ),
            "T_camera_object_aligned_json": (
                ""
                if T_camera_object_aligned is None
                else json.dumps(T_camera_object_aligned.tolist(), separators=(",", ":"))
            ),
            "preview_state": preview_state,
            "preview_calibration_count": len(self.calibration_samples),
            "cup_detected": bool(cup_detected),
            "cup_depth_valid": cup_depth_valid,
            "p_camera_cup_x": "" if p_camera_cup is None else float(p_camera_cup[0]),
            "p_camera_cup_y": "" if p_camera_cup is None else float(p_camera_cup[1]),
            "p_camera_cup_z": "" if p_camera_cup is None else float(p_camera_cup[2]),
            "p_world_cup_tag_x": "" if p_world_cup_tag is None else float(p_world_cup_tag[0]),
            "p_world_cup_tag_y": "" if p_world_cup_tag is None else float(p_world_cup_tag[1]),
            "p_world_cup_tag_z": "" if p_world_cup_tag is None else float(p_world_cup_tag[2]),
            "p_world_cup_object_x": (
                "" if p_world_cup_object is None else float(p_world_cup_object[0])
            ),
            "p_world_cup_object_y": (
                "" if p_world_cup_object is None else float(p_world_cup_object[1])
            ),
            "p_world_cup_object_z": (
                "" if p_world_cup_object is None else float(p_world_cup_object[2])
            ),
            "cup_diff_x_cm": "" if cup_diff is None else cup_diff["dx_cm"],
            "cup_diff_y_cm": "" if cup_diff is None else cup_diff["dy_cm"],
            "cup_diff_z_cm": "" if cup_diff is None else cup_diff["dz_cm"],
            "cup_diff_distance_cm": "" if cup_diff is None else cup_diff["distance_cm"],
            "camera_translation_diff_cm": (
                "" if camera_translation_diff_cm is None else camera_translation_diff_cm
            ),
            "camera_rotation_diff_deg": (
                "" if camera_rotation_diff_deg is None else camera_rotation_diff_deg
            ),
            "_oa_reason": oa_reason,
        }
        self.rows.append(row)
        if self._csv_path is not None:
            with self._csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=PREVIEW_CSV_FIELDS, extrasaction="ignore"
                )
                writer.writerow(row)

        if overlay_bgr is not None and self._rep_dir is not None:
            self._maybe_save_representative(overlay_bgr, view, frame_idx)

        return view

    def _maybe_save_representative(
        self,
        overlay_bgr: np.ndarray,
        view: PreviewFrameView,
        frame_idx: int,
    ) -> None:
        assert self._rep_dir is not None
        events: list[str] = []
        if (
            self.is_ready
            and "ready" not in self._rep_saved
            and view.p_world_cup_object_m is not None
        ):
            events.append("ready")
        if (
            view.april_status == "LOST"
            and view.p_world_cup_object_m is not None
            and "tag_lost_oa_active" not in self._rep_saved
        ):
            events.append("tag_lost_oa_active")
        if (
            view.cup_diff is not None
            and "both_active" not in self._rep_saved
        ):
            events.append("both_active")
        for event in events:
            path = self._rep_dir / f"{event}_f{frame_idx:06d}.jpg"
            annotated = draw_preview_banner(overlay_bgr.copy(), view)
            cv2.imwrite(str(path), annotated)
            self._rep_saved.add(event)

    def summary(self) -> dict[str, Any]:
        distances = [
            float(row["cup_diff_distance_cm"])
            for row in self.rows
            if row.get("cup_diff_distance_cm") not in ("", None)
        ]
        array = np.asarray(distances, dtype=np.float64) if distances else np.asarray([])
        return {
            "mode": "MVP_DEMO_PREVIEW_RECORD",
            "not_operational_performance_guarantee": True,
            "total_frames": self.frame_count,
            "session_calibration_success": self.is_ready,
            "calibration_sample_count": (
                self.settings.calibration_samples if self.is_ready else len(self.calibration_samples)
            ),
            "april_tag_and_object_anchor_joint_valid_frames": self.joint_valid_frames,
            "cup_both_world_comparable_frames": self.cup_compare_frames,
            "cup_difference_distance_cm": {
                "mean": float(np.mean(array)) if array.size else None,
                "median": float(np.median(array)) if array.size else None,
                "p90": float(np.percentile(array, 90)) if array.size else None,
                "max": float(np.max(array)) if array.size else None,
            },
            "apriltag_lost_oa_preview_valid_frames": self.tag_lost_oa_preview_frames,
            "apriltag_lost_oa_cup_world_frames": self.tag_lost_oa_cup_frames,
            "mean_fps": float(np.mean(self.fps_samples)) if self.fps_samples else None,
            "display_source_final": self.display_source,
            "auto_switch_world_source": self.settings.auto_switch_world_source,
            "persist_calibration": self.settings.persist_calibration,
            "object_frame_aligned_to_apriltag": (
                self.settings.align_object_frame_to_apriltag
            ),
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
        }

    def close(self) -> Path | None:
        if self.output_dir is None:
            return None
        summary = self.summary()
        path = self.output_dir / "preview_summary.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        readme = self.output_dir / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# Object Anchor MVP DEMO / PREVIEW logs",
                    "",
                    "Session-only preview records. Not production calibration.",
                    "Do not auto-load `session_calibration.json` on the next run.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path


def draw_preview_banner(image_bgr: np.ndarray, view: PreviewFrameView) -> np.ndarray:
    """Draw a three-column MVP DEMO comparison banner at the top of the frame."""
    if not view.show_panel:
        return image_bgr
    out = image_bgr
    height, width = out.shape[:2]
    panel_h = min(150, max(120, height // 6))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    col_w = width // 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55 if width >= 1000 else 0.45
    thickness = 1
    white = (240, 240, 240)
    accent = (0, 220, 255)
    warn = (80, 80, 255)
    ok = (80, 220, 120)

    def put(col: int, line: int, text: str, color: tuple[int, int, int] = white) -> None:
        x = 10 + col * col_w
        y = 22 + line * 22
        cv2.putText(out, text[:56], (x, y), font, scale, color, thickness, cv2.LINE_AA)

    title_color = accent
    put(0, 0, view.banner_title, title_color)
    put(1, 0, f"DISPLAY SOURCE: {view.display_source}", accent)
    fps_text = "FPS: N/A" if view.fps is None else f"FPS: {view.fps:.1f}"
    debug_text = " | DEBUG: ON" if view.debug_overlay_enabled else ""
    put(2, 0, fps_text + debug_text, accent)

    # AprilTag column
    april_color = ok if view.april_status == "ACTIVE" else warn
    put(0, 1, "[APRILTAG]", april_color)
    put(0, 2, f"Status: {view.april_status}", april_color)
    if view.april_status == "LOST":
        put(0, 3, "Cup World: N/A", white)
        if view.last_p_world_cup_tag_m is not None:
            put(0, 4, f"Last Cup World: {_fmt_xyz_cm(view.last_p_world_cup_tag_m)}", (180, 180, 180))
        put(0, 5, "APRILTAG: LOST", warn)
    elif view.p_world_cup_tag_m is not None:
        put(0, 3, f"Cup World: {_fmt_xyz_cm(view.p_world_cup_tag_m)}", white)
    else:
        put(0, 3, "Cup World: N/A", white)

    # Object Anchor column
    obj_color = (
        ok
        if view.object_status == "ACTIVE"
        else accent
        if view.object_status.startswith("CALIBRAT")
        else warn
    )
    put(1, 1, "[OBJECT ANCHOR - TISSUE]", obj_color)
    put(1, 2, f"Status: {view.object_status}", obj_color)
    if view.p_world_cup_object_m is not None and view.object_status == "ACTIVE":
        put(1, 3, f"Cup World: {_fmt_xyz_cm(view.p_world_cup_object_m)}", white)
        put(1, 4, "OBJECT ANCHOR PREVIEW: ACTIVE", ok)
    elif view.preview_state == "CALIBRATING":
        put(
            1,
            3,
            f"CALIBRATING {view.calibration_count}/{view.calibration_target}",
            accent,
        )
        put(1, 4, "Cup World: N/A", white)
    elif view.preview_state == "CALIBRATION_REQUIRED":
        put(1, 3, "Cup World: N/A", white)
        put(1, 4, "Press C to start session calibration", accent)
    else:
        put(1, 3, "Cup World: N/A", white)

    # Difference column
    put(2, 1, "[DIFFERENCE]", white)
    if view.cup_diff is not None:
        d = view.cup_diff
        put(2, 2, f"dX={d['dx_cm']:.1f}  dY={d['dy_cm']:.1f}  dZ={d['dz_cm']:.1f} cm", white)
        put(2, 3, f"Distance={d['distance_cm']:.2f} cm", ok)
        put(2, 4, "Status: ACTIVE", ok)
    else:
        put(2, 2, f"Status: {view.difference_status}", warn)
        put(2, 3, "dX/dY/dZ: N/A", white)
        put(2, 4, "Distance: N/A", white)

    # Keys reminder
    put(
        0,
        6 if panel_h >= 140 else 5,
        "Keys: C=calibrate R=reset O=source P=panel D=debug Q=quit",
        (160, 160, 160),
    )
    return out


def build_preview_session(
    settings: ObjectAnchorPreviewSettings,
    *,
    repo_root: Path,
    model_path: str,
    config_path: str,
) -> ObjectAnchorPreviewSession | None:
    if not settings.enabled:
        return None
    if settings.persist_calibration:
        raise ValueError("object_anchor_preview.persist_calibration must be false")
    if settings.auto_switch_world_source:
        # Hard-disable automatic production source switching.
        settings = ObjectAnchorPreviewSettings(
            **{
                **settings.__dict__,
                "auto_switch_world_source": False,
            }
        )
    output_dir = None
    if settings.save_preview_logs:
        root = Path(settings.output_root)
        if not root.is_absolute():
            root = repo_root / root
        output_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    return ObjectAnchorPreviewSession(
        settings=settings,
        model_path=model_path,
        config_path=config_path,
        output_dir=output_dir,
    )
