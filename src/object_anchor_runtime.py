"""Optional live Object Anchor inference, PnP validation, and overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_anchor_config import ObjectAnchorConfig, load_object_anchor_config
from object_anchor_detector import ObjectAnchorDetection, ObjectAnchorDetector, UltralyticsPoseDetector
from object_anchor_geometry import find_skeleton_crossings
from object_anchor_pose import ObjectPoseEstimate, estimate_object_pose
from object_anchor_visualizer import draw_object_anchor_keypoints, draw_object_pose_axes


@dataclass(frozen=True)
class ObjectAnchorRuntimeSettings:
    enabled: bool = False
    draw: bool = True
    camera_pose_only: bool = True
    max_translation_jump_m: float = 0.25
    max_rotation_jump_deg: float = 35.0
    reset_after_missed_frames: int = 15


@dataclass
class ObjectAnchorFrameResult:
    detection: ObjectAnchorDetection | None
    pose: ObjectPoseEstimate
    overlay_bgr: np.ndarray
    effective_visibility: np.ndarray | None = None
    skeleton_crossings: tuple[tuple[int, int], ...] = ()


def _invalid_pose(reason: str) -> ObjectPoseEstimate:
    return ObjectPoseEstimate(valid=False, reason=reason)


def _rotation_delta_deg(current: np.ndarray, previous: np.ndarray) -> float:
    relative = current @ previous.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


class ObjectPoseTemporalValidator:
    def __init__(self, settings: ObjectAnchorRuntimeSettings) -> None:
        self.settings = settings
        self._previous: ObjectPoseEstimate | None = None
        self._missed_frames = 0

    def _reject_jump(self, pose: ObjectPoseEstimate, reason: str) -> ObjectPoseEstimate:
        pose.valid = False
        pose.reason = reason
        self._missed_frames += 1
        if self._missed_frames >= self.settings.reset_after_missed_frames:
            self._previous = None
        return pose

    def validate(self, pose: ObjectPoseEstimate) -> ObjectPoseEstimate:
        if not pose.valid:
            self._missed_frames += 1
            if self._missed_frames >= self.settings.reset_after_missed_frames:
                self._previous = None
            return pose

        previous = self._previous
        if (
            previous is not None
            and previous.tvec is not None
            and previous.rotation_matrix is not None
            and pose.tvec is not None
            and pose.rotation_matrix is not None
        ):
            translation_jump = float(
                np.linalg.norm(pose.tvec.reshape(3) - previous.tvec.reshape(3))
            )
            rotation_jump = _rotation_delta_deg(
                pose.rotation_matrix, previous.rotation_matrix
            )
            if translation_jump > self.settings.max_translation_jump_m:
                return self._reject_jump(
                    pose,
                    f"translation_jump:{translation_jump:.3f}>"
                    f"{self.settings.max_translation_jump_m:.3f}m",
                )
            if rotation_jump > self.settings.max_rotation_jump_deg:
                return self._reject_jump(
                    pose,
                    f"rotation_jump:{rotation_jump:.1f}>"
                    f"{self.settings.max_rotation_jump_deg:.1f}deg",
                )

        self._previous = pose
        self._missed_frames = 0
        return pose


class ObjectAnchorRuntime:
    def __init__(
        self,
        config: ObjectAnchorConfig,
        detector: ObjectAnchorDetector,
        settings: ObjectAnchorRuntimeSettings,
    ) -> None:
        self.config = config
        self.detector = detector
        self.settings = settings
        self.temporal_validator = ObjectPoseTemporalValidator(settings)

    def process(
        self,
        bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None,
        *,
        draw_on_bgr: np.ndarray | None = None,
        debug_overlay: bool = True,
        draw_pose_axis: bool = True,
    ) -> ObjectAnchorFrameResult:
        overlay_source = draw_on_bgr if draw_on_bgr is not None else bgr
        detections = self.detector.predict(bgr)
        if not detections:
            pose = self.temporal_validator.validate(_invalid_pose("no_object_anchor_detection"))
            return ObjectAnchorFrameResult(None, pose, overlay_source.copy())

        detection = max(detections, key=lambda item: item.score)
        point_count = len(detection.keypoints_xy)
        if point_count != len(self.config.keypoints):
            pose = self.temporal_validator.validate(
                _invalid_pose(
                    f"keypoint_count_mismatch:{point_count}!={len(self.config.keypoints)}"
                )
            )
            return ObjectAnchorFrameResult(detection, pose, overlay_source.copy())

        effective_visibility = (
            np.asarray(detection.keypoint_visibility, dtype=np.int32).reshape(-1)
            if detection.keypoint_visibility is not None
            else np.where(
                detection.keypoint_confidences
                >= self.config.pose_settings.confidence_threshold,
                2,
                0,
            ).astype(np.int32)
        )
        pose = estimate_object_pose(
            detection.keypoints_xy,
            self.config.object_points,
            camera_matrix,
            dist_coeffs=dist_coeffs,
            confidences=detection.keypoint_confidences,
            visibility=effective_visibility,
            settings=self.config.pose_settings,
        )
        skeleton_crossings = (
            find_skeleton_crossings(
                detection.keypoints_xy,
                self.config.skeleton,
                valid_mask=effective_visibility >= 1,
            )
            if self.config.anchor_mode == "front_only"
            else ()
        )
        if skeleton_crossings and pose.valid:
            pose.valid = False
            pose.reason = "skeleton_crossing"
        pose = self.temporal_validator.validate(pose)

        overlay = overlay_source.copy()
        if self.settings.draw:
            if detection.bbox_xyxy is not None:
                x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 210, 0), 2)
            overlay = draw_object_anchor_keypoints(
                overlay,
                detection.keypoints_xy,
                self.config,
                confidences=detection.keypoint_confidences,
                visibility=effective_visibility,
                show_confidence_visibility=debug_overlay,
                show_names=debug_overlay,
                show_labels=debug_overlay,
                skeleton_crossed=bool(skeleton_crossings),
            )
            if pose.valid and draw_pose_axis:
                overlay = draw_object_pose_axes(
                    overlay,
                    pose,
                    camera_matrix,
                    dist_coeffs=dist_coeffs,
                    axis_length_m=min(self.config.size.values()) * 0.7,
                )
        return ObjectAnchorFrameResult(
            detection=detection,
            pose=pose,
            overlay_bgr=overlay,
            effective_visibility=effective_visibility,
            skeleton_crossings=skeleton_crossings,
        )

    def overlay_lines(self, result: ObjectAnchorFrameResult) -> list[str]:
        pose = result.pose
        if result.detection is None:
            return [f"ANCHOR {self.config.object_id}: {pose.reason}"]
        valid_kpts = (
            int(np.count_nonzero(result.effective_visibility >= 1))
            if result.effective_visibility is not None
            else 0
        )
        lines = [
            f"ANCHOR {self.config.object_id}: {'VALID' if pose.valid else 'INVALID'} {pose.reason}",
            f"kpts={valid_kpts}/{len(self.config.keypoints)} inliers={pose.inlier_count}",
            f"skeleton_crossed={bool(result.skeleton_crossings)}",
        ]
        if pose.tvec is not None:
            tx, ty, tz = pose.tvec.reshape(3)
            lines.append(f"T_cam_obj=({tx:.3f},{ty:.3f},{tz:.3f})m")
        if pose.rpy_deg is not None:
            roll, pitch, yaw = pose.rpy_deg
            lines.append(f"RPY=({roll:.1f},{pitch:.1f},{yaw:.1f})deg")
        if pose.mean_reprojection_error_px is not None:
            lines.append(f"reproj={pose.mean_reprojection_error_px:.2f}px")
        return lines


def build_optional_object_anchor_runtime(
    raw: dict[str, Any] | None,
    *,
    repo_root: Path,
) -> tuple[ObjectAnchorRuntime | None, str]:
    cfg = raw if isinstance(raw, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return None, "disabled_by_config"

    config_value = cfg.get("config_path", "configs/object_anchors/tissue_box_01.yaml")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.is_file():
        return None, f"config_not_found:{config_path}"

    model_value = str(cfg.get("model_path") or "").strip()
    if not model_value:
        return None, "model_path_empty"
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not model_path.is_file():
        return None, f"model_not_found:{model_path}"

    anchor_config = load_object_anchor_config(config_path)
    detector = UltralyticsPoseDetector(
        str(model_path),
        confidence_threshold=float(cfg.get("detector_conf", 0.25)),
        iou_threshold=float(cfg.get("detector_iou", 0.45)),
        imgsz=int(cfg.get("imgsz", 640)) if cfg.get("imgsz") is not None else None,
        device=cfg.get("device"),
    )
    settings = ObjectAnchorRuntimeSettings(
        enabled=True,
        draw=bool(cfg.get("draw", True)),
        camera_pose_only=bool(cfg.get("camera_pose_only", True)),
        max_translation_jump_m=float(cfg.get("max_translation_jump_m", 0.25)),
        max_rotation_jump_deg=float(cfg.get("max_rotation_jump_deg", 35.0)),
        reset_after_missed_frames=max(1, int(cfg.get("reset_after_missed_frames", 15))),
    )
    return ObjectAnchorRuntime(anchor_config, detector, settings), "enabled"
