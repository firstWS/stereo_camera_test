"""AprilTag-referenced Object Anchor world validation, registration, and diagnostics."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from apriltag_world import AprilTagWorldResult
from object_anchor_pose import rotation_matrix_to_rpy_deg
from object_anchor_registration import load_world_pose_registration, save_world_pose_registration
from object_anchor_runtime import ObjectAnchorFrameResult


TRANSFORM_FORMULA = (
    "T_world_object = T_world_camera @ T_camera_object; "
    "T_world_camera = T_world_tag @ inverse(T_camera_tag)"
)


def rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return an `(x, y, z, w)` unit quaternion."""
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
        )
    else:
        index = int(np.argmax(np.diag(R)))
        if index == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
        elif index == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
    q /= np.linalg.norm(q)
    return q


def quaternion_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64).reshape(-1, 4)
    accumulator = np.zeros((4, 4), dtype=np.float64)
    reference = quaternions[0]
    for quaternion in quaternions:
        q = quaternion / np.linalg.norm(quaternion)
        if float(np.dot(q, reference)) < 0.0:
            q = -q
        accumulator += np.outer(q, q)
    values, vectors = np.linalg.eigh(accumulator)
    result = vectors[:, int(np.argmax(values))]
    if result[3] < 0.0:
        result = -result
    return result / np.linalg.norm(result)


def average_transforms(transforms: list[np.ndarray], *, position_median: bool = False) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one transform is required")
    matrices = [np.asarray(transform, dtype=np.float64).reshape(4, 4) for transform in transforms]
    positions = np.vstack([matrix[:3, 3] for matrix in matrices])
    quaternions = np.vstack([rotation_matrix_to_quaternion(matrix[:3, :3]) for matrix in matrices])
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.median(positions, axis=0) if position_median else np.mean(positions, axis=0)
    result[:3, :3] = quaternion_to_rotation_matrix(average_quaternions(quaternions))
    return result


@dataclass(frozen=True)
class ObjectAnchorWorldSettings:
    registration_target_frames: int = 100
    registration_min_seed_frames: int = 5
    registration_max_position_outlier_m: float = 0.10
    registration_max_rotation_outlier_deg: float = 20.0
    compare_max_position_error_m: float = 0.15
    compare_max_rotation_error_deg: float = 25.0
    statistics_duration_s: float = 20.0
    failure_save_enabled: bool = True
    failure_save_cooldown_s: float = 0.5
    failure_save_max_frames: int = 200


@dataclass
class ObjectAnchorWorldFrameResult:
    valid: bool
    accepted: bool
    reason: str
    T_world_camera: np.ndarray | None = None
    T_camera_object: np.ndarray | None = None
    T_world_object: np.ndarray | None = None
    source_tag_ids: tuple[int, ...] = ()
    apriltag_reprojection_error_px: float | None = None
    position_error_xyz_m: np.ndarray | None = None
    position_error_norm_m: float | None = None
    rotation_error_deg: float | None = None


def estimate_object_anchor_world_pose(
    apriltag_result: AprilTagWorldResult | None,
    anchor_result: ObjectAnchorFrameResult | None,
    *,
    registered_world_pose: np.ndarray | None = None,
    max_position_error_m: float = 0.15,
    max_rotation_error_deg: float = 25.0,
) -> ObjectAnchorWorldFrameResult:
    if anchor_result is None:
        return ObjectAnchorWorldFrameResult(False, False, "anchor_not_processed")
    if anchor_result.detection is None:
        return ObjectAnchorWorldFrameResult(False, False, anchor_result.pose.reason)
    if anchor_result.skeleton_crossings:
        return ObjectAnchorWorldFrameResult(False, False, "skeleton_crossing")
    if not anchor_result.pose.valid or anchor_result.pose.T_camera_object is None:
        return ObjectAnchorWorldFrameResult(False, False, anchor_result.pose.reason)
    if apriltag_result is None or not apriltag_result.observations:
        reason = "apriltag_not_visible" if apriltag_result is None else apriltag_result.notes
        return ObjectAnchorWorldFrameResult(False, False, reason)

    observations = apriltag_result.observations
    T_world_camera = average_transforms([obs.T_world_camera for obs in observations])
    T_camera_object = np.asarray(anchor_result.pose.T_camera_object, dtype=np.float64)
    T_world_object = T_world_camera @ T_camera_object
    result = ObjectAnchorWorldFrameResult(
        valid=True,
        accepted=True,
        reason="ok",
        T_world_camera=T_world_camera,
        T_camera_object=T_camera_object,
        T_world_object=T_world_object,
        source_tag_ids=tuple(obs.tag_id for obs in observations),
        apriltag_reprojection_error_px=float(
            np.mean([obs.reprojection_error_px for obs in observations])
        ),
    )
    if registered_world_pose is None:
        return result

    reference = np.asarray(registered_world_pose, dtype=np.float64).reshape(4, 4)
    delta = T_world_object[:3, 3] - reference[:3, 3]
    result.position_error_xyz_m = delta
    result.position_error_norm_m = float(np.linalg.norm(delta))
    result.rotation_error_deg = rotation_delta_deg(
        T_world_object[:3, :3], reference[:3, :3]
    )
    if result.position_error_norm_m > max_position_error_m:
        result.accepted = False
        result.reason = (
            f"registered_position_error:{result.position_error_norm_m:.3f}>"
            f"{max_position_error_m:.3f}m"
        )
    elif result.rotation_error_deg > max_rotation_error_deg:
        result.accepted = False
        result.reason = (
            f"registered_rotation_error:{result.rotation_error_deg:.1f}>"
            f"{max_rotation_error_deg:.1f}deg"
        )
    return result


@dataclass
class WorldPoseRegistrationCollector:
    target_frames: int
    min_seed_frames: int
    max_position_outlier_m: float
    max_rotation_outlier_deg: float
    samples: list[np.ndarray] = field(default_factory=list)
    excluded: Counter[str] = field(default_factory=Counter)
    last_reason: str = ""

    def add(self, result: ObjectAnchorWorldFrameResult) -> bool:
        if not result.valid or result.T_world_object is None:
            self.excluded[result.reason] += 1
            self.last_reason = result.reason
            return False
        candidate = np.asarray(result.T_world_object, dtype=np.float64).reshape(4, 4)
        if len(self.samples) >= self.min_seed_frames:
            reference = average_transforms(self.samples, position_median=True)
            position_delta = float(np.linalg.norm(candidate[:3, 3] - reference[:3, 3]))
            rotation_delta = rotation_delta_deg(candidate[:3, :3], reference[:3, :3])
            if position_delta > self.max_position_outlier_m:
                self.excluded["registration_position_outlier"] += 1
                self.last_reason = "registration_position_outlier"
                return False
            if rotation_delta > self.max_rotation_outlier_deg:
                self.excluded["registration_rotation_outlier"] += 1
                self.last_reason = "registration_rotation_outlier"
                return False
        self.samples.append(candidate.copy())
        self.last_reason = "accepted"
        return True

    @property
    def complete(self) -> bool:
        return len(self.samples) >= self.target_frames

    def finalize(self) -> np.ndarray:
        if not self.complete:
            raise RuntimeError(f"registration incomplete: {len(self.samples)}/{self.target_frames}")
        return average_transforms(self.samples, position_median=True)


class ObjectAnchorWorldTracker:
    """Stateful live diagnostics that never feeds back into the AprilTag pipeline."""

    CSV_FIELDS = (
        "frame_idx", "elapsed_s", "fps", "apriltag_valid", "apriltag_tag_ids",
        "anchor_detected", "skeleton_crossed", "pnp_valid", "pnp_inliers",
        "reprojection_error_px", "world_valid", "accepted", "reason",
        "cam_obj_x", "cam_obj_y", "cam_obj_z", "cam_obj_roll", "cam_obj_pitch", "cam_obj_yaw",
        "world_obj_x", "world_obj_y", "world_obj_z", "world_obj_roll", "world_obj_pitch", "world_obj_yaw",
        "error_x", "error_y", "error_z", "error_norm", "rotation_error_deg",
        "bbox_xyxy", "keypoints_xy_conf", "transform_formula",
    )

    def __init__(
        self,
        *,
        object_id: str,
        settings: ObjectAnchorWorldSettings,
        registration_file: Path,
        session_dir: Path,
        start_registration: bool,
        keypoint_names: tuple[str, ...] = (),
    ) -> None:
        self.object_id = object_id
        self.keypoint_names = keypoint_names
        self.settings = settings
        self.registration_file = registration_file
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir = self.session_dir / "failures"
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.perf_counter()
        self.stats_written = False
        self.total_frames = 0
        self.apriltag_success = 0
        self.anchor_detection_success = 0
        self.pnp_success = 0
        self.world_success = 0
        self.skeleton_cross_count = 0
        self.reject_reasons: Counter[str] = Counter()
        self.translations: list[np.ndarray] = []
        self.rpy_values: list[np.ndarray] = []
        self.fps_values: list[float] = []
        self.position_error_xyz_values: list[np.ndarray] = []
        self.position_errors: list[float] = []
        self.rotation_errors: list[float] = []
        self.reprojection_errors: list[float] = []
        self.failure_saved = 0
        self._last_failure_save: dict[str, float] = {}
        self.registered_world_pose: np.ndarray | None = None
        if registration_file.is_file() and not start_registration:
            self.registered_world_pose = load_world_pose_registration(
                registration_file, expected_object_id=object_id
            )
        self.registration = (
            WorldPoseRegistrationCollector(
                target_frames=settings.registration_target_frames,
                min_seed_frames=settings.registration_min_seed_frames,
                max_position_outlier_m=settings.registration_max_position_outlier_m,
                max_rotation_outlier_deg=settings.registration_max_rotation_outlier_deg,
            )
            if start_registration
            else None
        )
        self.csv_path = self.session_dir / "object_anchor_world_frames.csv"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv.writeheader()

    def _failure_reason(self, result: ObjectAnchorWorldFrameResult, anchor: ObjectAnchorFrameResult | None) -> str | None:
        if anchor is None or anchor.detection is None:
            return None
        if anchor.skeleton_crossings:
            return "skeleton_crossing"
        if anchor.pose.reason.startswith("insufficient_correspondences"):
            return "keypoint_confidence"
        if anchor.pose.reason.startswith("reprojection_error"):
            return "reprojection_error"
        if not anchor.pose.valid:
            return "pnp_failure"
        if result.reason.startswith("registered_position_error"):
            return "registered_position_error"
        if result.reason.startswith("registered_rotation_error"):
            return "registered_rotation_error"
        return None

    def _save_failure(self, reason: str, frame_idx: int, raw_bgr: np.ndarray, overlay_bgr: np.ndarray) -> None:
        if not self.settings.failure_save_enabled or self.failure_saved >= self.settings.failure_save_max_frames:
            return
        now = time.perf_counter()
        if now - self._last_failure_save.get(reason, -1e9) < self.settings.failure_save_cooldown_s:
            return
        self._last_failure_save[reason] = now
        safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason)[:80]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"{stamp}_f{frame_idx:06d}_{safe_reason}"
        cv2.imwrite(str(self.failure_dir / f"{stem}_raw.jpg"), raw_bgr)
        cv2.imwrite(str(self.failure_dir / f"{stem}_overlay.jpg"), overlay_bgr)
        self.failure_saved += 1

    def process(
        self,
        *,
        frame_idx: int,
        fps: float,
        apriltag_result: AprilTagWorldResult | None,
        anchor_result: ObjectAnchorFrameResult | None,
        raw_bgr: np.ndarray,
        overlay_bgr: np.ndarray,
    ) -> tuple[ObjectAnchorWorldFrameResult, list[str]]:
        world = estimate_object_anchor_world_pose(
            apriltag_result,
            anchor_result,
            registered_world_pose=self.registered_world_pose,
            max_position_error_m=self.settings.compare_max_position_error_m,
            max_rotation_error_deg=self.settings.compare_max_rotation_error_deg,
        )
        self.total_frames += 1
        if fps > 0.0 and np.isfinite(fps):
            self.fps_values.append(float(fps))
        if apriltag_result is not None and apriltag_result.observations:
            self.apriltag_success += 1
        if anchor_result is not None and anchor_result.detection is not None:
            self.anchor_detection_success += 1
        if anchor_result is not None and anchor_result.pose.valid:
            self.pnp_success += 1
        if world.valid:
            self.world_success += 1
            assert world.T_world_object is not None
            self.translations.append(world.T_world_object[:3, 3].copy())
            self.rpy_values.append(np.asarray(rotation_matrix_to_rpy_deg(world.T_world_object[:3, :3])))
        if anchor_result is not None and anchor_result.skeleton_crossings:
            self.skeleton_cross_count += 1
        if anchor_result is not None and anchor_result.pose.mean_reprojection_error_px is not None:
            self.reprojection_errors.append(anchor_result.pose.mean_reprojection_error_px)
        if world.position_error_norm_m is not None:
            self.position_errors.append(world.position_error_norm_m)
        if world.position_error_xyz_m is not None:
            self.position_error_xyz_values.append(world.position_error_xyz_m.copy())
        if world.rotation_error_deg is not None:
            self.rotation_errors.append(world.rotation_error_deg)
        if not world.accepted:
            self.reject_reasons[world.reason] += 1

        registration_message = ""
        if self.registration is not None and not self.registration.complete:
            added = self.registration.add(world)
            registration_message = (
                f"REG {len(self.registration.samples)}/{self.registration.target_frames}"
                + (" accepted" if added else f" rejected:{self.registration.last_reason}")
            )
            if self.registration.complete:
                self.registered_world_pose = self.registration.finalize()
                save_world_pose_registration(
                    self.registration_file,
                    object_id=self.object_id,
                    T_world_object=self.registered_world_pose,
                    source="apriltag_world_plus_front_only_pnp",
                    metadata={
                        "formula": TRANSFORM_FORMULA,
                        "used_frames": len(self.registration.samples),
                        "excluded_frames": int(sum(self.registration.excluded.values())),
                        "excluded_reasons": dict(self.registration.excluded),
                    },
                )
                registration_message = f"REG SAVED {self.registration_file}"

        detection = anchor_result.detection if anchor_result is not None else None
        pose = anchor_result.pose if anchor_result is not None else None
        cam_t = pose.T_camera_object[:3, 3] if pose is not None and pose.T_camera_object is not None else None
        cam_rpy = rotation_matrix_to_rpy_deg(pose.T_camera_object[:3, :3]) if pose is not None and pose.T_camera_object is not None else None
        world_t = world.T_world_object[:3, 3] if world.T_world_object is not None else None
        world_rpy = rotation_matrix_to_rpy_deg(world.T_world_object[:3, :3]) if world.T_world_object is not None else None
        bbox = detection.bbox_xyxy if detection is not None else None
        keypoints = []
        if detection is not None:
            keypoints = [
                {
                    "id": index,
                    "name": self.keypoint_names[index] if index < len(self.keypoint_names) else str(index),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "confidence": float(confidence),
                }
                for index, (point, confidence) in enumerate(
                    zip(detection.keypoints_xy, detection.keypoint_confidences)
                )
            ]
        error_xyz = world.position_error_xyz_m
        self._csv.writerow(
            {
                "frame_idx": frame_idx,
                "elapsed_s": time.perf_counter() - self.start_time,
                "fps": fps,
                "apriltag_valid": bool(apriltag_result and apriltag_result.observations),
                "apriltag_tag_ids": ",".join(map(str, world.source_tag_ids)),
                "anchor_detected": detection is not None,
                "skeleton_crossed": bool(anchor_result and anchor_result.skeleton_crossings),
                "pnp_valid": bool(pose and pose.valid),
                "pnp_inliers": pose.inlier_count if pose else 0,
                "reprojection_error_px": pose.mean_reprojection_error_px if pose else "",
                "world_valid": world.valid,
                "accepted": world.accepted,
                "reason": world.reason,
                "cam_obj_x": cam_t[0] if cam_t is not None else "",
                "cam_obj_y": cam_t[1] if cam_t is not None else "",
                "cam_obj_z": cam_t[2] if cam_t is not None else "",
                "cam_obj_roll": cam_rpy[0] if cam_rpy else "",
                "cam_obj_pitch": cam_rpy[1] if cam_rpy else "",
                "cam_obj_yaw": cam_rpy[2] if cam_rpy else "",
                "world_obj_x": world_t[0] if world_t is not None else "",
                "world_obj_y": world_t[1] if world_t is not None else "",
                "world_obj_z": world_t[2] if world_t is not None else "",
                "world_obj_roll": world_rpy[0] if world_rpy else "",
                "world_obj_pitch": world_rpy[1] if world_rpy else "",
                "world_obj_yaw": world_rpy[2] if world_rpy else "",
                "error_x": error_xyz[0] if error_xyz is not None else "",
                "error_y": error_xyz[1] if error_xyz is not None else "",
                "error_z": error_xyz[2] if error_xyz is not None else "",
                "error_norm": world.position_error_norm_m if world.position_error_norm_m is not None else "",
                "rotation_error_deg": world.rotation_error_deg if world.rotation_error_deg is not None else "",
                "bbox_xyxy": json.dumps(bbox),
                "keypoints_xy_conf": json.dumps(keypoints),
                "transform_formula": TRANSFORM_FORMULA,
            }
        )
        self._csv_file.flush()

        failure_reason = self._failure_reason(world, anchor_result)
        if failure_reason:
            self._save_failure(failure_reason, frame_idx, raw_bgr, overlay_bgr)

        lines = [f"WORLD ANCHOR: {'VALID' if world.valid else 'INVALID'} {world.reason}"]
        lines.append(f"formula: {TRANSFORM_FORMULA}")
        if world.T_world_camera is not None:
            t = world.T_world_camera[:3, 3]
            lines.append(f"T_world_camera=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})m tags={world.source_tag_ids}")
        if world_t is not None and world_rpy is not None:
            lines.append(f"T_world_tissue=({world_t[0]:.3f},{world_t[1]:.3f},{world_t[2]:.3f})m")
            lines.append(f"WORLD RPY=({world_rpy[0]:.1f},{world_rpy[1]:.1f},{world_rpy[2]:.1f})deg")
        if error_xyz is not None:
            lines.append(
                f"ERR xyz=({error_xyz[0]:+.3f},{error_xyz[1]:+.3f},{error_xyz[2]:+.3f})m "
                f"norm={world.position_error_norm_m:.3f}m rot={world.rotation_error_deg:.1f}deg"
            )
        if registration_message:
            lines.append(registration_message)
        lines.append(f"FPS={fps:.1f}")

        if not self.stats_written and time.perf_counter() - self.start_time >= self.settings.statistics_duration_s:
            self.write_summary(final=False)
            self.stats_written = True
        return world, lines

    @staticmethod
    def _array_stats(values: list[np.ndarray]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "std": None}
        array = np.vstack(values)
        return {
            "count": len(array),
            "mean": [float(value) for value in np.mean(array, axis=0)],
            "std": [float(value) for value in np.std(array, axis=0)],
        }

    @staticmethod
    def _scalar_stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def summary(self) -> dict[str, Any]:
        total = max(1, self.total_frames)
        return {
            "object_id": self.object_id,
            "duration_s": time.perf_counter() - self.start_time,
            "total_frames": self.total_frames,
            "success_rates": {
                "apriltag": self.apriltag_success / total,
                "object_anchor_detection": self.anchor_detection_success / total,
                "pnp": self.pnp_success / total,
                "world_pose": self.world_success / total,
            },
            "translation_world_m": self._array_stats(self.translations),
            "rotation_world_rpy_deg": self._array_stats(self.rpy_values),
            "fps": self._scalar_stats(self.fps_values),
            "registered_position_error_xyz_m": self._array_stats(self.position_error_xyz_values),
            "registered_position_error_m": self._scalar_stats(self.position_errors),
            "registered_rotation_error_deg": self._scalar_stats(self.rotation_errors),
            "reprojection_error_px": self._scalar_stats(self.reprojection_errors),
            "skeleton_cross_count": self.skeleton_cross_count,
            "reject_count": int(sum(self.reject_reasons.values())),
            "reject_ratio": float(sum(self.reject_reasons.values()) / total),
            "reject_reasons": dict(self.reject_reasons),
            "failure_frames_saved": self.failure_saved,
            "registration": None if self.registration is None else {
                "target_frames": self.registration.target_frames,
                "used_frames": len(self.registration.samples),
                "excluded_frames": int(sum(self.registration.excluded.values())),
                "excluded_reasons": dict(self.registration.excluded),
                "file": str(self.registration_file),
            },
            "coordinate_convention": {
                "unit": "meter",
                "camera_axes": "+X right, +Y down, +Z forward (OpenCV)",
                "object_axes": "+X right, +Y front-to-back, +Z bottom-to-top",
                "world_axes": "configured by apriltag_world front_normal/top_direction",
                "formula": TRANSFORM_FORMULA,
            },
        }

    def write_summary(self, *, final: bool) -> Path:
        path = self.session_dir / ("object_anchor_world_summary_final.json" if final else "object_anchor_world_summary.json")
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return path

    def close(self) -> Path:
        summary_path = self.write_summary(final=True)
        self._csv_file.close()
        return summary_path


def build_world_settings(raw: dict[str, Any] | None) -> ObjectAnchorWorldSettings:
    cfg = raw if isinstance(raw, dict) else {}
    return ObjectAnchorWorldSettings(
        registration_target_frames=max(1, int(cfg.get("registration_target_frames", 100))),
        registration_min_seed_frames=max(1, int(cfg.get("registration_min_seed_frames", 5))),
        registration_max_position_outlier_m=float(cfg.get("registration_max_position_outlier_m", 0.10)),
        registration_max_rotation_outlier_deg=float(cfg.get("registration_max_rotation_outlier_deg", 20.0)),
        compare_max_position_error_m=float(cfg.get("compare_max_position_error_m", 0.15)),
        compare_max_rotation_error_deg=float(cfg.get("compare_max_rotation_error_deg", 25.0)),
        statistics_duration_s=max(1.0, float(cfg.get("statistics_duration_s", 20.0))),
        failure_save_enabled=bool(cfg.get("failure_save_enabled", True)),
        failure_save_cooldown_s=max(0.0, float(cfg.get("failure_save_cooldown_s", 0.5))),
        failure_save_max_frames=max(0, int(cfg.get("failure_save_max_frames", 200))),
    )
