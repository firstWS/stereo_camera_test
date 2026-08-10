"""IR AprilTag region masking for Phase 4.5-M2 visual-input ablation (mask generation only)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from apriltag_scale import APRILTAG_DICTIONARIES
from dataset_recorder.rgb_depth_geometry import (
    _find_extrinsic,
    _find_intrinsic,
    _intrinsic_matrix,
    extrinsic_to_homogeneous_4x4,
)

from .stereo_imu_vio_lite import StereoImuVioFrameInput

DEFAULT_TAG_DICTIONARY = "APRILTAG_36H11"
DEFAULT_MASK_MARGIN_PX = 14
DEFAULT_TAG_SIZE_M = 0.135
MASK_INTERVAL_START_FRAME = 202
MASK_INTERVAL_RECOVERY_FRAME = 352
TAG_ROI_SOURCE_RGB_PROJECT_TO_IR = "rgb_detect_aruco_project_to_stereo_ir"


@dataclass(frozen=True)
class RgbIrMaskCalibration:
    k_rgb: np.ndarray
    d_rgb: np.ndarray
    k_left: np.ndarray
    d_left: np.ndarray
    k_right: np.ndarray
    d_right: np.ndarray
    t_rgb_left: np.ndarray
    t_rgb_right: np.ndarray
    tag_size_m: float = DEFAULT_TAG_SIZE_M


@dataclass(frozen=True)
class TagMaskRoi:
    detected: bool
    corners_xy: tuple[tuple[float, float], ...] | None
    mask_corners_xy: tuple[tuple[float, float], ...] | None
    bbox_xyxy: tuple[int, int, int, int] | None
    expanded_bbox_xyxy: tuple[int, int, int, int] | None
    area_ratio: float
    holdover: bool = False
    roi_source: str = "ir_aruco_detect"


@dataclass(frozen=True)
class FrameTagMaskDiagnostics:
    frame_number: int
    tag_mask_active: bool
    left: TagMaskRoi
    right: TagMaskRoi
    fill_value: int
    left_detection_failed: bool
    right_detection_failed: bool
    rgb_detection_failed: bool = False


def load_rgb_ir_mask_calibration(
    intrinsics: Mapping[str, Any],
    extrinsics: Mapping[str, Any],
    *,
    tag_size_m: float = DEFAULT_TAG_SIZE_M,
) -> RgbIrMaskCalibration:
    rgb_entry = _find_intrinsic(intrinsics, "RGB")
    left_entry = _find_intrinsic(intrinsics, "LEFT_IR")
    right_entry = _find_intrinsic(intrinsics, "RIGHT_IR")
    rgb_left = _find_extrinsic(extrinsics, "RGB", "LEFT_IR")
    rgb_right = _find_extrinsic(extrinsics, "RGB", "RIGHT_IR")
    if None in (rgb_entry, left_entry, right_entry, rgb_left, rgb_right):
        raise ValueError("RGB/LEFT_IR/RIGHT_IR intrinsics and RGB->IR extrinsics are required")
    return RgbIrMaskCalibration(
        k_rgb=_intrinsic_matrix(rgb_entry),
        d_rgb=_distortion_vector(rgb_entry),
        k_left=_intrinsic_matrix(left_entry),
        d_left=_distortion_vector(left_entry),
        k_right=_intrinsic_matrix(right_entry),
        d_right=_distortion_vector(right_entry),
        t_rgb_left=extrinsic_to_homogeneous_4x4(rgb_left["extrinsic"]),
        t_rgb_right=extrinsic_to_homogeneous_4x4(rgb_right["extrinsic"]),
        tag_size_m=tag_size_m,
    )


def _distortion_vector(entry: Mapping[str, Any]) -> np.ndarray:
    dist = entry.get("distortion") or {}
    return np.array(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("p1", 0.0)),
            float(dist.get("p2", 0.0)),
            float(dist.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )


def _create_detector(dictionary: str) -> cv2.aruco.ArucoDetector:
    if dictionary not in APRILTAG_DICTIONARIES:
        raise ValueError(f"Unsupported AprilTag dictionary: {dictionary}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARIES[dictionary])
    return cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _expand_corners(corners: np.ndarray, margin_px: float) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    centroid = pts.mean(axis=0)
    expanded: list[list[float]] = []
    for point in pts:
        direction = point - centroid
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            offset = np.array([margin_px, margin_px], dtype=np.float64)
        else:
            offset = direction / norm * margin_px
        expanded.append((point + offset).tolist())
    return np.asarray(expanded, dtype=np.float64)


def _bbox_from_points(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    xs = points[:, 0]
    ys = points[:, 1]
    x1 = int(max(0, np.floor(xs.min())))
    y1 = int(max(0, np.floor(ys.min())))
    x2 = int(min(width, np.ceil(xs.max()) + 1))
    y2 = int(min(height, np.ceil(ys.max()) + 1))
    return x1, y1, x2, y2


def _corners_to_tuple(corners: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in np.asarray(corners, dtype=np.float64).reshape(-1, 2))


def _roi_from_corners(
    corners: np.ndarray,
    *,
    width: int,
    height: int,
    margin_px: int,
    roi_source: str,
    holdover: bool = False,
) -> TagMaskRoi:
    raw_bbox = _bbox_from_points(corners, width, height)
    mask_corners = _expand_corners(corners, float(margin_px))
    expanded_bbox = _bbox_from_points(mask_corners, width, height)
    area = max(0, expanded_bbox[2] - expanded_bbox[0]) * max(0, expanded_bbox[3] - expanded_bbox[1])
    area_ratio = float(area / max(width * height, 1))
    return TagMaskRoi(
        detected=True,
        corners_xy=_corners_to_tuple(corners),
        mask_corners_xy=_corners_to_tuple(mask_corners),
        bbox_xyxy=raw_bbox,
        expanded_bbox_xyxy=expanded_bbox,
        area_ratio=area_ratio,
        holdover=holdover,
        roi_source=roi_source,
    )


def detect_tag_roi(
    gray: np.ndarray,
    *,
    detector: cv2.aruco.ArucoDetector,
    margin_px: int = DEFAULT_MASK_MARGIN_PX,
    preferred_tag_id: int | None = 0,
) -> TagMaskRoi:
    height, width = gray.shape[:2]
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return TagMaskRoi(False, None, None, None, None, 0.0)

    selected = 0
    if preferred_tag_id is not None:
        flat_ids = [int(v) for v in ids.reshape(-1)]
        if preferred_tag_id in flat_ids:
            selected = flat_ids.index(preferred_tag_id)

    corners = np.asarray(corners_list[selected], dtype=np.float64).reshape(-1, 2)
    return _roi_from_corners(
        corners,
        width=width,
        height=height,
        margin_px=margin_px,
        roi_source="ir_aruco_detect",
    )


def detect_rgb_tag_corners(
    rgb_gray: np.ndarray,
    *,
    detector: cv2.aruco.ArucoDetector,
    preferred_tag_id: int | None = 0,
) -> np.ndarray | None:
    corners_list, ids, _ = detector.detectMarkers(rgb_gray)
    if ids is None or len(ids) == 0:
        return None
    selected = 0
    if preferred_tag_id is not None:
        flat_ids = [int(v) for v in ids.reshape(-1)]
        if preferred_tag_id in flat_ids:
            selected = flat_ids.index(preferred_tag_id)
    return np.asarray(corners_list[selected], dtype=np.float64).reshape(-1, 2)


def project_rgb_tag_corners_to_ir(
    rgb_corners: np.ndarray,
    *,
    gray_shape: tuple[int, int],
    mask_calib: RgbIrMaskCalibration,
    t_rgb_to_ir: np.ndarray,
    k_ir: np.ndarray,
    margin_px: int,
) -> TagMaskRoi | None:
    height, width = gray_shape
    object_points = _tag_object_points(mask_calib.tag_size_m).astype(np.float32)
    image_points = np.asarray(rgb_corners, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        mask_calib.k_rgb,
        mask_calib.d_rgb,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    t_rgb_tag = np.eye(4, dtype=np.float64)
    t_rgb_tag[:3, :3] = rmat
    t_rgb_tag[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

    projected: list[list[float]] = []
    for point in object_points:
        p_rgb = t_rgb_tag @ np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float64)
        p_ir = t_rgb_to_ir @ p_rgb
        if p_ir[2] <= 1e-6:
            return None
        uv = k_ir @ p_ir[:3]
        projected.append([float(uv[0] / uv[2]), float(uv[1] / uv[2])])

    corners = np.asarray(projected, dtype=np.float64)
    if not np.isfinite(corners).all():
        return None
    return _roi_from_corners(
        corners,
        width=width,
        height=height,
        margin_px=margin_px,
        roi_source=TAG_ROI_SOURCE_RGB_PROJECT_TO_IR,
    )


def apply_tag_roi_mask(gray: np.ndarray, roi: TagMaskRoi, *, fill_value: int) -> np.ndarray:
    if roi.mask_corners_xy is None:
        return gray
    masked = gray.copy()
    polygon = np.round(np.asarray(roi.mask_corners_xy, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(masked, polygon, int(fill_value))
    return masked


def _roi_from_holdover(previous: TagMaskRoi) -> TagMaskRoi:
    return TagMaskRoi(
        detected=previous.detected,
        corners_xy=previous.corners_xy,
        mask_corners_xy=previous.mask_corners_xy,
        bbox_xyxy=previous.bbox_xyxy,
        expanded_bbox_xyxy=previous.expanded_bbox_xyxy,
        area_ratio=previous.area_ratio,
        holdover=True,
        roi_source=previous.roi_source,
    )


def is_frame_tag_mask_active(
    frame_number: int,
    *,
    start_frame: int = MASK_INTERVAL_START_FRAME,
    recovery_frame: int = MASK_INTERVAL_RECOVERY_FRAME,
) -> bool:
    return start_frame <= frame_number < recovery_frame


def _resolve_stereo_rois(
    frame: StereoImuVioFrameInput,
    *,
    detector: cv2.aruco.ArucoDetector,
    margin_px: int,
    preferred_tag_id: int | None,
    rgb_gray: np.ndarray | None,
    mask_calib: RgbIrMaskCalibration | None,
) -> tuple[TagMaskRoi, TagMaskRoi, bool]:
    left_height, left_width = frame.left_gray.shape[:2]
    right_height, right_width = frame.right_gray.shape[:2]

    left_roi = detect_tag_roi(
        frame.left_gray,
        detector=detector,
        margin_px=margin_px,
        preferred_tag_id=preferred_tag_id,
    )
    right_roi = detect_tag_roi(
        frame.right_gray,
        detector=detector,
        margin_px=margin_px,
        preferred_tag_id=preferred_tag_id,
    )
    rgb_failed = False

    if (not left_roi.detected or not right_roi.detected) and rgb_gray is not None and mask_calib is not None:
        rgb_corners = detect_rgb_tag_corners(
            rgb_gray,
            detector=detector,
            preferred_tag_id=preferred_tag_id,
        )
        if rgb_corners is None:
            rgb_failed = True
        else:
            if not left_roi.detected:
                projected_left = project_rgb_tag_corners_to_ir(
                    rgb_corners,
                    gray_shape=(left_height, left_width),
                    mask_calib=mask_calib,
                    t_rgb_to_ir=mask_calib.t_rgb_left,
                    k_ir=mask_calib.k_left,
                    margin_px=margin_px,
                )
                if projected_left is not None:
                    left_roi = projected_left
            if not right_roi.detected:
                projected_right = project_rgb_tag_corners_to_ir(
                    rgb_corners,
                    gray_shape=(right_height, right_width),
                    mask_calib=mask_calib,
                    t_rgb_to_ir=mask_calib.t_rgb_right,
                    k_ir=mask_calib.k_right,
                    margin_px=margin_px,
                )
                if projected_right is not None:
                    right_roi = projected_right
            if not left_roi.detected or not right_roi.detected:
                rgb_failed = True

    return left_roi, right_roi, rgb_failed


def apply_tag_mask_to_stereo_frames(
    frames: Sequence[StereoImuVioFrameInput],
    *,
    rgb_gray_by_frame: Mapping[int, np.ndarray] | None = None,
    mask_calib: RgbIrMaskCalibration | None = None,
    start_frame: int = MASK_INTERVAL_START_FRAME,
    recovery_frame: int = MASK_INTERVAL_RECOVERY_FRAME,
    dictionary: str = DEFAULT_TAG_DICTIONARY,
    margin_px: int = DEFAULT_MASK_MARGIN_PX,
    preferred_tag_id: int | None = 0,
) -> tuple[list[StereoImuVioFrameInput], list[FrameTagMaskDiagnostics]]:
    detector = _create_detector(dictionary)
    masked_frames: list[StereoImuVioFrameInput] = []
    diagnostics: list[FrameTagMaskDiagnostics] = []
    last_left_roi: TagMaskRoi | None = None
    last_right_roi: TagMaskRoi | None = None

    for frame in frames:
        active = is_frame_tag_mask_active(
            frame.frame_number,
            start_frame=start_frame,
            recovery_frame=recovery_frame,
        )
        rgb_gray = None if rgb_gray_by_frame is None else rgb_gray_by_frame.get(frame.frame_number)
        left_detected, right_detected, rgb_failed = _resolve_stereo_rois(
            frame,
            detector=detector,
            margin_px=margin_px,
            preferred_tag_id=preferred_tag_id,
            rgb_gray=rgb_gray,
            mask_calib=mask_calib,
        )

        if left_detected.detected:
            last_left_roi = left_detected
        if right_detected.detected:
            last_right_roi = right_detected

        if not active:
            masked_frames.append(frame)
            diagnostics.append(
                FrameTagMaskDiagnostics(
                    frame_number=frame.frame_number,
                    tag_mask_active=False,
                    left=TagMaskRoi(False, None, None, None, None, 0.0),
                    right=TagMaskRoi(False, None, None, None, None, 0.0),
                    fill_value=0,
                    left_detection_failed=False,
                    right_detection_failed=False,
                    rgb_detection_failed=False,
                )
            )
            continue

        fill_value = int(np.median(frame.left_gray))
        left_failed = not left_detected.detected
        right_failed = not right_detected.detected
        left_roi = left_detected
        right_roi = right_detected

        if left_failed:
            if last_left_roi is None:
                raise ValueError(f"No prior valid LEFT tag ROI before frame {frame.frame_number}")
            left_roi = _roi_from_holdover(last_left_roi)
        if right_failed:
            if last_right_roi is None:
                raise ValueError(f"No prior valid RIGHT tag ROI before frame {frame.frame_number}")
            right_roi = _roi_from_holdover(last_right_roi)

        masked_left = apply_tag_roi_mask(frame.left_gray, left_roi, fill_value=fill_value)
        masked_right = apply_tag_roi_mask(frame.right_gray, right_roi, fill_value=fill_value)
        masked_frames.append(
            replace(
                frame,
                left_gray=masked_left,
                right_gray=masked_right,
            )
        )
        diagnostics.append(
            FrameTagMaskDiagnostics(
                frame_number=frame.frame_number,
                tag_mask_active=True,
                left=left_roi,
                right=right_roi,
                fill_value=fill_value,
                left_detection_failed=left_failed,
                right_detection_failed=right_failed,
                rgb_detection_failed=rgb_failed,
            )
        )

    return masked_frames, diagnostics


def summarize_mask_diagnostics(diagnostics: Sequence[FrameTagMaskDiagnostics]) -> dict[str, Any]:
    active = [row for row in diagnostics if row.tag_mask_active]
    if not active:
        return {
            "masked_frame_count": 0,
            "left_area_ratio_median": None,
            "right_area_ratio_median": None,
            "left_detection_failures": 0,
            "right_detection_failures": 0,
            "left_holdover_frames": 0,
            "right_holdover_frames": 0,
            "rgb_detection_failures": 0,
            "roi_source_counts": {},
        }
    roi_sources: dict[str, int] = {}
    for row in active:
        for roi in (row.left, row.right):
            if roi.detected:
                roi_sources[roi.roi_source] = roi_sources.get(roi.roi_source, 0) + 1
    return {
        "masked_frame_count": len(active),
        "left_area_ratio_median": float(np.median([row.left.area_ratio for row in active])),
        "right_area_ratio_median": float(np.median([row.right.area_ratio for row in active])),
        "left_area_ratio_p90": float(np.quantile([row.left.area_ratio for row in active], 0.9)),
        "right_area_ratio_p90": float(np.quantile([row.right.area_ratio for row in active], 0.9)),
        "left_detection_failures": sum(1 for row in active if row.left_detection_failed),
        "right_detection_failures": sum(1 for row in active if row.right_detection_failed),
        "left_holdover_frames": sum(1 for row in active if row.left.holdover),
        "right_holdover_frames": sum(1 for row in active if row.right.holdover),
        "rgb_detection_failures": sum(1 for row in active if row.rgb_detection_failed),
        "roi_source_counts": roi_sources,
    }


def diagnostics_to_jsonable(diagnostics: Sequence[FrameTagMaskDiagnostics]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in diagnostics:
        rows.append(
            {
                "frame_number": row.frame_number,
                "tag_mask_active": row.tag_mask_active,
                "fill_value": row.fill_value,
                "left_detection_failed": row.left_detection_failed,
                "right_detection_failed": row.right_detection_failed,
                "rgb_detection_failed": row.rgb_detection_failed,
                "left": _roi_to_json(row.left),
                "right": _roi_to_json(row.right),
            }
        )
    return rows


def _roi_to_json(roi: TagMaskRoi) -> dict[str, Any]:
    return {
        "detected": roi.detected,
        "corners_xy": roi.corners_xy,
        "mask_corners_xy": roi.mask_corners_xy,
        "bbox_xyxy": roi.bbox_xyxy,
        "expanded_bbox_xyxy": roi.expanded_bbox_xyxy,
        "area_ratio": roi.area_ratio,
        "holdover": roi.holdover,
        "roi_source": roi.roi_source,
    }


def load_rgb_gray_by_canonical_frame(session_dir: Any) -> dict[int, np.ndarray]:
    from dataset_recorder.reader import DatasetReader

    from .canonical_frames import load_canonical_frames_from_rgb_index

    reader = DatasetReader(session_dir)
    canonical_numbers = [frame.frame_number for frame in load_canonical_frames_from_rgb_index(session_dir)]
    by_frame: dict[int, np.ndarray] = {}
    for frame_number, record in zip(canonical_numbers, reader.iterate_rgb()):
        if record.file_path is None or not record.file_path.is_file():
            continue
        image = cv2.imread(str(record.file_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            by_frame[frame_number] = image
    return by_frame
