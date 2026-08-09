"""Offline AprilTag and cup detection for Phase 2 derived layer."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from apriltag_world import (
    AprilTagPoseSelectorState,
    build_apriltag_world_config,
    estimate_apriltag_world,
)
from detect import UltralyticsYOLODetector
from stereo_types import BBox

from .rgb_depth_geometry import (
    CupDepthEstimatorConfig,
    DepthPairingConfig,
    build_depth_timestamp_index,
    cup_depth_estimator_config_from_mapping,
    depth_meters_from_raw,
    depth_pairing_config_from_mapping,
    estimate_cup_xyz_from_rgb_bbox,
    load_rgb_depth_calibration,
    match_nearest_depth_timestamp,
)
from .cup_association import (
    AssociationConfig,
    CupDetectionRecord,
    associate_detections_to_tracks,
    association_config_from_mapping,
    order_cup_boxes,
)
from .object_annotations import (
    build_track_to_semantic_map,
    load_object_annotations,
    semantic_id_for_track,
)
from .apriltag_reference import (
    AprilTagPoseFrame,
    build_and_write_apriltag_reference,
    reference_config_from_mapping,
)
from .apriltag_runtime_pose import (
    RUNTIME_POSE_CSV_FIELDS,
    apriltag_observation_pose_columns,
)
from .reader import DatasetReader
from .session_metadata import write_json


def _rgb_intrinsic(calibration: Mapping[str, Any]) -> np.ndarray:
    for entry in calibration.get("intrinsics", []):
        if entry.get("frame") == "RGB" and entry.get("success"):
            intrinsic = entry.get("intrinsic") or {}
            fx = float(intrinsic.get("fx", 0.0))
            fy = float(intrinsic.get("fy", 0.0))
            cx = float(intrinsic.get("cx", 0.0))
            cy = float(intrinsic.get("cy", 0.0))
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError("RGB intrinsic unavailable in calibration snapshot")


def _estimate_camera_xyz_projected(
    bbox: BBox,
    depth_m: np.ndarray,
    calib: Any,
    *,
    min_valid_depth_ratio: float,
    z_min_m: float,
    z_max_m: float,
    cup_depth_config: CupDepthEstimatorConfig,
) -> tuple[bool, tuple[float | str, float | str, float | str]]:
    estimate = estimate_cup_xyz_from_rgb_bbox(
        depth_m,
        bbox,
        calib,
        min_valid_ratio=min_valid_depth_ratio,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        cup_depth_config=cup_depth_config,
    )
    if estimate.valid:
        return True, (estimate.X, estimate.Y, estimate.Z)
    return False, ("", "", "")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def derive_observations(
    session_dir: Path,
    *,
    apriltag_config: Mapping[str, Any] | None = None,
    detector: Any | None = None,
    min_valid_depth_ratio: float = 0.03,
    association_config: AssociationConfig | None = None,
    cup_mot_config: Mapping[str, Any] | None = None,
    depth_pairing_config: Mapping[str, Any] | None = None,
    cup_depth_config: Mapping[str, Any] | None = None,
    apriltag_reference_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import cv2

    session_dir = Path(session_dir)
    reader = DatasetReader(session_dir)
    derived_root = session_dir / "derived"
    apriltag_dir = derived_root / "apriltag"
    cups_dir = derived_root / "cups"
    annotations_dir = derived_root / "annotations"
    apriltag_dir.mkdir(parents=True, exist_ok=True)
    cups_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    atw_cfg = build_apriltag_world_config(dict(apriltag_config or {}))
    k = _rgb_intrinsic(reader.calibration_intrinsics())
    depth_cfg = depth_pairing_config_from_mapping(depth_pairing_config)
    cup_depth_cfg = cup_depth_estimator_config_from_mapping(cup_depth_config)
    rgb_depth_calib = None
    try:
        rgb_depth_calib = load_rgb_depth_calibration(
            reader.calibration_intrinsics(),
            reader.calibration_extrinsics(),
        )
    except ValueError as error:
        warnings_pre: list[str] = [f"rgb_depth_geometry unavailable: {error}"]
    else:
        warnings_pre = []
    yolo = detector or UltralyticsYOLODetector(class_ids=[41])
    assoc_cfg = association_config or association_config_from_mapping(cup_mot_config)
    reference_cfg = reference_config_from_mapping(apriltag_reference_config)

    apriltag_path = apriltag_dir / "observations.csv"
    detections_path = cups_dir / "detections.csv"
    tracks_path = cups_dir / "tracks.csv"
    observations_path = cups_dir / "observations.csv"
    track_summary_path = cups_dir / "track_summary.json"

    apriltag_fields = [
        "frame_number",
        "device_timestamp_us",
        "file_name",
        "tag_id",
        "visible",
        "reprojection_error_px",
        "notes",
        *RUNTIME_POSE_CSV_FIELDS,
    ]
    detection_fields = [
        "frame_number",
        "device_timestamp_us",
        "file_name",
        "detection_index",
        "class_id",
        "label",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "depth_valid",
        "camera_x",
        "camera_y",
        "camera_z",
        "depth_frame_number",
        "depth_device_timestamp_us",
        "rgb_depth_delta_us",
        "notes",
    ]
    track_fields = [
        "frame_number",
        "device_timestamp_us",
        "detection_index",
        "track_id",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    observation_fields = [
        "frame_number",
        "device_timestamp_us",
        "track_id",
        "semantic_id",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "depth_valid",
        "camera_x",
        "camera_y",
        "camera_z",
        "depth_frame_number",
        "depth_device_timestamp_us",
        "rgb_depth_delta_us",
        "notes",
    ]

    apriltag_rows: list[dict[str, Any]] = []
    detection_records: list[CupDetectionRecord] = []
    warnings: list[str] = []
    warnings.extend(warnings_pre)

    if not atw_cfg.enabled:
        warnings.append(
            "apriltag_world.enabled is false; AprilTag observations will not be generated."
        )

    depth_entries: list[tuple[dict[str, Any], Path | None]] = []
    for depth in reader.iterate_depth():
        depth_entries.append((dict(depth.row), depth.file_path))
    depth_timestamps, depth_rows, depth_paths = build_depth_timestamp_index(depth_entries)

    apriltag_pose_state = AprilTagPoseSelectorState() if atw_cfg.enabled else None
    apriltag_pose_frames: list[AprilTagPoseFrame] = []

    for rgb in reader.iterate_rgb():
        if rgb.file_path is None or not rgb.file_path.is_file():
            continue
        bgr = cv2.imread(str(rgb.file_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        frame_number = int(rgb.row.get("frame_number") or 0)
        device_ts = rgb.row.get("device_timestamp_us")
        file_name = rgb.row.get("file_name")
        image_h, image_w = bgr.shape[:2]

        t_world_camera: np.ndarray | None = None
        if atw_cfg.enabled:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            result = estimate_apriltag_world(
                gray,
                k,
                atw_cfg,
                pose_state=apriltag_pose_state,
                frame_number=frame_number,
                device_timestamp_us=int(device_ts) if device_ts not in (None, "") else None,
            )
            try:
                device_ts_i = int(device_ts) if device_ts not in (None, "") else None
            except (TypeError, ValueError):
                device_ts_i = None
            if result.observations:
                t_world_camera = np.asarray(
                    result.observations[0].T_world_camera, dtype=np.float64
                )
                for obs in result.observations:
                    notes = result.notes[:200]
                    if obs.pose_selection is not None:
                        sel = obs.pose_selection
                        notes = (
                            f"{notes};sel={sel.selection_reason};cand={sel.candidate_count}"
                        )[:200]
                    apriltag_rows.append(
                        {
                            "frame_number": frame_number,
                            "device_timestamp_us": device_ts,
                            "file_name": file_name,
                            "tag_id": obs.tag_id,
                            "visible": True,
                            "reprojection_error_px": obs.reprojection_error_px,
                            "notes": notes,
                            **apriltag_observation_pose_columns(
                                visible=True,
                                T_world_camera=t_world_camera,
                            ),
                        }
                    )
                apriltag_pose_frames.append(
                    AprilTagPoseFrame(
                        frame_number=frame_number,
                        device_timestamp_us=device_ts_i,
                        source_valid=True,
                        T_world_camera=t_world_camera.copy(),
                        reprojection_error_px=float(result.observations[0].reprojection_error_px),
                    )
                )
            else:
                apriltag_rows.append(
                    {
                        "frame_number": frame_number,
                        "device_timestamp_us": device_ts,
                        "file_name": file_name,
                        "tag_id": "",
                        "visible": False,
                        "reprojection_error_px": "",
                        "notes": result.notes[:200],
                        **apriltag_observation_pose_columns(
                            visible=False,
                            T_world_camera=None,
                        ),
                    }
                )
                apriltag_pose_frames.append(
                    AprilTagPoseFrame(
                        frame_number=frame_number,
                        device_timestamp_us=device_ts_i,
                        source_valid=False,
                        T_world_camera=None,
                        reprojection_error_px=None,
                    )
                )

        detection = yolo.predict(bgr)
        cup_boxes = [box for box in detection.boxes if int(box.class_id) == 41]
        ordered_boxes = order_cup_boxes(cup_boxes)

        depth_match = None
        depth_m = None
        try:
            rgb_ts = int(device_ts)
        except (TypeError, ValueError):
            rgb_ts = 0
        if rgb_ts > 0 and depth_timestamps.size > 0:
            depth_match = match_nearest_depth_timestamp(
                rgb_ts,
                depth_timestamps,
                depth_rows,
                depth_paths,
                max_delta_us=depth_cfg.max_rgb_depth_delta_us,
            )
        if depth_match is not None and depth_match.depth_file_path is not None:
            depth_path = Path(depth_match.depth_file_path)
            if depth_path.is_file():
                depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_image is not None:
                    depth_scale_raw = depth_match.depth_row.get("depth_scale")
                    depth_scale = (
                        None
                        if depth_scale_raw in (None, "")
                        else float(depth_scale_raw)
                    )
                    depth_m = depth_meters_from_raw(
                        depth_image,
                        depth_scale=depth_scale,
                        depth_is_millimeters=depth_cfg.depth_is_millimeters,
                    )

        for detection_index, box in enumerate(ordered_boxes):
            depth_valid = False
            camera_xyz: tuple[float | str, float | str, float | str] = ("", "", "")
            world_valid = False
            world_xyz: tuple[float | str, float | str, float | str] = ("", "", "")
            depth_frame_number: int | str = ""
            depth_device_timestamp_us: int | str = ""
            rgb_depth_delta_us: int | str = ""
            if depth_match is not None:
                depth_frame_number = depth_match.depth_frame_number
                depth_device_timestamp_us = depth_match.depth_device_timestamp_us
                rgb_depth_delta_us = depth_match.rgb_depth_delta_us
            if depth_m is not None and rgb_depth_calib is not None:
                depth_valid, camera_xyz = _estimate_camera_xyz_projected(
                    box,
                    depth_m,
                    rgb_depth_calib,
                    min_valid_depth_ratio=min_valid_depth_ratio,
                    z_min_m=depth_cfg.z_min_m,
                    z_max_m=depth_cfg.z_max_m,
                    cup_depth_config=cup_depth_cfg,
                )
            if depth_valid and t_world_camera is not None:
                try:
                    p_camera = np.array(
                        [float(camera_xyz[0]), float(camera_xyz[1]), float(camera_xyz[2]), 1.0],
                        dtype=np.float64,
                    )
                    p_world = t_world_camera @ p_camera
                    world_valid = True
                    world_xyz = (float(p_world[0]), float(p_world[1]), float(p_world[2]))
                except (TypeError, ValueError):
                    world_valid = False
            detection_records.append(
                CupDetectionRecord(
                    frame_number=frame_number,
                    device_timestamp_us=device_ts,
                    detection_index=detection_index,
                    class_id=int(box.class_id),
                    label=str(box.label or "cup"),
                    confidence=float(box.confidence),
                    bbox=tuple(float(v) for v in box.xyxy),
                    depth_valid=depth_valid,
                    camera_x=camera_xyz[0],
                    camera_y=camera_xyz[1],
                    camera_z=camera_xyz[2],
                    world_valid=world_valid,
                    world_x=world_xyz[0],
                    world_y=world_xyz[1],
                    world_z=world_xyz[2],
                    file_name=str(file_name or ""),
                    image_width=image_w,
                    image_height=image_h,
                    depth_frame_number=depth_frame_number,
                    depth_device_timestamp_us=depth_device_timestamp_us,
                    rgb_depth_delta_us=rgb_depth_delta_us,
                )
            )

    detection_rows = [
        {
            "frame_number": det.frame_number,
            "device_timestamp_us": det.device_timestamp_us,
            "file_name": det.file_name,
            "detection_index": det.detection_index,
            "class_id": det.class_id,
            "label": det.label,
            "confidence": det.confidence,
            "bbox_x1": det.bbox[0],
            "bbox_y1": det.bbox[1],
            "bbox_x2": det.bbox[2],
            "bbox_y2": det.bbox[3],
            "depth_valid": det.depth_valid,
            "camera_x": det.camera_x,
            "camera_y": det.camera_y,
            "camera_z": det.camera_z,
            "depth_frame_number": det.depth_frame_number,
            "depth_device_timestamp_us": det.depth_device_timestamp_us,
            "rgb_depth_delta_us": det.rgb_depth_delta_us,
            "notes": "derived_offline_not_ground_truth",
        }
        for det in detection_records
    ]
    _write_csv(apriltag_path, apriltag_fields, apriltag_rows)
    _write_csv(detections_path, detection_fields, detection_rows)

    reference_summary: dict[str, Any] | None = None
    if atw_cfg.enabled and reference_cfg.enabled and apriltag_pose_frames:
        reference_summary = build_and_write_apriltag_reference(
            session_dir,
            apriltag_pose_frames,
            reference_cfg,
        )

    assignments, track_summaries, mot_aggregate = associate_detections_to_tracks(
        detection_records,
        config=assoc_cfg,
    )
    detection_lookup = {
        (det.frame_number, det.detection_index): det for det in detection_records
    }
    track_rows = [
        {
            "frame_number": assignment.frame_number,
            "device_timestamp_us": assignment.device_timestamp_us,
            "detection_index": assignment.detection_index,
            "track_id": assignment.track_id,
            "confidence": assignment.confidence,
            "bbox_x1": assignment.bbox[0],
            "bbox_y1": assignment.bbox[1],
            "bbox_x2": assignment.bbox[2],
            "bbox_y2": assignment.bbox[3],
        }
        for assignment in assignments
    ]
    _write_csv(tracks_path, track_fields, track_rows)

    annotations = load_object_annotations(session_dir)
    track_to_semantic = build_track_to_semantic_map(annotations)
    if annotations is None:
        warnings.append(
            "objects.json annotation missing; semantic_id will be unknown until annotated."
        )

    observation_rows: list[dict[str, Any]] = []
    for assignment in assignments:
        det = detection_lookup[(assignment.frame_number, assignment.detection_index)]
        semantic_id = semantic_id_for_track(assignment.track_id, track_to_semantic)
        observation_rows.append(
            {
                "frame_number": assignment.frame_number,
                "device_timestamp_us": assignment.device_timestamp_us,
                "track_id": assignment.track_id,
                "semantic_id": semantic_id,
                "confidence": assignment.confidence,
                "bbox_x1": assignment.bbox[0],
                "bbox_y1": assignment.bbox[1],
                "bbox_x2": assignment.bbox[2],
                "bbox_y2": assignment.bbox[3],
                "depth_valid": det.depth_valid,
                "camera_x": det.camera_x,
                "camera_y": det.camera_y,
                "camera_z": det.camera_z,
                "depth_frame_number": det.depth_frame_number,
                "depth_device_timestamp_us": det.depth_device_timestamp_us,
                "rgb_depth_delta_us": det.rgb_depth_delta_us,
                "notes": "derived_offline_not_ground_truth",
            }
        )
    _write_csv(observations_path, observation_fields, observation_rows)

    track_summary_payload = {
        "schema_version": 2,
        "mot": mot_aggregate,
        "tracks": [summary.as_dict() for summary in track_summaries],
        "notes": "semantic_hint values are suggestions only; confirm via objects.json.",
    }
    write_json(track_summary_path, track_summary_payload)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_session": str(session_dir),
        "apriltag_enabled": atw_cfg.enabled,
        "apriltag_rows": len(apriltag_rows),
        "cup_detection_rows": len(detection_rows),
        "cup_track_rows": len(track_rows),
        "cup_observation_rows": len(observation_rows),
        "cup_rows": len(detection_rows),
        "annotation_present": annotations is not None,
        "apriltag_reference": reference_summary,
        "cup_depth": {
            "method": cup_depth_cfg.method,
            "near_quantile": cup_depth_cfg.near_quantile,
            "min_near_points": cup_depth_cfg.min_near_points,
        },
        "warnings": warnings,
        "notes": "Derived observations are not authoritative raw sensor data.",
    }
    write_json(derived_root / "manifest.json", manifest)
    return manifest
