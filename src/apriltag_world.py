"""AprilTag pose -> camera/world transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from apriltag_scale import _dict_id
from stereo_types import DepthEstimate

DEFAULT_MAX_CONTINUITY_FRAME_GAP = 90
DEFAULT_MAX_TRANSLATION_JUMP_M = 1.0
DEFAULT_MAX_ROTATION_JUMP_DEG = 60.0
TRANSLATION_SCORE_SCALE_M = 0.05
ROTATION_SCORE_SCALE_DEG = 5.0
WEIGHT_TRANSLATION = 1.0
WEIGHT_ROTATION = 1.0
WEIGHT_FRONT = 0.15
WEIGHT_REPROJ = 0.05
MIN_TAG_DEPTH_M = 0.05
ROTATION_DETERMINANT_TOLERANCE = 1e-3


SelectionReason = Literal[
    "initial_front_facing",
    "temporal_continuity",
    "single_candidate",
    "fallback",
]


@dataclass
class AprilTagWorldTag:
    tag_id: int
    T_world_tag: np.ndarray


@dataclass
class AprilTagWorldConfig:
    enabled: bool
    dictionary: str
    tag_size_m: float
    tags: dict[int, AprilTagWorldTag]
    dist_coeffs: np.ndarray
    draw: bool = True


@dataclass(frozen=True)
class PoseSelectionDiagnostics:
    candidate_count: int
    selected_candidate_index: int
    selection_reason: SelectionReason
    reprojection_error_px: float
    continuity_translation_m: float | None = None
    continuity_rotation_deg: float | None = None
    front_alignment: float | None = None


@dataclass
class AprilTagWorldObservation:
    tag_id: int
    T_camera_tag: np.ndarray
    T_world_tag: np.ndarray
    T_world_camera: np.ndarray
    reprojection_error_px: float
    pose_selection: PoseSelectionDiagnostics | None = None


@dataclass
class AprilTagPoseSelectorState:
    """Explicit per-session pose continuity state; caller-owned, not global."""

    T_world_camera: np.ndarray | None = None
    frame_number: int | None = None
    device_timestamp_us: int | None = None
    max_continuity_frame_gap: int = DEFAULT_MAX_CONTINUITY_FRAME_GAP

    def reset(self) -> None:
        self.T_world_camera = None
        self.frame_number = None
        self.device_timestamp_us = None

    def continuity_expired(
        self,
        frame_number: int | None,
        device_timestamp_us: int | None,
    ) -> bool:
        if self.T_world_camera is None:
            return True
        if frame_number is not None and self.frame_number is not None:
            return int(frame_number) - int(self.frame_number) > self.max_continuity_frame_gap
        if device_timestamp_us is not None and self.device_timestamp_us is not None:
            gap_us = int(device_timestamp_us) - int(self.device_timestamp_us)
            return gap_us > self.max_continuity_frame_gap * 33_333
        return False

    def update(
        self,
        T_world_camera: np.ndarray,
        *,
        frame_number: int | None,
        device_timestamp_us: int | None,
    ) -> None:
        self.T_world_camera = np.asarray(T_world_camera, dtype=np.float64)
        self.frame_number = frame_number
        self.device_timestamp_us = device_timestamp_us


@dataclass(frozen=True)
class _MarkerPoseCandidate:
    index: int
    T_camera_tag: np.ndarray
    rvec: np.ndarray
    reprojection_error_px: float
    tag_normal_camera: np.ndarray
    front_alignment: float
    determinant: float


@dataclass
class AprilTagWorldResult:
    observations: list[AprilTagWorldObservation]
    notes: str

    @property
    def visible_tag_ids(self) -> tuple[int, ...]:
        return tuple(obs.tag_id for obs in self.observations)


@dataclass
class WorldPointEstimate:
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    valid: bool = False
    source_tag_ids: tuple[int, ...] = ()
    notes: str = ""


def _axis_vector(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _unit(np.array(value, dtype=np.float64))

    text = str(value).strip().upper().replace("WORLD", "").replace(" ", "")
    if not text:
        raise ValueError("empty axis value")
    sign = -1.0 if text.startswith("-") else 1.0
    axis = text[-1]
    vec = np.zeros(3, dtype=np.float64)
    if axis == "X":
        vec[0] = sign
    elif axis == "Y":
        vec[1] = sign
    elif axis == "Z":
        vec[2] = sign
    else:
        raise ValueError(f"Unknown axis value {value!r}; expected +X, -X, +Y, -Y, +Z, or -Z")
    return vec


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return v / n


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.astype(np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _rotation_about_z_local(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _world_from_tag(
    position: Any,
    *,
    top_direction: Any,
    front_normal: Any,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    y_world = _axis_vector(top_direction)
    z_world = _axis_vector(front_normal)
    if abs(float(np.dot(y_world, z_world))) > 1e-6:
        raise ValueError("top_direction and front_normal must be perpendicular")
    x_world = _unit(np.cross(y_world, z_world))
    y_world = _unit(np.cross(z_world, x_world))
    R_world_tag = np.column_stack([x_world, y_world, z_world])
    if abs(float(yaw_deg)) > 1e-9:
        R_world_tag = R_world_tag @ _rotation_about_z_local(yaw_deg)
    return _homogeneous(R_world_tag, pos)


def build_apriltag_world_config(raw: dict[str, Any] | None) -> AprilTagWorldConfig:
    cfg = raw or {}
    enabled = bool(cfg.get("enabled", False))
    dictionary = str(cfg.get("dictionary", "APRILTAG_36H11"))
    tag_size_m = float(cfg.get("tag_size_m", cfg.get("tag_size", 0.0)))
    if enabled and tag_size_m <= 0.0:
        raise ValueError("apriltag_world.tag_size_m must be positive when enabled")

    default_top = cfg.get("top_direction", "+Y")
    default_normal = cfg.get("front_normal", "+Z")
    tags_raw = cfg.get("tags") or {}
    tags: dict[int, AprilTagWorldTag] = {}
    if isinstance(tags_raw, dict):
        items = tags_raw.items()
    else:
        items = [(entry.get("id"), entry) for entry in tags_raw]

    for raw_id, tag_cfg_any in items:
        if raw_id is None:
            continue
        tag_id = int(raw_id)
        tag_cfg = tag_cfg_any if isinstance(tag_cfg_any, dict) else {"position": tag_cfg_any}
        position = tag_cfg.get("position")
        if position is None:
            raise ValueError(f"apriltag_world.tags.{tag_id}.position is required")
        T_world_tag = _world_from_tag(
            position,
            top_direction=tag_cfg.get("top_direction", default_top),
            front_normal=tag_cfg.get("front_normal", default_normal),
            yaw_deg=float(tag_cfg.get("yaw_deg", 0.0)),
        )
        tags[tag_id] = AprilTagWorldTag(tag_id=tag_id, T_world_tag=T_world_tag)

    if enabled and not tags:
        raise ValueError("apriltag_world.tags must contain at least one tag when enabled")

    dist_raw = cfg.get("dist_coeffs", cfg.get("distortion_coeffs"))
    if dist_raw is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    else:
        dist_coeffs = np.asarray(dist_raw, dtype=np.float64).reshape(-1, 1)

    return AprilTagWorldConfig(
        enabled=enabled,
        dictionary=dictionary,
        tag_size_m=tag_size_m,
        tags=tags,
        dist_coeffs=dist_coeffs,
        draw=bool(cfg.get("draw", True)),
    )


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    s = float(tag_size_m) * 0.5
    return np.array(
        [
            [-s, s, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=np.float32,
    )


def _tag_front_normal_tag_frame() -> np.ndarray:
    """Tag-frame unit vector for configured front_normal (+Z convention)."""
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    diff = proj.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _rotation_delta_deg(R_prev: np.ndarray, R_cand: np.ndarray) -> float:
    R_delta = R_prev.T @ R_cand
    cosine = float(np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _candidate_from_rvec_tvec(
    index: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> _MarkerPoseCandidate | None:
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    R, _ = cv2.Rodrigues(rvec)
    determinant = float(np.linalg.det(R))
    if not np.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_DETERMINANT_TOLERANCE:
        return None

    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    object_cam = (R @ object_points.T).T + t.reshape(1, 3)
    if not np.all(np.isfinite(object_cam)) or float(np.min(object_cam[:, 2])) < MIN_TAG_DEPTH_M:
        return None

    tag_normal_camera = R @ _tag_front_normal_tag_frame()
    front_alignment = float(np.dot(tag_normal_camera, -_tag_front_normal_tag_frame()))
    reproj_err = _reprojection_error_px(object_points, image_points, rvec, tvec, K, dist_coeffs)
    return _MarkerPoseCandidate(
        index=index,
        T_camera_tag=_homogeneous(R, t),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
        reprojection_error_px=reproj_err,
        tag_normal_camera=tag_normal_camera,
        front_alignment=front_alignment,
        determinant=determinant,
    )


def _enumerate_marker_pose_candidates(
    image_points: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[_MarkerPoseCandidate]:
    candidates: list[_MarkerPoseCandidate] = []
    flags_ippe = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
    if flags_ippe is not None and hasattr(cv2, "solvePnPGeneric"):
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points,
                image_points,
                K,
                dist_coeffs,
                flags=flags_ippe,
            )
            if ok:
                for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    candidate = _candidate_from_rvec_tvec(
                        index,
                        rvec,
                        tvec,
                        object_points,
                        image_points,
                        K,
                        dist_coeffs,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        except cv2.error:
            pass

    if candidates:
        return candidates

    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return []
    if not ok:
        return []
    candidate = _candidate_from_rvec_tvec(
        0,
        rvec,
        tvec,
        object_points,
        image_points,
        K,
        dist_coeffs,
    )
    return [candidate] if candidate is not None else []


def _world_camera_from_candidate(
    candidate: _MarkerPoseCandidate,
    T_world_tag: np.ndarray,
) -> np.ndarray:
    return T_world_tag @ np.linalg.inv(candidate.T_camera_tag)


def _continuity_metrics(
    previous_T_world_camera: np.ndarray,
    candidate_T_world_camera: np.ndarray,
) -> tuple[float, float]:
    t_prev = previous_T_world_camera[:3, 3]
    t_cand = candidate_T_world_camera[:3, 3]
    translation_m = float(np.linalg.norm(t_cand - t_prev))
    rotation_deg = _rotation_delta_deg(
        previous_T_world_camera[:3, :3],
        candidate_T_world_camera[:3, :3],
    )
    return translation_m, rotation_deg


def _candidate_score(
    candidate: _MarkerPoseCandidate,
    *,
    previous_T_world_camera: np.ndarray | None,
    T_world_tag: np.ndarray,
    use_temporal: bool,
) -> tuple[float, float | None, float | None]:
    continuity_translation_m: float | None = None
    continuity_rotation_deg: float | None = None
    front_penalty = 1.0 - candidate.front_alignment
    score = WEIGHT_FRONT * front_penalty + WEIGHT_REPROJ * candidate.reprojection_error_px

    if use_temporal and previous_T_world_camera is not None:
        T_world_camera = _world_camera_from_candidate(candidate, T_world_tag)
        continuity_translation_m, continuity_rotation_deg = _continuity_metrics(
            previous_T_world_camera,
            T_world_camera,
        )
        score += WEIGHT_TRANSLATION * (
            continuity_translation_m / TRANSLATION_SCORE_SCALE_M
        )
        score += WEIGHT_ROTATION * (
            continuity_rotation_deg / ROTATION_SCORE_SCALE_DEG
        )
    return score, continuity_translation_m, continuity_rotation_deg


def _select_marker_pose_candidate(
    candidates: list[_MarkerPoseCandidate],
    T_world_tag: np.ndarray,
    *,
    previous_T_world_camera: np.ndarray | None,
    reinitialize: bool,
) -> tuple[_MarkerPoseCandidate | None, PoseSelectionDiagnostics | None]:
    if not candidates:
        return None, None

    use_temporal = previous_T_world_camera is not None and not reinitialize
    scored: list[tuple[float, _MarkerPoseCandidate, float | None, float | None]] = []
    for candidate in candidates:
        if use_temporal and previous_T_world_camera is not None:
            T_world_camera = _world_camera_from_candidate(candidate, T_world_tag)
            translation_m, rotation_deg = _continuity_metrics(
                previous_T_world_camera,
                T_world_camera,
            )
            if (
                translation_m > DEFAULT_MAX_TRANSLATION_JUMP_M
                or rotation_deg > DEFAULT_MAX_ROTATION_JUMP_DEG
            ):
                continue
        score, translation_m, rotation_deg = _candidate_score(
            candidate,
            previous_T_world_camera=previous_T_world_camera,
            T_world_tag=T_world_tag,
            use_temporal=use_temporal,
        )
        scored.append((score, candidate, translation_m, rotation_deg))

    reason: SelectionReason
    if scored:
        score, selected, translation_m, rotation_deg = min(scored, key=lambda item: item[0])
        reason = "temporal_continuity" if use_temporal else "initial_front_facing"
        if len(candidates) == 1:
            reason = "single_candidate"
    else:
        fallback_scored: list[tuple[float, _MarkerPoseCandidate, float | None, float | None]] = []
        for candidate in candidates:
            score, translation_m, rotation_deg = _candidate_score(
                candidate,
                previous_T_world_camera=previous_T_world_camera,
                T_world_tag=T_world_tag,
                use_temporal=False,
            )
            fallback_scored.append((score, candidate, translation_m, rotation_deg))
        score, selected, translation_m, rotation_deg = min(
            fallback_scored,
            key=lambda item: item[0],
        )
        reason = "fallback"

    diagnostics = PoseSelectionDiagnostics(
        candidate_count=len(candidates),
        selected_candidate_index=selected.index,
        selection_reason=reason,
        reprojection_error_px=selected.reprojection_error_px,
        continuity_translation_m=translation_m,
        continuity_rotation_deg=rotation_deg,
        front_alignment=selected.front_alignment,
    )
    return selected, diagnostics


def _solve_marker_pose(
    image_points: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    T_world_tag: np.ndarray,
    *,
    pose_state: AprilTagPoseSelectorState | None = None,
    frame_number: int | None = None,
    device_timestamp_us: int | None = None,
    commit_state: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, PoseSelectionDiagnostics] | None:
    candidates = _enumerate_marker_pose_candidates(
        image_points,
        object_points,
        K,
        dist_coeffs,
    )
    reinitialize = (
        pose_state is None
        or pose_state.continuity_expired(frame_number, device_timestamp_us)
    )
    previous = None if reinitialize else pose_state.T_world_camera if pose_state else None
    selected, diagnostics = _select_marker_pose_candidate(
        candidates,
        T_world_tag,
        previous_T_world_camera=previous,
        reinitialize=reinitialize,
    )
    if selected is None or diagnostics is None:
        return None

    T_world_camera = _world_camera_from_candidate(selected, T_world_tag)
    if pose_state is not None and commit_state:
        pose_state.update(
            T_world_camera,
            frame_number=frame_number,
            device_timestamp_us=device_timestamp_us,
        )
    return (
        selected.T_camera_tag,
        selected.rvec.reshape(3, 1),
        selected.reprojection_error_px,
        diagnostics,
    )


def rotation_delta_deg(R_prev: np.ndarray, R_cand: np.ndarray) -> float:
    """SO(3) geodesic angle in degrees between two rotation matrices."""
    return _rotation_delta_deg(R_prev, R_cand)


def estimate_apriltag_world(
    gray_u8: np.ndarray,
    K: np.ndarray,
    cfg: AprilTagWorldConfig,
    *,
    draw_on_bgr: np.ndarray | None = None,
    pose_state: AprilTagPoseSelectorState | None = None,
    frame_number: int | None = None,
    device_timestamp_us: int | None = None,
) -> AprilTagWorldResult:
    if not cfg.enabled:
        return AprilTagWorldResult([], "disabled")

    aruco_dict = cv2.aruco.getPredefinedDictionary(_dict_id(cfg.dictionary))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray_u8)
    if ids is None or len(ids) == 0:
        return AprilTagWorldResult([], "no_tags_visible")

    object_points = _tag_object_points(cfg.tag_size_m)
    observations: list[AprilTagWorldObservation] = []
    for i in range(len(ids)):
        tag_id = int(ids[i][0])
        tag_world = cfg.tags.get(tag_id)
        if tag_world is None:
            continue
        image_points = corners[i].reshape(4, 2).astype(np.float32)
        solved = _solve_marker_pose(
            image_points,
            object_points,
            K.astype(np.float64),
            cfg.dist_coeffs,
            tag_world.T_world_tag,
            pose_state=pose_state,
            frame_number=frame_number,
            device_timestamp_us=device_timestamp_us,
            commit_state=False,
        )
        if solved is None:
            continue
        T_camera_tag, rvec, reproj_err, pose_diag = solved
        try:
            T_world_camera = tag_world.T_world_tag @ np.linalg.inv(T_camera_tag)
        except np.linalg.LinAlgError:
            continue

        observations.append(
            AprilTagWorldObservation(
                tag_id=tag_id,
                T_camera_tag=T_camera_tag,
                T_world_tag=tag_world.T_world_tag,
                T_world_camera=T_world_camera,
                reprojection_error_px=reproj_err,
                pose_selection=pose_diag,
            )
        )

        if draw_on_bgr is not None:
            pts = image_points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(draw_on_bgr, [pts], True, (255, 128, 0), 2)
            c = np.mean(image_points, axis=0)
            cv2.putText(
                draw_on_bgr,
                f"WTag {tag_id}",
                (int(c[0]) + 6, int(c[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            try:
                cv2.drawFrameAxes(
                    draw_on_bgr,
                    K.astype(np.float64),
                    cfg.dist_coeffs,
                    rvec,
                    T_camera_tag[:3, 3].reshape(3, 1),
                    cfg.tag_size_m * 0.5,
                )
            except cv2.error:
                pass

    observations.sort(key=lambda obs: (obs.reprojection_error_px, obs.tag_id))
    if pose_state is not None and observations:
        best = observations[0]
        pose_state.update(
            best.T_world_camera,
            frame_number=frame_number,
            device_timestamp_us=device_timestamp_us,
        )
    if not observations:
        visible = ",".join(str(int(v[0])) for v in ids.reshape(-1))
        return AprilTagWorldResult([], f"no_configured_tags_visible visible={visible}")

    return AprilTagWorldResult(observations, "ok")


def world_point_from_camera_estimate(
    est: DepthEstimate,
    pose: AprilTagWorldResult | None,
) -> WorldPointEstimate:
    if pose is None:
        return WorldPointEstimate(valid=False, notes="world_pose_not_computed")
    if not pose.observations:
        return WorldPointEstimate(valid=False, notes=pose.notes)
    if not est.valid:
        return WorldPointEstimate(valid=False, notes="invalid_camera_point")

    p_camera = np.array([est.X, est.Y, est.Z, 1.0], dtype=np.float64)
    points = []
    tag_ids = []
    for obs in pose.observations:
        p_world = obs.T_world_camera @ p_camera
        points.append(p_world[:3])
        tag_ids.append(obs.tag_id)
    pts = np.vstack(points)
    mean = np.mean(pts, axis=0)

    notes = "ok"
    if len(points) > 1:
        spread = float(np.max(np.linalg.norm(pts - mean, axis=1)))
        notes = f"avg_from_tags max_spread_m={spread:.4f}"

    return WorldPointEstimate(
        X=float(mean[0]),
        Y=float(mean[1]),
        Z=float(mean[2]),
        valid=True,
        source_tag_ids=tuple(tag_ids),
        notes=notes,
    )
