"""AprilTag pose -> camera/world transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from apriltag_scale import _dict_id
from stereo_types import DepthEstimate


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


@dataclass
class AprilTagWorldObservation:
    tag_id: int
    T_camera_tag: np.ndarray
    T_world_tag: np.ndarray
    T_world_camera: np.ndarray
    reprojection_error_px: float


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


def _solve_marker_pose(
    image_points: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=flags)
    except cv2.error:
        ok = False
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
    if not ok:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
    if not ok:
        return None

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    diff = proj.reshape(-1, 2) - image_points.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(diff, axis=1)))
    R, _ = cv2.Rodrigues(rvec)
    return _homogeneous(R, tvec), rvec, reproj_err


def estimate_apriltag_world(
    gray_u8: np.ndarray,
    K: np.ndarray,
    cfg: AprilTagWorldConfig,
    *,
    draw_on_bgr: np.ndarray | None = None,
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
        solved = _solve_marker_pose(image_points, object_points, K.astype(np.float64), cfg.dist_coeffs)
        if solved is None:
            continue
        T_camera_tag, rvec, reproj_err = solved
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
