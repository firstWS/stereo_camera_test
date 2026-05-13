"""Stereo calibration from checkerboard image pairs + epipolar sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from calibration_repository import (
    RectificationMaps,
    StereoCalibration,
    save_calibration,
    save_maps_npy,
)


def _board_object_points(
    board_size: tuple[int, int], square_size_m: float
) -> np.ndarray:
    """board_size: inner corners (cols, rows)."""
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid.astype(np.float32) * square_size_m
    return objp


def find_corners_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    ok_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, flags)
    ok_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, flags)
    if not ok_l or not ok_r:
        return None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), criteria)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), criteria)
    return corners_l, corners_r


def calibrate_stereo_from_pairs(
    left_images: Sequence[np.ndarray],
    right_images: Sequence[np.ndarray],
    board_size: tuple[int, int],
    square_size_m: float,
) -> tuple[StereoCalibration, list[float]]:
    """
    board_size: (cols, rows) inner corners.
    Returns calibration and per-pair RMS corner errors after stereoCalibrate (placeholder list).
    """
    if len(left_images) != len(right_images):
        raise ValueError("left/right image counts must match")
    obj_point = _board_object_points(board_size, square_size_m)
    obj_points: list[np.ndarray] = []
    img_points_l: list[np.ndarray] = []
    img_points_r: list[np.ndarray] = []
    pair_rms: list[float] = []

    h, w = left_images[0].shape[:2]

    for li, ri in zip(left_images, right_images):
        cl, cr = find_corners_pair(li, ri, board_size)
        if cl is None or cr is None:
            continue
        pair_rms.append(0.0)
        obj_points.append(obj_point)
        img_points_l.append(cl)
        img_points_r.append(cr)

    if len(obj_points) < 3:
        raise RuntimeError(
            f"Need at least 3 valid stereo pairs; found {len(obj_points)}"
        )

    term = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    _, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, img_points_l, (w, h), None, None, criteria=term
    )
    _, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, img_points_r, (w, h), None, None, criteria=term
    )
    flags = cv2.CALIB_FIX_INTRINSIC
    _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_l,
        img_points_r,
        K1,
        D1,
        K2,
        D2,
        (w, h),
        criteria=term,
        flags=flags,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)

    calib = StereoCalibration(
        image_size=(w, h),
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        maps=None,
    )
    return calib, pair_rms


def mean_epipolar_y_shift_rectified(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    board_size: tuple[int, int],
) -> float | None:
    """After rectification, checkerboard corners should share same v in L/R."""
    cl, cr = find_corners_pair(left_rect, right_rect, board_size)
    if cl is None or cr is None:
        return None
    dy = np.abs(cl[:, 0, 1] - cr[:, 0, 1])
    return float(np.mean(dy))


def rectify_pair(
    left_bgr: np.ndarray, right_bgr: np.ndarray, maps: RectificationMaps
) -> tuple[np.ndarray, np.ndarray]:
    l = cv2.remap(left_bgr, maps.map1_left, maps.map2_left, cv2.INTER_LINEAR)
    r = cv2.remap(right_bgr, maps.map1_right, maps.map2_right, cv2.INTER_LINEAR)
    return l, r


def save_calib_bundle(
    out_yaml: str | Path,
    calib: StereoCalibration,
    maps_prefix: str | Path | None = None,
) -> None:
    out_yaml = Path(out_yaml)
    save_calibration(out_yaml, calib)
    if maps_prefix is not None:
        m = calib.ensure_maps()
        save_maps_npy(maps_prefix, m)
