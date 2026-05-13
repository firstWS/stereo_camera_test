"""Undistort + rectify stereo pair using maps from calibration."""

from __future__ import annotations

import cv2
import numpy as np

from calibration_repository import RectificationMaps, StereoCalibration
from stereo_types import StereoFrame


def rectify_stereo_frame(frame: StereoFrame, maps: RectificationMaps) -> StereoFrame:
    left = cv2.remap(frame.left_bgr, maps.map1_left, maps.map2_left, cv2.INTER_LINEAR)
    right = cv2.remap(
        frame.right_bgr, maps.map1_right, maps.map2_right, cv2.INTER_LINEAR
    )
    return StereoFrame(left_bgr=left, right_bgr=right)


def prepare_maps_if_needed(calib: StereoCalibration) -> RectificationMaps:
    return calib.ensure_maps()


def crop_to_calib_size(frame: StereoFrame, calib: StereoCalibration) -> StereoFrame:
    w, h = calib.image_size
    hl, wl = frame.left_bgr.shape[:2]
    if wl == w and hl == h:
        return frame
    return StereoFrame(
        left_bgr=cv2.resize(frame.left_bgr, (w, h), interpolation=cv2.INTER_AREA),
        right_bgr=cv2.resize(frame.right_bgr, (w, h), interpolation=cv2.INTER_AREA),
    )
