from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class RectificationMaps:
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray


@dataclass
class StereoCalibration:
    image_size: tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    maps: RectificationMaps | None = None

    def ensure_maps(self) -> RectificationMaps:
        if self.maps is not None:
            return self.maps
        w, h = self.image_size
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, (w, h), cv2.CV_32FC1
        )
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, (w, h), cv2.CV_32FC1
        )
        self.maps = RectificationMaps(map1_l, map2_l, map1_r, map2_r)
        return self.maps


def save_calibration(path: str | Path, calib: StereoCalibration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    w, h = calib.image_size
    fs.write("image_width", int(w))
    fs.write("image_height", int(h))
    fs.write("K1", calib.K1)
    fs.write("D1", calib.D1)
    fs.write("K2", calib.K2)
    fs.write("D2", calib.D2)
    fs.write("R", calib.R)
    fs.write("T", calib.T)
    fs.write("E", calib.E)
    fs.write("F", calib.F)
    fs.write("R1", calib.R1)
    fs.write("R2", calib.R2)
    fs.write("P1", calib.P1)
    fs.write("P2", calib.P2)
    fs.write("Q", calib.Q)
    fs.release()


def load_calibration(path: str | Path) -> StereoCalibration:
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    w_node = fs.getNode("image_width")
    h_node = fs.getNode("image_height")
    if w_node.empty() or h_node.empty():
        w_node = fs.getNode("imageSize_width")
        h_node = fs.getNode("imageSize_height")
    w = int(w_node.real())
    h = int(h_node.real())
    def mat(name: str) -> np.ndarray:
        m = fs.getNode(name).mat()
        if m is None or m.size == 0:
            raise KeyError(f"Missing matrix {name} in {path}")
        return m
    calib = StereoCalibration(
        image_size=(w, h),
        K1=mat("K1"),
        D1=mat("D1"),
        K2=mat("K2"),
        D2=mat("D2"),
        R=mat("R"),
        T=mat("T"),
        E=mat("E"),
        F=mat("F"),
        R1=mat("R1"),
        R2=mat("R2"),
        P1=mat("P1"),
        P2=mat("P2"),
        Q=mat("Q"),
        maps=None,
    )
    fs.release()
    return calib


def save_maps_npy(prefix: str | Path, maps: RectificationMaps) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(prefix) + "_map1_left.npy", maps.map1_left)
    np.save(str(prefix) + "_map2_left.npy", maps.map2_left)
    np.save(str(prefix) + "_map1_right.npy", maps.map1_right)
    np.save(str(prefix) + "_map2_right.npy", maps.map2_right)


def load_maps_npy(prefix: str | Path) -> RectificationMaps:
    prefix = Path(prefix)
    return RectificationMaps(
        map1_left=np.load(str(prefix) + "_map1_left.npy"),
        map2_left=np.load(str(prefix) + "_map2_left.npy"),
        map1_right=np.load(str(prefix) + "_map1_right.npy"),
        map2_right=np.load(str(prefix) + "_map2_right.npy"),
    )
