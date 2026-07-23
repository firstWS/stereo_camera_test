#!/usr/bin/env python3
"""Verify the single Python environment used by camera and Object Anchor code."""

from __future__ import annotations

import platform
import sys

import cv2
import numpy
import pytest
import torch
import ultralytics
import yaml
from importlib.metadata import version

import pyorbbecsdk


def main() -> None:
    values = {
        "Python": platform.python_version(),
        "OpenCV": cv2.__version__,
        "Ultralytics": ultralytics.__version__,
        "Orbbec SDK Python binding": version("pyorbbecsdk2"),
        "NumPy": numpy.__version__,
        "PyYAML": yaml.__version__,
        "pytest": pytest.__version__,
        "PyTorch": torch.__version__,
    }
    for name, value in values.items():
        print(f"{name}: {value}")
    print(f"Python executable: {sys.executable}")
    print(f"pyorbbecsdk module: {pyorbbecsdk.__file__}")

    errors: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 is required")
    if not hasattr(cv2, "imshow") or not hasattr(cv2, "waitKey"):
        errors.append("OpenCV GUI functions are unavailable")
    if not hasattr(cv2, "aruco"):
        errors.append("cv2.aruco is unavailable")
    if errors:
        raise SystemExit("Environment verification failed: " + "; ".join(errors))
    print("environment_ok")


if __name__ == "__main__":
    main()
