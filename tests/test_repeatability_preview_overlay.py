from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "repeatability_run_preview_test",
    ROOT / "experiments" / "repeatability_run.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_default_overlay_hides_red_development_lines(monkeypatch) -> None:
    texts: list[str] = []
    original = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        texts.append(str(text))
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    RUNNER._annotate_left(
        image,
        [],
        extra_lines=[
            "solver=IPPE",
            "rvec=(...)",
            "T_world_camera=(...)",
            "Pose difference diagnostic",
        ],
        debug_overlay_enabled=False,
    )
    assert texts == []


def test_debug_overlay_restores_development_lines(monkeypatch) -> None:
    texts: list[str] = []
    original = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        texts.append(str(text))
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    RUNNER._annotate_left(
        image,
        [],
        extra_lines=["solver=IPPE", "T_world_camera=(...)"],
        debug_overlay_enabled=True,
    )
    assert "solver=IPPE" in texts
    assert "T_world_camera=(...)" in texts
