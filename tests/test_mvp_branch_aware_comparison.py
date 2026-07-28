from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULE_PATH = ROOT / "scripts" / "analyze_mvp_branch_aware_comparison.py"
SPEC = importlib.util.spec_from_file_location("analyze_mvp_branch_aware_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _T(x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    result[0, 3] = x
    return result


def test_relative_and_predict_roundtrip() -> None:
    T_camera_tag = _T(0.2, 10.0)
    T_camera_object = _T(0.5, -5.0)
    relative = MODULE.relative_tag_object(T_camera_tag, T_camera_object)
    predicted = MODULE.predict_camera_tag(T_camera_object, relative)
    np.testing.assert_allclose(predicted, T_camera_tag, atol=1e-9)


def test_decision_b_when_pose_ok_without_cup() -> None:
    summaries = {
        0: {
            "validation_frames": 40,
            "mvp_pose_passed": True,
        },
        1: {
            "validation_frames": 20,
            "mvp_pose_passed": True,
        },
    }
    decision = MODULE.decide_mvp(summaries, cup_data_present=False)
    assert decision["decision"] == "B"


def test_decision_c_when_within_branch_fails() -> None:
    summaries = {
        0: {"validation_frames": 40, "mvp_pose_passed": False},
        1: {"validation_frames": 20, "mvp_pose_passed": False},
    }
    decision = MODULE.decide_mvp(summaries, cup_data_present=False)
    assert decision["decision"] == "C"


def test_inventory_marks_cup_missing_on_saved_mvp_run() -> None:
    source = ROOT / "out/object_anchor_full99/mvp_final_comparison/20260726_163325"
    inventory = MODULE.inventory_source(source)
    assert inventory["sufficient_for_pose_analysis"] is True
    assert inventory["cup_data_present"] is False
    assert inventory["camera_required"] is False
