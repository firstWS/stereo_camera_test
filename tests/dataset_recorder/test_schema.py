from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.session_metadata import (  # noqa: E402
    load_scenario_file,
    validate_scenario_payload,
)
from dataset_recorder.types import SUPPORTED_SCENARIOS  # noqa: E402


def test_scenario_yaml_files_are_valid() -> None:
    for slug in SUPPORTED_SCENARIOS:
        payload = load_scenario_file(ROOT / "configs" / "dataset" / "scenarios" / f"{slug}.yaml")
        assert payload["scenario_slug"] == slug
        assert validate_scenario_payload(payload) == []


def test_scenario_a_is_yaw_pan_without_planned_translation() -> None:
    payload = load_scenario_file(ROOT / "configs" / "dataset" / "scenarios" / "scenario_a.yaml")
    assert payload["camera_motion"] == "rightward_yaw"
    assert payload["planned_translation_m"] is None
    assert 20.0 <= float(payload["planned_yaw_deg"]) <= 30.0
    assert payload["cup2"]["initial_state"] == "initially_hidden"
    windows = payload["planned_motion_windows"]
    assert windows["initial_hold_sec"] == [0.0, 3.0]
    assert windows["yaw_pan_sec"] == [3.0, 5.0]
    assert windows["final_hold_sec"] == [5.0, 15.0]


def test_scenario_b_has_yaw_range_metadata() -> None:
    payload = load_scenario_file(ROOT / "configs" / "dataset" / "scenarios" / "scenario_b.yaml")
    assert payload["camera_motion"] == "translation_yaw"
    assert 15.0 <= float(payload["planned_yaw_deg"]) <= 30.0
    assert payload["cup2"]["initial_state"] == "initially_hidden"


def test_missing_scenario_field_is_reported() -> None:
    errors = validate_scenario_payload({"scenario_slug": "scenario_a"})
    assert any("missing scenario field" in error for error in errors)


def test_null_planned_translation_is_allowed() -> None:
    payload = {
        "scenario_name": "x",
        "scenario_slug": "scenario_a",
        "camera_motion": "rightward_yaw",
        "anchor_visibility": "visible",
        "cup1_visibility": "visible",
        "cup2": {"initial_state": "initially_hidden"},
        "planned_translation_m": None,
        "planned_yaw_deg": 25.0,
        "planned_duration_sec": 15,
    }
    assert validate_scenario_payload(payload) == []


def test_null_planned_yaw_is_rejected() -> None:
    payload = {
        "scenario_name": "x",
        "scenario_slug": "scenario_a",
        "camera_motion": "rightward_yaw",
        "anchor_visibility": "visible",
        "cup1_visibility": "visible",
        "cup2": {"initial_state": "initially_hidden"},
        "planned_translation_m": None,
        "planned_yaw_deg": None,
        "planned_duration_sec": 15,
    }
    errors = validate_scenario_payload(payload)
    assert any("planned_yaw_deg" in error for error in errors)
