from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    Cup2FirstAppearance,
    build_dropout_manifest_payload,
    generate_dropout_windows,
    load_dropout_protocol_config,
    load_frame_timestamps_from_reference_csv,
    resolve_cup2_first_appearance,
    window_includes_cup2_first_appearance,
)

OFFICIAL_SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
OFFICIAL_CONFIG = ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml"

OFFICIAL_WINDOWS = {
    "B_motion_start__0.5s": (81, 95, 96),
    "B_motion_start__1.0s": (81, 110, 111),
    "B_motion_start__2.0s": (81, 140, 141),
    "B_motion_start__3.0s": (81, 170, 171),
    "B_motion_start__5.0s": (81, 230, 231),
    "C_pre_cup2__0.5s": (202, 216, 217),
    "C_pre_cup2__1.0s": (202, 231, 232),
    "C_pre_cup2__2.0s": (202, 261, 262),
    "C_pre_cup2__3.0s": (202, 291, 292),
    "C_pre_cup2__5.0s": (202, 351, 352),
    "D_active_with_cup2__0.5s": (248, 262, 263),
    "D_active_with_cup2__1.0s": (248, 277, 278),
    "D_active_with_cup2__2.0s": (248, 307, 308),
    "D_active_with_cup2__3.0s": (248, 337, 338),
    "D_active_with_cup2__5.0s": (248, 397, 398),
}

EXPECTED_INCLUDES = {
    "B_motion_start__0.5s": False,
    "B_motion_start__1.0s": False,
    "B_motion_start__2.0s": False,
    "B_motion_start__3.0s": False,
    "B_motion_start__5.0s": True,
    "C_pre_cup2__0.5s": True,
    "C_pre_cup2__1.0s": True,
    "C_pre_cup2__2.0s": True,
    "C_pre_cup2__3.0s": True,
    "C_pre_cup2__5.0s": True,
    "D_active_with_cup2__0.5s": False,
    "D_active_with_cup2__1.0s": False,
    "D_active_with_cup2__2.0s": False,
    "D_active_with_cup2__3.0s": False,
    "D_active_with_cup2__5.0s": False,
}


@pytest.fixture(scope="module")
def official_cup2_first_appearance():
    if not OFFICIAL_SESSION.is_dir():
        pytest.skip("official Scenario A session not available")
    observations_csv = OFFICIAL_SESSION / "derived/cups/observations.csv"
    if not observations_csv.is_file():
        pytest.skip("official cup observations not available")
    appearance = resolve_cup2_first_appearance(observations_csv, semantic_id="cup2")
    assert appearance is not None
    return appearance


@pytest.fixture(scope="module")
def official_windows_with_cup2(official_config, official_frames, official_cup2_first_appearance):
    return generate_dropout_windows(
        official_config,
        official_frames,
        cup2_first_appearance=official_cup2_first_appearance,
    )


def test_resolve_cup2_first_appearance_from_observations(official_cup2_first_appearance) -> None:
    assert official_cup2_first_appearance.frame_number == 203
    assert official_cup2_first_appearance.device_timestamp_us > 0


@pytest.mark.parametrize(
    ("window_id", "expected"),
    list(EXPECTED_INCLUDES.items()),
    ids=list(EXPECTED_INCLUDES.keys()),
)
def test_official_includes_cup2_appearance_per_window(
    official_windows_with_cup2,
    window_id: str,
    expected: bool,
) -> None:
    by_id = {window.window_id: window for window in official_windows_with_cup2}
    assert by_id[window_id].includes_cup2_appearance is expected


def test_half_open_event_semantics_at_start_boundary() -> None:
    assert window_includes_cup2_first_appearance(
        start_timestamp_us=1000,
        boundary_timestamp_us=2000,
        event_timestamp_us=1000,
    )


def test_half_open_event_semantics_at_boundary_is_false() -> None:
    assert not window_includes_cup2_first_appearance(
        start_timestamp_us=1000,
        boundary_timestamp_us=2000,
        event_timestamp_us=2000,
    )


def test_half_open_event_semantics_just_before_boundary() -> None:
    assert window_includes_cup2_first_appearance(
        start_timestamp_us=1000,
        boundary_timestamp_us=2000,
        event_timestamp_us=1999,
    )


def test_official_window_ranges_unchanged(official_windows_with_cup2) -> None:
    by_id = {window.window_id: window for window in official_windows_with_cup2}
    for window_id, (start, end, recovery) in OFFICIAL_WINDOWS.items():
        window = by_id[window_id]
        assert window.start_frame == start
        assert window.end_frame == end
        assert window.recovery_frame == recovery


def test_manifest_includes_cup2_first_appearance_provenance(
    official_config,
    official_windows_with_cup2,
    official_cup2_first_appearance,
) -> None:
    payload = build_dropout_manifest_payload(
        official_config,
        official_windows_with_cup2,
        cup2_first_appearance=official_cup2_first_appearance,
    )
    cup2 = payload["cup2"]
    assert cup2["first_appearance_frame"] == official_cup2_first_appearance.frame_number
    assert (
        cup2["first_appearance_device_timestamp_us"]
        == official_cup2_first_appearance.device_timestamp_us
    )
    assert "includes_cup2_appearance" not in payload["anchors"][0]


def test_manifest_determinism_with_cup2_metadata(
    official_config,
    official_windows_with_cup2,
    official_cup2_first_appearance,
) -> None:
    payload_a = build_dropout_manifest_payload(
        official_config,
        official_windows_with_cup2,
        cup2_first_appearance=official_cup2_first_appearance,
    )
    payload_b = build_dropout_manifest_payload(
        official_config,
        official_windows_with_cup2,
        cup2_first_appearance=official_cup2_first_appearance,
    )
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)
