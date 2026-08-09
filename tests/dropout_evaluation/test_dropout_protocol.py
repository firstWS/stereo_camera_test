from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    DropoutAnchorDefinition,
    DropoutProtocolConfig,
    FrameTimestamp,
    SuccessThresholds,
    build_dropout_manifest_payload,
    compute_dropout_window,
    format_duration_for_window_id,
    generate_dropout_windows,
    is_runtime_tag_available,
    is_runtime_tag_masked,
    load_dropout_protocol_config,
    write_dropout_manifest,
)
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


def test_format_duration_for_window_id_is_deterministic() -> None:
    assert format_duration_for_window_id(0.5) == "0.5s"
    assert format_duration_for_window_id(1.0) == "1.0s"
    assert format_duration_for_window_id(2.0) == "2.0s"
    assert format_duration_for_window_id(3.0) == "3.0s"
    assert format_duration_for_window_id(5.0) == "5.0s"


def test_half_open_interval_masks_boundary_excluded() -> None:
    anchor = DropoutAnchorDefinition(
        anchor_id="B_motion_start",
        start_frame=81,
        start_device_timestamp_us=1000,
        convention="first_sustained_motion_frame",
        motion_class="motion_start",
    )
    frames = [
        FrameTimestamp(frame_number=81, device_timestamp_us=1000),
        FrameTimestamp(frame_number=82, device_timestamp_us=1200),
        FrameTimestamp(frame_number=83, device_timestamp_us=1400),
        FrameTimestamp(frame_number=84, device_timestamp_us=1600),
        FrameTimestamp(frame_number=85, device_timestamp_us=1800),
        FrameTimestamp(frame_number=86, device_timestamp_us=2000),
    ]
    window = compute_dropout_window(
        anchor=anchor,
        duration_sec=0.001,
        session_id="test",
        frames=frames,
    )
    assert window.boundary_timestamp_us == 2000
    assert is_runtime_tag_masked(1000, window)
    assert is_runtime_tag_masked(1800, window)
    assert not is_runtime_tag_masked(2000, window)
    assert is_runtime_tag_available(2000, window)


def test_recovery_frame_is_first_at_or_after_boundary(official_windows) -> None:
    by_id = {window.window_id: window for window in official_windows}
    for window_id, (_, _, recovery) in OFFICIAL_WINDOWS.items():
        window = by_id[window_id]
        assert window.recovery_frame == recovery


def test_official_window_count_and_anchors(official_config, official_windows) -> None:
    assert len(official_windows) == 15
    assert tuple(duration for duration in official_config.durations_sec) == (0.5, 1.0, 2.0, 3.0, 5.0)
    anchor_frames = {anchor.start_frame for anchor in official_config.anchors}
    assert anchor_frames == {81, 202, 248}


def test_official_windows_match_expected_frames(official_windows) -> None:
    by_id = {window.window_id: window for window in official_windows}
    for window_id, (start, end, recovery) in OFFICIAL_WINDOWS.items():
        window = by_id[window_id]
        assert window.start_frame == start
        assert window.end_frame == end
        assert window.recovery_frame == recovery
        assert window.target_duration_sec > window.masked_sample_span_sec


def test_windows_are_generated_from_timestamps_not_hardcoded_frames(official_config, official_frames) -> None:
    shifted_frames = [
        FrameTimestamp(
            frame_number=frame.frame_number + 10_000,
            device_timestamp_us=frame.device_timestamp_us,
        )
        for frame in official_frames
    ]
    shifted_config = DropoutProtocolConfig(
        schema_version=official_config.schema_version,
        session_id=official_config.session_id,
        session_path=official_config.session_path,
        anchors=tuple(
            DropoutAnchorDefinition(
                anchor_id=anchor.anchor_id,
                start_frame=anchor.start_frame + 10_000,
                start_device_timestamp_us=anchor.start_device_timestamp_us,
                convention=anchor.convention,
                motion_class=anchor.motion_class,
            )
            for anchor in official_config.anchors
        ),
        durations_sec=official_config.durations_sec,
        mask_interval=official_config.mask_interval,
        reference_source=official_config.reference_source,
        reference_role=official_config.reference_role,
        cup2_semantic_id=official_config.cup2_semantic_id,
        cup2_observations_csv=official_config.cup2_observations_csv,
        success_thresholds=official_config.success_thresholds,
        output_root=official_config.output_root,
    )
    windows = generate_dropout_windows(shifted_config, shifted_frames)
    b_half = next(window for window in windows if window.window_id == "B_motion_start__0.5s")
    assert b_half.start_frame == 10_081
    assert b_half.end_frame == 10_095
    assert b_half.recovery_frame == 10_096


def test_b_anchor_half_open_recovery_boundary(official_windows) -> None:
    window = next(window for window in official_windows if window.window_id == "B_motion_start__0.5s")
    assert window.start_frame == 81
    assert window.end_frame == 95
    assert window.recovery_frame == 96
    assert window.frame_count == 15
    assert window.masked_sample_span_sec == pytest.approx(0.467, abs=0.01)


def test_missing_recovery_at_dataset_end() -> None:
    anchor = DropoutAnchorDefinition(
        anchor_id="tail",
        start_frame=1,
        start_device_timestamp_us=1_000_000,
        convention="test",
        motion_class="test",
    )
    frames = [
        FrameTimestamp(frame_number=1, device_timestamp_us=1_000_000),
        FrameTimestamp(frame_number=2, device_timestamp_us=1_500_000),
    ]
    window = compute_dropout_window(
        anchor=anchor,
        duration_sec=5.0,
        session_id="tail",
        frames=frames,
    )
    assert window.recovery_frame is None
    assert window.recovery_device_timestamp_us is None


def test_reference_mask_independence() -> None:
    anchor = DropoutAnchorDefinition(
        anchor_id="B_motion_start",
        start_frame=81,
        start_device_timestamp_us=29634275450,
        convention="first_sustained_motion_frame",
        motion_class="motion_start",
    )
    frames = [FrameTimestamp(frame_number=81, device_timestamp_us=29634275450)]
    window = compute_dropout_window(
        anchor=anchor,
        duration_sec=0.5,
        session_id="scenario_a",
        frames=frames * 20,
    )
    assert is_runtime_tag_masked(29634275450, window)
    assert not is_runtime_tag_available(29634275450, window)


def test_manifest_payload_is_deterministic(official_config, official_windows) -> None:
    payload_a = build_dropout_manifest_payload(official_config, official_windows)
    payload_b = build_dropout_manifest_payload(official_config, official_windows)
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)
    assert len(payload_a["windows"]) == 15
    assert payload_a["success_thresholds"]["pose_availability_min"] == 0.90


def test_write_dropout_manifest_to_evaluation_output(tmp_path, official_config, official_windows) -> None:
    output_path = tmp_path / "dropout_windows.json"
    write_dropout_manifest(config=official_config, windows=official_windows, output_path=output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "20260807_161354_scenario_a"
    assert payload["mask_semantics"]["interval"] == "half_open"
    assert payload["reference"]["role"] == "evaluation_only"


def test_config_loads_success_thresholds() -> None:
    config = load_dropout_protocol_config(OFFICIAL_CONFIG)
    assert config.success_thresholds.pose_availability_min == 0.90
    assert config.success_thresholds.cup2_world_median_max_m == 0.10
    assert config.cup2_semantic_id == "cup2"
