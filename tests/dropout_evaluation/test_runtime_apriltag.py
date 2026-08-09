"""Tests for runtime AprilTag loader (candidate path only)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    DropoutAnchorDefinition,
    FrameTimestamp,
    compute_dropout_window,
)
from dropout_evaluation.evaluation_metrics import PoseReference, evaluate_window  # noqa: E402
from dropout_evaluation.hold_last_pose import (  # noqa: E402
    generate_hold_last_pose_candidates,
    select_pre_window_anchor,
)
from dropout_evaluation.runtime_apriltag import (  # noqa: E402
    RuntimeAprilTagPose,
    RuntimeAprilTagPoseUnavailableError,
    load_runtime_apriltag_poses,
    observations_csv_supports_runtime_pose,
)

OFFICIAL_OBS = (
    ROOT
    / "out/datasets/gemini335l/20260807_161354_scenario_a/derived/apriltag/observations.csv"
)


def _write_observations_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_official_observations_schema_has_runtime_pose_columns() -> None:
    if not OFFICIAL_OBS.is_file():
        pytest.skip("official observations not available")
    assert observations_csv_supports_runtime_pose(OFFICIAL_OBS) is True


def test_official_observations_loader_succeeds_without_reference_csv(tmp_path) -> None:
    if not OFFICIAL_OBS.is_file():
        pytest.skip("official observations not available")
    copied = tmp_path / "observations.csv"
    copied.write_text(OFFICIAL_OBS.read_text(encoding="utf-8"), encoding="utf-8")
    poses = load_runtime_apriltag_poses(observations_csv=copied)
    assert len(poses) == 429


def test_loader_works_from_observations_csv_only(tmp_path) -> None:
    path = tmp_path / "observations.csv"
    _write_observations_csv(
        path,
        [
            {
                "frame_number": "80",
                "device_timestamp_us": "970000",
                "visible": "True",
                "pose_valid": "True",
                "world_tx": "1.0",
                "world_ty": "2.0",
                "world_tz": "3.0",
                "world_qw": "1.0",
                "world_qx": "0.0",
                "world_qy": "0.0",
                "world_qz": "0.0",
            },
            {
                "frame_number": "81",
                "device_timestamp_us": "1000000",
                "visible": "True",
                "pose_valid": "True",
                "world_tx": "9.0",
                "world_ty": "0.0",
                "world_tz": "0.0",
                "world_qw": "1.0",
                "world_qx": "0.0",
                "world_qy": "0.0",
                "world_qz": "0.0",
            },
        ],
        [
            "frame_number",
            "device_timestamp_us",
            "visible",
            "pose_valid",
            "world_tx",
            "world_ty",
            "world_tz",
            "world_qw",
            "world_qx",
            "world_qy",
            "world_qz",
        ],
    )
    poses = load_runtime_apriltag_poses(observations_csv=path)
    assert len(poses) == 2
    assert poses[0].frame_number == 80
    assert poses[0].T_world_camera[0, 3] == pytest.approx(1.0)
    assert poses[0].source == "runtime_apriltag"


def test_loader_skips_pose_valid_false_rows(tmp_path) -> None:
    path = tmp_path / "observations.csv"
    _write_observations_csv(
        path,
        [
            {
                "frame_number": "80",
                "device_timestamp_us": "970000",
                "visible": "True",
                "pose_valid": "False",
                "world_tx": "",
                "world_ty": "",
                "world_tz": "",
                "world_qw": "",
                "world_qx": "",
                "world_qy": "",
                "world_qz": "",
            },
            {
                "frame_number": "81",
                "device_timestamp_us": "1000000",
                "visible": "True",
                "pose_valid": "True",
                "world_tx": "1.0",
                "world_ty": "0.0",
                "world_tz": "0.0",
                "world_qw": "1.0",
                "world_qx": "0.0",
                "world_qy": "0.0",
                "world_qz": "0.0",
            },
        ],
        [
            "frame_number",
            "device_timestamp_us",
            "visible",
            "pose_valid",
            "world_tx",
            "world_ty",
            "world_tz",
            "world_qw",
            "world_qx",
            "world_qy",
            "world_qz",
        ],
    )
    poses = load_runtime_apriltag_poses(observations_csv=path)
    assert len(poses) == 1
    assert poses[0].frame_number == 81


def test_loader_does_not_require_reference_columns(tmp_path) -> None:
    path = tmp_path / "observations.csv"
    _write_observations_csv(
        path,
        [
            {
                "frame_number": "80",
                "device_timestamp_us": "970000",
                "visible": "True",
                "pose_valid": "True",
                "world_tx": "0.0",
                "world_ty": "0.0",
                "world_tz": "0.0",
                "world_qw": "1.0",
                "world_qx": "0.0",
                "world_qy": "0.0",
                "world_qz": "0.0",
            }
        ],
        [
            "frame_number",
            "device_timestamp_us",
            "visible",
            "pose_valid",
            "world_tx",
            "world_ty",
            "world_tz",
            "world_qw",
            "world_qx",
            "world_qy",
            "world_qz",
        ],
    )
    assert observations_csv_supports_runtime_pose(path) is True
    poses = load_runtime_apriltag_poses(observations_csv=path)
    assert len(poses) == 1


def test_loader_supports_anchor_and_recovery_selection(tmp_path) -> None:
    path = tmp_path / "observations.csv"
    _write_observations_csv(
        path,
        [
            {
                "frame_number": "80",
                "device_timestamp_us": "970000",
                "visible": "True",
                "pose_valid": "True",
                "world_tx": "1.0",
                "world_ty": "0.0",
                "world_tz": "0.0",
                "world_qw": "1.0",
                "world_qx": "0.0",
                "world_qy": "0.0",
                "world_qz": "0.0",
            },
            {
                "frame_number": "85",
                "device_timestamp_us": "1510000",
                "visible": "True",
                "pose_valid": "True",
                "world_tx": "2.0",
                "world_ty": "0.0",
                "world_tz": "0.0",
                "world_qw": "1.0",
                "world_qx": "0.0",
                "world_qy": "0.0",
                "world_qz": "0.0",
            },
        ],
        [
            "frame_number",
            "device_timestamp_us",
            "visible",
            "pose_valid",
            "world_tx",
            "world_ty",
            "world_tz",
            "world_qw",
            "world_qx",
            "world_qy",
            "world_qz",
        ],
    )
    poses = load_runtime_apriltag_poses(observations_csv=path)
    window = compute_dropout_window(
        anchor=DropoutAnchorDefinition(
            anchor_id="test",
            start_frame=81,
            start_device_timestamp_us=1_000_000,
            convention="test",
            motion_class="test",
        ),
        duration_sec=0.5,
        session_id="test",
        frames=[
            FrameTimestamp(80, 970_000),
            FrameTimestamp(81, 1_000_000),
            FrameTimestamp(82, 1_030_000),
            FrameTimestamp(83, 1_060_000),
            FrameTimestamp(84, 1_090_000),
            FrameTimestamp(85, 1_510_000),
        ],
    )
    anchor = select_pre_window_anchor(window=window, runtime_poses=poses, max_anchor_age_frames=120)
    assert anchor is not None
    assert anchor.frame_number == 80
    generation = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=poses,
        frame_timestamps=[
            FrameTimestamp(80, 970_000),
            FrameTimestamp(81, 1_000_000),
            FrameTimestamp(82, 1_030_000),
            FrameTimestamp(83, 1_060_000),
            FrameTimestamp(84, 1_090_000),
            FrameTimestamp(85, 1_510_000),
        ],
    )
    assert generation.provenance.recovery_actual_frame == 85


def test_candidate_generator_has_no_reference_arguments() -> None:
    signature = inspect.signature(generate_hold_last_pose_candidates)
    assert "references" not in signature.parameters
    assert "reference" not in signature.parameters


def test_candidate_output_invariant_when_reference_changes() -> None:
    anchor = DropoutAnchorDefinition(
        anchor_id="test",
        start_frame=81,
        start_device_timestamp_us=1_000_000,
        convention="test",
        motion_class="test",
    )
    frames = [
        FrameTimestamp(80, 970_000),
        FrameTimestamp(81, 1_000_000),
        FrameTimestamp(82, 1_030_000),
        FrameTimestamp(83, 1_060_000),
        FrameTimestamp(84, 1_090_000),
        FrameTimestamp(85, 1_510_000),
    ]
    window = compute_dropout_window(
        anchor=anchor,
        duration_sec=0.5,
        session_id="test",
        frames=frames,
    )
    runtime = [
        RuntimeAprilTagPose(
            frame_number=80,
            device_timestamp_us=970_000,
            T_world_camera=np.eye(4),
            valid=True,
        ),
        RuntimeAprilTagPose(
            frame_number=85,
            device_timestamp_us=1_510_000,
            T_world_camera=np.eye(4),
            valid=True,
        ),
    ]
    generation_a = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    generation_b = generate_hold_last_pose_candidates(
        window=window,
        runtime_poses=runtime,
        frame_timestamps=frames,
    )
    assert len(generation_a.candidates) == len(generation_b.candidates)
    for left, right in zip(generation_a.candidates, generation_b.candidates):
        assert left.frame_number == right.frame_number
        assert left.state == right.state
        assert np.allclose(left.T_world_camera, right.T_world_camera)

    references_a = [
        PoseReference(81, 1_000_000, np.eye(4), True),
        PoseReference(82, 1_030_000, np.eye(4), True),
        PoseReference(83, 1_060_000, np.eye(4), True),
        PoseReference(84, 1_090_000, np.eye(4), True),
    ]
    T_shift = np.eye(4)
    T_shift[0, 3] = 1.0
    references_b = [
        PoseReference(81, 1_000_000, T_shift.copy(), True),
        PoseReference(82, 1_030_000, T_shift.copy(), True),
        PoseReference(83, 1_060_000, T_shift.copy(), True),
        PoseReference(84, 1_090_000, T_shift.copy(), True),
    ]
    eval_a = evaluate_window(window=window, references=references_a, candidates=generation_a.candidates)
    eval_b = evaluate_window(window=window, references=references_b, candidates=generation_a.candidates)
    assert eval_a.pose.translation_error.median == pytest.approx(0.0, abs=1e-9)
    assert eval_b.pose.translation_error.median == pytest.approx(1.0, abs=1e-9)


def test_anchor_selected_from_runtime_pose_only() -> None:
    window = compute_dropout_window(
        anchor=DropoutAnchorDefinition(
            anchor_id="test",
            start_frame=81,
            start_device_timestamp_us=1_000_000,
            convention="test",
            motion_class="test",
        ),
        duration_sec=0.5,
        session_id="test",
        frames=[
            FrameTimestamp(80, 970_000),
            FrameTimestamp(81, 1_000_000),
            FrameTimestamp(82, 1_030_000),
            FrameTimestamp(83, 1_060_000),
            FrameTimestamp(84, 1_090_000),
            FrameTimestamp(85, 1_510_000),
        ],
    )
    runtime = [
        RuntimeAprilTagPose(
            frame_number=80,
            device_timestamp_us=970_000,
            T_world_camera=np.diag([1.0, 1.0, 1.0, 1.0]),
            valid=True,
        )
    ]
    anchor = select_pre_window_anchor(
        window=window,
        runtime_poses=runtime,
        max_anchor_age_frames=120,
    )
    assert anchor is not None
    assert anchor.frame_number == 80
