from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.dropout_protocol import (  # noqa: E402
    DropoutAnchorDefinition,
    DropoutProtocolConfig,
    SuccessThresholds,
    compute_dropout_window,
    format_duration_for_window_id,
    generate_dropout_windows,
    is_runtime_tag_available,
    is_runtime_tag_masked,
    load_dropout_protocol_config,
    load_frame_timestamps_from_reference_csv,
)
from dropout_evaluation.evaluation_metrics import (  # noqa: E402
    CupObservation,
    MetricStatus,
    PoseEstimate,
    PoseReference,
    PoseTrackingState,
    apply_known_rotation_offset_deg,
    apply_known_translation_offset,
    compute_cup2_world_metrics,
    cup_world_position_error,
    evaluate_window,
    references_to_perfect_candidates,
    rotation_error_deg,
    translation_error,
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


@pytest.fixture(scope="module")
def official_frames():
    if not OFFICIAL_SESSION.is_dir():
        pytest.skip("official Scenario A session not available")
    reference_csv = OFFICIAL_SESSION / "derived/reference/apriltag_pose_smoothed.csv"
    if not reference_csv.is_file():
        pytest.skip("official reference CSV not available")
    return load_frame_timestamps_from_reference_csv(reference_csv)


@pytest.fixture(scope="module")
def official_config():
    return load_dropout_protocol_config(OFFICIAL_CONFIG)


@pytest.fixture(scope="module")
def official_windows(official_config, official_frames):
    return generate_dropout_windows(
        official_config,
        official_frames,
        session_dir=OFFICIAL_SESSION,
    )


def _T(translation: tuple[float, float, float], yaw_deg: float = 0.0) -> np.ndarray:
    theta = np.deg2rad(yaw_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, -s, 0.0, translation[0]],
            [s, c, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _reference(fn: int, ts: int, t=(0.0, 0.0, 0.0), yaw_deg=0.0) -> PoseReference:
    return PoseReference(
        frame_number=fn,
        device_timestamp_us=ts,
        T_world_camera=_T(t, yaw_deg),
        valid=True,
    )
