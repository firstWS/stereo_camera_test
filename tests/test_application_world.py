"""Application World compatibility layer tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from application_world import (  # noqa: E402
    DEFAULT_APPLICATION_WORLD_CONFIG_PATH,
    EXPECTED_TRANSLATION_DELTA_M,
    application_world_point_to_tag0_world,
    application_world_pose_to_tag0_world,
    load_application_world_contract,
    load_T_application_tag0,
    rotation_unchanged,
    tag0_world_point_to_application_world,
    tag0_world_pose_to_application_world,
)


def test_tag0_origin_maps_to_application_tag0_position() -> None:
    T = load_T_application_tag0()
    point = tag0_world_point_to_application_world(np.zeros(3), T_application_tag0=T)
    np.testing.assert_allclose(point, [1.0, 2.0, 0.0], atol=1e-12)


def test_arbitrary_point_uses_se3_transform() -> None:
    T = load_T_application_tag0()
    source = np.array([-0.073, -1.070, 2.161], dtype=np.float64)
    expected = T[:3, :3] @ source + T[:3, 3]
    actual = tag0_world_point_to_application_world(source, T_application_tag0=T)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual - source, EXPECTED_TRANSLATION_DELTA_M, atol=1e-12)


def test_pose_transform_composes_with_se3() -> None:
    T_application_tag0 = load_T_application_tag0()
    T_tag0_camera = np.eye(4, dtype=np.float64)
    T_tag0_camera[:3, 3] = np.array([-0.705, -0.870, 2.735], dtype=np.float64)
    T_application_camera = tag0_world_pose_to_application_world(
        T_tag0_camera,
        T_application_tag0=T_application_tag0,
    )
    np.testing.assert_allclose(
        T_application_camera,
        T_application_tag0 @ T_tag0_camera,
        atol=1e-12,
    )


def test_current_config_keeps_orientation_unchanged() -> None:
    T_application_tag0 = load_T_application_tag0()
    T_tag0_camera = np.eye(4, dtype=np.float64)
    T_tag0_camera[:3, :3] = np.array(
        [
            [0.98, -0.10, 0.05],
            [0.10, 0.99, 0.02],
            [-0.05, -0.01, 0.99],
        ],
        dtype=np.float64,
    )
    T_tag0_camera[:3, 3] = np.array([-0.707, -0.972, 2.685], dtype=np.float64)
    T_application_camera = tag0_world_pose_to_application_world(
        T_tag0_camera,
        T_application_tag0=T_application_tag0,
    )
    assert rotation_unchanged(T_tag0_camera[:3, :3], T_application_camera[:3, :3])
    np.testing.assert_allclose(
        T_application_camera[:3, 3] - T_tag0_camera[:3, 3],
        EXPECTED_TRANSLATION_DELTA_M,
        atol=1e-12,
    )


def test_round_trip_point_and_pose() -> None:
    T_application_tag0 = load_T_application_tag0()
    point = np.array([0.2, -0.4, 1.7], dtype=np.float64)
    round_trip_point = application_world_point_to_tag0_world(
        tag0_world_point_to_application_world(point, T_application_tag0=T_application_tag0),
        T_application_tag0=T_application_tag0,
    )
    np.testing.assert_allclose(round_trip_point, point, atol=1e-12)

    T_tag0 = np.eye(4, dtype=np.float64)
    T_tag0[:3, 3] = np.array([0.1, -0.2, 3.0], dtype=np.float64)
    round_trip_pose = application_world_pose_to_tag0_world(
        tag0_world_pose_to_application_world(T_tag0, T_application_tag0=T_application_tag0),
        T_application_tag0=T_application_tag0,
    )
    np.testing.assert_allclose(round_trip_pose, T_tag0, atol=1e-12)


def test_authoritative_config_contract() -> None:
    contract = load_application_world_contract()
    assert contract.config_path.endswith("orbbec_gemini.yaml")
    np.testing.assert_allclose(contract.T_application_tag0[:3, 3], [1.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(contract.T_application_tag0[:3, :3], np.eye(3), atol=1e-12)


@pytest.mark.skipif(
    not (ROOT / "out/analysis/phase45_application_world_sanity/application_world_sanity.json").is_file(),
    reason="sanity artifact not generated yet",
)
def test_frozen_artifact_smoke() -> None:
    payload = json.loads(
        (ROOT / "out/analysis/phase45_application_world_sanity/application_world_sanity.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["gate"] in {
        "APPLICATION_WORLD_COMPAT_READY",
        "APPLICATION_WORLD_COMPAT_READY_WITH_WARNING",
    }
    assert payload["tag0_origin_sanity"]["passes"] is True
    frame_248 = payload["representative_frames"]["248"]
    cup2 = frame_248["cups"]["cup2"]
    np.testing.assert_allclose(cup2["tag0_world_xyz_m"], [-0.0727, -1.0699, 2.1608], atol=1e-3)
    np.testing.assert_allclose(cup2["application_world_xyz_m"], [0.9273, 0.9301, 2.1608], atol=1e-3)
    assert cup2["delta_matches_expected"] is True
