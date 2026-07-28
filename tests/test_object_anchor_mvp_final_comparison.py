from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULE_PATH = ROOT / "experiments" / "object_anchor_mvp_final_comparison.py"
SPEC = importlib.util.spec_from_file_location(
    "object_anchor_mvp_final_comparison", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _transform(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    result[0, 3] = x
    return result


def test_mvp_config_uses_only_apriltag_id_0_and_branch_aware() -> None:
    config_path = ROOT / "configs/experiments/orbbec_gemini_object_anchor_mvp_final.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    MODULE.assert_single_apriltag_id0(config)
    assert sorted(int(key) for key in config["apriltag_world"]["tags"]) == [0]
    assert config["apriltag_scale"]["enabled"] is False
    settings = MODULE.load_mvp_settings(config)
    assert settings["branch_aware"] is True
    assert settings["use_single_reference_cluster_gate"] is False
    assert int(settings["temporal_filter_window"]) == 3
    assert int(settings["branch_min_joint_valid"]) == 30
    assert int(settings["branch_min_inliers"]) == 20


def test_assign_branches_does_not_require_95pct_majority() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "frame_idx": index,
                "T_world_camera_tag_json": MODULE._json_matrix(_transform(0.0)),
            }
        )
    for index in range(10, 16):
        rows.append(
            {
                "frame_idx": index,
                "T_world_camera_tag_json": MODULE._json_matrix(_transform(2.5, 55.0)),
            }
        )
    settings = MODULE.load_mvp_settings({})
    assigned, info, centroids = MODULE.assign_april_tag_branches(rows, settings)
    assert info["single_reference_cluster_gate_used"] is False
    assert info["pose_clusters"]["cluster_count"] == 2
    assert len(centroids) == 2
    assert any(row["pose_cluster_id"] == 0 for row in assigned)
    assert any(row["pose_cluster_id"] == 1 for row in assigned)


def test_relative_predict_roundtrip_and_branch_assignment() -> None:
    T_camera_tag = _transform(0.2, 8.0)
    T_camera_object = _transform(0.4, -3.0)
    relative = MODULE.relative_tag_object(T_camera_tag, T_camera_object)
    predicted = MODULE.predict_camera_tag(T_camera_object, relative)
    np.testing.assert_allclose(predicted, T_camera_tag, atol=1e-9)

    centroids = {0: _transform(1.0), 1: _transform(3.5, 50.0)}
    assert MODULE.assign_branch_id(
        _transform(1.02, 1.0),
        centroids,
        translation_threshold_m=0.25,
        rotation_threshold_deg=20.0,
    ) == 0
    assert (
        MODULE.assign_branch_id(
            _transform(10.0, 0.0),
            centroids,
            translation_threshold_m=0.25,
            rotation_threshold_deg=20.0,
        )
        is None
    )


def test_register_branch_relative_accepts_small_branch() -> None:
    import json

    import cv2

    rows = []
    T_camera_tag = _transform(0.1)
    rvec, _ = cv2.Rodrigues(T_camera_tag[:3, :3])
    for index in range(35):
        T_camera_object = _transform(0.5 + (0.001 if index % 2 else -0.001))
        rows.append(
            {
                "frame_idx": index,
                "timestamp_utc": "t",
                "pose_cluster_id": 0,
                "pnp_operational_valid": True,
                "valid_keypoints": 4,
                "reprojection_error_px": 1.5,
                "apriltag_rvec_json": json.dumps(rvec.reshape(3).tolist()),
                "apriltag_tvec_json": json.dumps(T_camera_tag[:3, 3].tolist()),
                "T_camera_object_filtered_json": MODULE._json_matrix(T_camera_object),
                "T_world_camera_tag_json": MODULE._json_matrix(_transform(1.0)),
            }
        )

    settings = MODULE.load_mvp_settings({})
    results = MODULE.register_branch_relative_poses(rows, settings)
    assert 0 in results
    assert results[0]["accepted"] is True
    assert results[0]["inlier_count"] >= 20


def test_branch_aware_decision_b_without_enough_cup() -> None:
    pose_summary = {
        "total_frames": 300,
        "weighted": {
            "total_frames": 300,
            "pose_comparable_count": 240,
            "object_temporal_jump_count": 0,
            "translation_difference_cm": {
                "mean": 3.0,
                "median": 2.5,
                "p90": 6.0,
                "max": 9.0,
            },
            "rotation_difference_deg": {
                "mean": 1.5,
                "median": 1.2,
                "p90": 3.0,
                "max": 5.0,
            },
        },
    }
    cup_summary = {
        "weighted": {
            "cup_comparable_count": 40,
            "cup_world_difference_cm": {
                "mean": 2.0,
                "median": 1.5,
                "p90": 4.0,
                "max": 6.0,
            },
        }
    }
    decision = MODULE.evaluate_mvp_decision_branch_aware(
        pose_summary=pose_summary,
        cup_summary=cup_summary,
        settings=MODULE.load_mvp_settings({}),
    )
    assert decision["decision"] == "B"


def test_cup_and_filtered_fields_present_in_exports() -> None:
    assert "P_camera_cup_x" in MODULE.REFERENCE_CLUSTER_FIELDS
    assert "T_camera_object_filtered_json" in MODULE.REFERENCE_CLUSTER_FIELDS
    assert "apriltag_branch_id" in MODULE.FRAME_COMPARE_FIELDS
    assert "T_camera_tag_predicted_json" in MODULE.FRAME_COMPARE_FIELDS
    assert "P_camera_cup_x" in MODULE.CUP_COMPARE_FIELDS
    assert "apriltag_branch_id" in MODULE.CUP_COMPARE_FIELDS


def test_live_run_refuses_without_explicit_confirmation() -> None:
    try:
        MODULE.run_mvp_final(
            config_path=ROOT
            / "configs/experiments/orbbec_gemini_object_anchor_mvp_final.yaml",
            confirm_live=False,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Refusing to open the camera" in str(exc)


def test_causal_median3_default_in_settings() -> None:
    settings = MODULE.load_mvp_settings({})
    assert int(settings["temporal_filter_window"]) == 3
    history = [_transform(0.0), _transform(0.1), _transform(0.2)]
    filtered = MODULE.causal_filter_pose(history, 3)
    assert abs(float(filtered[0, 3]) - 0.1) < 1e-9
