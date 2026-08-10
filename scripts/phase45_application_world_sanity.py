"""Read-only sanity check: Tag0 world -> Application World on frozen Scenario A artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from application_world import (  # noqa: E402
    DEFAULT_APPLICATION_WORLD_CONFIG_PATH,
    DEFAULT_TAG0_WORLD_CONFIG_PATH,
    EXPECTED_TRANSLATION_DELTA_M,
    application_world_axis_definitions,
    load_application_world_contract,
    rotation_unchanged,
    tag0_world_point_to_application_world,
    tag0_world_pose_to_application_world,
    translation_delta,
)
from dropout_evaluation.dropout_protocol import load_frame_timestamps_from_rgb_index  # noqa: E402
from dropout_evaluation.evaluation_io import load_cup_observations_from_csv  # noqa: E402
from dropout_evaluation.evaluation_metrics import transform_point_camera_to_world  # noqa: E402
from dropout_evaluation.hold_last_pose_runner import load_dropout_windows_from_manifest  # noqa: E402
from dropout_evaluation.rgbd_odometry_adapter import replay_session_for_window  # noqa: E402
from dropout_evaluation.runtime_apriltag import load_runtime_apriltag_poses_from_session  # noqa: E402
from dropout_evaluation.stereo_imu_vio_adapter import (  # noqa: E402
    load_vio_trajectory_from_csv,
    vio_trajectory_to_local_trajectory,
)

DEFAULT_SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
DEFAULT_TRAJECTORY = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/trajectory.csv"
DEFAULT_MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"
DEFAULT_OUTPUT_DIR = ROOT / "out/analysis/phase45_application_world_sanity"
REPRESENTATIVE_FRAMES = (180, 248, 352)
FIRST_MVP_CUP1_REFERENCE_MEDIAN = (0.599, 1.005, 2.129)


def _xyz(point: np.ndarray | None) -> list[float] | None:
    if point is None:
        return None
    return [round(float(v), 6) for v in np.asarray(point, dtype=np.float64).reshape(3)]


def _pose_dict(T: np.ndarray | None) -> dict[str, Any] | None:
    if T is None:
        return None
    matrix = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return {
        "translation_m": _xyz(matrix[:3, 3]),
        "rotation": matrix[:3, :3].tolist(),
    }


def _point_record(
    label: str,
    point_tag0_world: np.ndarray | None,
    *,
    T_application_tag0: np.ndarray,
) -> dict[str, Any] | None:
    if point_tag0_world is None:
        return None
    point_application = tag0_world_point_to_application_world(
        point_tag0_world,
        T_application_tag0=T_application_tag0,
    )
    delta = translation_delta(point_application, point_tag0_world)
    return {
        "label": label,
        "tag0_world_xyz_m": _xyz(point_tag0_world),
        "application_world_xyz_m": _xyz(point_application),
        "delta_application_minus_tag0_m": _xyz(delta),
        "delta_matches_expected": bool(np.allclose(delta, EXPECTED_TRANSLATION_DELTA_M, atol=1e-6)),
    }


def _pose_record(
    label: str,
    T_tag0: np.ndarray | None,
    *,
    T_application_tag0: np.ndarray,
) -> dict[str, Any] | None:
    if T_tag0 is None:
        return None
    T_application = tag0_world_pose_to_application_world(T_tag0, T_application_tag0=T_application_tag0)
    tag0_t = np.asarray(T_tag0, dtype=np.float64)[:3, 3]
    app_t = np.asarray(T_application, dtype=np.float64)[:3, 3]
    delta = app_t - tag0_t
    return {
        "label": label,
        "tag0_world_pose": _pose_dict(T_tag0),
        "application_world_pose": _pose_dict(T_application),
        "translation_delta_m": _xyz(delta),
        "delta_matches_expected": bool(np.allclose(delta, EXPECTED_TRANSLATION_DELTA_M, atol=1e-6)),
        "rotation_unchanged": rotation_unchanged(T_tag0[:3, :3], T_application[:3, :3]),
    }


def build_sanity_report(
    *,
    session_dir: Path,
    trajectory_csv: Path,
    manifest_path: Path,
    application_config_path: Path,
    tag0_config_path: Path,
    representative_frames: tuple[int, ...] = REPRESENTATIVE_FRAMES,
) -> dict[str, Any]:
    contract = load_application_world_contract(config_path=application_config_path)
    T_application_tag0 = contract.T_application_tag0

    windows = load_dropout_windows_from_manifest(manifest_path)
    c5 = next(w for w in windows if w.window_id == "C_pre_cup2__5.0s")
    traj = vio_trajectory_to_local_trajectory(load_vio_trajectory_from_csv(trajectory_csv))
    runtime = load_runtime_apriltag_poses_from_session(session_dir)
    frame_timestamps = load_frame_timestamps_from_rgb_index(session_dir)
    cups = load_cup_observations_from_csv(session_dir / "derived/cups/observations.csv")
    cups_by_frame: dict[int, list[Any]] = {}
    for cup in cups:
        cups_by_frame.setdefault(cup.frame_number, []).append(cup)

    replay = replay_session_for_window(
        window=c5,
        local_trajectory=traj,
        runtime_poses=runtime,
        frame_timestamps=frame_timestamps,
    )

    tag0_origin_application = tag0_world_point_to_application_world(
        np.zeros(3, dtype=np.float64),
        T_application_tag0=T_application_tag0,
    )
    tag0_pose_application = tag0_world_pose_to_application_world(
        np.eye(4, dtype=np.float64),
        T_application_tag0=T_application_tag0,
    )

    frame_records: dict[str, Any] = {}
    all_delta_ok = True
    all_rotation_ok = True

    for frame_number in representative_frames:
        state = replay.frames.get(frame_number)
        camera_tag0 = (
            state.T_world_camera.copy()
            if state is not None and state.world_valid and state.T_world_camera is not None
            else None
        )
        camera_record = _pose_record("camera", camera_tag0, T_application_tag0=T_application_tag0)
        if camera_record is not None:
            all_delta_ok &= camera_record["delta_matches_expected"]
            all_rotation_ok &= camera_record["rotation_unchanged"]

        cup_records: dict[str, Any] = {}
        if camera_tag0 is not None:
            for cup in cups_by_frame.get(frame_number, []):
                if not cup.valid or cup.P_camera is None:
                    continue
                point_tag0 = transform_point_camera_to_world(camera_tag0, cup.P_camera)
                record = _point_record(cup.semantic_id, point_tag0, T_application_tag0=T_application_tag0)
                if record is not None:
                    cup_records[cup.semantic_id] = record
                    all_delta_ok &= record["delta_matches_expected"]

        frame_records[str(frame_number)] = {
            "camera": camera_record,
            "cups": cup_records,
        }

    gate = "APPLICATION_WORLD_COMPAT_READY"
    warnings: list[str] = []
    if not np.allclose(tag0_origin_application, contract.tag0_position_application_world_m, atol=1e-6):
        gate = "APPLICATION_WORLD_COMPAT_FAILED"
    if not np.allclose(tag0_origin_application, EXPECTED_TRANSLATION_DELTA_M, atol=1e-6):
        gate = "APPLICATION_WORLD_COMPAT_FAILED"
    if not all_delta_ok:
        gate = "APPLICATION_WORLD_COMPAT_FAILED"
    if not all_rotation_ok:
        gate = "APPLICATION_WORLD_COMPAT_FAILED"
    if not np.allclose(T_application_tag0[:3, :3], np.eye(3), atol=1e-9):
        warnings.append("T_application_tag0 rotation is not identity; orientation-change checks are config-specific")
        if gate == "APPLICATION_WORLD_COMPAT_READY":
            gate = "APPLICATION_WORLD_COMPAT_READY_WITH_WARNING"

    return {
        "gate": gate,
        "warnings": warnings,
        "source_application_world_config": str(application_config_path),
        "source_tag0_world_config": str(tag0_config_path),
        "application_world_contract": contract.as_dict(),
        "tag0_world_contract_note": "Phase 2/3/4 internal evaluation frame; Tag0 at origin",
        "T_application_tag0": T_application_tag0.tolist(),
        "axis_definitions": application_world_axis_definitions(contract.front_normal, contract.top_direction),
        "tag0_origin_sanity": {
            "tag0_world_xyz_m": [0.0, 0.0, 0.0],
            "application_world_xyz_m": _xyz(tag0_origin_application),
            "expected_application_world_xyz_m": list(contract.tag0_position_application_world_m),
            "passes": bool(
                np.allclose(tag0_origin_application, contract.tag0_position_application_world_m, atol=1e-6)
            ),
        },
        "tag0_pose_sanity": {
            "tag0_world_pose": _pose_dict(np.eye(4)),
            "application_world_pose": _pose_dict(tag0_pose_application),
            "rotation_unchanged": rotation_unchanged(np.eye(3), tag0_pose_application[:3, :3]),
            "translation_matches_tag0_position": bool(
                np.allclose(tag0_pose_application[:3, 3], contract.tag0_position_application_world_m, atol=1e-6)
            ),
        },
        "expected_translation_delta_m": EXPECTED_TRANSLATION_DELTA_M.tolist(),
        "representative_frames": frame_records,
        "first_mvp_cup1_reference_only_median_xyz_m": list(FIRST_MVP_CUP1_REFERENCE_MEDIAN),
        "artifact_sources": {
            "session_dir": str(session_dir),
            "trajectory_csv": str(trajectory_csv),
            "manifest_path": str(manifest_path),
            "cups_csv": str(session_dir / "derived/cups/observations.csv"),
        },
        "existing_artifacts_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Application World compatibility sanity (frozen artifacts only)")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--application-config", type=Path, default=DEFAULT_APPLICATION_WORLD_CONFIG_PATH)
    parser.add_argument("--tag0-config", type=Path, default=DEFAULT_TAG0_WORLD_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_sanity_report(
        session_dir=args.session,
        trajectory_csv=args.trajectory,
        manifest_path=args.manifest,
        application_config_path=args.application_config,
        tag0_config_path=args.tag0_config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "application_world_sanity.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] != "APPLICATION_WORLD_COMPAT_FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
