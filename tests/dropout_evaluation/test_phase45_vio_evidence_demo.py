"""Tests for Phase 4.5-M3 evidence demo renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.ir_tag_mask import MASK_INTERVAL_RECOVERY_FRAME, MASK_INTERVAL_START_FRAME, is_frame_tag_mask_active
from dropout_evaluation.phase45_vio_evidence_demo import (  # noqa: E402
    COLOR_RED,
    COLOR_VIO_TRACKING,
    IR_INSET_HEIGHT,
    IR_INSET_WIDTH,
    LEFT_BOTTOM_X,
    LEFT_BOTTOM_Y_START,
    TOP_VIEW_SIZE,
    WorldTopViewBounds,
    application_world_contract_lines,
    find_cup2_first_valid_frame,
    is_tag_dropout_red_state,
    left_bottom_layout_y_positions,
    phase_status_content,
    phase_status_lines,
    render_evidence_frame,
    render_world_top_view,
    status_line_color,
    tag0_world_point_to_display,
    tag0_world_pose_to_display,
    visual_update_metadata_from_trajectory,
)
from dropout_evaluation.phase45_vio_demo import (  # noqa: E402
    CupBbox,
    build_demo_replay_states,
    cup_world_position_tag0,
)
from application_world import load_application_world_contract  # noqa: E402

SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
MASK_DIAG = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/tag_mask_diagnostics.json"
TRAJECTORY = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/trajectory.csv"
MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"


def test_mask_interval_state() -> None:
    assert not is_frame_tag_mask_active(201)
    assert is_frame_tag_mask_active(202)
    assert is_frame_tag_mask_active(351)
    assert not is_frame_tag_mask_active(352)


def test_tag_dropout_red_state_interval() -> None:
    assert not is_tag_dropout_red_state(201)
    assert is_tag_dropout_red_state(202)
    assert is_tag_dropout_red_state(351)
    assert not is_tag_dropout_red_state(352)


def test_dropout_red_status_colors() -> None:
    assert status_line_color("TAG INPUT REMOVED", frame_number=202, highlight_tag_dropout_red=True) == COLOR_RED
    assert status_line_color("Tag Pose: BLOCKED", frame_number=250, highlight_tag_dropout_red=True) == COLOR_RED
    assert status_line_color("STEREO + IMU VIO TRACKING", frame_number=250, highlight_tag_dropout_red=True) == COLOR_VIO_TRACKING
    assert status_line_color("TAG INPUT REMOVED", frame_number=202, highlight_tag_dropout_red=False) != COLOR_RED


def test_simple_render_without_world_top_view() -> None:
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    out = render_evidence_frame(
        image_bgr=image,
        rgb_gray=np.zeros((800, 1280), dtype=np.uint8),
        frame_number=248,
        relative_time_sec=8.0,
        replay=None,
        cups=[],
        mask_diag=None,
        original_left_ir_bgr=np.full((480, 848, 3), 120, dtype=np.uint8),
        masked_left_ir_bgr=np.full((480, 848, 3), 90, dtype=np.uint8),
        top_view_panel=None,
        cup2_first_frame=203,
        show_world_top_view=False,
        highlight_tag_dropout_red=True,
    )
    # bottom-right region should remain black (no top-view panel pasted)
    assert out[520:, 900:].mean() < 5


def test_phase_status_lines() -> None:
    assert "APRILTAG" in phase_status_lines(100, cup2_highlight=False)[0]
    assert "TAG INPUT REMOVED" in phase_status_lines(202, cup2_highlight=False)[0]
    assert "RE-ANCHORED" in phase_status_lines(352, cup2_highlight=False)[0]
    secondary = phase_status_content(250, cup2_highlight=False).secondary
    assert any("MASKED" in line for line in secondary)


def test_world_source_panel_removed() -> None:
    content = phase_status_content(250, cup2_highlight=False)
    joined = " ".join(content.primary + content.secondary)
    assert "WORLD SOURCE" not in joined


def test_left_bottom_layout_no_overlap_with_panel_region() -> None:
    camera_y, cup2_y = left_bottom_layout_y_positions(has_cup2=True)
    assert camera_y >= LEFT_BOTTOM_Y_START
    assert cup2_y > camera_y
    assert cup2_y + 30 < 800
    app_camera_y, app_cup1_y, app_cup2_y = left_bottom_layout_y_positions(
        has_cup2=True,
        use_application_world=True,
    )
    assert app_cup1_y > app_camera_y
    assert app_cup2_y > app_cup1_y
    assert app_cup2_y + 30 < 800


def test_top_view_labels_present() -> None:
    panel = render_world_top_view(
        bounds=WorldTopViewBounds(-1.0, 1.0, 0.0, 3.0),
        camera_path=[(-0.5, 1.0), (0.0, 1.5), (0.5, 2.0)],
        current_camera=(0.5, 2.0),
        heading_xz=(1.0, 0.0),
        cup2_trail_xz=[(-0.1, 2.1), (-0.08, 2.12)],
        current_cup2=(-0.07, 2.16),
    )
    assert panel.shape[:2] == (TOP_VIEW_SIZE[1], TOP_VIEW_SIZE[0])


def test_top_view_coordinate_conversion() -> None:
    bounds = WorldTopViewBounds(-1.0, 1.0, 0.0, 2.0)
    px, py = bounds.to_plot(0.0, 1.0, 300, 240)
    assert 0 <= px < 300
    assert 0 <= py < 240


def test_world_top_view_render_smoke() -> None:
    test_top_view_labels_present()


def test_visual_update_metadata_from_trajectory() -> None:
    traj = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/trajectory.csv"
    if not traj.is_file():
        pytest.skip("trajectory missing")
    meta = visual_update_metadata_from_trajectory(traj)
    assert meta["visual_update_eligible_count"] == 435
    assert meta["visual_update_success_ratio_eligible_only"] == 1.0


def test_cup2_first_appearance_event() -> None:
    class _Replay:
        world_valid = True
        T_world_camera = np.eye(4)

    replay = {203: _Replay()}
    cups = {
        203: [
            CupBbox(
                frame_number=203,
                semantic_id="cup2",
                x1=10,
                y1=10,
                x2=40,
                y2=40,
                P_camera=np.array([0.1, 0.0, 1.0]),
                valid=True,
            )
        ]
    }
    assert find_cup2_first_valid_frame(cups, replay) == 203
    lines = phase_status_lines(203, cup2_highlight=True)
    assert any("CUP2" in line for line in lines)


@pytest.mark.skipif(not MASK_DIAG.is_file(), reason="mask diagnostics missing")
def test_original_masked_ir_same_frame_mapping() -> None:
    from dropout_evaluation.phase45_vio_tag_mask_demo import load_mask_diagnostics

    diag = load_mask_diagnostics(MASK_DIAG)
    assert 202 in diag
    assert diag[202].tag_mask_active is True
    assert diag[202].left.mask_corners_xy is not None


def test_render_evidence_frame_smoke() -> None:
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    rgb_gray = np.zeros((800, 1280), dtype=np.uint8)
    original = np.full((480, 848, 3), 120, dtype=np.uint8)
    masked = np.full((480, 848, 3), 90, dtype=np.uint8)
    out = render_evidence_frame(
        image_bgr=image,
        rgb_gray=rgb_gray,
        frame_number=202,
        relative_time_sec=6.7,
        replay=None,
        cups=[],
        mask_diag=None,
        original_left_ir_bgr=original,
        masked_left_ir_bgr=masked,
        top_view_panel=None,
        cup2_first_frame=203,
        show_world_top_view=False,
        highlight_tag_dropout_red=True,
    )
    assert out.shape == image.shape
    # left-bottom coordinate block should not place dark panel overlay in that region
    assert out[LEFT_BOTTOM_Y_START : LEFT_BOTTOM_Y_START + 120, LEFT_BOTTOM_X : LEFT_BOTTOM_X + 500].mean() < 20


@pytest.mark.skipif(
    not (ROOT / "out/demo/phase45_vio_application_world_final/scenario_a_vio_application_world_final.mp4").is_file(),
    reason="application world final demo mp4 missing",
)
def test_application_world_final_demo_video_smoke_openable() -> None:
    import cv2

    mp4 = ROOT / "out/demo/phase45_vio_application_world_final/scenario_a_vio_application_world_final.mp4"
    cap = cv2.VideoCapture(str(mp4))
    assert cap.isOpened()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert count == 436
    cap.release()


def test_application_world_contract_lines() -> None:
    contract = load_application_world_contract()
    title, detail = application_world_contract_lines(contract)
    assert title == "APPLICATION WORLD"
    assert "Tag0 = (1.00, 2.00, 0.00) m" in detail
    assert "+X right" in detail


def test_tag0_origin_maps_to_application_world() -> None:
    contract = load_application_world_contract()
    point = tag0_world_point_to_display(
        np.zeros(3),
        use_application_world=True,
        T_application_tag0=contract.T_application_tag0,
    )
    np.testing.assert_allclose(point, [1.0, 2.0, 0.0], atol=1e-9)


@pytest.mark.skipif(not TRAJECTORY.is_file(), reason="trajectory missing")
def test_application_world_representative_values() -> None:
    contract = load_application_world_contract()
    replay_by_frame = build_demo_replay_states(
        session_dir=SESSION,
        trajectory_csv=TRAJECTORY,
        manifest_path=MANIFEST,
    )
    from dropout_evaluation.phase45_vio_demo import load_cup_bboxes_by_frame

    cups_by_frame = load_cup_bboxes_by_frame(SESSION / "derived/cups/observations.csv")

    def _display(frame_number: int) -> dict[str, np.ndarray | None]:
        replay = replay_by_frame.get(frame_number)
        cups = cups_by_frame.get(frame_number, [])
        camera = tag0_world_pose_to_display(
            replay.T_world_camera if replay is not None and replay.world_valid else None,
            use_application_world=True,
            T_application_tag0=contract.T_application_tag0,
        )
        cup1 = tag0_world_point_to_display(
            cup_world_position_tag0(replay, cups, "cup1"),
            use_application_world=True,
            T_application_tag0=contract.T_application_tag0,
        )
        cup2 = tag0_world_point_to_display(
            cup_world_position_tag0(replay, cups, "cup2"),
            use_application_world=True,
            T_application_tag0=contract.T_application_tag0,
        )
        return {"camera": camera, "cup1": cup1, "cup2": cup2}

    f180 = _display(180)
    assert f180["camera"] is not None
    np.testing.assert_allclose(f180["camera"][:3, 3], [0.105, 1.023, 2.611], atol=0.01)
    assert f180["cup1"] is not None
    np.testing.assert_allclose(f180["cup1"], [0.227, 0.897, 2.291], atol=0.01)
    assert f180["cup2"] is None

    f248 = _display(248)
    assert f248["camera"] is not None
    np.testing.assert_allclose(f248["camera"][:3, 3], [0.295, 1.130, 2.734], atol=0.01)
    assert f248["cup1"] is not None
    np.testing.assert_allclose(f248["cup1"], [0.373, 0.988, 2.412], atol=0.01)
    assert f248["cup2"] is not None
    np.testing.assert_allclose(f248["cup2"], [0.927, 0.930, 2.161], atol=0.01)

    f352 = _display(352)
    assert f352["camera"] is not None
    np.testing.assert_allclose(f352["camera"][:3, 3], [0.293, 1.028, 2.685], atol=0.01)
    assert f352["cup1"] is not None
    np.testing.assert_allclose(f352["cup1"], [0.384, 0.898, 2.361], atol=0.01)
    assert f352["cup2"] is not None
    np.testing.assert_allclose(f352["cup2"], [0.959, 0.838, 2.138], atol=0.01)

    camera_tag0 = replay_by_frame[248].T_world_camera
    camera_app = f248["camera"]
    np.testing.assert_allclose(camera_app[:3, :3], camera_tag0[:3, :3], atol=1e-9)


def test_cup_positions_are_not_pose_sources() -> None:
    import inspect

    from dropout_evaluation import rgbd_odometry_adapter

    replay_source = inspect.getsource(rgbd_odometry_adapter.replay_session_for_window)
    assert "cup" not in replay_source.lower()
    assert "semantic_id" not in replay_source


def test_application_world_render_smoke() -> None:
    contract = load_application_world_contract()
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    out = render_evidence_frame(
        image_bgr=image,
        rgb_gray=np.zeros((800, 1280), dtype=np.uint8),
        frame_number=248,
        relative_time_sec=8.0,
        replay=None,
        cups=[],
        mask_diag=None,
        original_left_ir_bgr=np.full((480, 848, 3), 120, dtype=np.uint8),
        masked_left_ir_bgr=np.full((480, 848, 3), 90, dtype=np.uint8),
        top_view_panel=None,
        cup2_first_frame=203,
        show_world_top_view=False,
        highlight_tag_dropout_red=True,
        use_application_world=True,
        T_application_tag0=contract.T_application_tag0,
        application_world_contract=contract,
    )
    assert out.shape == image.shape
    assert out[520:, 900:].mean() < 5
