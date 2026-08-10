"""Tests for Phase 4.5-M2 IR tag masking ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dropout_evaluation.ir_tag_mask import (  # noqa: E402
    MASK_INTERVAL_RECOVERY_FRAME,
    MASK_INTERVAL_START_FRAME,
    apply_tag_mask_to_stereo_frames,
    apply_tag_roi_mask,
    detect_tag_roi,
    is_frame_tag_mask_active,
)
from dropout_evaluation.stereo_imu_vio_lite import StereoImuVioFrameInput  # noqa: E402
from dropout_evaluation.stereo_imu_vio_tag_mask import (  # noqa: E402
    TAG_MASK_WINDOW_ID,
    build_tag_mask_provenance,
    evaluate_tag_mask_c5s_window,
)

SESSION = ROOT / "out/datasets/gemini335l/20260807_161354_scenario_a"
MANIFEST = ROOT / "out/evaluation/phase3/20260807_161354_scenario_a/dropout_windows.json"
PROTOCOL = ROOT / "configs/evaluation/phase3_dropout_scenario_a.yaml"


def _make_frame(frame_number: int, *, with_tag: bool = True) -> StereoImuVioFrameInput:
    left = np.full((200, 300), 120, dtype=np.uint8)
    right = np.full((200, 300), 120, dtype=np.uint8)
    if with_tag:
        cv2.rectangle(left, (120, 70), (180, 130), 20, -1)
        cv2.rectangle(right, (120, 70), (180, 130), 20, -1)
    return StereoImuVioFrameInput(
        frame_number=frame_number,
        device_timestamp_us=frame_number * 33_333,
        left_gray=left,
        right_gray=right,
        native_left_frame_number=frame_number,
        native_right_frame_number=frame_number,
    )


def test_mask_interval_half_open() -> None:
    assert not is_frame_tag_mask_active(201)
    assert is_frame_tag_mask_active(202)
    assert is_frame_tag_mask_active(351)
    assert not is_frame_tag_mask_active(352)


def test_mask_applied_to_left_and_right_pixels() -> None:
    from dropout_evaluation.ir_tag_mask import TagMaskRoi

    left = np.full((240, 320), 100, dtype=np.uint8)
    right = np.full((240, 320), 100, dtype=np.uint8)
    roi = TagMaskRoi(
        detected=True,
        corners_xy=((140.0, 90.0), (180.0, 90.0), (180.0, 130.0), (140.0, 130.0)),
        mask_corners_xy=((132.0, 82.0), (188.0, 82.0), (188.0, 138.0), (132.0, 138.0)),
        bbox_xyxy=(140, 90, 180, 130),
        expanded_bbox_xyxy=(132, 82, 188, 138),
        area_ratio=0.05,
    )
    masked_left = apply_tag_roi_mask(left, roi, fill_value=77)
    masked_right = apply_tag_roi_mask(right, roi, fill_value=77)
    assert masked_left[110, 160] == 77
    assert masked_right[110, 160] == 77
    assert left[110, 160] == 100


def test_apply_tag_mask_frame_state_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    from dropout_evaluation.ir_tag_mask import TagMaskRoi

    fake_roi = TagMaskRoi(
        detected=True,
        corners_xy=((140.0, 90.0), (180.0, 90.0), (180.0, 130.0), (140.0, 130.0)),
        mask_corners_xy=((132.0, 82.0), (188.0, 82.0), (188.0, 138.0), (132.0, 138.0)),
        bbox_xyxy=(140, 90, 180, 130),
        expanded_bbox_xyxy=(132, 82, 188, 138),
        area_ratio=0.05,
    )
    monkeypatch.setattr(
        "dropout_evaluation.ir_tag_mask._resolve_stereo_rois",
        lambda *args, **kwargs: (fake_roi, fake_roi, False),
    )
    frames = [_make_frame(n) for n in range(200, 354)]
    masked_frames, diagnostics = apply_tag_mask_to_stereo_frames(frames, start_frame=202, recovery_frame=352)
    by_frame = {row.frame_number: row for row in diagnostics}
    assert by_frame[201].tag_mask_active is False
    assert by_frame[202].tag_mask_active is True
    assert by_frame[351].tag_mask_active is True
    assert by_frame[352].tag_mask_active is False

    original_202_left = frames[2].left_gray.copy()
    masked_202_left = masked_frames[2].left_gray
    assert not np.array_equal(original_202_left, masked_202_left)
    assert np.array_equal(frames[0].left_gray, masked_frames[0].left_gray)


def test_roi_margin_expands_bbox() -> None:
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    roi_small = detect_tag_roi(
        _make_frame(1).left_gray,
        detector=detector,
        margin_px=4,
    )
    roi_large = detect_tag_roi(
        _make_frame(1).left_gray,
        detector=detector,
        margin_px=20,
    )
    if roi_small.detected and roi_large.detected:
        assert roi_large.area_ratio >= roi_small.area_ratio


def test_estimator_module_has_no_tag_pose_imports() -> None:
    import dropout_evaluation.stereo_imu_vio_lite as lite

    source = Path(lite.__file__).read_text(encoding="utf-8")
    assert "apriltag" not in source.lower()
    assert "runtime_apriltag" not in source


def test_tag_mask_provenance_flags() -> None:
    provenance = build_tag_mask_provenance(
        base_provenance={"algorithm_id": "stereo_imu_vio_lite"},
        mask_summary={"masked_frame_count": 150},
        margin_px=14,
        dictionary="APRILTAG_36H11",
    )
    assert provenance["tag_detector_used_for_mask_generation_only"] is True
    assert provenance["estimator_uses_tag_pose"] is False
    assert provenance["estimator_receives_tag_roi"] is False


@pytest.mark.skipif(not SESSION.is_dir(), reason="Scenario A missing")
def test_canonical_frame_count_for_mask_run_prereq() -> None:
    from dataset_recorder.reader import DatasetReader
    from dropout_evaluation.stereo_imu_vio_continuous import load_stereo_frames

    reader = DatasetReader(SESSION)
    frames, _ = load_stereo_frames(reader)
    assert len(frames) == 436
    assert frames[0].frame_number == 1
    assert frames[-1].frame_number == 436


@pytest.mark.skipif(
    not (
        SESSION.is_dir()
        and MANIFEST.is_file()
        and (ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/trajectory.csv").is_file()
    ),
    reason="tag-mask trajectory missing",
)
def test_c5_single_window_evaluation_wiring() -> None:
    trajectory = ROOT / "out/analysis/phase4_stereo_imu_vio_lite_tag_mask_c5s/trajectory.csv"
    row, metrics = evaluate_tag_mask_c5s_window(
        session_dir=SESSION,
        manifest_path=MANIFEST,
        trajectory_csv=trajectory,
        protocol_config_path=PROTOCOL,
    )
    assert row["window_id"] == TAG_MASK_WINDOW_ID
    assert metrics["pose_availability"] is not None


@pytest.mark.skipif(
    not (ROOT / "out/demo/phase45_vio_tag_mask/scenario_a_vio_tag_mask_c5s_demo.mp4").is_file(),
    reason="tag-mask demo missing",
)
def test_tag_mask_demo_video_smoke() -> None:
    cap = cv2.VideoCapture(str(ROOT / "out/demo/phase45_vio_tag_mask/scenario_a_vio_tag_mask_c5s_demo.mp4"))
    assert cap.isOpened()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 400 <= count <= 436
    cap.release()
