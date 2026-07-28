from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from apriltag_world import AprilTagWorldObservation, AprilTagWorldResult
from object_anchor_pose import ObjectPoseEstimate
from object_anchor_preview import (
    ObjectAnchorPreviewSession,
    align_object_pose,
    build_preview_session,
    cup_difference_cm,
    draw_object_preview_axes,
    draw_preview_banner,
    load_preview_settings,
    object_frame_alignment_transform,
    transform_point,
)
from object_anchor_runtime import ObjectAnchorFrameResult
from stereo_types import DepthEstimate

ROOT = Path(__file__).resolve().parents[1]


def _T(x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out[0, 3] = x
    return out


def _anchor(
    *,
    valid: bool = True,
    T: np.ndarray | None = None,
    reproj: float = 2.0,
    detected: bool = True,
) -> ObjectAnchorFrameResult:
    pose = ObjectPoseEstimate(
        valid=valid,
        reason="ok" if valid else "invalid",
        T_camera_object=T if T is not None else _T(0.4),
        mean_reprojection_error_px=reproj,
        inlier_indices=(0, 1, 2, 3),
    )
    detection = object() if detected else None
    if detected:
        # Minimal duck-typed detection; runtime only checks None.
        class _Det:
            pass

        detection = _Det()
    return ObjectAnchorFrameResult(
        detection=detection,  # type: ignore[arg-type]
        pose=pose,
        overlay_bgr=np.zeros((80, 120, 3), dtype=np.uint8),
        effective_visibility=np.array([2, 2, 2, 2], dtype=np.int32),
    )


def _tag(T_world_camera: np.ndarray, tag_id: int = 0) -> AprilTagWorldResult:
    obs = AprilTagWorldObservation(
        tag_id=tag_id,
        T_camera_tag=_T(0.1),
        T_world_tag=np.eye(4),
        T_world_camera=T_world_camera,
        reprojection_error_px=0.5,
    )
    return AprilTagWorldResult([obs], "ok")


def _cup(valid: bool = True) -> DepthEstimate:
    return DepthEstimate(
        track="A",
        X=0.1,
        Y=-0.05,
        Z=0.9,
        disparity=None,
        valid=valid,
        notes="ok" if valid else "invalid_depth",
    )


def _session(tmp_path: Path, **overrides: object) -> ObjectAnchorPreviewSession:
    raw = {
        "enabled": True,
        "calibration_samples": 30,
        "temporal_filter_window": 3,
        "save_preview_logs": True,
        "auto_switch_world_source": False,
        "persist_calibration": False,
        "output_root": str(tmp_path / "preview"),
    }
    raw.update(overrides)
    settings = load_preview_settings(raw)
    session = build_preview_session(
        settings,
        repo_root=ROOT,
        model_path="models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt",
        config_path="configs/orbbec_gemini.yaml",
    )
    assert session is not None
    return session


def test_ready_after_30_stable_samples(tmp_path: Path) -> None:
    session = _session(tmp_path, calibration_samples=30)
    session.start_calibration()
    T_tag = _T(1.0)
    for i in range(30):
        view = session.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(T_tag),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    assert session.is_ready
    assert view.preview_state == "ACTIVE"
    assert view.p_world_cup_object_m is not None
    assert (tmp_path / "preview").exists() or session.output_dir is not None
    assert (session.output_dir / "session_calibration.json").is_file()


def test_no_object_cup_world_before_ready(tmp_path: Path) -> None:
    session = _session(tmp_path, calibration_samples=30)
    session.start_calibration()
    for i in range(10):
        view = session.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(_T(1.0)),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    assert not session.is_ready
    assert view.p_world_cup_object_m is None
    assert "CALIBRATING" in view.object_status


def test_apriltag_jump_resets_calibration_count(tmp_path: Path) -> None:
    session = _session(tmp_path, calibration_samples=30)
    session.start_calibration()
    for i in range(10):
        session.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(_T(1.0)),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    assert len(session.calibration_samples) == 10
    session.update(
        frame_idx=10,
        timestamp=10.0,
        fps=30.0,
        apriltag_result=_tag(_T(1.0 + 0.6)),  # 60cm jump
        anchor_result=_anchor(T=_T(0.4)),
        cup_estimate=_cup(),
        cup_detected=True,
    )
    # Jump clears previous samples, then current valid frame is appended.
    assert len(session.calibration_samples) == 1


def test_new_session_starts_uncalibrated(tmp_path: Path) -> None:
    first = _session(tmp_path / "a")
    first.start_calibration()
    for i in range(30):
        first.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(_T(1.0)),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    assert first.is_ready
    second = _session(tmp_path / "b")
    assert not second.is_ready
    assert second.T_world_object_preview is None


def test_apriltag_loss_keeps_oa_preview(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.start_calibration()
    for i in range(30):
        session.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(_T(1.0)),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    view = session.update(
        frame_idx=40,
        timestamp=40.0,
        fps=30.0,
        apriltag_result=AprilTagWorldResult([], "no_tags"),
        anchor_result=_anchor(T=_T(0.41)),
        cup_estimate=_cup(),
        cup_detected=True,
    )
    assert view.april_status == "LOST"
    assert view.p_world_cup_tag_m is None
    assert view.p_world_cup_object_m is not None
    assert view.object_status == "ACTIVE"
    assert view.difference_status == "N/A"
    assert session.tag_lost_oa_cup_frames >= 1


def test_object_anchor_loss_hides_oa_cup(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.T_world_object_preview = _T(2.0)
    view = session.update(
        frame_idx=1,
        timestamp=1.0,
        fps=30.0,
        apriltag_result=_tag(_T(1.0)),
        anchor_result=_anchor(detected=False, valid=False, T=None),
        cup_estimate=_cup(),
        cup_detected=True,
    )
    assert view.p_world_cup_object_m is None
    assert view.object_status == "LOST OBJECT"


def test_invalid_cup_depth_hides_both_worlds(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.T_world_object_preview = _T(2.0)
    view = session.update(
        frame_idx=1,
        timestamp=1.0,
        fps=30.0,
        apriltag_result=_tag(_T(1.0)),
        anchor_result=_anchor(T=_T(0.4)),
        cup_estimate=_cup(valid=False),
        cup_detected=True,
    )
    assert view.p_world_cup_tag_m is None
    assert view.p_world_cup_object_m is None
    assert "INVALID DEPTH" in view.object_status or view.difference_status == "INVALID DEPTH"


def test_same_p_camera_cup_used_for_both_transforms(tmp_path: Path) -> None:
    session = _session(tmp_path)
    T_world_object = _T(2.0)
    T_camera_object = _T(0.4)
    session.T_world_object_preview = T_world_object
    cup = _cup()
    view = session.update(
        frame_idx=1,
        timestamp=1.0,
        fps=30.0,
        apriltag_result=_tag(_T(1.0)),
        anchor_result=_anchor(T=T_camera_object),
        cup_estimate=cup,
        cup_detected=True,
    )
    p_cam = np.array([cup.X, cup.Y, cup.Z])
    expected_tag = transform_point(_T(1.0), p_cam)
    expected_obj = transform_point(
        T_world_object
        @ np.linalg.inv(align_object_pose(T_camera_object, session.settings)),
        p_cam,
    )
    np.testing.assert_allclose(view.p_world_cup_tag_m, expected_tag, atol=1e-9)
    np.testing.assert_allclose(view.p_world_cup_object_m, expected_obj, atol=1e-9)


def test_difference_euclidean_distance() -> None:
    tag = np.array([0.124, -0.081, 0.965])
    obj = np.array([0.140, -0.073, 0.942])
    diff = cup_difference_cm(tag, obj)
    assert abs(diff["dx_cm"] - 1.6) < 1e-9
    assert abs(diff["dy_cm"] - 0.8) < 1e-9
    assert abs(diff["dz_cm"] - (-2.3)) < 1e-9
    assert abs(diff["distance_cm"] - np.sqrt(1.6**2 + 0.8**2 + 2.3**2)) < 1e-9


def test_banner_uses_na_not_zeros() -> None:
    from object_anchor_preview import PreviewFrameView

    image = np.zeros((400, 1200, 3), dtype=np.uint8)
    view = PreviewFrameView(
        preview_state="CALIBRATION_REQUIRED",
        april_status="LOST",
        object_status="CALIBRATION REQUIRED",
        difference_status="N/A",
        p_world_cup_tag_m=None,
        p_world_cup_object_m=None,
    )
    out = draw_preview_banner(image, view)
    # Ensure banner painted something non-zero and text path did not crash.
    assert out.shape == image.shape
    assert int(out[:120].sum()) > 0


def test_banner_keeps_fps_and_default_debug_is_off(monkeypatch) -> None:
    from object_anchor_preview import PreviewFrameView

    texts: list[str] = []
    original = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        texts.append(str(text))
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    view = PreviewFrameView(
        preview_state="ACTIVE",
        april_status="ACTIVE",
        object_status="ACTIVE",
        difference_status="N/A",
        fps=29.7,
        debug_overlay_enabled=False,
    )
    draw_preview_banner(np.zeros((400, 1200, 3), dtype=np.uint8), view)
    assert any(text == "FPS: 29.7" for text in texts)
    assert not any("DEBUG: ON" in text for text in texts)


def test_display_source_does_not_enable_auto_switch(tmp_path: Path) -> None:
    session = _session(tmp_path, auto_switch_world_source=True)
    assert session.settings.auto_switch_world_source is False
    assert session.display_source == "APRILTAG"
    session.handle_key(ord("o"))
    assert session.display_source == "OBJECT_ANCHOR_PREVIEW"
    # Operational flag remains false; only display toggled.
    assert session.settings.auto_switch_world_source is False


def test_debug_overlay_default_and_d_toggle(tmp_path: Path) -> None:
    session = _session(tmp_path)
    assert session.debug_overlay_enabled is False
    assert session.handle_key(ord("d")) == "debug_overlay:on"
    assert session.debug_overlay_enabled is True
    assert session.handle_key(ord("D")) == "debug_overlay:off"
    assert session.debug_overlay_enabled is False


def test_convention_alignment_is_rx_plus_90() -> None:
    settings = load_preview_settings({"enabled": True, "save_preview_logs": False})
    alignment = object_frame_alignment_transform(settings)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(alignment, expected, atol=1e-12)
    # Aligned +X=raw +X, aligned +Y=raw +Z, aligned +Z=-raw +Y.
    np.testing.assert_allclose(alignment[:3, 0], [1, 0, 0])
    np.testing.assert_allclose(alignment[:3, 1], [0, 0, 1])
    np.testing.assert_allclose(alignment[:3, 2], [0, -1, 0])
    assert np.linalg.det(alignment[:3, :3]) == 1.0


def test_apriltag_and_object_raw_axis_conventions_from_definitions() -> None:
    from apriltag_world import _tag_object_points, _world_from_tag

    tag_points = _tag_object_points(2.0)
    # Detector corner order is top-left, top-right, bottom-right, bottom-left:
    # tag +X points right, tag +Y points up, and right-handed +Z is front/outward.
    np.testing.assert_allclose(tag_points[0], [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(tag_points[1], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(tag_points[2], [1.0, -1.0, 0.0])
    tag_world = _world_from_tag(
        [0.0, 0.0, 0.0], top_direction="+Y", front_normal="+Z"
    )
    np.testing.assert_allclose(tag_world[:3, :3], np.eye(3), atol=1e-12)

    anchor_cfg = yaml.safe_load(
        (
            ROOT / "configs/object_anchors/tissue_box_01_front_only.yaml"
        ).read_text(encoding="utf-8")
    )["object_anchor"]
    points = np.asarray(
        [entry["xyz"] for entry in anchor_cfg["keypoints_3d"]], dtype=np.float64
    )
    # TL->TR is raw +X; BL->TL is raw +Z. All front points lie at raw -Y.
    np.testing.assert_allclose(points[1] - points[0], [0.235, 0.0, 0.0])
    np.testing.assert_allclose(points[0] - points[3], [0.0, 0.0, 0.110])
    assert np.all(points[:, 1] < 0.0)
    assert anchor_cfg["coordinate_system"]["y_axis"] == "front_to_back"


def test_raw_and_aligned_registration_restore_same_camera_and_cup() -> None:
    settings = load_preview_settings({"enabled": True, "save_preview_logs": False})
    T_world_camera = _T(1.2, 17.0)
    T_camera_object_raw = _T(0.4, -8.0)
    T_camera_object_aligned = align_object_pose(T_camera_object_raw, settings)

    T_world_object_raw = T_world_camera @ T_camera_object_raw
    restored_raw = T_world_object_raw @ np.linalg.inv(T_camera_object_raw)

    T_world_object_aligned = T_world_camera @ T_camera_object_aligned
    restored_aligned = T_world_object_aligned @ np.linalg.inv(
        T_camera_object_aligned
    )
    np.testing.assert_allclose(restored_aligned, restored_raw, atol=1e-10)
    np.testing.assert_allclose(restored_aligned, T_world_camera, atol=1e-10)

    p_camera_cup = np.array([0.15, -0.04, 0.95])
    np.testing.assert_allclose(
        transform_point(restored_aligned, p_camera_cup),
        transform_point(restored_raw, p_camera_cup),
        atol=1e-10,
    )


def test_preview_calibration_and_runtime_both_use_aligned_pose(tmp_path: Path) -> None:
    session = _session(tmp_path, calibration_samples=1)
    T_world_camera = _T(1.0, 12.0)
    T_camera_object_raw = _T(0.4, -5.0)
    session.start_calibration()
    view = session.update(
        frame_idx=0,
        timestamp=0.0,
        fps=30.0,
        apriltag_result=_tag(T_world_camera),
        anchor_result=_anchor(T=T_camera_object_raw),
        cup_estimate=_cup(),
        cup_detected=True,
    )
    expected_aligned = align_object_pose(T_camera_object_raw, session.settings)
    np.testing.assert_allclose(view.T_camera_object_aligned, expected_aligned)
    np.testing.assert_allclose(
        session.T_world_object_preview,
        T_world_camera @ expected_aligned,
        atol=1e-10,
    )
    restored = session.T_world_object_preview @ np.linalg.inv(
        view.T_camera_object_aligned
    )
    np.testing.assert_allclose(restored, T_world_camera, atol=1e-10)


def test_axis_renderer_distinguishes_raw_and_aligned_in_debug() -> None:
    from object_anchor_preview import PreviewFrameView

    settings = load_preview_settings({"enabled": True, "save_preview_logs": False})
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    raw = np.eye(4)
    raw[2, 3] = 1.0
    aligned = align_object_pose(raw, settings)
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    normal = PreviewFrameView(
        preview_state="ACTIVE",
        april_status="ACTIVE",
        object_status="ACTIVE",
        difference_status="N/A",
        T_camera_object_raw=raw,
        T_camera_object_aligned=aligned,
        debug_overlay_enabled=False,
    )
    debug = PreviewFrameView(
        preview_state="ACTIVE",
        april_status="ACTIVE",
        object_status="ACTIVE",
        difference_status="N/A",
        T_camera_object_raw=raw,
        T_camera_object_aligned=aligned,
        debug_overlay_enabled=True,
    )
    normal_image = draw_object_preview_axes(base.copy(), normal, K, None, settings)
    debug_image = draw_object_preview_axes(base.copy(), debug, K, None, settings)
    assert int(normal_image.sum()) > 0  # aligned axis remains in default view
    assert int(debug_image.sum()) > int(normal_image.sum())  # raw axis + label added


def test_persist_calibration_rejected() -> None:
    settings = load_preview_settings(
        {"enabled": True, "persist_calibration": True, "save_preview_logs": False}
    )
    try:
        build_preview_session(
            settings,
            repo_root=ROOT,
            model_path="m",
            config_path="c",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "persist_calibration" in str(exc)


def test_session_calibration_not_autoload_marked(tmp_path: Path) -> None:
    session = _session(tmp_path, calibration_samples=3)
    session.start_calibration()
    for i in range(3):
        session.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(_T(1.0)),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    payload = json.loads(
        (session.output_dir / "session_calibration.json").read_text(encoding="utf-8")
    )
    assert payload["auto_load_on_next_run"] is False
    assert payload["not_production_calibration"] is True


def test_orbbec_config_preview_enabled_and_run_ps1_path() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/orbbec_gemini.yaml").read_text(encoding="utf-8")
    )
    preview = config["object_anchor_preview"]
    assert preview["enabled"] is True
    assert preview["auto_switch_world_source"] is False
    assert preview["persist_calibration"] is False
    assert preview["calibration_samples"] == 30
    assert preview["temporal_filter_window"] == 3
    assert preview["debug_overlay_enabled"] is False
    assert preview["debug_overlay_toggle_key"] == "d"
    assert preview["align_object_frame_to_apriltag"] is True
    assert preview["object_frame_alignment"]["translation_m"] == [0.0, 0.0, 0.0]
    assert preview["object_frame_alignment"]["rotation_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
    run_ps1 = (ROOT / "run.ps1").read_text(encoding="utf-8")
    assert "experiments\\repeatability_run.py" in run_ps1
    assert "configs\\orbbec_gemini.yaml" in run_ps1


def test_preview_logs_do_not_touch_production_calibration(tmp_path: Path) -> None:
    production = ROOT / "out/object_anchor_calibration/tissue_box_01_world_pose.yaml"
    before = production.read_bytes() if production.is_file() else None
    session = _session(tmp_path, calibration_samples=2)
    session.start_calibration()
    for i in range(2):
        session.update(
            frame_idx=i,
            timestamp=float(i),
            fps=30.0,
            apriltag_result=_tag(_T(1.0)),
            anchor_result=_anchor(T=_T(0.4)),
            cup_estimate=_cup(),
            cup_detected=True,
        )
    session.close()
    after = production.read_bytes() if production.is_file() else None
    assert before == after
    assert session.output_dir is not None
    assert session.output_dir != production.parent
    assert (session.output_dir / "preview_summary.json").is_file()
    assert load_preview_settings(
        yaml.safe_load((ROOT / "configs/orbbec_gemini.yaml").read_text(encoding="utf-8"))[
            "object_anchor_preview"
        ]
    ).output_root == "out/object_anchor_preview"
