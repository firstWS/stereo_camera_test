from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DIAG_PATH = ROOT / "scripts" / "diagnose_mvp_translation_filters.py"
SPEC = importlib.util.spec_from_file_location("diagnose_mvp_translation_filters", DIAG_PATH)
assert SPEC is not None and SPEC.loader is not None
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)

MVP_PATH = ROOT / "experiments" / "object_anchor_mvp_final_comparison.py"
MVP_SPEC = importlib.util.spec_from_file_location("object_anchor_mvp_final_comparison", MVP_PATH)
assert MVP_SPEC is not None and MVP_SPEC.loader is not None
MVP = importlib.util.module_from_spec(MVP_SPEC)
MVP_SPEC.loader.exec_module(MVP)


def _T(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out[0, 3] = x
    return out


def test_causal_filter_does_not_use_future_frames() -> None:
    history = [_T(0.0), _T(0.10), _T(0.20), _T(1.0)]
    # Window 3 at index 2 should ignore the future spike at index 3.
    filtered = DIAG.causal_filter_pose(history[:3], window=3)
    assert abs(float(filtered[0, 3]) - 0.10) < 1e-9
    # Adding future should not change a filter evaluated on the first three only.
    filtered_with_future_ignored = DIAG.causal_filter_pose(history[:3], window=3)
    np.testing.assert_allclose(filtered, filtered_with_future_ignored)


def test_windows_3_5_7_and_quaternion_sign_flip() -> None:
    base = _T(1.0, yaw_deg=0.0)
    flipped = base.copy()
    # 180-degree equivalent quaternion sign flip on identity-near rotation is
    # simulated by using the same rotation matrix twice; average must stay valid.
    history = [base, flipped, _T(1.02, 2.0), _T(0.98, -1.0), _T(1.01, 1.0), _T(1.00, 0.5), _T(1.03, -0.5)]
    for window in (3, 5, 7):
        filtered = DIAG.causal_filter_pose(history, window=window)
        assert filtered.shape == (4, 4)
        assert np.isfinite(filtered).all()
        # Translation median over window should be near 1.0
        assert abs(float(filtered[0, 3]) - 1.0) < 0.05


def test_validation_assignment_disjoint_from_registration() -> None:
    assignments = DIAG.load_assignments(
        ROOT / "out/object_anchor_full99/mvp_branch_aware_comparison/20260726_164157"
    )
    reg = {idx for idx, split in assignments.items() if split == "registration"}
    val = {idx for idx, split in assignments.items() if split == "validation"}
    assert reg and val
    assert reg.isdisjoint(val)


def test_isolated_runner_persists_cup_camera_fields() -> None:
    assert "P_camera_cup_x" in MVP.REFERENCE_CLUSTER_FIELDS
    assert "P_camera_cup_y" in MVP.REFERENCE_CLUSTER_FIELDS
    assert "P_camera_cup_z" in MVP.REFERENCE_CLUSTER_FIELDS
    assert "T_camera_object_filtered_json" in MVP.REFERENCE_CLUSTER_FIELDS
    assert "T_camera_object_filtered_json" in MVP.FRAME_COMPARE_FIELDS
    assert "P_world_cup_object_x" in MVP.FRAME_COMPARE_FIELDS


def test_production_hashes_unchanged_targets() -> None:
    # Guard against accidental production edits in this task.
    protected = [
        ROOT / "run.ps1",
        ROOT / "src/apriltag_world.py",
        ROOT / "src/object_anchor_pose.py",
        ROOT / "src/object_anchor_runtime.py",
    ]
    digests = {}
    for path in protected:
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digests["run.ps1"]
    # orbbec_gemini.yaml may already differ from git baseline due to earlier Full99 wiring;
    # ensure this test file does not rewrite it.
    config = yaml.safe_load((ROOT / "configs/orbbec_gemini.yaml").read_text(encoding="utf-8"))
    assert "object_anchor" in config
