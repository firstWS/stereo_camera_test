from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MVP_PATH = ROOT / "experiments" / "object_anchor_mvp_final_comparison.py"
SPEC = importlib.util.spec_from_file_location("object_anchor_mvp_final_comparison", MVP_PATH)
assert SPEC is not None and SPEC.loader is not None
MVP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MVP)

DIAG_PATH = ROOT / "scripts" / "diagnose_mvp_branch_mapping.py"
DIAG_SPEC = importlib.util.spec_from_file_location("diagnose_mvp_branch_mapping", DIAG_PATH)
assert DIAG_SPEC is not None and DIAG_SPEC.loader is not None
DIAG = importlib.util.module_from_spec(DIAG_SPEC)
DIAG_SPEC.loader.exec_module(DIAG)


def _T(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out[0, 3] = x
    return out


def test_label_swap_still_matches_same_prototype_pose() -> None:
    proto_a = _T(1.0)
    proto_b = _T(3.5, 50.0)
    # Registration labels: 0=A, 1=B
    reg = {0: proto_a, 1: proto_b}
    # Swapped numeric labels with identical poses
    swapped = {0: proto_b, 1: proto_a}
    sample = _T(1.01, 1.0)
    m1 = MVP.match_pose_to_prototypes(
        sample, reg, translation_threshold_m=0.25, rotation_threshold_deg=20.0
    )
    m2 = MVP.match_pose_to_prototypes(
        sample, swapped, translation_threshold_m=0.25, rotation_threshold_deg=20.0
    )
    assert m1["status"] == "matched" and m1["branch_id"] == 0
    assert m2["status"] == "matched" and m2["branch_id"] == 1
    # Same physical prototype (near x=1) despite different IDs.
    assert np.allclose(reg[m1["branch_id"]][:3, 3], swapped[m2["branch_id"]][:3, 3], atol=1e-9)


def test_nearest_prototype_and_ambiguous_unknown() -> None:
    prototypes = {0: _T(0.0), 1: _T(2.5, 55.0)}
    near0 = MVP.match_pose_to_prototypes(
        _T(0.02), prototypes, translation_threshold_m=0.25, rotation_threshold_deg=20.0
    )
    assert near0["branch_id"] == 0
    far = MVP.match_pose_to_prototypes(
        _T(10.0), prototypes, translation_threshold_m=0.25, rotation_threshold_deg=20.0
    )
    assert far["branch_id"] is None
    assert far["status"] == "unknown_too_far"
    # Two nearly equidistant prototypes both in-threshold => ambiguous.
    close = {0: _T(0.0), 1: _T(0.04)}
    amb = MVP.match_pose_to_prototypes(
        _T(0.02),
        close,
        translation_threshold_m=0.25,
        rotation_threshold_deg=20.0,
        ambiguous_margin_translation_m=0.05,
        ambiguous_margin_rotation_deg=5.0,
    )
    assert amb["branch_id"] is None
    assert amb["status"] == "unknown_ambiguous"


def test_calibration_missing_branch_excluded_safely() -> None:
    # Pose matches internal id 0, but only id 1 has calibration.
    prototypes = {0: _T(1.0), 1: _T(3.5, 50.0)}
    accepted = {1: {"T_tag_object": _T(0.5)}}
    match = MVP.match_pose_to_prototypes(
        _T(1.01), prototypes, translation_threshold_m=0.25, rotation_threshold_deg=20.0
    )
    assert match["branch_id"] == 0
    assert match["branch_id"] not in accepted


def test_count_audit_explains_152_vs_118_paradox() -> None:
    source = ROOT / "out/object_anchor_full99/mvp_final_comparison/20260726_172325"
    audit = DIAG.audit_registration_counts(source)
    assert audit["frames_by_label"]["0"] == 118
    assert audit["frames_by_label"]["1"] == 182
    assert audit["candidates_by_label"]["0"] == 100
    assert audit["candidates_by_label"]["1"] == 152
    assert audit["duplicate_append_detected"] is False
    assert audit["verdict"] in {"A", "D"}
    assert audit["per_branch"]["1"]["candidates_le_cluster_frames"] is True


def test_production_hashes_stable() -> None:
    for name, digest in DIAG.production_hashes().items():
        assert len(digest) == 64
        path = ROOT / ("run.ps1" if name == "run.ps1" else f"src/{name}")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
