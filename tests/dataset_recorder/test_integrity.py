from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.integrity import validate_dataset_session  # noqa: E402

_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures.py"
_SPEC = importlib.util.spec_from_file_location("dataset_recorder_fixtures", _FIXTURES_PATH)
assert _SPEC and _SPEC.loader
_FIXTURES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURES)
build_synthetic_session = _FIXTURES.build_synthetic_session


def test_validate_synthetic_session_is_valid(synthetic_session: Path) -> None:
    result = validate_dataset_session(synthetic_session)
    assert result["overall_status"] == "WARNING"
    assert "camera_imu_extrinsic_not_available" in result["warnings"]
    assert result["blockers"] == []


def test_missing_required_file_is_invalid(tmp_path: Path) -> None:
    session_dir = build_synthetic_session(tmp_path)
    (session_dir / "scenario.json").unlink()
    result = validate_dataset_session(session_dir)
    assert result["overall_status"] == "INVALID"
    assert any("missing:scenario.json" in blocker for blocker in result["blockers"])


def test_frame_file_mismatch_is_invalid(tmp_path: Path) -> None:
    session_dir = build_synthetic_session(tmp_path)
    frame = session_dir / "streams" / "rgb" / "frames" / "frame_000000.png"
    frame.unlink()
    result = validate_dataset_session(session_dir)
    assert result["overall_status"] == "INVALID"
    assert any("frame_file_mismatch:RGB" in blocker for blocker in result["blockers"])


def test_camera_imu_available_yields_valid(tmp_path: Path) -> None:
    session_dir = build_synthetic_session(tmp_path, camera_imu_available=True)
    result = validate_dataset_session(session_dir)
    assert result["overall_status"] == "VALID"
