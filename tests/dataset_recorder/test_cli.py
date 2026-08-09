from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.device_inspector import InspectionResult  # noqa: E402
from sensor_validation.profile_selector import ProfileDecision  # noqa: E402
from sensor_validation.sdk_adapter import ProfileHandle  # noqa: E402


def _load_cli_module():
    script = ROOT / "scripts" / "dataset" / "record_gemini335l.py"
    spec = importlib.util.spec_from_file_location("record_gemini335l_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


def _record_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "i_confirm_device_access": True,
        "config": str(ROOT / "configs" / "dataset" / "gemini335l_phase2.yaml"),
        "scenario": "scenario_a",
        "scenario_file": None,
        "output": None,
        "serial": None,
        "device_index": 0,
        "allow_fallback": False,
        "duration": None,
        "preview": False,
        "no_preview": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _rgb_profile() -> ProfileHandle:
    return ProfileHandle(
        profile_id="rgb",
        sensor_type="COLOR",
        stream_type="COLOR",
        kind="video",
        format="RGB",
        width=1280,
        height=800,
        fps=30,
        raw=object(),
    )


def _decision(sensor: str, profile_id: str) -> ProfileDecision:
    return ProfileDecision(
        sensor=sensor,
        requested={},
        selected_profile_id=profile_id,
        selected={"profile_id": profile_id},
        exact_match=True,
        fallback_used=False,
        reason="test",
        candidates_considered=1,
    )


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli_module()
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "record" in output and "validate" in output and "derive" in output


def test_record_refuses_without_confirmation(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli_module()
    monkeypatch.setattr(
        module,
        "OrbbecSdkAdapter",
        lambda: (_ for _ in ()).throw(AssertionError("SDK must not load")),
    )
    assert module.main(["record", "--scenario", "scenario_a"]) == 2
    assert "--i-confirm-device-access" in capsys.readouterr().err


def test_validate_cli_writes_integrity_json(synthetic_session: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli_module()
    code = module.main(["validate", "--session", str(synthetic_session)])
    assert code in (0, 1)
    payload = json.loads((synthetic_session / "integrity.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] in {"VALID", "WARNING", "INVALID"}
    assert "overall_status" in capsys.readouterr().out


def test_record_inspect_uses_serial_number_keywords_device_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_cli_module()
    captured: dict[str, object] = {}
    open_calls = 0

    class FakeAdapter:
        def open_device(self, **_kwargs: object) -> object:
            nonlocal open_calls
            open_calls += 1
            raise AssertionError("cmd_record must not call adapter.open_device directly")

    class StopInspect(Exception):
        pass

    def fake_inspect(adapter: object, *, serial_number: object, device_index: object) -> object:
        captured["adapter"] = adapter
        captured["serial_number"] = serial_number
        captured["device_index"] = device_index
        raise StopInspect()

    monkeypatch.setattr(module, "OrbbecSdkAdapter", FakeAdapter)
    monkeypatch.setattr(module, "inspect_connected_device", fake_inspect)
    monkeypatch.setattr(module, "_session_dir", lambda *_args, **_kwargs: tmp_path / "session")

    with pytest.raises(StopInspect):
        module.cmd_record(_record_args(serial=None, device_index=0))

    assert open_calls == 0
    assert captured["serial_number"] is None
    assert captured["device_index"] == 0


def test_record_inspect_uses_explicit_serial_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_cli_module()
    captured: dict[str, object] = {}

    class FakeAdapter:
        def open_device(self, **_kwargs: object) -> object:
            raise AssertionError("cmd_record must not call adapter.open_device directly")

    class StopInspect(Exception):
        pass

    def fake_inspect(adapter: object, *, serial_number: object, device_index: object) -> object:
        captured["serial_number"] = serial_number
        captured["device_index"] = device_index
        raise StopInspect()

    monkeypatch.setattr(module, "OrbbecSdkAdapter", FakeAdapter)
    monkeypatch.setattr(module, "inspect_connected_device", fake_inspect)
    monkeypatch.setattr(module, "_session_dir", lambda *_args, **_kwargs: tmp_path / "session")

    with pytest.raises(StopInspect):
        module.cmd_record(_record_args(serial="ABC123", device_index=None))

    assert captured["serial_number"] == "ABC123"
    assert captured["device_index"] == 0


def test_derive_help(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli_module()
    with pytest.raises(SystemExit) as exc:
        module.main(["derive", "--help"])
    assert exc.value.code == 0
    assert "derive" in capsys.readouterr().out


def test_derive_cli_warns_when_apriltag_disabled(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_session: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli_module()

    def fake_derive(*_args, **_kwargs):
        return {
            "apriltag_enabled": False,
            "warnings": [
                "apriltag_world.enabled is false; AprilTag observations will not be generated.",
                "objects.json annotation missing; semantic_id will be unknown until annotated.",
            ],
        }

    monkeypatch.setattr(module, "derive_observations", fake_derive)
    code = module.main(["derive", "--session", str(synthetic_session)])
    assert code == 0
    err = capsys.readouterr().err
    assert "apriltag_world.enabled is false" in err


def test_record_help_includes_preview_flags(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_cli_module()
    with pytest.raises(SystemExit) as exc:
        module.main(["record", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--preview" in output
    assert "--no-preview" in output


def test_record_flow_validates_after_session_json_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_cli_module()
    device = object()
    validate_state: dict[str, object] = {}

    class FakeAdapter:
        def open_device(self, **_kwargs: object) -> object:
            raise AssertionError("cmd_record must not call adapter.open_device directly")

    opened = SimpleNamespace(device=device, index=0)
    inspection = InspectionResult(
        opened=opened,
        profiles=[_rgb_profile()],
        device_info={"connected_device": {"name": "Gemini 335L"}},
        stream_profiles={"profiles": []},
    )

    class FakeRecorder:
        def __init__(self, adapter: object, **kwargs: object) -> None:
            pass

        def record(self, **kwargs: object) -> dict[str, object]:
            return {
                "overall_status": "COMPLETE",
                "elapsed_seconds": 15.0,
                "queue_overflow_counts": {},
                "writer_errors": [],
                "callback_errors": [],
                "stop_errors": [],
            }

    def fake_validate(session_dir: Path) -> dict[str, object]:
        session_path = session_dir / "session.json"
        validate_state["session_exists"] = session_path.is_file()
        provisional = json.loads(session_path.read_text(encoding="utf-8"))
        validate_state["provisional_integrity"] = provisional.get("integrity_status")
        return {"overall_status": "VALID"}

    monkeypatch.setattr(module, "OrbbecSdkAdapter", FakeAdapter)
    monkeypatch.setattr(
        module,
        "inspect_connected_device",
        lambda adapter, *, serial_number, device_index: inspection,
    )
    monkeypatch.setattr(
        module,
        "_select_profiles",
        lambda *_args, **_kwargs: ([_decision("RGB", "rgb")], {"RGB": _rgb_profile()}),
    )
    monkeypatch.setattr(
        module,
        "export_profile_calibration",
        lambda _selected: {"source": "test", "intrinsics": [], "extrinsics": []},
    )
    monkeypatch.setattr(module, "DatasetRecorder", FakeRecorder)
    monkeypatch.setattr(module, "validate_dataset_session", fake_validate)
    monkeypatch.setattr(module, "_session_dir", lambda *_args, **_kwargs: tmp_path / "session")

    code = module.cmd_record(_record_args())
    assert code == 0
    assert validate_state["session_exists"] is True
    assert validate_state["provisional_integrity"] is None
    final = json.loads((tmp_path / "session" / "session.json").read_text(encoding="utf-8"))
    assert final["integrity_status"] == "VALID"
    integrity = json.loads((tmp_path / "session" / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["overall_status"] == "VALID"
    assert "missing:session.json" not in integrity.get("blockers", [])


def test_record_flow_reaches_dataset_recorder_without_double_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_cli_module()
    device = object()
    open_calls = 0
    recorder_kwargs: dict[str, object] = {}

    class FakeAdapter:
        def open_device(self, **_kwargs: object) -> object:
            nonlocal open_calls
            open_calls += 1
            raise AssertionError("cmd_record must not call adapter.open_device directly")

    opened = SimpleNamespace(device=device, index=0)
    inspection = InspectionResult(
        opened=opened,
        profiles=[_rgb_profile()],
        device_info={"connected_device": {"name": "Gemini 335L"}},
        stream_profiles={"profiles": []},
    )

    class FakeRecorder:
        def __init__(self, adapter: object, **kwargs: object) -> None:
            recorder_kwargs["adapter"] = adapter
            recorder_kwargs.update(kwargs)

        def record(self, **kwargs: object) -> dict[str, object]:
            recorder_kwargs["record_kwargs"] = kwargs
            return {
                "overall_status": "COMPLETE",
                "elapsed_seconds": 15.0,
                "queue_overflow_counts": {},
                "writer_errors": [],
                "callback_errors": [],
                "stop_errors": [],
            }

    monkeypatch.setattr(module, "OrbbecSdkAdapter", FakeAdapter)
    monkeypatch.setattr(
        module,
        "inspect_connected_device",
        lambda adapter, *, serial_number, device_index: inspection,
    )
    monkeypatch.setattr(
        module,
        "_select_profiles",
        lambda *_args, **_kwargs: ([_decision("RGB", "rgb")], {"RGB": _rgb_profile()}),
    )
    monkeypatch.setattr(
        module,
        "export_profile_calibration",
        lambda _selected: {"source": "test", "intrinsics": [], "extrinsics": []},
    )
    monkeypatch.setattr(module, "DatasetRecorder", FakeRecorder)
    monkeypatch.setattr(
        module,
        "validate_dataset_session",
        lambda *_args, **_kwargs: {"overall_status": "VALID"},
    )
    monkeypatch.setattr(module, "_session_dir", lambda *_args, **_kwargs: tmp_path / "session")

    code = module.cmd_record(_record_args())
    assert code == 0
    assert open_calls == 0
    assert recorder_kwargs["device"] is device
    assert recorder_kwargs["session_dir"] == tmp_path / "session"
    assert recorder_kwargs["preview_enabled"] is False
    assert recorder_kwargs["record_kwargs"]["scenario_slug"] == "scenario_a"
    final = json.loads((tmp_path / "session" / "session.json").read_text(encoding="utf-8"))
    assert final["integrity_status"] == "VALID"
    assert (tmp_path / "session" / "recording_state.json").is_file()
