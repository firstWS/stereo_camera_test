from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.sdk_adapter import ProfileHandle  # noqa: E402
from sensor_validation.sensor_recorder import CsvWriterThread, SensorRecorder  # noqa: E402


class FakeFrame:
    def __init__(self, index: int, *, imu: bool = False) -> None:
        self.index = index
        self.imu = imu

    def get_index(self) -> int:
        return self.index

    def get_timestamp_us(self) -> int:
        return 1000 + self.index * 100

    def get_system_timestamp_us(self) -> int:
        return 2000 + self.index * 100

    def get_global_timestamp_us(self) -> int:
        return 3000 + self.index * 100

    def get_width(self) -> int:
        return 2

    def get_height(self) -> int:
        return 2

    def get_format(self) -> SimpleNamespace:
        return SimpleNamespace(name="Y8")

    def get_data(self) -> bytes:
        return bytes([1, 2, 3, 4])

    def get_data_size(self) -> int:
        return 4

    def get_temperature(self) -> float:
        return 25.0

    def get_x(self) -> float:
        return 0.1

    def get_y(self) -> float:
        return 0.2

    def get_z(self) -> float:
        return 0.3


class FakeFrameSet:
    def get_color_frame(self) -> FakeFrame:
        return FakeFrame(1)

    def get_depth_frame(self) -> None:
        return None

    def get_left_ir_frame(self) -> None:
        return None

    def get_right_ir_frame(self) -> None:
        return None

    def get_accel_frame(self) -> FakeFrame:
        return FakeFrame(2, imu=True)

    def get_gyro_frame(self) -> FakeFrame:
        return FakeFrame(3, imu=True)


class FakePipeline:
    def __init__(self, kind: str, *, fail_start: bool = False) -> None:
        self.kind = kind
        self.fail_start = fail_start
        self.started = False
        self.stopped = False

    def start(self, _config: object, callback: object) -> None:
        if self.fail_start:
            raise RuntimeError("start failed")
        self.started = True
        callback(FakeFrameSet())

    def stop(self) -> None:
        self.stopped = True


class FakeAdapter:
    def __init__(self, *, fail_second_start: bool = False) -> None:
        self.sdk = SimpleNamespace()
        self.created: list[FakePipeline] = []
        self.fail_second_start = fail_second_start

    def create_pipeline(self, _device: object) -> FakePipeline:
        kind = "video" if not self.created else "imu"
        pipeline = FakePipeline(
            kind,
            fail_start=self.fail_second_start and kind == "imu",
        )
        self.created.append(pipeline)
        return pipeline

    def create_config(self) -> SimpleNamespace:
        return SimpleNamespace(enable_stream=lambda _profile: None)

    @staticmethod
    def enable_profiles(config: object, profiles: object) -> None:
        for profile in profiles:
            config.enable_stream(profile.raw)

    @staticmethod
    def stop_pipeline(pipeline: FakePipeline) -> None:
        pipeline.stop()
        return None


def _selected() -> dict[str, ProfileHandle]:
    return {
        "RGB": ProfileHandle(
            profile_id="rgb",
            sensor_type="COLOR",
            stream_type="COLOR",
            kind="video",
            format="Y8",
            width=2,
            height=2,
            fps=30,
            raw=object(),
        ),
        "ACCEL": ProfileHandle(
            profile_id="accel",
            sensor_type="ACCEL",
            stream_type="ACCEL",
            kind="accel",
            format="ACCEL",
            sample_rate="200_HZ",
            raw=object(),
        ),
        "GYRO": ProfileHandle(
            profile_id="gyro",
            sensor_type="GYRO",
            stream_type="GYRO",
            kind="gyro",
            format="GYRO",
            sample_rate="200_HZ",
            raw=object(),
        ),
    }


def test_recorder_stops_both_pipelines_and_flushes_files(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    recorder = SensorRecorder(
        adapter,
        device=object(),
        selected_profiles=_selected(),
        output_dir=tmp_path,
        no_preview=True,
    )
    state = recorder.record(duration_seconds=0.01, mode="static", progress=lambda _: None)
    assert state["overall_status"] == "COMPLETE"
    assert all(pipeline.stopped for pipeline in adapter.created)
    assert "RGB" in (tmp_path / "video_frames.csv").read_text(encoding="utf-8")
    imu_csv = (tmp_path / "imu_samples.csv").read_text(encoding="utf-8")
    assert "ACCEL" in imu_csv and "GYRO" in imu_csv
    assert list((tmp_path / "frames").glob("rgb_first_*.png"))


def test_second_pipeline_start_failure_still_stops_first(tmp_path: Path) -> None:
    adapter = FakeAdapter(fail_second_start=True)
    recorder = SensorRecorder(
        adapter,
        device=object(),
        selected_profiles=_selected(),
        output_dir=tmp_path,
        no_preview=True,
    )
    state = recorder.record(duration_seconds=0.01, mode="static", progress=lambda _: None)
    assert state["overall_status"] == "INCOMPLETE"
    assert "start failed" in state["fatal_error"]
    assert adapter.created[0].stopped is True
    assert adapter.created[1].stopped is True
    assert (tmp_path / "video_frames.csv").is_file()


def test_queue_overflow_is_counted_without_writer(tmp_path: Path) -> None:
    writer = CsvWriterThread(tmp_path, queue_size=1)
    assert writer.submit("video", {"stream": "RGB"}) is True
    assert writer.submit("video", {"stream": "RGB"}) is False
    assert writer.overflow_counts["video"] == 1


def test_live_cli_refuses_access_without_confirmation(
    monkeypatch: object, capsys: object
) -> None:
    script = ROOT / "scripts" / "sensor_validation" / "validate_gemini335l.py"
    spec = importlib.util.spec_from_file_location("validate_gemini335l_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "OrbbecSdkAdapter",
        lambda: (_ for _ in ()).throw(AssertionError("SDK must not load")),
    )
    assert module.main(["inspect"]) == 2
    assert "--i-confirm-device-access" in capsys.readouterr().err


def test_static_mode_does_not_call_motion_beep(
    tmp_path: Path, monkeypatch: object
) -> None:
    import sensor_validation.sensor_recorder as recorder_mod

    calls: list[str] = []
    monkeypatch.setattr(
        recorder_mod, "_play_motion_beep", lambda cue: calls.append(cue)
    )
    adapter = FakeAdapter()
    recorder = SensorRecorder(
        adapter,
        device=object(),
        selected_profiles=_selected(),
        output_dir=tmp_path,
        no_preview=True,
    )
    state = recorder.record(duration_seconds=0.05, mode="static", progress=lambda _: None)
    assert state["overall_status"] == "COMPLETE"
    assert calls == []


def test_translation_modes_call_motion_beep(
    tmp_path: Path, monkeypatch: object
) -> None:
    import sensor_validation.sensor_recorder as recorder_mod

    for mode in ("translation", "translation-yaw"):
        calls: list[str] = []
        monkeypatch.setattr(
            recorder_mod, "_play_motion_beep", lambda cue, c=calls: c.append(cue)
        )
        adapter = FakeAdapter()
        recorder = SensorRecorder(
            adapter,
            device=object(),
            selected_profiles=_selected(),
            output_dir=tmp_path / mode,
            no_preview=True,
        )
        state = recorder.record(
            duration_seconds=0.05, mode=mode, progress=lambda _: None
        )
        assert state["overall_status"] == "COMPLETE"
        assert "ready" in calls
        assert state["requested_duration_seconds"] == 0.05


def test_play_motion_beep_swallows_winsound_and_bell_failures(
    monkeypatch: object,
) -> None:
    import builtins
    import types

    import sensor_validation.sensor_recorder as recorder_mod

    broken = types.ModuleType("winsound")

    def _failing_beep(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("beep failed")

    broken.Beep = _failing_beep  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "winsound", broken)
    monkeypatch.setattr(
        builtins,
        "print",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("print failed")),
    )
    recorder_mod._play_motion_beep("ready")
    recorder_mod._play_motion_beep("move")


def test_play_motion_beep_falls_back_when_winsound_missing(
    monkeypatch: object,
) -> None:
    import sensor_validation.sensor_recorder as recorder_mod

    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "winsound":
            raise ImportError("no winsound")
        return real_import(name, *args, **kwargs)

    bells: list[str] = []

    def capture_print(*args: object, **_kwargs: object) -> None:
        bells.append("".join(str(a) for a in args))

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr("builtins.print", capture_print)
    recorder_mod._play_motion_beep("countdown")
    assert any("\a" in text for text in bells)
