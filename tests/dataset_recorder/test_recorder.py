from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensor_validation.sdk_adapter import ProfileHandle  # noqa: E402

from dataset_recorder.frame_writer import FrameSnapshot, save_frame_file  # noqa: E402
from dataset_recorder.recorder import (  # noqa: E402
    DatasetRecorder,
    DatasetWriterThread,
    emit_scenario_motion_cues,
    scenario_a_cue_for_elapsed,
)


class FakeFrame:
    def __init__(self, index: int, *, imu: bool = False) -> None:
        self.index = index
        self.imu = imu

    def get_index(self) -> int:
        return self.index

    def get_timestamp_us(self) -> int:
        return 1000 + self.index * 33_333

    def get_system_timestamp_us(self) -> int:
        return 2000 + self.index * 33_333

    def get_global_timestamp_us(self) -> int:
        return 3000 + self.index * 33_333

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

    def get_depth_scale(self) -> float:
        return 0.001

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

    def get_depth_frame(self) -> FakeFrame:
        return FakeFrame(2)

    def get_left_ir_frame(self) -> FakeFrame:
        return FakeFrame(3)

    def get_right_ir_frame(self) -> FakeFrame:
        return FakeFrame(4)

    def get_accel_frame(self) -> FakeFrame:
        return FakeFrame(5, imu=True)

    def get_gyro_frame(self) -> FakeFrame:
        return FakeFrame(6, imu=True)


class FakePipeline:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.started = False
        self.stopped = False

    def start(self, _config: object, callback: object) -> None:
        self.started = True
        callback(FakeFrameSet())

    def stop(self) -> None:
        self.stopped = True


class FakeAdapter:
    def __init__(self) -> None:
        self.sdk = SimpleNamespace(
            OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE=object())
        )
        self.created: list[FakePipeline] = []

    def create_pipeline(self, _device: object) -> FakePipeline:
        kind = "video" if not self.created else "imu"
        pipeline = FakePipeline(kind)
        self.created.append(pipeline)
        return pipeline

    def create_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            enable_stream=lambda _profile: None,
            set_frame_aggregate_output_mode=lambda _mode: None,
        )

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
        "DEPTH": ProfileHandle(
            profile_id="depth",
            sensor_type="DEPTH",
            stream_type="DEPTH",
            kind="video",
            format="Y16",
            width=2,
            height=2,
            fps=30,
            raw=object(),
        ),
        "LEFT_IR": ProfileHandle(
            profile_id="left_ir",
            sensor_type="IR",
            stream_type="LEFT_IR",
            kind="video",
            format="Y8",
            width=2,
            height=2,
            fps=30,
            raw=object(),
        ),
        "RIGHT_IR": ProfileHandle(
            profile_id="right_ir",
            sensor_type="IR",
            stream_type="RIGHT_IR",
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


def test_writer_thread_creates_stream_layout(tmp_path: Path) -> None:
    writer = DatasetWriterThread(tmp_path, queue_size=8)
    writer.start()
    snapshot = FrameSnapshot(
        stream="RGB",
        sequence=1,
        width=2,
        height=2,
        format_name="Y8",
        data=bytes([1, 2, 3, 4]),
        depth_scale=None,
    )
    writer.submit(
        "video",
        (
            {
                "stream": "RGB",
                "frame_number": 0,
                "received_sequence": 1,
                "callback_sequence": 1,
            },
            snapshot,
        ),
    )
    writer.close()
    assert (tmp_path / "streams" / "rgb" / "index.csv").is_file()
    assert list((tmp_path / "streams" / "rgb" / "frames").glob("*.png"))


def test_dataset_recorder_writes_all_selected_streams(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    recorder = DatasetRecorder(
        adapter,
        device=object(),
        selected_profiles=_selected(),
        session_dir=tmp_path,
        queue_size=64,
    )
    state = recorder.record(duration_seconds=0.01, progress=lambda _: None)
    assert state["overall_status"] == "COMPLETE"
    for stream in ("rgb", "depth", "left_ir", "right_ir"):
        assert (tmp_path / "streams" / stream / "index.csv").is_file()
    assert (tmp_path / "streams" / "accel.csv").is_file()
    assert (tmp_path / "streams" / "gyro.csv").is_file()
    assert (tmp_path / "events.csv").is_file()


def test_save_depth_frame_uses_uint16_png(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    snapshot = FrameSnapshot(
        stream="DEPTH",
        sequence=1,
        width=2,
        height=2,
        format_name="Y16",
        data=np.array([[100, 200], [300, 400]], dtype=np.uint16).tobytes(),
        depth_scale=0.001,
    )
    saved = save_frame_file(snapshot, tmp_path / "frames", frame_number=0)
    image = cv2.imread(str(tmp_path / "frames" / saved["file_name"]), cv2.IMREAD_UNCHANGED)
    assert image.dtype == np.uint16
    assert image.shape == (2, 2)


def test_flush_progress_uses_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataset_recorder.recorder import flush_progress

    calls: list[dict[str, object]] = []

    def fake_print(*args: object, **kwargs: object) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr("builtins.print", fake_print)
    flush_progress("hello")
    assert calls[0]["kwargs"].get("flush") is True


def test_scenario_a_cue_phases() -> None:
    assert scenario_a_cue_for_elapsed(0.0)[0] == "hold"
    assert scenario_a_cue_for_elapsed(2.0)[0] == "prepare"
    assert scenario_a_cue_for_elapsed(3.0)[0] == "yaw_start"
    assert scenario_a_cue_for_elapsed(5.0)[0] == "final_hold"
    yaw_msg = scenario_a_cue_for_elapsed(3.5)[1]
    assert "20~30" in yaw_msg
    assert "0.30" not in yaw_msg
    assert "0.5m" not in yaw_msg


def test_scenario_a_cues_emit_once_without_translation_text(monkeypatch: object) -> None:
    messages: list[str] = []
    events: list[tuple[str, float]] = []
    monkeypatch.setattr(
        "dataset_recorder.recorder._play_scenario_beep",
        lambda _cue: None,
    )
    last = None
    for elapsed in (0.0, 0.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0):
        last = emit_scenario_motion_cues(
            scenario_slug="scenario_a",
            elapsed_seconds=elapsed,
            last_cue_id=last,
            progress=messages.append,
            on_cue=lambda event_type, _message, cue_elapsed: events.append(
                (event_type, cue_elapsed)
            ),
        )
    joined = "\n".join(messages)
    assert joined.count("[정지]") == 2
    assert "[준비]" in joined
    assert "[회전 시작]" in joined
    assert "0.30m" not in joined
    assert "횡이동" not in joined
    assert [event for event, _elapsed in events] == [
        "motion_prepare",
        "motion_start",
        "motion_end",
    ]
    assert events[0][1] == pytest.approx(2.0)
    assert events[1][1] == pytest.approx(3.0)
    assert events[2][1] == pytest.approx(5.0)
    assert emit_scenario_motion_cues(
        scenario_slug="scenario_b",
        elapsed_seconds=3.0,
        last_cue_id=None,
        progress=messages.append,
    ) is None
