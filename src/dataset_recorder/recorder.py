"""Full-frame Gemini 335L dataset recorder for Phase 2."""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from sensor_validation.sdk_adapter import OrbbecSdkAdapter, ProfileHandle, enum_name
from sensor_validation.sensor_recorder import (
    _frame_common,
    _imu_xyz,
    _safe_method,
    _snapshot,
)

from .frame_writer import FrameSnapshot, save_frame_file
from .rgb_preview import RgbPreview
from .types import EVENT_FIELDS, IMU_CSV_FIELDS, STREAM_DIR_NAMES, STREAM_INDEX_FIELDS


# Scenario A planned operator cues (elapsed seconds). Not ground truth.
_SCENARIO_A_CUES: tuple[tuple[float, str, str], ...] = (
    (
        0.0,
        "hold",
        "[정지]\n카메라를 움직이지 마세요.",
    ),
    (
        2.0,
        "prepare",
        "[준비]\n1초 후 카메라를 오른쪽으로 돌립니다.",
    ),
    (
        3.0,
        "yaw_start",
        (
            "[회전 시작]\n"
            "카메라를 오른쪽으로 천천히 돌리세요.\n"
            "목표 Yaw 약 20~30도.\n"
            "약간의 자연스러운 이동은 허용됩니다."
        ),
    ),
    (
        5.0,
        "final_hold",
        "[정지]\n회전을 멈추고 현재 자세를 유지하세요.",
    ),
)

_MOTION_CUE_EVENT_TYPES: dict[str, str] = {
    "prepare": "motion_prepare",
    "yaw_start": "motion_start",
    "final_hold": "motion_end",
}


def flush_progress(message: str) -> None:
    print(message, flush=True)


def _play_scenario_beep(cue: str) -> None:
    """Optional audible cue for Scenario A phases; never raises into the recorder loop."""

    def _terminal_bell() -> None:
        try:
            print("\a", flush=True)
        except Exception:
            return

    try:
        import winsound
    except Exception:
        _terminal_bell()
        return
    try:
        if cue == "hold":
            winsound.Beep(880, 120)
        elif cue == "prepare":
            winsound.Beep(660, 90)
        elif cue == "yaw_start":
            for _ in range(3):
                winsound.Beep(1320, 220)
                time.sleep(0.05)
        elif cue == "final_hold":
            for _ in range(2):
                winsound.Beep(520, 180)
                time.sleep(0.05)
        else:
            winsound.Beep(880, 100)
    except Exception:
        _terminal_bell()


def scenario_a_cue_for_elapsed(elapsed_seconds: float) -> tuple[str, str] | None:
    """Return the latest Scenario A cue that should be active at ``elapsed_seconds``."""

    active: tuple[str, str] | None = None
    for threshold, cue_id, message in _SCENARIO_A_CUES:
        if elapsed_seconds + 1e-9 >= threshold:
            active = (cue_id, message)
        else:
            break
    return active


def emit_scenario_motion_cues(
    *,
    scenario_slug: str | None,
    elapsed_seconds: float,
    last_cue_id: str | None,
    progress: Callable[[str], None],
    on_cue: Callable[[str, str, float], None] | None = None,
) -> str | None:
    """Emit Scenario A motion cues once per phase. Returns the active cue id."""

    if scenario_slug != "scenario_a":
        return last_cue_id
    active = scenario_a_cue_for_elapsed(elapsed_seconds)
    if active is None:
        return last_cue_id
    cue_id, message = active
    if cue_id == last_cue_id:
        return last_cue_id
    progress("=" * 64)
    progress(message)
    progress("=" * 64)
    _play_scenario_beep(cue_id)
    event_type = _MOTION_CUE_EVENT_TYPES.get(cue_id)
    if on_cue is not None and event_type is not None:
        on_cue(event_type, f"cue:{cue_id}:elapsed={elapsed_seconds:.3f}s", elapsed_seconds)
    return cue_id


class DatasetWriterThread:
    """Bounded queue writer for per-stream index CSV, IMU CSV, and frame files."""

    def __init__(self, session_dir: Path, *, queue_size: int) -> None:
        self.session_dir = session_dir
        self.streams_dir = session_dir / "streams"
        self.queue: queue.Queue[tuple[str, Any] | None] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self.overflow_counts: dict[str, int] = defaultdict(int)
        self.write_errors: list[str] = []
        self._index_writers: dict[str, csv.DictWriter[str, Any]] = {}
        self._index_handles: dict[str, Any] = {}
        self._imu_writers: dict[str, csv.DictWriter[str, Any]] = {}
        self._imu_handles: dict[str, Any] = {}
        self._events_handle: Any = None
        self._events_writer: csv.DictWriter[str, Any] | None = None
        self._thread = threading.Thread(target=self._run, name="dataset-writer")

    def start(self) -> None:
        self.streams_dir.mkdir(parents=True, exist_ok=True)
        for stream, folder in STREAM_DIR_NAMES.items():
            stream_dir = self.streams_dir / folder
            stream_dir.mkdir(parents=True, exist_ok=True)
            (stream_dir / "frames").mkdir(parents=True, exist_ok=True)
            handle = (stream_dir / "index.csv").open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=STREAM_INDEX_FIELDS, extrasaction="ignore")
            writer.writeheader()
            handle.flush()
            self._index_handles[stream] = handle
            self._index_writers[stream] = writer
        for stream in ("ACCEL", "GYRO"):
            handle = (self.streams_dir / f"{stream.lower()}.csv").open(
                "w", encoding="utf-8", newline=""
            )
            writer = csv.DictWriter(handle, fieldnames=IMU_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            handle.flush()
            self._imu_handles[stream] = handle
            self._imu_writers[stream] = writer
        self._events_handle = (self.session_dir / "events.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._events_writer = csv.DictWriter(
            self._events_handle, fieldnames=EVENT_FIELDS, extrasaction="ignore"
        )
        self._events_writer.writeheader()
        self._events_handle.flush()
        self._thread.start()

    def submit(self, kind: str, payload: Any) -> bool:
        try:
            self.queue.put_nowait((kind, payload))
            return True
        except queue.Full:
            self.overflow_counts[kind] += 1
            return False

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            kind, payload = item
            try:
                if kind == "video":
                    self._write_video(payload)
                elif kind == "imu":
                    self._write_imu(payload)
                elif kind == "event":
                    assert self._events_writer is not None
                    self._events_writer.writerow(payload)
            except Exception as error:
                self.write_errors.append(f"{type(error).__name__}: {error}")
            finally:
                self.queue.task_done()

    def _write_video(self, payload: tuple[dict[str, Any], FrameSnapshot]) -> None:
        row, snapshot = payload
        stream = str(row["stream"])
        folder = STREAM_DIR_NAMES[stream]
        frames_dir = self.streams_dir / folder / "frames"
        frame_number = row.get("frame_number")
        saved = save_frame_file(
            snapshot,
            frames_dir,
            frame_number=int(frame_number) if frame_number is not None else None,
        )
        row = dict(row)
        row["file_name"] = saved["file_name"]
        self._index_writers[stream].writerow(row)

    def _write_imu(self, row: dict[str, Any]) -> None:
        stream = str(row["stream"])
        self._imu_writers[stream].writerow(row)

    def close(self) -> None:
        if self._thread.is_alive():
            self.queue.put(None)
            self._thread.join(timeout=30.0)
        for handle in list(self._index_handles.values()) + list(self._imu_handles.values()):
            try:
                handle.flush()
                handle.close()
            except Exception as error:
                self.write_errors.append(f"{type(error).__name__}: {error}")
        if self._events_handle is not None:
            try:
                self._events_handle.flush()
                self._events_handle.close()
            except Exception as error:
                self.write_errors.append(f"{type(error).__name__}: {error}")


def _to_frame_snapshot(snapshot: Any) -> FrameSnapshot:
    return FrameSnapshot(
        stream=snapshot.stream,
        sequence=snapshot.sequence,
        width=snapshot.width,
        height=snapshot.height,
        format_name=snapshot.format_name,
        data=snapshot.data,
        depth_scale=snapshot.depth_scale,
    )


class DatasetRecorder:
    """Record all video frames and IMU samples into a Phase 2 dataset session."""

    def __init__(
        self,
        adapter: OrbbecSdkAdapter,
        *,
        device: Any,
        selected_profiles: Mapping[str, ProfileHandle],
        session_dir: Path,
        queue_size: int = 8192,
        preview_enabled: bool = False,
    ) -> None:
        self.adapter = adapter
        self.device = device
        self.selected_profiles = dict(selected_profiles)
        self.session_dir = session_dir
        self.writer = DatasetWriterThread(session_dir, queue_size=queue_size)
        self.preview = RgbPreview(enabled=preview_enabled)
        self.sequences: dict[str, int] = defaultdict(int)
        self.callback_sequences: dict[str, int] = defaultdict(int)
        self.callback_errors: list[str] = []
        self._record_start_perf_ns = 0

    def log_event(
        self,
        event_type: str,
        message: str,
        *,
        host_monotonic_ns: int | None = None,
        device_timestamp_us: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        monotonic_ns = host_monotonic_ns
        if monotonic_ns is None and elapsed_seconds is not None and self._record_start_perf_ns:
            monotonic_ns = self._record_start_perf_ns + int(elapsed_seconds * 1_000_000_000)
        self.writer.submit(
            "event",
            {
                "event_time_host_monotonic_ns": monotonic_ns or time.perf_counter_ns(),
                "event_time_device_us": device_timestamp_us,
                "event_type": event_type,
                "message": message,
            },
        )

    def _video_callback(self, frames: Any) -> None:
        host_monotonic_ns = time.perf_counter_ns()
        host_wall_time_ns = time.time_ns()
        self.callback_sequences["video"] += 1
        callback_sequence = self.callback_sequences["video"]
        getters = {
            "RGB": "get_color_frame",
            "DEPTH": "get_depth_frame",
            "LEFT_IR": "get_left_ir_frame",
            "RIGHT_IR": "get_right_ir_frame",
        }
        for stream, getter in getters.items():
            if stream not in self.selected_profiles:
                continue
            try:
                frame = _safe_method(frames, getter)
                if frame is None:
                    continue
                snapshot = _snapshot(frame, stream, 0)
                if snapshot is None:
                    continue
                self.sequences[stream] += 1
                sequence = self.sequences[stream]
                snapshot.sequence = sequence
                row = _frame_common(
                    frame,
                    stream=stream,
                    sequence=sequence,
                    callback_sequence=callback_sequence,
                    host_monotonic_ns=host_monotonic_ns,
                    host_wall_time_ns=host_wall_time_ns,
                    sdk=self.adapter.sdk,
                )
                row.update(
                    {
                        "width": _safe_method(frame, "get_width"),
                        "height": _safe_method(frame, "get_height"),
                        "format": enum_name(_safe_method(frame, "get_format")),
                        "data_size_bytes": _safe_method(frame, "get_data_size"),
                        "depth_scale": _safe_method(frame, "get_depth_scale"),
                    }
                )
                frame_snapshot = _to_frame_snapshot(snapshot)
                if stream == "RGB":
                    self.preview.update_from_snapshot(frame_snapshot)
                if not self.writer.submit("video", (row, frame_snapshot)):
                    self.callback_errors.append(f"video:{stream}:{sequence}:queue_full")
            except Exception as error:
                self.callback_errors.append(
                    f"video:{stream}:{type(error).__name__}: {error}"
                )

    def _imu_callback(self, frames: Any) -> None:
        host_monotonic_ns = time.perf_counter_ns()
        host_wall_time_ns = time.time_ns()
        self.callback_sequences["imu"] += 1
        callback_sequence = self.callback_sequences["imu"]
        for stream, getter in (("ACCEL", "get_accel_frame"), ("GYRO", "get_gyro_frame")):
            profile = self.selected_profiles.get(stream)
            if profile is None:
                continue
            try:
                frame = _safe_method(frames, getter)
                if frame is None:
                    continue
                self.sequences[stream] += 1
                sequence = self.sequences[stream]
                x, y, z = _imu_xyz(frame)
                row = _frame_common(
                    frame,
                    stream=stream,
                    sequence=sequence,
                    callback_sequence=callback_sequence,
                    host_monotonic_ns=host_monotonic_ns,
                    host_wall_time_ns=host_wall_time_ns,
                    sdk=self.adapter.sdk,
                )
                row.update(
                    {
                        "sample_rate": profile.sample_rate,
                        "full_scale_range": profile.full_scale_range,
                        "temperature": _safe_method(frame, "get_temperature"),
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )
                if not self.writer.submit("imu", row):
                    self.callback_errors.append(f"imu:{stream}:{sequence}:queue_full")
            except Exception as error:
                self.callback_errors.append(
                    f"imu:{stream}:{type(error).__name__}: {error}"
                )

    def record(
        self,
        *,
        duration_seconds: float,
        progress: Callable[[str], None] = flush_progress,
        scenario_slug: str | None = None,
    ) -> dict[str, Any]:
        video_profiles = [
            self.selected_profiles[name]
            for name in ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR")
            if name in self.selected_profiles
        ]
        imu_profiles = [
            self.selected_profiles[name]
            for name in ("ACCEL", "GYRO")
            if name in self.selected_profiles
        ]
        video_pipeline = None
        imu_pipeline = None
        started_video = False
        started_imu = False
        interrupted = False
        fatal_error: str | None = None
        stop_errors: list[str] = []
        start_perf_ns = time.perf_counter_ns()
        self._record_start_perf_ns = start_perf_ns
        deadline_ns = start_perf_ns + int(duration_seconds * 1_000_000_000)

        def _on_motion_cue(event_type: str, message: str, elapsed: float) -> None:
            self.log_event(event_type, message, elapsed_seconds=elapsed)

        self.writer.start()
        self.log_event("recording_start", "dataset recording started")
        try:
            if video_profiles:
                video_pipeline = self.adapter.create_pipeline(self.device)
                video_config = self.adapter.create_config()
                self.adapter.enable_profiles(video_config, video_profiles)
                aggregate = getattr(
                    getattr(self.adapter.sdk, "OBFrameAggregateOutputMode", None),
                    "FULL_FRAME_REQUIRE",
                    None,
                )
                if aggregate is not None and callable(
                    getattr(video_config, "set_frame_aggregate_output_mode", None)
                ):
                    video_config.set_frame_aggregate_output_mode(aggregate)
                video_pipeline.start(video_config, self._video_callback)
                started_video = True
            if imu_profiles:
                imu_pipeline = self.adapter.create_pipeline(self.device)
                imu_config = self.adapter.create_config()
                self.adapter.enable_profiles(imu_config, imu_profiles)
                imu_pipeline.start(imu_config, self._imu_callback)
                started_imu = True
            if not started_video and not started_imu:
                raise RuntimeError("No selected video or IMU profile can be recorded.")
            last_second: int | None = None
            last_cue_id: str | None = None
            last_cue_id = emit_scenario_motion_cues(
                scenario_slug=scenario_slug,
                elapsed_seconds=0.0,
                last_cue_id=last_cue_id,
                progress=progress,
                on_cue=_on_motion_cue,
            )
            while True:
                remaining = max(0.0, (deadline_ns - time.perf_counter_ns()) / 1_000_000_000)
                if remaining <= 0.0:
                    break
                elapsed = max(0.0, duration_seconds - remaining)
                last_cue_id = emit_scenario_motion_cues(
                    scenario_slug=scenario_slug,
                    elapsed_seconds=elapsed,
                    last_cue_id=last_cue_id,
                    progress=progress,
                    on_cue=_on_motion_cue,
                )
                self.preview.render(
                    scenario_slug=scenario_slug,
                    elapsed_seconds=elapsed,
                    phase_cue_id=last_cue_id,
                )
                integer_remaining = int(remaining)
                if integer_remaining != last_second and integer_remaining % 5 == 0:
                    progress(f"남은 시간: {integer_remaining + 1}초")
                    last_second = integer_remaining
                time.sleep(min(0.05, remaining))
            if scenario_slug == "scenario_a":
                progress("Recording 종료")
        except KeyboardInterrupt:
            interrupted = True
            progress("사용자 중단을 감지했습니다. 부분 결과를 저장합니다.")
        except Exception as error:
            fatal_error = f"{type(error).__name__}: {error}"
        finally:
            self.log_event("recording_stop", "dataset recording stopped")
            if imu_pipeline is not None:
                if error := self.adapter.stop_pipeline(imu_pipeline):
                    stop_errors.append(f"imu:{error}")
            if video_pipeline is not None:
                if error := self.adapter.stop_pipeline(video_pipeline):
                    stop_errors.append(f"video:{error}")
            self.writer.close()
            self.preview.close()
        elapsed_seconds = (time.perf_counter_ns() - start_perf_ns) / 1_000_000_000
        complete = (
            fatal_error is None
            and not interrupted
            and not stop_errors
            and not self.writer.write_errors
            and not self.callback_errors
            and elapsed_seconds + 0.05 >= duration_seconds
        )
        return {
            "overall_status": "COMPLETE" if complete else "INCOMPLETE",
            "requested_duration_seconds": duration_seconds,
            "elapsed_seconds": elapsed_seconds,
            "video_pipeline_started": started_video,
            "imu_pipeline_started": started_imu,
            "interrupted": interrupted,
            "fatal_error": fatal_error,
            "stop_errors": stop_errors,
            "received_counts": dict(self.sequences),
            "callback_counts": dict(self.callback_sequences),
            "queue_overflow_counts": dict(self.writer.overflow_counts),
            "writer_errors": self.writer.write_errors,
            "callback_errors": self.callback_errors,
        }
