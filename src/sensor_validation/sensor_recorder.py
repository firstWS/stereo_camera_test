"""Bounded-queue video/IMU recording with explicit partial-session handling."""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .profile_selector import build_video_probe_matrix
from .sdk_adapter import OrbbecSdkAdapter, ProfileHandle, enum_name, safe_call

VIDEO_FIELDS = (
    "stream",
    "received_sequence",
    "callback_sequence",
    "frame_number",
    "device_timestamp_us",
    "system_timestamp_us",
    "global_timestamp_us",
    "host_monotonic_ns",
    "host_wall_time_ns",
    "width",
    "height",
    "format",
    "data_size_bytes",
    "depth_scale",
    "metadata_json",
)
IMU_FIELDS = (
    "stream",
    "received_sequence",
    "callback_sequence",
    "frame_number",
    "device_timestamp_us",
    "system_timestamp_us",
    "global_timestamp_us",
    "host_monotonic_ns",
    "host_wall_time_ns",
    "sample_rate",
    "full_scale_range",
    "temperature",
    "x",
    "y",
    "z",
    "metadata_json",
)


@dataclass
class FrameSnapshot:
    stream: str
    sequence: int
    width: int
    height: int
    format_name: str
    data: bytes
    depth_scale: float | None


class CsvWriterThread:
    """One writer owns both handles, so callbacks never perform disk I/O."""

    def __init__(self, output_dir: Path, *, queue_size: int) -> None:
        self.output_dir = output_dir
        self.queue: queue.Queue[tuple[str, Any] | None] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self.overflow_counts: dict[str, int] = defaultdict(int)
        self.write_errors: list[str] = []
        self.snapshot_results: list[dict[str, Any]] = []
        self._video_handle: Any = None
        self._imu_handle: Any = None
        self._thread = threading.Thread(target=self._run, name="sensor-csv-writer")

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._video_handle = (self.output_dir / "video_frames.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._imu_handle = (self.output_dir / "imu_samples.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._video_writer = csv.DictWriter(
            self._video_handle, fieldnames=VIDEO_FIELDS, extrasaction="ignore"
        )
        self._imu_writer = csv.DictWriter(
            self._imu_handle, fieldnames=IMU_FIELDS, extrasaction="ignore"
        )
        self._video_writer.writeheader()
        self._imu_writer.writeheader()
        self._video_handle.flush()
        self._imu_handle.flush()
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
                    self._video_writer.writerow(payload)
                elif kind == "imu":
                    self._imu_writer.writerow(payload)
                elif kind == "snapshot":
                    snapshot, label = payload
                    self.snapshot_results.append(
                        save_snapshot(snapshot, self.output_dir / "frames", label)
                    )
            except Exception as error:
                self.write_errors.append(f"{type(error).__name__}: {error}")
            finally:
                self.queue.task_done()

    def close(self) -> None:
        if self._thread.is_alive():
            self.queue.put(None)
            self._thread.join(timeout=10.0)
        for handle in (self._video_handle, self._imu_handle):
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except Exception as error:
                    self.write_errors.append(f"{type(error).__name__}: {error}")


def _safe_method(frame: Any, name: str) -> Any:
    method = getattr(frame, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _metadata_values(frame: Any, sdk: Any) -> dict[str, Any]:
    enum_type = getattr(sdk, "OBFrameMetadataType", None)
    if enum_type is None or not callable(getattr(frame, "has_metadata", None)):
        return {}
    members = getattr(enum_type, "__members__", {})
    result: dict[str, Any] = {}
    for name, value in members.items():
        try:
            if frame.has_metadata(value):
                result[name] = frame.get_metadata_value(value)
        except Exception:
            continue
    return result


def _frame_common(
    frame: Any,
    *,
    stream: str,
    sequence: int,
    callback_sequence: int,
    host_monotonic_ns: int,
    host_wall_time_ns: int,
    sdk: Any,
) -> dict[str, Any]:
    metadata = _metadata_values(frame, sdk)
    return {
        "stream": stream,
        "received_sequence": sequence,
        "callback_sequence": callback_sequence,
        "frame_number": _safe_method(frame, "get_index"),
        "device_timestamp_us": _safe_method(frame, "get_timestamp_us"),
        "system_timestamp_us": _safe_method(frame, "get_system_timestamp_us"),
        "global_timestamp_us": _safe_method(frame, "get_global_timestamp_us"),
        "host_monotonic_ns": host_monotonic_ns,
        "host_wall_time_ns": host_wall_time_ns,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    }


def _imu_xyz(frame: Any) -> tuple[Any, Any, Any]:
    direct = tuple(_safe_method(frame, f"get_{axis}") for axis in ("x", "y", "z"))
    if all(value is not None for value in direct):
        return direct
    value = _safe_method(frame, "get_value")
    if value is None:
        return direct
    return tuple(getattr(value, axis, None) for axis in ("x", "y", "z"))


def _snapshot(frame: Any, stream: str, sequence: int) -> FrameSnapshot | None:
    width = _safe_method(frame, "get_width")
    height = _safe_method(frame, "get_height")
    data = _safe_method(frame, "get_data")
    if width is None or height is None or data is None:
        return None
    try:
        copied = bytes(data)
    except Exception:
        try:
            copied = np.asarray(data).tobytes()
        except Exception:
            return None
    return FrameSnapshot(
        stream=stream,
        sequence=sequence,
        width=int(width),
        height=int(height),
        format_name=enum_name(_safe_method(frame, "get_format")) or "UNKNOWN",
        data=copied,
        depth_scale=(
            float(scale)
            if (scale := _safe_method(frame, "get_depth_scale")) is not None
            else None
        ),
    )


def _decode_snapshot(snapshot: FrameSnapshot) -> np.ndarray | None:
    import cv2

    format_name = snapshot.format_name.upper()
    image: np.ndarray | None = None
    if "MJPG" in format_name or "MJPEG" in format_name:
        image = cv2.imdecode(np.frombuffer(snapshot.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    elif snapshot.stream == "DEPTH" or any(
        token in format_name for token in ("Y16", "Z16", "Y10", "Y12")
    ):
        expected = snapshot.width * snapshot.height
        values = np.frombuffer(snapshot.data, dtype=np.uint16)
        if values.size >= expected:
            image = values[:expected].reshape(snapshot.height, snapshot.width)
    elif any(token in format_name for token in ("RGB", "BGR")):
        expected = snapshot.width * snapshot.height * 3
        values = np.frombuffer(snapshot.data, dtype=np.uint8)
        if values.size >= expected:
            image = values[:expected].reshape(snapshot.height, snapshot.width, 3)
            if "RGB" in format_name and "BGR" not in format_name:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif "Y8" in format_name:
        expected = snapshot.width * snapshot.height
        values = np.frombuffer(snapshot.data, dtype=np.uint8)
        if values.size >= expected:
            image = values[:expected].reshape(snapshot.height, snapshot.width)
    return image


def save_snapshot(snapshot: FrameSnapshot, output_dir: Path, label: str) -> dict[str, Any]:
    """Save lossless PNG when the format is known, otherwise raw bytes plus JSON."""

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{snapshot.stream.lower()}_{label}_{snapshot.sequence:06d}"
    image = _decode_snapshot(snapshot)

    sidecar = {
        "stream": snapshot.stream,
        "received_sequence": snapshot.sequence,
        "width": snapshot.width,
        "height": snapshot.height,
        "format": snapshot.format_name,
        "depth_scale": snapshot.depth_scale,
        "depth_storage": (
            "raw_uint16_png_no_resize" if snapshot.stream == "DEPTH" and image is not None else None
        ),
    }
    if image is not None:
        path = output_dir / f"{stem}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"cv2.imwrite failed: {path}")
    else:
        path = output_dir / f"{stem}.bin"
        path.write_bytes(snapshot.data)
        sidecar["fallback_reason"] = "format_not_losslessly_decoded"
    sidecar_path = output_dir / f"{stem}.json"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"data_path": str(path), "sidecar_path": str(sidecar_path)}


class SensorRecorder:
    def __init__(
        self,
        adapter: OrbbecSdkAdapter,
        *,
        device: Any,
        selected_profiles: Mapping[str, ProfileHandle],
        output_dir: Path,
        queue_size: int = 8192,
        sample_fps: float = 1.0,
        save_all_frames: bool = False,
        no_preview: bool = True,
    ) -> None:
        self.adapter = adapter
        self.device = device
        self.selected_profiles = dict(selected_profiles)
        self.output_dir = output_dir
        self.writer = CsvWriterThread(output_dir, queue_size=queue_size)
        self.sample_fps = max(0.0, float(sample_fps))
        self.save_all_frames = save_all_frames
        self.no_preview = no_preview
        self.sequences: dict[str, int] = defaultdict(int)
        self.callback_sequences: dict[str, int] = defaultdict(int)
        self.last_sample_ns: dict[str, int] = defaultdict(int)
        self.first_snapshots: dict[str, FrameSnapshot] = {}
        self.last_snapshots: dict[str, FrameSnapshot] = {}
        self.snapshot_results: list[dict[str, Any]] = []
        self.snapshot_errors: list[str] = []
        self.callback_errors: list[str] = []
        self._lock = threading.Lock()
        self._deadline_ns: int | None = None

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
                self.sequences[stream] += 1
                sequence = self.sequences[stream]
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
                self.writer.submit("video", row)
                self._consider_snapshot(frame, stream, sequence, host_monotonic_ns)
            except Exception as error:
                self.callback_errors.append(
                    f"video:{stream}:{type(error).__name__}: {error}"
                )

    def _consider_snapshot(
        self,
        frame: Any,
        stream: str,
        sequence: int,
        host_monotonic_ns: int,
    ) -> None:
        interval_ns = int(1_000_000_000 / self.sample_fps) if self.sample_fps > 0 else None
        first = stream not in self.first_snapshots
        sampled = (
            interval_ns is not None
            and host_monotonic_ns - self.last_sample_ns[stream] >= interval_ns
        )
        near_deadline = (
            self._deadline_ns is not None
            and 0 <= self._deadline_ns - host_monotonic_ns <= 200_000_000
        )
        if not (first or sampled or near_deadline or self.save_all_frames):
            return
        snapshot = _snapshot(frame, stream, sequence)
        if snapshot is None:
            return
        with self._lock:
            if first:
                self.first_snapshots[stream] = snapshot
            if near_deadline or sampled:
                self.last_snapshots[stream] = snapshot
            if sampled:
                self.last_sample_ns[stream] = host_monotonic_ns
        if self.save_all_frames or (sampled and not first and not near_deadline):
            label = "all" if self.save_all_frames else "sample"
            try:
                if not self.writer.submit("snapshot", (snapshot, label)):
                    self.snapshot_errors.append(f"{stream}:{sequence}:snapshot_queue_full")
            except Exception as error:
                self.snapshot_errors.append(f"{stream}:{sequence}:{type(error).__name__}: {error}")

    def _imu_callback(self, frames: Any) -> None:
        host_monotonic_ns = time.perf_counter_ns()
        host_wall_time_ns = time.time_ns()
        self.callback_sequences["imu"] += 1
        callback_sequence = self.callback_sequences["imu"]
        for stream, getter in (
            ("ACCEL", "get_accel_frame"),
            ("GYRO", "get_gyro_frame"),
        ):
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
                self.writer.submit("imu", row)
            except Exception as error:
                self.callback_errors.append(
                    f"imu:{stream}:{type(error).__name__}: {error}"
                )

    def _flush_representative_snapshots(self) -> None:
        for stream, snapshot in self.first_snapshots.items():
            try:
                self.snapshot_results.append(
                    save_snapshot(snapshot, self.output_dir / "frames", "first")
                )
            except Exception as error:
                self.snapshot_errors.append(
                    f"{stream}:first:{type(error).__name__}: {error}"
                )
        for stream, snapshot in self.last_snapshots.items():
            if self.first_snapshots.get(stream) is snapshot:
                continue
            try:
                self.snapshot_results.append(
                    save_snapshot(snapshot, self.output_dir / "frames", "last")
                )
            except Exception as error:
                self.snapshot_errors.append(
                    f"{stream}:last:{type(error).__name__}: {error}"
                )

    def _show_preview(self) -> bool:
        import cv2

        with self._lock:
            snapshots = dict(self.last_snapshots or self.first_snapshots)
        preferred = snapshots.get("RGB") or snapshots.get("DEPTH")
        if preferred is not None:
            image = _decode_snapshot(preferred)
            if image is not None:
                if preferred.stream == "DEPTH" and image.ndim == 2:
                    display = cv2.normalize(
                        image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                    image = cv2.applyColorMap(display, cv2.COLORMAP_JET)
                cv2.imshow("Gemini 335L Phase 1 Validation", image)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)

    def record(
        self,
        *,
        duration_seconds: float,
        mode: str,
        progress: Callable[[str], None] = print,
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
        self._deadline_ns = start_perf_ns + int(duration_seconds * 1_000_000_000)
        self.writer.start()
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
            progress(_mode_instruction(mode, "start"))
            if mode in {"translation", "translation-yaw"}:
                progress(
                    "청각 안내: 짧은 비프=정지 카운트다운, 긴 비프 3회=지금 이동, "
                    "중간 비프 2회=이동 종료 후 정지"
                )
                _play_motion_beep("ready")
            last_second = None
            last_motion_message: str | None = None
            while True:
                remaining = max(
                    0.0, (self._deadline_ns - time.perf_counter_ns()) / 1_000_000_000
                )
                if remaining <= 0.0:
                    break
                elapsed = max(0.0, duration_seconds - remaining)
                if mode in {"translation", "translation-yaw"}:
                    if elapsed < 3.0:
                        countdown = int(max(0.0, 2.999 - elapsed)) + 1
                        motion_message = f"초기 정지 유지: {countdown}초"
                        motion_cue = "countdown"
                    elif remaining <= 3.0:
                        motion_message = (
                            "!!!! 이동 종료 — 지금 위치에서 완전히 정지하세요 !!!!"
                        )
                        motion_cue = "stop"
                    else:
                        motion_message = (
                            "!!!! 지금 오른쪽으로 약 0.5m 부드럽게 횡이동 시작 !!!!"
                            if mode == "translation"
                            else "!!!! 지금 병진 이동과 Yaw 회전을 시작하세요 !!!!"
                        )
                        motion_cue = "move"
                    if motion_message != last_motion_message:
                        progress("=" * 64)
                        progress(motion_message)
                        progress("=" * 64)
                        _play_motion_beep(motion_cue)
                        last_motion_message = motion_message
                integer_remaining = int(remaining)
                if integer_remaining != last_second and (
                    integer_remaining < 4 or integer_remaining % 10 == 0
                ):
                    progress(f"남은 시간: {integer_remaining + 1}초")
                    last_second = integer_remaining
                if not self.no_preview and self._show_preview():
                    interrupted = True
                    progress("Preview 종료 키를 감지했습니다. 부분 결과를 저장합니다.")
                    break
                time.sleep(min(0.05, remaining))
        except KeyboardInterrupt:
            interrupted = True
            progress("사용자 중단을 감지했습니다. 부분 결과를 안전하게 저장합니다.")
        except Exception as error:
            fatal_error = f"{type(error).__name__}: {error}"
        finally:
            if imu_pipeline is not None:
                if error := self.adapter.stop_pipeline(imu_pipeline):
                    stop_errors.append(f"imu:{error}")
            if video_pipeline is not None:
                if error := self.adapter.stop_pipeline(video_pipeline):
                    stop_errors.append(f"video:{error}")
            self.writer.close()
            self._flush_representative_snapshots()
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
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
            "mode": mode,
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
            "snapshot_errors": self.snapshot_errors,
            "saved_snapshots": self.writer.snapshot_results + self.snapshot_results,
        }


def _play_motion_beep(cue: str) -> None:
    """Windows audible cue for motion phases; never raises into the recorder loop."""

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
        if cue == "ready":
            winsound.Beep(880, 120)
        elif cue == "countdown":
            winsound.Beep(660, 90)
        elif cue == "move":
            for _ in range(3):
                winsound.Beep(1320, 220)
                time.sleep(0.05)
        elif cue == "stop":
            for _ in range(2):
                winsound.Beep(520, 180)
                time.sleep(0.05)
        else:
            winsound.Beep(880, 100)
    except Exception:
        _terminal_bell()


def _mode_instruction(mode: str, phase: str) -> str:
    if phase != "start":
        return ""
    if mode == "static":
        return "정적 기록 시작: 카메라를 움직이지 말고 고정 상태를 유지하세요."
    if mode == "translation":
        return (
            "이동 기록 시작: 3초 정지(짧은 비프) → 이동 시작(긴 비프 3회) → "
            "종료 3초 전부터 정지(중간 비프 2회)."
        )
    return (
        "이동+Yaw 기록 시작: 3초 정지(짧은 비프) → 이동+Yaw 시작(긴 비프 3회) → "
        "종료 3초 전부터 정지(중간 비프 2회)."
    )


def probe_profile_combinations(
    adapter: OrbbecSdkAdapter,
    *,
    device: Any,
    selected_profiles: Mapping[str, ProfileHandle],
    duration_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Bounded start/stop probes; no fallback is performed inside a probe."""

    attempts: list[dict[str, Any]] = []

    def run_attempt(
        name: str,
        video_names: list[str],
        imu_names: list[str],
    ) -> None:
        video_pipeline = None
        imu_pipeline = None
        video_started = False
        imu_started = False
        counts = {"video_callbacks": 0, "imu_callbacks": 0}
        errors: list[str] = []

        def video_callback(_frames: Any) -> None:
            counts["video_callbacks"] += 1

        def imu_callback(_frames: Any) -> None:
            counts["imu_callbacks"] += 1

        try:
            if video_names:
                video_pipeline = adapter.create_pipeline(device)
                config = adapter.create_config()
                adapter.enable_profiles(
                    config, [selected_profiles[item] for item in video_names]
                )
                video_pipeline.start(config, video_callback)
                video_started = True
            if imu_names:
                imu_pipeline = adapter.create_pipeline(device)
                config = adapter.create_config()
                adapter.enable_profiles(
                    config, [selected_profiles[item] for item in imu_names]
                )
                imu_pipeline.start(config, imu_callback)
                imu_started = True
            time.sleep(max(0.0, duration_seconds))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            if imu_pipeline is not None:
                if error := adapter.stop_pipeline(imu_pipeline):
                    errors.append(f"imu_stop:{error}")
            if video_pipeline is not None:
                if error := adapter.stop_pipeline(video_pipeline):
                    errors.append(f"video_stop:{error}")
        attempts.append(
            {
                "name": name,
                "video_sensors": video_names,
                "imu_sensors": imu_names,
                "requested_profile_ids": [
                    selected_profiles[item].profile_id
                    for item in video_names + imu_names
                ],
                "video_started": video_started,
                "imu_started": imu_started,
                "callback_counts": counts,
                "status": "success" if not errors else "combination_failed",
                "errors": errors,
            }
        )

    video_matrix = build_video_probe_matrix(selected_profiles)
    for item in video_matrix:
        run_attempt(item["name"], list(item["sensors"]), [])
    imu_names = [
        item for item in ("ACCEL", "GYRO") if item in selected_profiles
    ]
    if imu_names:
        run_attempt("imu_only", [], imu_names)
    if video_matrix and imu_names:
        run_attempt(
            "max_video_plus_imu",
            list(video_matrix[-1]["sensors"]),
            imu_names,
        )
    return attempts
