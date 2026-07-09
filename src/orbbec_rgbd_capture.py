"""
Orbbec SDK v2 (Python ``pyorbbecsdk``) RGB + depth aligned to color.

Requires Orbbec host drivers + SDK and: ``pip install pyorbbecsdk2``
(import 모듈명은 여전히 ``pyorbbecsdk`` 입니다.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _import_pyorbbec():
    try:
        from pyorbbecsdk import (  # type: ignore
            AlignFilter,
            Config,
            Context,
            FormatConvertFilter,
            OBConvertFormat,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBSensorType,
            OBStreamType,
            Pipeline,
            VideoFrame,
        )

        return {
            "AlignFilter": AlignFilter,
            "Config": Config,
            "Context": Context,
            "FormatConvertFilter": FormatConvertFilter,
            "OBConvertFormat": OBConvertFormat,
            "OBFormat": OBFormat,
            "OBFrameAggregateOutputMode": OBFrameAggregateOutputMode,
            "OBSensorType": OBSensorType,
            "OBStreamType": OBStreamType,
            "Pipeline": Pipeline,
            "VideoFrame": VideoFrame,
        }
    except ImportError as e:
        raise SystemExit(
            "Orbbec mode needs the Python wrapper. Install with:\n"
            "  pip install pyorbbecsdk2\n"
            "and install Orbbec SDK / drivers per "
            "https://github.com/orbbec/pyorbbecsdk\n"
            f"Original error: {e}"
        ) from e


def frame_to_bgr_image(frame: Any, ob_format_mod: Any) -> np.ndarray | None:
    """Convert Orbbec color ``VideoFrame`` to BGR uint8 (OpenCV)."""
    import cv2

    OBFormat = ob_format_mod
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return data.reshape((height, width, 3))
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if color_format == OBFormat.UYVY:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)

    from pyorbbecsdk import FormatConvertFilter, OBConvertFormat  # type: ignore

    convert_filter = FormatConvertFilter()
    format_map = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
    }
    if color_format in format_map:
        convert_filter.set_format_convert_format(format_map[color_format])
        rgb_frame = convert_filter.process(frame)
        if rgb_frame:
            rgb_data = np.asanyarray(rgb_frame.get_data())
            rgb_image = rgb_data.reshape((height, width, 3))
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    print(f"[orbbec] Unsupported color format: {color_format}")
    return None


def _intrinsic_to_K(intr: Any) -> np.ndarray:
    return np.array(
        [[float(intr.fx), 0.0, float(intr.cx)], [0.0, float(intr.fy), float(intr.cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pick_video_profile(profile_list: Any, width: int | None, height: int | None, fmt: Any, fps: int | None) -> Any:
    if width is None or height is None or fps is None:
        return profile_list.get_default_video_stream_profile()
    return profile_list.get_video_stream_profile(int(width), int(height), fmt, int(fps))


@dataclass
class OrbbecFrame:
    bgr: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray


class OrbbecRGBDCapture:
    """Starts color+depth streams; optional software align depth → color."""

    def __init__(self, ob_cfg: dict[str, Any]) -> None:
        self._cfg = ob_cfg
        self._sdk = _import_pyorbbec()
        self._pipeline: Any = None
        self._align_filter: Any = None
        self._K: np.ndarray | None = None
        self._wait_ms = max(1, int(ob_cfg.get("wait_for_frames_ms", 100)))
        self._depth_additional = float(ob_cfg.get("depth_scale_additional", 1.0))
        self._depth_is_mm = bool(ob_cfg.get("depth_is_millimeters", True))

    def start(self) -> None:
        ob = self._cfg
        serial = str(ob.get("serial") or "").strip()
        dev_index = int(ob.get("device_index", 0))

        Context = self._sdk["Context"]
        Pipeline = self._sdk["Pipeline"]
        Config = self._sdk["Config"]
        OBSensorType = self._sdk["OBSensorType"]
        OBFormat = self._sdk["OBFormat"]
        OBFrameAggregateOutputMode = self._sdk["OBFrameAggregateOutputMode"]
        AlignFilter = self._sdk["AlignFilter"]
        OBStreamType = self._sdk["OBStreamType"]

        ctx = Context()
        device_list = ctx.query_devices()
        if device_list.get_count() < 1:
            raise SystemExit("Orbbec: no devices found (query_devices empty).")
        if serial:
            device = device_list.get_device_by_serial_number(serial)
        else:
            device = device_list.get_device_by_index(dev_index)

        pipeline = Pipeline(device)
        config = Config()

        cw = ob.get("color_width")
        ch = ob.get("color_height")
        cfps = ob.get("color_fps")
        color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = _pick_video_profile(color_list, cw, ch, OBFormat.RGB, cfps)
        except Exception:
            color_profile = color_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        dw = ob.get("depth_width")
        dh = ob.get("depth_height")
        dfps = ob.get("depth_fps")
        depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = _pick_video_profile(depth_list, dw, dh, OBFormat.Y16, dfps)
        except Exception:
            depth_profile = depth_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        try:
            pipeline.start(config)
        except Exception as e:
            msg = str(e)
            if "already been started" in msg.lower():
                raise SystemExit(
                    "Orbbec: camera is already in use (another python.exe or Orbbec Viewer?).\n"
                    "Close the other app or end the previous run.ps1 -Orbbec with Ctrl+C, then retry.\n"
                    f"SDK error: {msg}"
                ) from e
            raise

        intr = color_profile.get_intrinsic()
        self._K = _intrinsic_to_K(intr)
        self._pipeline = pipeline

        if bool(ob.get("align_depth_to_color", True)):
            self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        else:
            self._align_filter = None

    def release(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align_filter = None

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("OrbbecRGBDCapture.start() first")
        return self._K

    def read_rgbd(self) -> tuple[bool, OrbbecFrame | None]:
        """Returns ``(ok, OrbbecFrame)`` with ``depth_m`` in meters (float32)."""
        if self._pipeline is None:
            raise RuntimeError("OrbbecRGBDCapture.start() first")

        frames = self._pipeline.wait_for_frames(self._wait_ms)
        if frames is None:
            return False, None

        if self._align_filter is not None:
            frames = self._align_filter.process(frames)
            if not frames:
                return False, None
        fs = frames.as_frame_set() if hasattr(frames, "as_frame_set") else frames

        color_frame = fs.get_color_frame()
        depth_frame = fs.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return False, None

        bgr = frame_to_bgr_image(color_frame, self._sdk["OBFormat"])
        if bgr is None:
            return False, None

        h, w = depth_frame.get_height(), depth_frame.get_width()
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
        scale = float(depth_frame.get_depth_scale())
        depth = raw.astype(np.float32) * scale
        depth *= float(self._depth_additional)
        if self._depth_is_mm:
            depth /= 1000.0

        dh, dw = depth.shape[:2]
        bh, bw = bgr.shape[:2]
        if (dh, dw) != (bh, bw):
            import cv2

            depth = cv2.resize(depth, (bw, bh), interpolation=cv2.INTER_NEAREST)

        if self._K is None:
            return False, None
        return True, OrbbecFrame(bgr=bgr, depth_m=depth, K=self._K.copy())


def placeholder_stereo_frames(rgb_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right placeholder for APIs that still expect StereoFrame RGB|empty."""
    return rgb_bgr, np.zeros_like(rgb_bgr)
