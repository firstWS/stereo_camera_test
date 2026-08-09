"""Shared constants for Phase 2 dataset recording."""

from __future__ import annotations

RECORD_TOOL_NAME = "record_gemini335l"
RECORD_TOOL_VERSION = "0.1.0"
DATASET_SCHEMA_VERSION = 1
SCENARIO_SCHEMA_VERSION = 1

VIDEO_STREAMS = ("RGB", "DEPTH", "LEFT_IR", "RIGHT_IR")
IMU_STREAMS = ("ACCEL", "GYRO")

STREAM_DIR_NAMES = {
    "RGB": "rgb",
    "DEPTH": "depth",
    "LEFT_IR": "left_ir",
    "RIGHT_IR": "right_ir",
}

STREAM_INDEX_FIELDS = (
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
    "file_name",
)

IMU_CSV_FIELDS = (
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

EVENT_FIELDS = (
    "event_time_host_monotonic_ns",
    "event_time_device_us",
    "event_type",
    "message",
)

SUPPORTED_SCENARIOS = ("scenario_a", "scenario_b")
