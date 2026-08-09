"""Software AprilTag dropout protocol: windows, masks, and manifest generation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from dataset_recorder.session_metadata import write_json

PROTOCOL_SCHEMA_VERSION = 1
MASK_INTERVAL_HALF_OPEN = "half_open"
DEFAULT_DURATIONS_SEC: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class FrameTimestamp:
    frame_number: int
    device_timestamp_us: int


@dataclass(frozen=True)
class Cup2FirstAppearance:
    frame_number: int
    device_timestamp_us: int


@dataclass(frozen=True)
class DropoutAnchorDefinition:
    anchor_id: str
    start_frame: int
    start_device_timestamp_us: int
    convention: str
    motion_class: str


@dataclass(frozen=True)
class SuccessThresholds:
    pose_availability_min: float = 0.90
    major_tracking_lost_max: int = 0
    cup2_world_median_max_m: float = 0.10
    cup2_world_p90_max_m: float = 0.20
    dropout_normal_max_sec: float = 3.0
    dropout_stress_sec: float = 5.0


@dataclass(frozen=True)
class DropoutProtocolConfig:
    schema_version: int
    session_id: str
    session_path: str
    anchors: tuple[DropoutAnchorDefinition, ...]
    durations_sec: tuple[float, ...]
    mask_interval: str
    reference_source: str
    reference_role: str
    cup2_semantic_id: str
    cup2_observations_csv: str
    success_thresholds: SuccessThresholds
    output_root: str


@dataclass(frozen=True)
class DropoutWindow:
    window_id: str
    anchor_id: str
    session_id: str
    start_frame: int
    start_device_timestamp_us: int
    boundary_timestamp_us: int
    end_frame: int
    end_device_timestamp_us: int
    recovery_frame: int | None
    recovery_device_timestamp_us: int | None
    target_duration_sec: float
    masked_sample_span_sec: float
    frame_count: int
    motion_class: str
    includes_cup2_appearance: bool
    anchor_convention: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "anchor_id": self.anchor_id,
            "session_id": self.session_id,
            "start_frame": self.start_frame,
            "start_device_timestamp_us": self.start_device_timestamp_us,
            "boundary_timestamp_us": self.boundary_timestamp_us,
            "end_frame": self.end_frame,
            "end_device_timestamp_us": self.end_device_timestamp_us,
            "recovery_frame": self.recovery_frame,
            "recovery_device_timestamp_us": self.recovery_device_timestamp_us,
            "target_duration_sec": self.target_duration_sec,
            "masked_sample_span_sec": self.masked_sample_span_sec,
            "frame_count": self.frame_count,
            "motion_class": self.motion_class,
            "includes_cup2_appearance": self.includes_cup2_appearance,
            "anchor_convention": self.anchor_convention,
            "mask_semantics": {
                "interval": MASK_INTERVAL_HALF_OPEN,
                "rule": "start_timestamp <= device_timestamp_us < boundary_timestamp_us",
            },
        }


def format_duration_for_window_id(duration_sec: float) -> str:
    """Deterministic duration label for window IDs across environments."""
    rounded = round(float(duration_sec), 1)
    if math.isclose(rounded, round(rounded)):
        return f"{rounded:.1f}s"
    return f"{rounded:.1f}s"


def _duration_to_boundary_us(start_timestamp_us: int, duration_sec: float) -> int:
    return start_timestamp_us + int(round(float(duration_sec) * 1_000_000))


def _parse_bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def success_thresholds_from_mapping(raw: Mapping[str, Any] | None) -> SuccessThresholds:
    cfg = dict(raw or {})
    defaults = SuccessThresholds()
    return SuccessThresholds(
        pose_availability_min=float(cfg.get("pose_availability_min", defaults.pose_availability_min)),
        major_tracking_lost_max=int(cfg.get("major_tracking_lost_max", defaults.major_tracking_lost_max)),
        cup2_world_median_max_m=float(cfg.get("cup2_world_median_max_m", defaults.cup2_world_median_max_m)),
        cup2_world_p90_max_m=float(cfg.get("cup2_world_p90_max_m", defaults.cup2_world_p90_max_m)),
        dropout_normal_max_sec=float(cfg.get("dropout_normal_max_sec", defaults.dropout_normal_max_sec)),
        dropout_stress_sec=float(cfg.get("dropout_stress_sec", defaults.dropout_stress_sec)),
    )


def dropout_protocol_config_from_mapping(payload: Mapping[str, Any]) -> DropoutProtocolConfig:
    session = dict(payload.get("session") or {})
    mask_semantics = dict(payload.get("mask_semantics") or {})
    reference = dict(payload.get("reference") or {})
    cup2 = dict(payload.get("cup2") or {})
    output = dict(payload.get("output") or {})

    anchors_raw = payload.get("anchors")
    if not isinstance(anchors_raw, Mapping):
        raise ValueError("dropout config requires anchors mapping")

    anchors: list[DropoutAnchorDefinition] = []
    for anchor_id, raw in anchors_raw.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"anchor {anchor_id} must be a mapping")
        anchors.append(
            DropoutAnchorDefinition(
                anchor_id=str(anchor_id),
                start_frame=int(raw["start_frame"]),
                start_device_timestamp_us=int(raw["start_device_timestamp_us"]),
                convention=str(raw.get("convention", "")),
                motion_class=str(raw.get("motion_class", anchor_id)),
            )
        )

    durations_raw = payload.get("durations_sec", list(DEFAULT_DURATIONS_SEC))
    if not isinstance(durations_raw, Sequence) or isinstance(durations_raw, (str, bytes)):
        raise ValueError("durations_sec must be a sequence")
    durations_sec = tuple(float(value) for value in durations_raw)

    return DropoutProtocolConfig(
        schema_version=int(payload.get("schema_version", PROTOCOL_SCHEMA_VERSION)),
        session_id=str(session.get("session_id", "")),
        session_path=str(session.get("path", "")),
        anchors=tuple(anchors),
        durations_sec=durations_sec,
        mask_interval=str(mask_semantics.get("interval", MASK_INTERVAL_HALF_OPEN)),
        reference_source=str(reference.get("source", "derived/reference/apriltag_pose_smoothed.csv")),
        reference_role=str(reference.get("role", "evaluation_only")),
        cup2_semantic_id=str(cup2.get("semantic_id", "cup2")),
        cup2_observations_csv=str(cup2.get("observations_csv", "derived/cups/observations.csv")),
        success_thresholds=success_thresholds_from_mapping(payload.get("success_thresholds")),
        output_root=str(output.get("root", "out/evaluation/phase3")),
    )


def load_dropout_protocol_config(path: Path) -> DropoutProtocolConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return dropout_protocol_config_from_mapping(payload)


def load_frame_timestamps_from_rgb_index(session_dir: Path) -> list[FrameTimestamp]:
    index_path = session_dir / "streams" / "rgb" / "index.csv"
    frames: list[FrameTimestamp] = []
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frames.append(
                FrameTimestamp(
                    frame_number=int(row["frame_number"]),
                    device_timestamp_us=int(row["device_timestamp_us"]),
                )
            )
    frames.sort(key=lambda item: item.device_timestamp_us)
    return frames


def load_frame_timestamps_from_reference_csv(reference_csv: Path) -> list[FrameTimestamp]:
    frames: list[FrameTimestamp] = []
    with reference_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if "reference_valid" in row and not _parse_bool(row["reference_valid"]):
                continue
            ts = row.get("device_timestamp_us")
            if ts in (None, ""):
                continue
            frames.append(
                FrameTimestamp(
                    frame_number=int(row["frame_number"]),
                    device_timestamp_us=int(ts),
                )
            )
    frames.sort(key=lambda item: item.device_timestamp_us)
    return frames


def resolve_cup2_first_appearance(
    observations_csv: Path,
    semantic_id: str = "cup2",
) -> Cup2FirstAppearance | None:
    """Resolve earliest valid Cup2 observation from Phase 2 derived observations."""
    earliest: Cup2FirstAppearance | None = None
    with observations_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("semantic_id") != semantic_id:
                continue
            if not _parse_bool(row.get("depth_valid", "false")):
                continue
            ts = row.get("device_timestamp_us")
            if ts in (None, ""):
                continue
            frame_number = int(row["frame_number"])
            device_timestamp_us = int(ts)
            if earliest is None or device_timestamp_us < earliest.device_timestamp_us:
                earliest = Cup2FirstAppearance(
                    frame_number=frame_number,
                    device_timestamp_us=device_timestamp_us,
                )
    return earliest


def resolve_cup2_first_appearance_for_config(
    config: DropoutProtocolConfig,
    session_dir: Path,
) -> Cup2FirstAppearance | None:
    return resolve_cup2_first_appearance(
        session_dir / config.cup2_observations_csv,
        config.cup2_semantic_id,
    )


def window_includes_cup2_first_appearance(
    *,
    start_timestamp_us: int,
    boundary_timestamp_us: int,
    event_timestamp_us: int,
) -> bool:
    """True when Cup2 first-appearance event falls inside the masked half-open window."""
    return start_timestamp_us <= event_timestamp_us < boundary_timestamp_us


def compute_dropout_window(
    *,
    anchor: DropoutAnchorDefinition,
    duration_sec: float,
    session_id: str,
    frames: Sequence[FrameTimestamp],
    cup2_first_appearance: Cup2FirstAppearance | None = None,
) -> DropoutWindow:
    if not frames:
        raise ValueError("frame timestamps are required to compute dropout windows")

    start_ts = anchor.start_device_timestamp_us
    boundary_ts = _duration_to_boundary_us(start_ts, duration_sec)

    masked = [frame for frame in frames if start_ts <= frame.device_timestamp_us < boundary_ts]
    if not masked:
        raise ValueError(
            f"no masked frames for anchor={anchor.anchor_id} duration={duration_sec}s "
            f"start_ts={start_ts} boundary_ts={boundary_ts}"
        )

    end_frame = masked[-1].frame_number
    end_ts = masked[-1].device_timestamp_us

    recovery_candidates = [frame for frame in frames if frame.device_timestamp_us >= boundary_ts]
    recovery_frame = recovery_candidates[0].frame_number if recovery_candidates else None
    recovery_ts = recovery_candidates[0].device_timestamp_us if recovery_candidates else None

    includes_cup2_appearance = False
    if cup2_first_appearance is not None:
        includes_cup2_appearance = window_includes_cup2_first_appearance(
            start_timestamp_us=start_ts,
            boundary_timestamp_us=boundary_ts,
            event_timestamp_us=cup2_first_appearance.device_timestamp_us,
        )

    window_id = f"{anchor.anchor_id}__{format_duration_for_window_id(duration_sec)}"
    return DropoutWindow(
        window_id=window_id,
        anchor_id=anchor.anchor_id,
        session_id=session_id,
        start_frame=anchor.start_frame,
        start_device_timestamp_us=start_ts,
        boundary_timestamp_us=boundary_ts,
        end_frame=end_frame,
        end_device_timestamp_us=end_ts,
        recovery_frame=recovery_frame,
        recovery_device_timestamp_us=recovery_ts,
        target_duration_sec=float(duration_sec),
        masked_sample_span_sec=(end_ts - start_ts) / 1_000_000.0,
        frame_count=len(masked),
        motion_class=anchor.motion_class,
        includes_cup2_appearance=includes_cup2_appearance,
        anchor_convention=anchor.convention,
    )


def generate_dropout_windows(
    config: DropoutProtocolConfig,
    frames: Sequence[FrameTimestamp],
    *,
    session_dir: Path | None = None,
    cup2_first_appearance: Cup2FirstAppearance | None = None,
) -> list[DropoutWindow]:
    resolved_cup2 = cup2_first_appearance
    if resolved_cup2 is None and session_dir is not None:
        resolved_cup2 = resolve_cup2_first_appearance_for_config(config, session_dir)

    windows: list[DropoutWindow] = []
    for anchor in config.anchors:
        for duration_sec in config.durations_sec:
            windows.append(
                compute_dropout_window(
                    anchor=anchor,
                    duration_sec=duration_sec,
                    session_id=config.session_id,
                    frames=frames,
                    cup2_first_appearance=resolved_cup2,
                )
            )
    return windows


def is_runtime_tag_masked(device_timestamp_us: int, window: DropoutWindow) -> bool:
    """Return True when runtime AprilTag observation must be masked."""
    return window.start_device_timestamp_us <= device_timestamp_us < window.boundary_timestamp_us


def is_runtime_tag_available(device_timestamp_us: int, window: DropoutWindow) -> bool:
    """Runtime tag availability is independent of reference validity."""
    return not is_runtime_tag_masked(device_timestamp_us, window)


def build_dropout_manifest_payload(
    config: DropoutProtocolConfig,
    windows: Sequence[DropoutWindow],
    *,
    cup2_first_appearance: Cup2FirstAppearance | None = None,
) -> dict[str, Any]:
    cup2_payload: dict[str, Any] = {
        "semantic_id": config.cup2_semantic_id,
        "observations_csv": config.cup2_observations_csv,
    }
    if cup2_first_appearance is not None:
        cup2_payload["first_appearance_frame"] = cup2_first_appearance.frame_number
        cup2_payload["first_appearance_device_timestamp_us"] = (
            cup2_first_appearance.device_timestamp_us
        )

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "session_id": config.session_id,
        "session_path": config.session_path,
        "mask_semantics": {
            "interval": config.mask_interval,
            "rule": "start_timestamp <= device_timestamp_us < boundary_timestamp_us",
            "runtime_dropout_definition": (
                "Runtime/Candidate path cannot consume AprilTag pose observations "
                "while masked. Reference path remains evaluation-only."
            ),
        },
        "reference": {
            "source": config.reference_source,
            "role": config.reference_role,
        },
        "cup2": cup2_payload,
        "success_thresholds": {
            "pose_availability_min": config.success_thresholds.pose_availability_min,
            "major_tracking_lost_max": config.success_thresholds.major_tracking_lost_max,
            "cup2_world_median_max_m": config.success_thresholds.cup2_world_median_max_m,
            "cup2_world_p90_max_m": config.success_thresholds.cup2_world_p90_max_m,
            "dropout_normal_max_sec": config.success_thresholds.dropout_normal_max_sec,
            "dropout_stress_sec": config.success_thresholds.dropout_stress_sec,
        },
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "start_frame": anchor.start_frame,
                "start_device_timestamp_us": anchor.start_device_timestamp_us,
                "convention": anchor.convention,
                "motion_class": anchor.motion_class,
            }
            for anchor in config.anchors
        ],
        "durations_sec": list(config.durations_sec),
        "windows": [window.as_dict() for window in windows],
    }


def write_dropout_manifest(
    *,
    config: DropoutProtocolConfig,
    windows: Sequence[DropoutWindow],
    output_path: Path,
    cup2_first_appearance: Cup2FirstAppearance | None = None,
) -> Path:
    payload = build_dropout_manifest_payload(
        config,
        windows,
        cup2_first_appearance=cup2_first_appearance,
    )
    write_json(output_path, payload)
    return output_path
