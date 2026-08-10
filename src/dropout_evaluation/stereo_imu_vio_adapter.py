"""Phase 4.5-A/B VIO local trajectory to Phase 3 PoseEstimate adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rgbd_odometry_adapter import (
    RgbdAdapterStatus,
    RgbdAdapterWindowResult,
    RgbdOdometryAdapterConfig,
    generate_rgbd_odometry_candidates,
)
from .rgbd_odometry_continuous import TrajectorySample
from .stereo_imu_vio_lite import STEREO_IMU_VIO_LITE_ALGORITHM_ID, StereoImuTrajectorySample


@dataclass(frozen=True)
class StereoImuVioAdapterConfig:
    algorithm_id: str = STEREO_IMU_VIO_LITE_ALGORITHM_ID


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def load_vio_trajectory_from_csv(path: str | Path) -> list[StereoImuTrajectorySample]:
    samples: list[StereoImuTrajectorySample] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                StereoImuTrajectorySample(
                    frame_number=int(row["frame_number"]),
                    device_timestamp_us=int(row["device_timestamp_us"]),
                    valid=_parse_bool(row["valid"]),
                    state=str(row["state"]),
                    native_left_frame_number=int(row["native_left_frame_number"]),
                    native_right_frame_number=int(row["native_right_frame_number"]),
                    tx=_parse_optional_float(row.get("tx")),
                    ty=_parse_optional_float(row.get("ty")),
                    tz=_parse_optional_float(row.get("tz")),
                    qw=_parse_optional_float(row.get("qw")),
                    qx=_parse_optional_float(row.get("qx")),
                    qy=_parse_optional_float(row.get("qy")),
                    qz=_parse_optional_float(row.get("qz")),
                    visual_inliers=int(row.get("visual_inliers") or 0),
                    stereo_points=int(row.get("stereo_points") or 0),
                    imu_samples_used=int(row.get("imu_samples_used") or 0),
                    visual_update_success=_parse_bool(row.get("visual_update_success", "false")),
                    imu_propagated=_parse_bool(row.get("imu_propagated", "false")),
                    propagated_only=_parse_bool(row.get("propagated_only", "false")),
                    failure_reason=row.get("failure_reason") or None,
                )
            )
    samples.sort(key=lambda item: item.frame_number)
    return samples


def vio_sample_to_trajectory_sample(sample: StereoImuTrajectorySample) -> TrajectorySample:
    tracking_state = "LOCAL_TRACKING" if sample.valid else "TRACKING_LOST"
    return TrajectorySample(
        frame_number=sample.frame_number,
        device_timestamp_us=sample.device_timestamp_us,
        valid=sample.valid,
        tracking_state=tracking_state,
        segment_id=0,
        segment_start=sample.state == "init",
        continuity_from_previous_segment=True,
        tx=sample.tx,
        ty=sample.ty,
        tz=sample.tz,
        qw=sample.qw,
        qx=sample.qx,
        qy=sample.qy,
        qz=sample.qz,
        source_frame=None,
        pair_gap_frames=None,
        bridge_recovered=False,
    )


def vio_trajectory_to_local_trajectory(
    samples: list[StereoImuTrajectorySample],
) -> list[TrajectorySample]:
    return [vio_sample_to_trajectory_sample(sample) for sample in samples]


def generate_stereo_imu_vio_candidates(
    *,
    window,
    local_trajectory,
    runtime_poses,
    frame_timestamps,
    config: StereoImuVioAdapterConfig | None = None,
) -> RgbdAdapterWindowResult:
    """Generate Phase 3 PoseEstimate candidates by reusing RGB-D adapter semantics."""
    cfg = config or StereoImuVioAdapterConfig()
    rgbd_config = RgbdOdometryAdapterConfig(algorithm_id=cfg.algorithm_id)
    return generate_rgbd_odometry_candidates(
        window=window,
        local_trajectory=local_trajectory,
        runtime_poses=runtime_poses,
        frame_timestamps=frame_timestamps,
        config=rgbd_config,
    )
