"""Phase 4.1 dense Depth-to-RGB alignment for RGB-D odometry prerequisites."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_recorder.rgb_depth_geometry import (
    DepthPairingConfig,
    DepthPixelRgbProjection,
    RgbDepthCalibration,
    build_depth_timestamp_index,
    depth_meters_from_raw,
    depth_pairing_config_from_mapping,
    match_nearest_depth_timestamp,
    normalize_depth_image,
    project_depth_pixels_to_rgb,
)

ALIGNMENT_SCHEMA_VERSION = 1
DEPTH_UNIT_METERS = "meters"
INVALID_DEPTH_VALUE = 0.0
Z_BUFFER_POLICY = "nearest_positive_z_rgb"
HOLE_FILL_POLICY = "none"
EXTRINSIC_DIRECTION = "RGB_to_DEPTH_stored_inverse_for_depth_to_rgb"
TIMESTAMP_PAIRING_POLICY = "nearest_device_timestamp"
DEFAULT_ODOMETRY_Z_MIN_M = 0.05
DEFAULT_ODOMETRY_Z_MAX_M = 4.0
ALIGNED_DEPTH_DTYPE = np.float32
ALIGNED_DEPTH_FILE_SUFFIX = ".npy"


@dataclass(frozen=True)
class AlignmentDiagnostics:
    source_depth_pixels: int
    invalid_source_depth: int
    source_valid_depth: int
    behind_rgb_camera: int
    projected_out_of_bounds: int
    valid_projected: int
    projected_point_count: int
    unique_rgb_pixel_count: int
    collision_count: int
    collision_ratio: float
    z_buffer_collisions: int


@dataclass(frozen=True)
class CoverageMetrics:
    valid_pixel_count: int
    valid_pixel_ratio: float
    odometry_valid_pixel_count: int
    odometry_valid_pixel_ratio: float
    z_min_m: float | None = None
    z_max_m: float | None = None
    z_median_m: float | None = None
    z_p90_m: float | None = None
    z_max_observed_m: float | None = None


@dataclass(frozen=True)
class SpatialCoverageMetrics:
    center_valid_ratio: float
    left_valid_ratio: float
    right_valid_ratio: float
    top_valid_ratio: float
    bottom_valid_ratio: float
    grid_3x3_valid_ratio: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class AlignmentResult:
    aligned_depth: np.ndarray
    diagnostics: AlignmentDiagnostics
    coverage: CoverageMetrics
    spatial_coverage: SpatialCoverageMetrics


def _rgb_pixel_indices(
    u_rgb: np.ndarray,
    v_rgb: np.ndarray,
    *,
    rgb_width: int,
    rgb_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_i = np.floor(u_rgb).astype(np.int64)
    v_i = np.floor(v_rgb).astype(np.int64)
    in_bounds = (u_i >= 0) & (u_i < rgb_width) & (v_i >= 0) & (v_i < rgb_height)
    return u_i, v_i, in_bounds


def _z_buffer_projected_points(
    z_rgb: np.ndarray,
    u_i: np.ndarray,
    v_i: np.ndarray,
    *,
    rgb_width: int,
    rgb_height: int,
) -> tuple[np.ndarray, int, int]:
    """Scatter projected RGB-camera Z values onto the RGB pixel grid."""
    aligned = np.full((rgb_height, rgb_width), np.inf, dtype=np.float64)
    flat = aligned.ravel()
    flat_indices = (v_i * rgb_width + u_i).astype(np.int64)
    np.minimum.at(flat, flat_indices, z_rgb.astype(np.float64))
    valid_mask = np.isfinite(aligned) & (aligned < np.inf)
    aligned_depth = np.where(valid_mask, aligned, INVALID_DEPTH_VALUE).astype(ALIGNED_DEPTH_DTYPE)
    unique_pixel_count = int(np.unique(flat_indices).size)
    collision_count = int(z_rgb.size - unique_pixel_count)
    return aligned_depth, collision_count, unique_pixel_count


def align_depth_to_rgb(
    depth_image: np.ndarray,
    depth_intrinsics: np.ndarray,
    rgb_intrinsics: np.ndarray,
    t_rgb_to_depth: np.ndarray,
    depth_scale: float | None,
    rgb_width: int,
    rgb_height: int,
    *,
    depth_is_millimeters: bool = True,
    z_min_m: float = DEFAULT_ODOMETRY_Z_MIN_M,
    z_max_m: float = 40.0,
    odometry_z_min_m: float = DEFAULT_ODOMETRY_Z_MIN_M,
    odometry_z_max_m: float = DEFAULT_ODOMETRY_Z_MAX_M,
    depth_width: int | None = None,
    depth_height: int | None = None,
) -> AlignmentResult:
    """Project native depth pixels onto the RGB camera pixel grid."""
    calib = RgbDepthCalibration(
        k_rgb=np.asarray(rgb_intrinsics, dtype=np.float64),
        k_depth=np.asarray(depth_intrinsics, dtype=np.float64),
        t_rgb_to_depth=np.asarray(t_rgb_to_depth, dtype=np.float64),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        depth_width=int(depth_width or normalize_depth_image(depth_image).shape[1]),
        depth_height=int(depth_height or normalize_depth_image(depth_image).shape[0]),
    )
    depth_m = depth_meters_from_raw(
        depth_image,
        depth_scale=depth_scale,
        depth_is_millimeters=depth_is_millimeters,
    )
    return align_depth_meters_to_rgb(
        depth_m,
        calib,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        odometry_z_min_m=odometry_z_min_m,
        odometry_z_max_m=odometry_z_max_m,
    )


def align_depth_meters_to_rgb(
    depth_m: np.ndarray,
    calib: RgbDepthCalibration,
    *,
    z_min_m: float = DEFAULT_ODOMETRY_Z_MIN_M,
    z_max_m: float = 40.0,
    odometry_z_min_m: float = DEFAULT_ODOMETRY_Z_MIN_M,
    odometry_z_max_m: float = DEFAULT_ODOMETRY_Z_MAX_M,
) -> AlignmentResult:
    """Align metric depth (meters) from the native depth grid to the RGB grid."""
    depth_m = normalize_depth_image(depth_m)
    rgb_width = calib.rgb_width
    rgb_height = calib.rgb_height
    source_depth_pixels = int(depth_m.size)

    projection = project_depth_pixels_to_rgb(
        depth_m,
        calib,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
    )
    diagnostics = _diagnostics_from_projection(
        projection,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        source_depth_pixels=source_depth_pixels,
    )

    if diagnostics.valid_projected == 0:
        aligned_depth = np.zeros((rgb_height, rgb_width), dtype=ALIGNED_DEPTH_DTYPE)
    else:
        positive = projection.z_rgb > 0.0
        u_i, v_i, in_bounds = _rgb_pixel_indices(
            projection.u_rgb[positive],
            projection.v_rgb[positive],
            rgb_width=rgb_width,
            rgb_height=rgb_height,
        )
        z_sel = projection.z_rgb[positive][in_bounds]
        u_sel = u_i[in_bounds]
        v_sel = v_i[in_bounds]
        aligned_depth, _, _ = _z_buffer_projected_points(
            z_sel,
            u_sel,
            v_sel,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
        )

    coverage = compute_coverage_metrics(
        aligned_depth,
        odometry_z_min_m=odometry_z_min_m,
        odometry_z_max_m=odometry_z_max_m,
    )
    spatial_coverage = compute_spatial_coverage(aligned_depth)
    return AlignmentResult(
        aligned_depth=aligned_depth,
        diagnostics=diagnostics,
        coverage=coverage,
        spatial_coverage=spatial_coverage,
    )


def _diagnostics_from_projection(
    projection: DepthPixelRgbProjection,
    *,
    rgb_width: int,
    rgb_height: int,
    source_depth_pixels: int,
) -> AlignmentDiagnostics:
    projected_point_count = int(projection.z_rgb.size)
    if projected_point_count == 0:
        return AlignmentDiagnostics(
            source_depth_pixels=source_depth_pixels,
            invalid_source_depth=projection.invalid_source_count,
            source_valid_depth=projection.source_valid_count,
            behind_rgb_camera=0,
            projected_out_of_bounds=0,
            valid_projected=0,
            projected_point_count=0,
            unique_rgb_pixel_count=0,
            collision_count=0,
            collision_ratio=0.0,
            z_buffer_collisions=0,
        )

    positive = projection.z_rgb > 0.0
    behind_rgb_camera = int((~positive).sum())
    u_i, v_i, in_bounds = _rgb_pixel_indices(
        projection.u_rgb[positive],
        projection.v_rgb[positive],
        rgb_width=rgb_width,
        rgb_height=rgb_height,
    )
    projected_out_of_bounds = int((~in_bounds).sum())
    valid_projected = int(in_bounds.sum())
    if valid_projected == 0:
        unique_rgb_pixel_count = 0
        collision_count = 0
    else:
        flat_indices = (v_i[in_bounds] * rgb_width + u_i[in_bounds]).astype(np.int64)
        unique_rgb_pixel_count = int(np.unique(flat_indices).size)
        collision_count = valid_projected - unique_rgb_pixel_count
    collision_ratio = float(collision_count / projected_point_count) if projected_point_count else 0.0
    return AlignmentDiagnostics(
        source_depth_pixels=source_depth_pixels,
        invalid_source_depth=projection.invalid_source_count,
        source_valid_depth=projection.source_valid_count,
        behind_rgb_camera=behind_rgb_camera,
        projected_out_of_bounds=projected_out_of_bounds,
        valid_projected=valid_projected,
        projected_point_count=projected_point_count,
        unique_rgb_pixel_count=unique_rgb_pixel_count,
        collision_count=collision_count,
        collision_ratio=collision_ratio,
        z_buffer_collisions=collision_count,
    )


def compute_coverage_metrics(
    aligned_depth: np.ndarray,
    *,
    odometry_z_min_m: float = DEFAULT_ODOMETRY_Z_MIN_M,
    odometry_z_max_m: float = DEFAULT_ODOMETRY_Z_MAX_M,
) -> CoverageMetrics:
    depth = aligned_depth.astype(np.float64, copy=False)
    valid = depth > 0.0
    total = int(depth.size)
    valid_count = int(valid.sum())
    valid_ratio = float(valid_count / total) if total else 0.0
    odometry_valid = valid & (depth >= odometry_z_min_m) & (depth <= odometry_z_max_m)
    odometry_count = int(odometry_valid.sum())
    odometry_ratio = float(odometry_count / total) if total else 0.0
    z_values = depth[valid]
    if z_values.size == 0:
        return CoverageMetrics(
            valid_pixel_count=valid_count,
            valid_pixel_ratio=valid_ratio,
            odometry_valid_pixel_count=odometry_count,
            odometry_valid_pixel_ratio=odometry_ratio,
        )
    return CoverageMetrics(
        valid_pixel_count=valid_count,
        valid_pixel_ratio=valid_ratio,
        odometry_valid_pixel_count=odometry_count,
        odometry_valid_pixel_ratio=odometry_ratio,
        z_min_m=float(np.min(z_values)),
        z_median_m=float(np.median(z_values)),
        z_p90_m=float(np.quantile(z_values, 0.9)),
        z_max_observed_m=float(np.max(z_values)),
    )


def compute_spatial_coverage(aligned_depth: np.ndarray) -> SpatialCoverageMetrics:
    depth = aligned_depth
    height, width = depth.shape[:2]
    valid = depth > 0.0

    def _region_ratio(y0: int, y1: int, x0: int, x1: int) -> float:
        region = valid[y0:y1, x0:x1]
        if region.size == 0:
            return 0.0
        return float(region.sum() / region.size)

    cy0, cy1 = height // 3, 2 * height // 3
    cx0, cx1 = width // 3, 2 * width // 3
    center = _region_ratio(cy0, cy1, cx0, cx1)
    left = _region_ratio(0, height, 0, cx0)
    right = _region_ratio(0, height, cx1, width)
    top = _region_ratio(0, cy0, 0, width)
    bottom = _region_ratio(cy1, height, 0, width)

    grid: list[list[float]] = []
    for gy in range(3):
        y0 = gy * height // 3
        y1 = (gy + 1) * height // 3
        row: list[float] = []
        for gx in range(3):
            x0 = gx * width // 3
            x1 = (gx + 1) * width // 3
            row.append(_region_ratio(y0, y1, x0, x1))
        grid.append(row)
    return SpatialCoverageMetrics(
        center_valid_ratio=center,
        left_valid_ratio=left,
        right_valid_ratio=right,
        top_valid_ratio=top,
        bottom_valid_ratio=bottom,
        grid_3x3_valid_ratio=tuple(tuple(row) for row in grid),
    )


def build_alignment_manifest_provenance(
    *,
    session_id: str,
    source_rgb_resolution: tuple[int, int],
    source_depth_resolution: tuple[int, int],
    output_resolution: tuple[int, int],
    pairing_config: DepthPairingConfig | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pairing = pairing_config or DepthPairingConfig()
    manifest: dict[str, Any] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "session_id": session_id,
        "source_rgb_resolution": {
            "width": int(source_rgb_resolution[0]),
            "height": int(source_rgb_resolution[1]),
        },
        "source_depth_resolution": {
            "width": int(source_depth_resolution[0]),
            "height": int(source_depth_resolution[1]),
        },
        "output_resolution": {
            "width": int(output_resolution[0]),
            "height": int(output_resolution[1]),
        },
        "depth_unit": DEPTH_UNIT_METERS,
        "invalid_value": INVALID_DEPTH_VALUE,
        "aligned_depth_dtype": str(ALIGNED_DEPTH_DTYPE),
        "aligned_depth_file_suffix": ALIGNED_DEPTH_FILE_SUFFIX,
        "z_buffer_policy": Z_BUFFER_POLICY,
        "hole_fill_policy": HOLE_FILL_POLICY,
        "extrinsic_direction": EXTRINSIC_DIRECTION,
        "timestamp_pairing_policy": TIMESTAMP_PAIRING_POLICY,
        "max_pair_delta_us": pairing.max_rgb_depth_delta_us,
        "depth_is_millimeters": pairing.depth_is_millimeters,
        "source_z_min_m": pairing.z_min_m,
        "source_z_max_m": pairing.z_max_m,
        "odometry_z_min_m": DEFAULT_ODOMETRY_Z_MIN_M,
        "odometry_z_max_m": DEFAULT_ODOMETRY_Z_MAX_M,
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def save_aligned_depth_npy(path: Path, aligned_depth: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, aligned_depth.astype(ALIGNED_DEPTH_DTYPE, copy=False))


def write_alignment_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def depth_pairing_config_from_yaml_mapping(cfg: Mapping[str, Any] | None) -> DepthPairingConfig:
    return depth_pairing_config_from_mapping(cfg)


def match_rgb_depth_pair(
    rgb_timestamp_us: int,
    depth_rows: Sequence[tuple[dict[str, Any], Any]],
    *,
    pairing_config: DepthPairingConfig | None = None,
):
    pairing = pairing_config or DepthPairingConfig()
    timestamps, rows, paths = build_depth_timestamp_index(depth_rows)
    return match_nearest_depth_timestamp(
        rgb_timestamp_us,
        timestamps,
        rows,
        paths,
        max_delta_us=pairing.max_rgb_depth_delta_us,
    )


def alignment_result_to_report_dict(result: AlignmentResult) -> dict[str, Any]:
    return {
        "diagnostics": asdict(result.diagnostics),
        "coverage": asdict(result.coverage),
        "spatial_coverage": asdict(result.spatial_coverage),
    }
