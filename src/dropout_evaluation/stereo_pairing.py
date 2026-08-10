"""Stereo IR pairing policy for Phase 4.5-A Scenario A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_STEREO_PAIR_TOLERANCE_US = 1_000


@dataclass(frozen=True)
class StereoPairRecord:
    canonical_frame_number: int
    device_timestamp_us: int
    native_left_frame_number: int
    native_right_frame_number: int
    left_timestamp_us: int
    right_timestamp_us: int
    timestamp_skew_us: int
    pairing_method: str


def _record_ts(record: Mapping[str, Any]) -> int:
    return int(record.get("device_timestamp_us") or 0)


def _record_frame_number(record: Mapping[str, Any]) -> int:
    return int(record.get("frame_number") or 0)


def pair_stereo_records(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    canonical_frame_numbers: Sequence[int],
    *,
    tolerance_us: int = DEFAULT_STEREO_PAIR_TOLERANCE_US,
) -> tuple[list[StereoPairRecord], list[int]]:
    """Pair left/right IR by device_timestamp_us: exact first, then nearest within tolerance."""
    if len(left_records) != len(canonical_frame_numbers):
        raise ValueError(
            "left record count must match canonical frame count: "
            f"{len(left_records)} vs {len(canonical_frame_numbers)}"
        )

    right_entries = [
        (index, _record_ts(record), _record_frame_number(record), record)
        for index, record in enumerate(right_records)
    ]
    used_right: set[int] = set()
    pairs: list[StereoPairRecord] = []
    unpaired_left: list[int] = []

    for left_index, (left_record, canonical_frame_number) in enumerate(
        zip(left_records, canonical_frame_numbers)
    ):
        left_ts = _record_ts(left_record)
        native_left = _record_frame_number(left_record)

        exact_candidates = [
            entry for entry in right_entries if entry[0] not in used_right and entry[1] == left_ts
        ]
        if exact_candidates:
            right_index, right_ts, native_right, _ = exact_candidates[0]
            method = "exact"
        else:
            nearest: tuple[int, int, int, Mapping[str, Any]] | None = None
            best_skew: int | None = None
            for entry in right_entries:
                right_index, right_ts, native_right, right_record = entry
                if right_index in used_right:
                    continue
                skew = abs(right_ts - left_ts)
                if skew > tolerance_us:
                    continue
                if best_skew is None or skew < best_skew:
                    best_skew = skew
                    nearest = entry
            if nearest is None:
                unpaired_left.append(left_index)
                continue
            right_index, right_ts, native_right, _ = nearest
            method = "nearest"

        used_right.add(right_index)
        pairs.append(
            StereoPairRecord(
                canonical_frame_number=int(canonical_frame_number),
                device_timestamp_us=left_ts,
                native_left_frame_number=native_left,
                native_right_frame_number=native_right,
                left_timestamp_us=left_ts,
                right_timestamp_us=right_ts,
                timestamp_skew_us=abs(right_ts - left_ts),
                pairing_method=method,
            )
        )

    return pairs, unpaired_left


def summarize_pairing(pairs: Sequence[StereoPairRecord], unpaired_left: Sequence[int]) -> dict[str, Any]:
    exact = sum(1 for pair in pairs if pair.pairing_method == "exact")
    nearest = sum(1 for pair in pairs if pair.pairing_method == "nearest")
    max_skew = max((pair.timestamp_skew_us for pair in pairs), default=0)
    return {
        "paired_frames": len(pairs),
        "unpaired_left_frames": len(unpaired_left),
        "exact_pairs": exact,
        "nearest_pairs": nearest,
        "max_timestamp_skew_us": max_skew,
    }
