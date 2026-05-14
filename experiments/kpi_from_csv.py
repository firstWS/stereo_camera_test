#!/usr/bin/env python3
"""Compute repeatability KPIs (std, IQR, failure rate) from repeatability CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def _parse_float(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _row_is_primary_detection(r: dict) -> bool:
    """Legacy CSV has no det_idx; multi-object CSV uses det_idx 0 for highest-confidence box."""
    raw = r.get("det_idx")
    if raw is None or str(raw).strip() == "":
        return True
    return str(raw).strip() == "0"


def _iqr(values: list[float]) -> float:
    if len(values) < 4:
        return float("nan")
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return float(q3 - q1)


def summarize_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    def collect(col: str, valid_flag: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            if not _row_is_primary_detection(r):
                continue
            if str(r.get(valid_flag, "")).lower() not in ("1", "true", "yes"):
                continue
            v = _parse_float(str(r.get(col, "")).strip())
            if v is not None:
                out.append(v)
        return out

    for track, prefix in [("A", "A_"), ("B", "B_")]:
        xs = collect(prefix + "X", track + "_valid")
        ys = collect(prefix + "Y", track + "_valid")
        zs = collect(prefix + "Z", track + "_valid")

        def stats(vals: list[float]) -> dict:
            if not vals:
                return {"n": 0}
            return {
                "n": len(vals),
                "mean": statistics.mean(vals),
                "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "iqr": _iqr(vals),
            }

        total = sum(1 for r in rows if _row_is_primary_detection(r))
        ok = sum(
            1
            for r in rows
            if _row_is_primary_detection(r)
            and str(r.get(f"{track}_valid", "")).lower() in ("1", "true", "yes")
        )
        print(f"\n== Track {track} == (primary det_idx rows only)")
        print(f"valid_frames: {ok} / {total} ({(ok/total*100) if total else 0:.1f}%)")
        for name, vals in [("X", xs), ("Y", ys), ("Z", zs)]:
            st = stats(vals)
            print(f"  {name}: {st}")

    # Frame-to-frame jump Z (track A)
    zseq: list[tuple[int, float]] = []
    for r in rows:
        if not _row_is_primary_detection(r):
            continue
        if str(r.get("A_valid", "")).lower() not in ("1", "true", "yes"):
            continue
        fi = int(float(r["frame_idx"]))
        z = _parse_float(str(r.get("A_Z", "")))
        if z is not None:
            zseq.append((fi, z))
    zseq.sort(key=lambda t: t[0])
    jumps = [
        abs(zseq[i][1] - zseq[i - 1][1]) for i in range(1, len(zseq))
    ]
    if jumps:
        sj = sorted(jumps)
        p95 = sj[min(len(sj) - 1, int(0.95 * (len(sj) - 1)))]
        print("\n== A_Z consecutive |delta| ==")
        print({
            "n": len(jumps),
            "mean_jump": statistics.mean(jumps),
            "p95_jump": p95,
        })

    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    summarize_csv(args.csv)


if __name__ == "__main__":
    main()
