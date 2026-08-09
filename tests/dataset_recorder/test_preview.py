from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset_recorder.frame_writer import FrameSnapshot  # noqa: E402
from dataset_recorder.rgb_preview import RgbPreview  # noqa: E402


def test_preview_disabled_does_not_call_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    preview = RgbPreview(enabled=False)
    monkeypatch.setattr(
        "dataset_recorder.rgb_preview.decode_snapshot",
        lambda _snapshot: (_ for _ in ()).throw(AssertionError("decode must not run")),
    )
    preview.update_from_snapshot(
        FrameSnapshot(
            stream="RGB",
            sequence=1,
            width=2,
            height=2,
            format_name="RGB",
            data=bytes([1, 2, 3, 4]),
            depth_scale=None,
        )
    )
    preview.render(scenario_slug="scenario_a", elapsed_seconds=1.0, phase_cue_id="hold")
    preview.close()


def test_preview_render_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    preview = RgbPreview(enabled=True)

    def _broken_decode(_snapshot: FrameSnapshot) -> np.ndarray:
        raise RuntimeError("preview decode failed")

    monkeypatch.setattr("dataset_recorder.rgb_preview.decode_snapshot", _broken_decode)
    preview.update_from_snapshot(
        FrameSnapshot(
            stream="RGB",
            sequence=1,
            width=4,
            height=4,
            format_name="RGB",
            data=bytes(range(48)),
            depth_scale=None,
        )
    )
    preview.render(scenario_slug="scenario_a", elapsed_seconds=3.0, phase_cue_id="yaw_start")
    preview.close()


def test_preview_resize_does_not_mutate_snapshot_bytes() -> None:
    data = np.arange(48, dtype=np.uint8).tobytes()
    snapshot = FrameSnapshot(
        stream="RGB",
        sequence=1,
        width=4,
        height=4,
        format_name="RGB",
        data=data,
        depth_scale=None,
    )
    preview = RgbPreview(enabled=True, max_width=2)
    preview.update_from_snapshot(snapshot)
    assert snapshot.data == data
    preview.render(scenario_slug="scenario_a", elapsed_seconds=0.0, phase_cue_id="hold")
    assert snapshot.data == data
    preview.close()


def test_preview_cleanup_swallows_destroy_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    preview = RgbPreview(enabled=True)
    preview._opened = True

    def _raise_destroy(_window: str) -> None:
        raise RuntimeError("destroy failed")

    monkeypatch.setattr(cv2, "destroyWindow", _raise_destroy)
    preview.close()
