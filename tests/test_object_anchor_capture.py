from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_anchor_capture import (  # noqa: E402
    ObjectAnchorCaptureSession,
    ObjectAnchorCaptureSettings,
)


def _frame() -> np.ndarray:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 15
    image[:, :, 1] = 80
    image[:, :, 2] = 220
    return image


def test_capture_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        ObjectAnchorCaptureSettings("other")
    with pytest.raises(ValueError):
        ObjectAnchorCaptureSettings("positive", target_count=0)
    with pytest.raises(ValueError):
        ObjectAnchorCaptureSettings("negative", interval_seconds=0.0)


def test_positive_capture_is_immediate_timed_and_exact(tmp_path: Path) -> None:
    session = ObjectAnchorCaptureSession(
        tmp_path,
        ObjectAnchorCaptureSettings("positive", target_count=3, interval_seconds=1.0),
        camera_serial="SERIAL-1",
        loaded_model_path="model.pt",
        session_time=datetime(2026, 7, 22, 16, 0, 0, tzinfo=timezone.utc),
    )

    assert session.save(_frame(), now_monotonic=10.0) is not None
    assert session.save(_frame(), now_monotonic=10.9) is None
    assert session.save(_frame(), now_monotonic=11.0) is not None
    assert session.save(_frame(), now_monotonic=12.0) is not None
    assert session.complete
    assert session.save(_frame(), now_monotonic=13.0) is None
    assert len(list(session.image_dir.glob("*.jpg"))) == 3
    assert not (tmp_path / "positive" / "labels").exists()

    saved = cv2.imread(str(sorted(session.image_dir.glob("*.jpg"))[0]))
    assert saved is not None
    assert saved.shape == _frame().shape
    assert int(saved[0, 0, 2]) > int(saved[0, 0, 0])


def test_negative_capture_creates_empty_matching_label_and_manifest(tmp_path: Path) -> None:
    session = ObjectAnchorCaptureSession(
        tmp_path,
        ObjectAnchorCaptureSettings("negative", target_count=1, interval_seconds=0.5),
        camera_serial="SERIAL-2",
        loaded_model_path="weights/best.pt",
        session_time=datetime(2026, 7, 22, 16, 15, 0, tzinfo=timezone.utc),
    )
    image_path = session.save(
        _frame(),
        now_monotonic=1.0,
        captured_at=datetime(2026, 7, 22, 16, 15, 1, tzinfo=timezone.utc),
        object_anchor_detected=True,
        object_anchor_confidence=0.42,
        apriltag_detected=True,
    )
    assert image_path is not None
    label_path = session.label_dir / f"{image_path.stem}.txt"
    assert label_path.is_file()
    assert label_path.stat().st_size == 0

    with session.manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["filename"] == image_path.name
    assert rows[0]["capture_type"] == "negative"
    assert rows[0]["frame_width"] == "32"
    assert rows[0]["frame_height"] == "24"
    assert rows[0]["camera_serial"] == "SERIAL-2"
    assert rows[0]["object_anchor_detected"] == "True"
    assert rows[0]["apriltag_detected"] == "True"


def test_failed_image_write_does_not_advance_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = ObjectAnchorCaptureSession(
        tmp_path,
        ObjectAnchorCaptureSettings("positive", target_count=1, interval_seconds=1.0),
        camera_serial="",
        loaded_model_path="",
    )
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)
    assert session.save(_frame(), now_monotonic=2.0) is None
    assert session.saved_count == 0
    assert session.last_saved_monotonic is None
    assert not session.manifest_path.exists()


def test_repeated_session_stamp_never_overwrites_existing_file(tmp_path: Path) -> None:
    stamp = datetime(2026, 7, 22, 16, 20, 0, tzinfo=timezone.utc)
    first = ObjectAnchorCaptureSession(
        tmp_path,
        ObjectAnchorCaptureSettings("positive", target_count=1, interval_seconds=1.0),
        camera_serial="",
        loaded_model_path="",
        session_time=stamp,
    )
    first_path = first.save(_frame(), now_monotonic=1.0)
    second = ObjectAnchorCaptureSession(
        tmp_path,
        ObjectAnchorCaptureSettings("positive", target_count=1, interval_seconds=1.0),
        camera_serial="",
        loaded_model_path="",
        session_time=stamp,
    )
    second_path = second.save(_frame(), now_monotonic=1.0)
    assert first_path is not None and second_path is not None
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()


def test_default_target_count_stops_at_exactly_100_saved_images(tmp_path: Path) -> None:
    session = ObjectAnchorCaptureSession(
        tmp_path,
        ObjectAnchorCaptureSettings("positive"),
        camera_serial="SERIAL-100",
        loaded_model_path="model.pt",
    )
    for index in range(100):
        assert session.save(_frame(), now_monotonic=float(index)) is not None
    assert session.complete
    assert session.saved_count == 100
    assert session.save(_frame(), now_monotonic=100.0) is None
    assert len(list(session.image_dir.glob("*.jpg"))) == 100
