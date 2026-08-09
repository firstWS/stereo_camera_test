from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures.py"
_SPEC = importlib.util.spec_from_file_location("dataset_recorder_fixtures", _FIXTURES_PATH)
assert _SPEC and _SPEC.loader
_FIXTURES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURES)
build_synthetic_session = _FIXTURES.build_synthetic_session


@pytest.fixture
def synthetic_session(tmp_path: Path) -> Path:
    return build_synthetic_session(tmp_path)
