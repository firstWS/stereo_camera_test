"""Session-level semantic object annotations for derived cup observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .session_metadata import load_json, write_json

ANNOTATIONS_REL_PATH = Path("derived") / "annotations" / "objects.json"
VALID_SEMANTIC_IDS = frozenset({"cup1", "cup2", "unknown"})


def annotations_path(session_dir: Path) -> Path:
    return Path(session_dir) / ANNOTATIONS_REL_PATH


def load_object_annotations(session_dir: Path) -> dict[str, Any] | None:
    path = annotations_path(session_dir)
    if not path.is_file():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid annotations payload: {path}")
    return payload


def write_object_annotations(session_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = annotations_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(payload))
    return path


def validate_object_annotations(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("objects.json schema_version must be 1")
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        errors.append("objects.json missing objects mapping")
        return errors
    for semantic_id, entry in objects.items():
        if semantic_id not in VALID_SEMANTIC_IDS - {"unknown"}:
            errors.append(f"unsupported semantic_id: {semantic_id}")
        if not isinstance(entry, dict):
            errors.append(f"objects.{semantic_id} must be a mapping")
            continue
        track_id = entry.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            errors.append(f"objects.{semantic_id}.track_id must be a non-empty string")
    return errors


def build_track_to_semantic_map(payload: Mapping[str, Any] | None) -> dict[str, str]:
    if payload is None:
        return {}
    errors = validate_object_annotations(payload)
    if errors:
        raise ValueError("; ".join(errors))
    mapping: dict[str, str] = {}
    objects = payload.get("objects") or {}
    for semantic_id, entry in objects.items():
        if semantic_id == "unknown":
            continue
        if not isinstance(entry, dict):
            continue
        track_id = entry.get("track_id")
        if isinstance(track_id, str) and track_id:
            mapping[track_id] = semantic_id
    return mapping


def semantic_id_for_track(track_id: str, track_to_semantic: Mapping[str, str]) -> str:
    return track_to_semantic.get(track_id, "unknown")
