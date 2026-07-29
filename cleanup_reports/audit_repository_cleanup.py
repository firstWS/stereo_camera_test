#!/usr/bin/env python3
"""Generate a read-only repository cleanup audit (no deletion or moving)."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


TEXT_EXTENSIONS = {
    ".py", ".ps1", ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".csv", ".gitignore", ".bat", ".cmd",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".torchscript"}
CACHE_PARTS = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".ipynb_checkpoints",
}
RUNTIME_SEEDS = {
    "run.ps1",
    "experiments/repeatability_run.py",
    "configs/orbbec_gemini.yaml",
    "src/object_anchor_preview.py",
    "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt",
    "configs/object_anchors/tissue_box_01_front_only.yaml",
    "yolo11s.pt",
}
PROTECTED_EXACT = {
    *RUNTIME_SEEDS,
    "src/object_anchor_detector.py",
    "src/object_anchor_pose.py",
    "src/object_anchor_visualizer.py",
    "src/object_anchor_runtime.py",
    "src/apriltag_world.py",
    "src/apriltag_scale.py",
    "src/orbbec_rgbd_capture.py",
    "src/rgbd_geometry.py",
    "src/detect.py",
}
TRAINING_SCRIPT_NAMES = {
    "train_object_anchor_pose.py",
    "prepare_object_anchor_full99_dataset.py",
    "prepare_object_anchor_valid99_source.py",
    "sanitize_yolo_pose_dataset.py",
    "verify_object_anchor_full99_protection.py",
    "summarize_object_anchor_full99.py",
}
RND_SCRIPT_MARKERS = (
    "diagnose_", "analyze_", "feasibility", "mvp_final_comparison",
    "evaluate_object_anchor_pilot", "synthetic_object_anchor_test",
    "prepare_object_anchor_orbbec_pilot", "kpi_from_csv",
)
TEST_FIXTURE_PREFIXES = (
    "out/object_anchor_full99/mvp_final_comparison/20260726_163325/",
    "out/object_anchor_full99/mvp_branch_aware_comparison/20260726_164157/",
    "out/object_anchor_full99/mvp_final_comparison/20260726_172325/",
)


def run_git(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name == ".gitignore"


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MODEL_EXTENSIONS:
        return "model"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
        return "config_or_metadata"
    if suffix == ".py":
        return "python"
    if suffix == ".ps1":
        return "powershell"
    if suffix in {".csv", ".tsv"}:
        return "tabular_result"
    if suffix in {".zip", ".7z", ".rar", ".tar", ".gz"}:
        return "archive"
    if suffix in {".log"} or ".log." in path.name.lower():
        return "log"
    if suffix in {".pyc", ".pyo"}:
        return "python_cache"
    if suffix in {".md", ".txt"}:
        return "documentation"
    return suffix.lstrip(".") or "no_extension"


def enumerate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        relative_parts = current.relative_to(root).parts
        dirnames[:] = [
            name
            for name in dirnames
            if name != ".git"
            and not (
                (relative_parts and relative_parts[0] == "cleanup_reports")
                or (not relative_parts and name == "cleanup_reports")
            )
        ]
        for filename in filenames:
            path = current / filename
            if path.is_file() and not path.is_symlink():
                files.append(path)
    return sorted(files, key=lambda item: rel(root, item).lower())


def git_state_sets(root: Path) -> tuple[set[str], set[str], set[str], dict[str, str]]:
    tracked = set(run_git(root, "ls-files"))
    untracked = set(run_git(root, "ls-files", "--others", "--exclude-standard"))
    ignored = set(
        run_git(root, "ls-files", "--others", "--ignored", "--exclude-standard")
    )
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    porcelain = [line.rstrip() for line in result.stdout.splitlines() if line]
    changed: dict[str, str] = {}
    for line in porcelain:
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed[path] = status
    return tracked, untracked, ignored, changed


def local_python_modules(root: Path, paths: list[Path]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for path in paths:
        rp = rel(root, path)
        if path.suffix.lower() != ".py":
            continue
        if rp.startswith((".venv/", "out/", "runs/", "data/")):
            continue
        stem = path.stem
        modules.setdefault(stem, rp)
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.setdefault(".".join(parts), rp)
            if parts[0] in {"src", "experiments", "scripts", "tests"} and len(parts) > 1:
                modules.setdefault(".".join(parts[1:]), rp)
    return modules


def parse_python_imports(path: Path, modules: dict[str, str]) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        for name in names:
            candidates = [name]
            bits = name.split(".")
            candidates.extend(".".join(bits[:index]) for index in range(len(bits), 0, -1))
            for candidate in candidates:
                if candidate in modules:
                    found.add(modules[candidate])
                    break
                if candidate.split(".")[-1] in modules:
                    found.add(modules[candidate.split(".")[-1]])
                    break
    return found


def existing_paths_from_yaml(root: Path, path: Path) -> set[str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    values: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            values.append(value)

    walk(payload)
    found: set[str] = set()
    for value in values:
        candidate = Path(value)
        candidates = [candidate] if candidate.is_absolute() else [root / candidate, path.parent / candidate]
        for item in candidates:
            if item.is_file():
                try:
                    found.add(rel(root, item.resolve()))
                except ValueError:
                    pass
                break
    return found


def build_references(
    root: Path, paths: list[Path], modules: dict[str, str]
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    references: dict[str, set[str]] = defaultdict(set)
    imports: dict[str, set[str]] = defaultdict(set)
    text_cache: dict[str, str] = {}
    relative_set = {rel(root, path) for path in paths}
    basename_map: dict[str, list[str]] = defaultdict(list)
    for rp in relative_set:
        basename_map[Path(rp).name.lower()].append(rp)

    path_token = re.compile(
        r"[A-Za-z0-9_@.+-]+(?:[\\/][A-Za-z0-9_@.+ -]+)+"
        r"(?:\.[A-Za-z0-9_-]+)?|[A-Za-z0-9_@.+-]+\.[A-Za-z0-9_-]+"
    )
    for path in paths:
        rp = rel(root, path)
        if not is_text_candidate(path) or path.stat().st_size > 5 * 1024 * 1024:
            continue
        if rp.startswith((".venv/", "out/", "runs/", "data/")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text_cache[rp] = text
        if path.suffix.lower() == ".py":
            for target in parse_python_imports(path, modules):
                references[target].add(rp)
                imports[rp].add(target)
        if path.suffix.lower() in {".yaml", ".yml"}:
            for target in existing_paths_from_yaml(root, path):
                references[target].add(rp)
        for token in path_token.findall(text):
            normalized = token.strip("'\"`()[]{}:,").replace("\\", "/")
            candidates = [normalized, normalized.lstrip("./")]
            for candidate in candidates:
                if candidate in relative_set:
                    references[candidate].add(rp)
            base = Path(normalized).name.lower()
            matches = basename_map.get(base, [])
            if len(matches) == 1:
                references[matches[0]].add(rp)
    return references, imports, text_cache


def runtime_dependencies(
    root: Path,
    relative_set: set[str],
    imports: dict[str, set[str]],
    references: dict[str, set[str]],
    text_cache: dict[str, str],
) -> tuple[set[str], set[str], list[tuple[str, str, str]]]:
    direct_runtime = {item for item in RUNTIME_SEEDS if item in relative_set}
    edges: list[tuple[str, str, str]] = []

    # Direct no-setup path: selected entrypoint and its transitive local imports.
    queue = deque(direct_runtime)
    visited: set[str] = set()
    while queue:
        source = queue.popleft()
        if source in visited:
            continue
        visited.add(source)
        for target in imports.get(source, set()):
            edges.append((source, target, "direct_python_import"))
            if target not in direct_runtime:
                direct_runtime.add(target)
                queue.append(target)
    if "scripts/sync-session-path.ps1" in relative_set:
        direct_runtime.add("scripts/sync-session-path.ps1")
        edges.append(
            ("run.ps1", "scripts/sync-session-path.ps1", "direct_powershell_source")
        )

    config_assets = {
        target
        for target, sources in references.items()
        if "configs/orbbec_gemini.yaml" in sources
        and (
            target.startswith(("models/", "configs/"))
            or Path(target).suffix.lower() in MODEL_EXTENSIONS
        )
    }
    for target in sorted(config_assets):
        direct_runtime.add(target)
        edges.append(
            ("configs/orbbec_gemini.yaml", target, "selected_config_asset")
        )

    runtime = set(direct_runtime)
    explicit = {
        "setup.ps1",
        "scripts/sync-session-path.ps1",
        "requirements.txt",
        "scripts/create_placeholder_calibration.py",
        "scripts/smoke_test.py",
        "scripts/verify_environment.py",
        "scripts/synthetic_object_anchor_test.py",
        "calibration/stereo_calib.yaml",
        "configs/object_anchors/tissue_box_01_front_only.yaml",
    }
    for item in explicit:
        if item in relative_set:
            runtime.add(item)
            edges.append(("run.ps1", item, "setup_or_explicit_path"))

    # setup.ps1 executes the complete current test suite.
    for rp in relative_set:
        if rp.startswith("tests/") and rp.endswith(".py"):
            runtime.add(rp)
            edges.append(("setup.ps1", rp, "full_test_suite"))
        if any(rp.startswith(prefix) for prefix in TEST_FIXTURE_PREFIXES):
            runtime.add(rp)
            edges.append(("setup.ps1", rp, "regression_fixture"))

    # Resolve Python imports and explicit Python file paths from runtime/setup/tests.
    queue = deque(runtime)
    visited = set()
    while queue:
        source = queue.popleft()
        if source in visited:
            continue
        visited.add(source)
        targets = set(imports.get(source, set()))
        if source.endswith(".py"):
            targets.update(
                target
                for target, sources in references.items()
                if source in sources and target.endswith(".py")
            )
        for target in targets:
            edges.append((source, target, "python_import_or_explicit_module"))
            if target not in runtime:
                runtime.add(target)
                queue.append(target)

    # Local venv is a recreatable environment, not repository runtime source.
    runtime = {item for item in runtime if not item.startswith(".venv/")}
    direct_runtime = {
        item for item in direct_runtime if not item.startswith(".venv/")
    }
    return runtime, direct_runtime, sorted(set(edges))


def latest_timestamp_dirs(root: Path) -> set[str]:
    latest: set[str] = set()
    timestamp_re = re.compile(r"^\d{8}_\d{6}$")
    for base_name in ("out", "runs"):
        base = root / base_name
        if not base.is_dir():
            continue
        groups: dict[Path, list[Path]] = defaultdict(list)
        for path in base.rglob("*"):
            if path.is_dir() and timestamp_re.match(path.name):
                groups[path.parent].append(path)
        for paths in groups.values():
            chosen = max(paths, key=lambda item: item.name)
            latest.add(rel(root, chosen))
    return latest


def classify(
    rp: str,
    size: int,
    git_status: str,
    runtime: set[str],
    latest_dirs: set[str],
    references: dict[str, set[str]],
) -> tuple[str, str, str, str, str]:
    path = Path(rp)
    parts = set(path.parts)
    lower = rp.lower()
    name = path.name.lower()
    referenced = bool(references.get(rp))

    if rp in runtime or rp in PROTECTED_EXACT:
        return (
            "KEEP_RUNTIME", "current Orbbec/Preview runtime or regression dependency",
            "no", "keep in place", "low",
        )
    if git_status.startswith("tracked_modified") or (
        git_status == "untracked" and path.suffix.lower() in {".py", ".ps1", ".yaml", ".yml", ".md"}
    ):
        return (
            "MANUAL_REVIEW", "uncommitted user work must be protected",
            "uncertain", "review/commit before cleanup", "high",
        )
    if rp == ".gitignore" or rp.startswith("samples/"):
        return (
            "KEEP_REPRODUCIBILITY", "repository configuration or tracked sample input",
            "yes_from_source_control", "keep pending explicit sample ownership review", "low",
        )
    if rp.startswith("data/"):
        return (
            "KEEP_REPRODUCIBILITY", "training/source dataset; empty labels may intentionally mean negative samples",
            "no", "keep; only reorganize after dataset-specific approval", "low",
        )
    if any(part in CACHE_PARTS for part in path.parts) or path.suffix.lower() in {".pyc", ".pyo"}:
        return (
            "SAFE_DELETE_CANDIDATE", "interpreter/test cache; regenerated automatically",
            "yes", "delete only after user approval", "low",
        )
    if rp.startswith(".venv/"):
        return (
            "SAFE_DELETE_CANDIDATE", "local virtual environment; reproducible from setup/requirements",
            "yes", "optionally recreate after approval", "medium",
        )
    if rp.startswith("Log/") or name in {"thumbs.db", ".ds_store"} or ".tmp" in name:
        return (
            "SAFE_DELETE_CANDIDATE", "generated log or temporary OS/tool artifact",
            "yes", "delete after confirming no diagnostic retention need", "low",
        )
    if size == 0 and git_status in {"ignored", "untracked"}:
        return (
            "SAFE_DELETE_CANDIDATE", "empty generated/untracked file",
            "yes", "delete after approval", "low",
        )
    if rp == "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt":
        return (
            "KEEP_RUNTIME", "active Full99 Object Anchor model",
            "no", "keep in place", "low",
        )
    if (
        rp.startswith("models/object_anchor/")
        and Path(rp).name.lower() == "best.pt"
    ):
        return (
            "KEEP_REPRODUCIBILITY", "canonical baseline/pilot model in the Full99 training lineage",
            "limited", "keep with training lineage and hashes", "low",
        )
    if rp == "yolo11n-pose.pt":
        return (
            "KEEP_REPRODUCIBILITY", "base pose model referenced by Full99 training script",
            "downloadable", "keep with training metadata", "low",
        )
    if rp.startswith("out/object_anchor_full99/training/weights/"):
        return (
            "ARCHIVE_RND", "duplicate export of canonical runs/models training weights",
            "yes_if_canonical_runs_and_models_remain", "archive intact before pruning", "medium",
        )
    if (
        rp.startswith("runs/object_anchor_pose/")
        and Path(rp).name.lower() == "best.pt"
    ):
        return (
            "KEEP_REPRODUCIBILITY", "best checkpoint retained as canonical training audit",
            "limited", "retain with args.yaml and results.csv", "low",
        )
    if (
        rp.startswith("runs/object_anchor_pose/")
        and Path(rp).name.lower() == "last.pt"
    ):
        return (
            "ARCHIVE_RND", "non-canonical last checkpoint; best and training audit remain",
            "yes_if_best_and_training_metadata_remain", "archive before any pruning", "medium",
        )
    if rp.startswith("models/") or path.suffix.lower() in MODEL_EXTENSIONS:
        if "full99" in lower and ("best.pt" in lower or "args.yaml" in lower):
            return (
                "KEEP_REPRODUCIBILITY", "Full99 model/training reproduction artifact",
                "limited", "keep with final dataset and training metadata", "low",
            )
        return (
            "ARCHIVE_RND", "non-active model/checkpoint; not referenced by current runtime",
            "uncertain", "archive after model provenance review", "medium",
        )
    if rp.startswith("runs/"):
        if (
            rp.startswith("runs/object_anchor_pose/")
            and Path(rp).name.lower() in {"best.pt", "args.yaml", "results.csv"}
        ):
            return (
                "KEEP_REPRODUCIBILITY", "canonical training audit or best checkpoint",
                "limited", "retain with corresponding model lineage", "low",
            )
        if "full99" in lower:
            return (
                "KEEP_REPRODUCIBILITY", "final Full99 training run evidence",
                "limited", "retain compact training metadata/weights; propose archiving bulk plots", "low",
            )
        return (
            "ARCHIVE_RND", "historical training/validation run",
            "usually", "archive intact before considering pruning", "medium",
        )
    if referenced:
        return (
            "KEEP_REPRODUCIBILITY", "referenced by current code, tests, config, or documentation",
            "varies", "keep until the reference is intentionally retired", "low",
        )
    if rp.startswith("out/object_anchor_calibration/"):
        return (
            "KEEP_RUNTIME", "operational calibration path",
            "no", "keep in place", "high",
        )
    if rp.startswith("out/"):
        if rp.startswith("out/object_anchor_full99/training/"):
            return (
                "ARCHIVE_RND", "duplicate export of canonical runs/models training artifacts",
                "yes_if_canonical_runs_and_models_remain", "archive intact before pruning", "medium",
            )
        if rp.startswith(
            (
                "out/tissue_box_front_only_val_predictions/",
                "out/tissue_box_front_only_val_predictions_ids/",
                "out/tissue_box_front_only_val_predictions_max1/",
            )
        ):
            return (
                "SAFE_DELETE_CANDIDATE", "regenerable validation visualization output",
                "yes", "delete only after confirming retained summary/model", "low",
            )
        in_latest = any(rp == item or rp.startswith(item + "/") for item in latest_dirs)
        is_summary = (
            name in {"readme.md", "config_snapshot.yaml", "session_calibration.json"}
            or "summary" in name
            or "decision" in name
        )
        final_evidence = any(
            marker in lower
            for marker in (
                "translation_diagnostics", "mvp_translation_diagnostics",
                "object_anchor_full99", "object_anchor_preview", "comparison_summary",
            )
        )
        if is_summary and (in_latest or final_evidence):
            return (
                "KEEP_REPRODUCIBILITY", "compact MVP/Full99/Preview result evidence",
                "recomputable_with_hardware_or_training", "retain summary/config/representative evidence", "low",
            )
        return (
            "ARCHIVE_RND", "historical or bulk experiment output not used by runtime",
            "varies", "archive run directory intact; prune only after report review", "medium",
        )
    if rp.startswith("scripts/"):
        if path.name in TRAINING_SCRIPT_NAMES:
            return (
                "KEEP_REPRODUCIBILITY", "Full99 data preparation/training reproducibility utility",
                "yes_from_source_control", "keep in scripts", "low",
            )
        if any(marker in lower for marker in RND_SCRIPT_MARKERS):
            return (
                "ARCHIVE_RND", "completed diagnostic/pilot analysis utility",
                "yes_from_source_control", "archive with matching results", "low",
            )
        return (
            "KEEP_REPRODUCIBILITY", "general build/calibration/environment utility",
            "yes_from_source_control", "keep pending manual structure cleanup", "low",
        )
    if rp.startswith("experiments/") and rp != "experiments/repeatability_run.py":
        return (
            "ARCHIVE_RND", "historical experiment entrypoint not in default runtime",
            "yes_from_source_control", "archive with experiment config/results", "low",
        )
    if rp.startswith("configs/experiments/"):
        return (
            "ARCHIVE_RND", "historical experiment configuration",
            "yes_from_source_control", "archive with corresponding runner/results", "low",
        )
    if rp.startswith("tests/"):
        return (
            "ARCHIVE_RND", "test tied to non-runtime R&D path",
            "yes_from_source_control", "archive together with covered R&D code", "low",
        )
    if rp.startswith(("docs/", "configs/", "calibration/")) or name.startswith("requirements"):
        return (
            "KEEP_REPRODUCIBILITY", "configuration/documentation/calibration provenance",
            "limited", "keep pending manual documentation restructuring", "low",
        )
    if rp.startswith("src/"):
        return (
            "KEEP_RUNTIME", "shared production/reusable source module (conservative protection)",
            "yes_from_source_control", "keep in place", "low",
        )
    if path.suffix.lower() in {".zip", ".7z", ".rar"}:
        return (
            "MANUAL_REVIEW", "archive may be a unique source/export; uniqueness unconfirmed",
            "uncertain", "verify extraction/provenance before any deletion", "high",
        )
    return (
        "MANUAL_REVIEW", "role or reproducibility value not established",
        "uncertain", "manual owner review", "medium",
    )


def dataset_roots(root: Path) -> list[Path]:
    base = root / "data"
    if not base.is_dir():
        return []
    roots: set[Path] = set()
    for path in base.rglob("*"):
        if not path.is_dir():
            continue
        if path.name.lower() in {"images", "labels", "train", "val", "test"}:
            continue
        children = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        if {"images", "labels"} <= children or {"train", "val"} <= children:
            roots.add(path)
    for child in base.iterdir():
        if child.is_dir():
            roots.add(child)
    return sorted(roots)


def summarize_dataset(root: Path, dataset: Path, hash_by_path: dict[str, str]) -> dict[str, Any]:
    files = [path for path in dataset.rglob("*") if path.is_file()]
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [
        path
        for path in files
        if path.suffix.lower() == ".txt"
        and "labels" in {part.lower() for part in path.relative_to(dataset).parts}
    ]
    # Compare by basename because YOLO images/labels live in parallel directories.
    image_names = Counter(path.stem for path in images)
    label_names = Counter(path.stem for path in labels)
    missing_labels = sum(max(0, count - label_names.get(stem, 0)) for stem, count in image_names.items())
    orphan_labels = sum(max(0, count - image_names.get(stem, 0)) for stem, count in label_names.items())
    train_hashes = {
        hash_by_path.get(rel(root, path))
        for path in images
        if "train" in {part.lower() for part in path.relative_to(dataset).parts}
    }
    val_hashes = {
        hash_by_path.get(rel(root, path))
        for path in images
        if "val" in {part.lower() for part in path.relative_to(dataset).parts}
    }
    train_hashes.discard(None)
    val_hashes.discard(None)
    leakage = train_hashes & val_hashes
    lower = rel(root, dataset).lower()
    role = (
        "Full99 final dataset"
        if "full99" in lower
        else "CVAT source/export"
        if "cvat" in lower
        else "capture/source"
        if "capture" in lower or "source" in lower
        else "legacy/pilot/intermediate dataset"
    )
    notes = [
        "Do not delete or move automatically; duplicate hashes may be intentional split/export copies."
    ]
    if "tissue_box_front_orbbec_full99" in lower:
        notes.append(
            "Internal train/val exact-hash overlap is zero, but both Orbbec splits "
            "come from the same capture session; this is not an external test."
        )
    if "tissue_box_front_orbbec_pilot" in lower:
        notes.append(
            "Pilot is a designed subset of Full99; pilot validation images appear "
            "in Full99 training, so cross-experiment metric comparisons need care."
        )
    if "cvat" in lower:
        notes.append("Preserve as source/export provenance, including archives.")
    return {
        "dataset_path": rel(root, dataset),
        "role": role,
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "image_count": len(images),
        "label_count": len(labels),
        "yaml_count": sum(path.suffix.lower() in {".yaml", ".yml"} for path in files),
        "archive_count": sum(path.suffix.lower() in {".zip", ".7z", ".rar"} for path in files),
        "missing_label_name_count": missing_labels,
        "orphan_label_name_count": orphan_labels,
        "train_val_duplicate_hash_count": len(leakage),
        "recommended_category": "KEEP_REPRODUCIBILITY",
        "note": " ".join(notes),
    }


def output_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp_re = re.compile(r"^\d{8}_\d{6}$")
    candidates: set[Path] = set()
    for base_name in ("out", "runs"):
        base = root / base_name
        if not base.is_dir():
            continue
        candidates.update(path for path in base.rglob("*") if path.is_dir() and timestamp_re.match(path.name))
        candidates.update(path for path in base.iterdir() if path.is_dir())
    for directory in sorted(candidates):
        files = [path for path in directory.rglob("*") if path.is_file()]
        summaries = [
            path for path in files
            if "summary" in path.name.lower()
            or "decision" in path.name.lower()
            or path.name.lower() == "readme.md"
        ]
        configs = [
            path for path in files
            if path.name.lower() in {"config_snapshot.yaml", "args.yaml"}
        ]
        images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
        raw_media = [
            path for path in files
            if path.suffix.lower() in IMAGE_EXTENSIONS | {".mp4", ".avi", ".mkv", ".npy"}
        ]
        status = "unknown"
        summary_payloads: dict[str, Any] = {}
        for summary in summaries[:20]:
            if summary.suffix.lower() != ".json" or summary.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                summary_payloads[summary.name.lower()] = json.loads(
                    summary.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                continue
        preview_summary = summary_payloads.get("preview_summary.json")
        decision = summary_payloads.get("mvp_final_decision.json")
        if isinstance(preview_summary, dict):
            status = (
                "successful"
                if preview_summary.get("session_calibration_success") is True
                else "incomplete"
            )
        elif isinstance(decision, dict):
            status = (
                "successful"
                if str(decision.get("decision", "")).upper() == "A"
                else "failed_or_aborted"
            )
        elif "diagnostic_summary.json" in summary_payloads:
            status = "completed_diagnostic"
        elif rel(root, directory).startswith("runs/") and any(
            path.name == "results.csv" for path in files
        ):
            status = "completed_training"
        else:
            status_text = ""
            for summary in summaries[:10]:
                if summary.stat().st_size <= 2 * 1024 * 1024:
                    status_text += summary.read_text(
                        encoding="utf-8", errors="replace"
                    ).lower()
            status = (
                "failed_or_aborted"
                if any(token in status_text for token in ("failed", "aborted"))
                else "successful"
                if any(token in status_text for token in ("completed", "success"))
                else "unknown"
            )
        rows.append(
            {
                "path": rel(root, directory),
                "kind": "training_run" if rel(root, directory).startswith("runs/") else "experiment_output",
                "status": status,
                "file_count": len(files),
                "size_bytes": sum(path.stat().st_size for path in files),
                "summary_files": ";".join(rel(root, path) for path in summaries[:20]),
                "config_files": ";".join(rel(root, path) for path in configs[:20]),
                "representative_image_count": len(images),
                "raw_media_file_count": len(raw_media),
                "raw_media_size_bytes": sum(path.stat().st_size for path in raw_media),
                "runtime_referenced": "no",
                "regenerable": "code yes; live/training inputs and hardware may be required",
                "retention_value": (
                    "high"
                    if any(marker in rel(root, directory).lower() for marker in ("full99", "preview"))
                    else "medium"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    tracked, untracked, ignored, changed = git_state_sets(root)
    paths = enumerate_files(root)
    relative_set = {rel(root, path) for path in paths}
    modules = local_python_modules(root, paths)
    references, imports, text_cache = build_references(root, paths, modules)
    runtime, direct_runtime, runtime_edges = runtime_dependencies(
        root, relative_set, imports, references, text_cache
    )
    latest_dirs = latest_timestamp_dirs(root)

    inventory: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    hash_groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for index, path in enumerate(paths, start=1):
        rp = rel(root, path)
        stat = path.stat()
        digest = sha256(path)
        hashes[rp] = digest
        hash_groups[(stat.st_size, digest)].append(rp)
        if rp in tracked:
            git_status = (
                f"tracked_modified:{changed[rp].strip()}"
                if rp in changed
                else "tracked"
            )
        elif rp in untracked:
            git_status = "untracked"
        elif rp in ignored:
            git_status = "ignored"
        else:
            git_status = "outside_git_listing"
        category, reason, recoverable, action, risk = classify(
            rp, stat.st_size, git_status, runtime, latest_dirs, references
        )
        refs = sorted(references.get(rp, set()))
        inventory.append(
            {
                "path": rp,
                "file_type": file_kind(path),
                "size_bytes": stat.st_size,
                "modified_time_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "git_status": git_status,
                "sha256": digest,
                "current_code_import": any(
                    rp in targets for targets in imports.values()
                ),
                "config_reference": any(
                    source.endswith((".yaml", ".yml", ".json"))
                    for source in refs
                ),
                "run_ps1_runtime_dependency": rp in runtime,
                "test_reference": any(source.startswith("tests/") for source in refs),
                "subprocess_or_command_reference": any(
                    source.endswith((".ps1", ".bat", ".cmd")) for source in refs
                ),
                "referenced_by": ";".join(refs),
                "duplicate_group_size": len(hash_groups[(stat.st_size, digest)]),
                "category": category,
                "recoverable": recoverable,
                "preservation_reason": reason,
                "recommended_action": action,
                "risk_level": risk,
                "requires_user_approval": category in {
                    "ARCHIVE_RND", "SAFE_DELETE_CANDIDATE", "MANUAL_REVIEW"
                },
            }
        )

    # Duplicate group size is known only after all hashes.
    for row in inventory:
        row["duplicate_group_size"] = len(
            hash_groups[(int(row["size_bytes"]), str(row["sha256"]))]
        )

    inventory_fields = [
        "path", "file_type", "size_bytes", "modified_time_utc", "git_status",
        "sha256", "current_code_import", "config_reference",
        "run_ps1_runtime_dependency", "test_reference",
        "subprocess_or_command_reference", "referenced_by",
        "duplicate_group_size", "category", "recoverable",
        "preservation_reason", "recommended_action", "risk_level",
        "requires_user_approval",
    ]
    write_csv(output / "repository_inventory.csv", inventory, inventory_fields)

    classification_rows = [
        {
            "path": row["path"],
            "category": row["category"],
            "size_bytes": row["size_bytes"],
            "git_status": row["git_status"],
            "referenced_by": row["referenced_by"],
            "reason": row["preservation_reason"],
            "recoverable": row["recoverable"],
            "recommended_action": row["recommended_action"],
            "risk_level": row["risk_level"],
            "requires_user_approval": row["requires_user_approval"],
        }
        for row in inventory
    ]
    write_csv(
        output / "cleanup_classification.csv",
        classification_rows,
        [
            "path", "category", "size_bytes", "git_status", "referenced_by",
            "reason", "recoverable", "recommended_action", "risk_level",
            "requires_user_approval",
        ],
    )

    duplicate_rows: list[dict[str, Any]] = []
    duplicate_group_count = 0
    duplicate_file_count = 0
    duplicate_savings = 0
    for group_index, ((size, digest), members) in enumerate(
        sorted(hash_groups.items(), key=lambda item: (-item[0][0], item[0][1])),
        start=1,
    ):
        if size <= 0 or len(members) < 2:
            continue
        duplicate_group_count += 1
        duplicate_file_count += len(members)
        duplicate_savings += size * (len(members) - 1)
        likely_intentional = any(
            member.startswith(("data/", ".venv/")) for member in members
        )
        for member in members:
            duplicate_rows.append(
                {
                    "group_id": group_index,
                    "sha256": digest,
                    "size_bytes_each": size,
                    "group_file_count": len(members),
                    "potential_savings_bytes": size * (len(members) - 1),
                    "path": member,
                    "likely_intentional_structure_copy": likely_intentional,
                    "automatic_delete_recommended": False,
                    "note": "Keep one is not automatically safe; verify dataset/model provenance.",
                }
            )
    write_csv(
        output / "duplicate_files.csv",
        duplicate_rows,
        [
            "group_id", "sha256", "size_bytes_each", "group_file_count",
            "potential_savings_bytes", "path",
            "likely_intentional_structure_copy", "automatic_delete_recommended",
            "note",
        ],
    )

    large_rows = sorted(inventory, key=lambda row: int(row["size_bytes"]), reverse=True)
    write_csv(
        output / "large_files.csv",
        [
            {
                "rank": index,
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "size_mib": round(int(row["size_bytes"]) / 1024 / 1024, 3),
                "file_type": row["file_type"],
                "git_status": row["git_status"],
                "category": row["category"],
                "sha256": row["sha256"],
                "referenced_by": row["referenced_by"],
            }
            for index, row in enumerate(large_rows, start=1)
            if int(row["size_bytes"]) >= 1024 * 1024
        ],
        [
            "rank", "path", "size_bytes", "size_mib", "file_type",
            "git_status", "category", "sha256", "referenced_by",
        ],
    )

    model_rows: list[dict[str, Any]] = []
    for row in inventory:
        rp = str(row["path"])
        if (
            Path(rp).suffix.lower() not in MODEL_EXTENSIONS
            or rp.startswith(".venv/")
        ):
            continue
        lower = rp.lower()
        model_type = (
            "best" if Path(rp).name.lower() == "best.pt"
            else "last" if Path(rp).name.lower() == "last.pt"
            else "intermediate_or_base"
        )
        relation = (
            "active_full99" if rp == "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt"
            else "full99_related" if "full99" in lower
            else "pilot_or_legacy" if any(token in lower for token in ("pilot", "front_only"))
            else "base_detector_or_other"
        )
        refs = sorted(references.get(rp, set()))
        model_rows.append(
            {
                "path": rp,
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "current_config_references": ";".join(
                    item for item in refs if item.endswith((".yaml", ".yml", ".json"))
                ),
                "all_references": ";".join(refs),
                "generation_experiment": str(Path(rp).parent),
                "checkpoint_kind": model_type,
                "full99_relationship": relation,
                "category": row["category"],
                "archive_possible": relation not in {"active_full99"},
                "recoverable": (
                    "duplicate hash or training data/settings required"
                    if row["duplicate_group_size"] == 1
                    else "identical hash copy exists"
                ),
                "recommended_action": row["recommended_action"],
            }
        )
    write_csv(
        output / "model_inventory.csv",
        model_rows,
        [
            "path", "size_bytes", "sha256", "current_config_references",
            "all_references", "generation_experiment", "checkpoint_kind",
            "full99_relationship", "category", "archive_possible",
            "recoverable", "recommended_action",
        ],
    )

    dataset_rows = [
        summarize_dataset(root, dataset, hashes) for dataset in dataset_roots(root)
    ]
    write_csv(
        output / "dataset_inventory.csv",
        dataset_rows,
        [
            "dataset_path", "role", "file_count", "size_bytes", "image_count",
            "label_count", "yaml_count", "archive_count",
            "missing_label_name_count", "orphan_label_name_count",
            "train_val_duplicate_hash_count", "recommended_category", "note",
        ],
    )

    output_rows = output_runs(root)
    write_csv(
        output / "experiment_output_inventory.csv",
        output_rows,
        [
            "path", "kind", "status", "file_count", "size_bytes",
            "summary_files", "config_files", "representative_image_count",
            "raw_media_file_count", "raw_media_size_bytes",
            "runtime_referenced", "regenerable", "retention_value",
        ],
    )

    referenced_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "python_imports": "AST import/import-from resolution against local modules",
            "config_paths": "recursive YAML scalar path resolution",
            "string_paths": "path-like token and unique basename matching",
            "limitations": [
                "Computed runtime plugin names and arbitrary runtime-generated paths may require manual review.",
                "References inside binary files and text files over 5 MiB are not scanned.",
            ],
        },
        "runtime_files": sorted(runtime),
        "direct_live_runtime_files": sorted(direct_runtime),
        "setup_and_regression_support_files": sorted(runtime - direct_runtime),
        "references": {
            target: sorted(sources) for target, sources in sorted(references.items())
        },
        "runtime_edges": [
            {"source": source, "target": target, "kind": kind}
            for source, target, kind in runtime_edges
        ],
    }
    write_json(output / "referenced_files.json", referenced_payload)

    category_counts = Counter(str(row["category"]) for row in inventory)
    category_bytes = Counter()
    git_counts = Counter()
    for row in inventory:
        category_bytes[str(row["category"])] += int(row["size_bytes"])
        status = str(row["git_status"])
        key = "tracked" if status.startswith("tracked") else status
        git_counts[key] += 1

    protected = sorted(
        {
            str(row["path"])
            for row in inventory
            if row["category"] in {"KEEP_RUNTIME", "KEEP_REPRODUCIBILITY"}
            or str(row["git_status"]).startswith("tracked_modified")
            or row["git_status"] == "untracked"
        }
    )
    (output / "protected_files.txt").write_text(
        "\n".join(protected) + "\n", encoding="utf-8"
    )

    graph_lines = [
        "# Runtime dependency graph",
        "",
        "Scope: static dependency closure for `.\\run.ps1 -Orbbec`; camera was not opened.",
        "",
        "```text",
        "run.ps1",
        "  -> configs/orbbec_gemini.yaml",
        "  -> experiments/repeatability_run.py",
        "       -> src/* local runtime imports",
        "       -> Full99 Object Anchor model/config",
        "       -> YOLO Cup model",
        "       -> AprilTag/OpenCV pose pipeline",
        "       -> Orbbec RGB-D capture",
        "       -> session-only Object Anchor Preview logs",
        "```",
        "",
        f"Direct live runtime files: **{len(direct_runtime)}**",
        f"Setup + current regression closure: **{len(runtime - direct_runtime)}**",
        f"Combined KEEP_RUNTIME dependency closure: **{len(runtime)}**",
        "",
        "## Direct live runtime files",
        *[f"- `{item}`" for item in sorted(direct_runtime)],
        "",
        "## Setup and current regression support",
        *[f"- `{item}`" for item in sorted(runtime - direct_runtime)],
        "",
        "## Static edges",
        *[
            f"- `{source}` --{kind}--> `{target}`"
            for source, target, kind in runtime_edges
        ],
        "",
        "## Important boundary",
        "- Display source switching is Preview-only; operational world CSV/source remains AprilTag.",
        "- `session_calibration.json` is an output record and is never auto-loaded as operational calibration.",
    ]
    (output / "runtime_dependency_graph.md").write_text(
        "\n".join(graph_lines) + "\n", encoding="utf-8"
    )

    archive_lines = [
        "# Proposed archive plan",
        "",
        "No files were moved or deleted in phase 1.",
        "",
        f"- ARCHIVE_RND candidates: {category_counts['ARCHIVE_RND']} files",
        f"- Candidate size: {category_bytes['ARCHIVE_RND']} bytes",
        "",
        "## Proposed phase-2 grouping",
        "- `archive_local/rnd_<timestamp>/experiments/`: historical feasibility/MVP gate runners.",
        "- `archive_local/rnd_<timestamp>/diagnostics/`: branch, intrinsic, IPPE, and translation diagnostics.",
        "- `archive_local/rnd_<timestamp>/configs/`: matching historical experiment configs.",
        "- `archive_local/rnd_<timestamp>/results/`: failed/aborted and bulk historical outputs, kept intact initially.",
        "- `archive_local/rnd_<timestamp>/models/`: non-active pilot/legacy checkpoints after provenance confirmation.",
        "",
        "## Safety sequence",
        "1. User approves exact rows from `cleanup_classification.csv`.",
        "2. Create cleanup branch and backup tag/commit.",
        "3. Copy/move to archive without compression.",
        "4. Re-run import/config checks and all tests.",
        "5. Hold archive for one user review cycle before any pruning.",
    ]
    (output / "proposed_archive_plan.md").write_text(
        "\n".join(archive_lines) + "\n", encoding="utf-8"
    )

    delete_lines = [
        "# Proposed delete plan",
        "",
        "No files were deleted in phase 1.",
        "",
        f"- SAFE_DELETE_CANDIDATE files: {category_counts['SAFE_DELETE_CANDIDATE']}",
        f"- Candidate size: {category_bytes['SAFE_DELETE_CANDIDATE']} bytes",
        "",
        "## Low-risk categories after approval",
        "- `__pycache__`, `*.pyc`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`.",
        "- Generated SDK/application logs after confirming no investigation need.",
        "- Empty ignored/generated files.",
        "",
        "## Separate approval required",
        "- `.venv/` is reproducible but deleting it interrupts local execution until setup is rerun.",
        "- Exact-hash duplicates are not automatically deletable; dataset split/export copies may be intentional.",
        "- ZIP/CVAT exports and all source images require provenance confirmation.",
        "- Models/checkpoints require model inventory review even when another hash-identical copy exists.",
    ]
    (output / "proposed_delete_plan.md").write_text(
        "\n".join(delete_lines) + "\n", encoding="utf-8"
    )

    structure_lines = [
        "# Proposed final structure (proposal only)",
        "",
        "No directory moves were performed. Moving these paths requires coordinated import/config/doc updates.",
        "",
        "```text",
        "src/                         # runtime and reusable modules",
        "experiments/                 # maintained entrypoints (repeatability_run.py)",
        "scripts/",
        "  data/                      # preparation/sanitation",
        "  training/                  # Full99 training/reproduction",
        "  diagnostics/               # retained active diagnostics only",
        "configs/",
        "  runtime/                   # Orbbec/default runtime configs",
        "  training/                  # dataset/training configs",
        "  archived_experiments/      # historical MVP/branch configs",
        "models/",
        "  active/                    # Full99 best + active Cup detector assets",
        "  archived/                  # pilot/legacy checkpoints",
        "data/",
        "  source/                    # original captures/CVAT exports",
        "  active/                    # final Full99 training dataset",
        "  archived/                  # legacy/pilot datasets",
        "tests/",
        "  runtime/",
        "  training/",
        "  archived_rnd/",
        "docs/",
        "  mvp/",
        "  runtime/",
        "  model_data/",
        "out/",
        "  latest_preview/",
        "  retained_summaries/",
        "archive_local/rnd_<timestamp>/",
        "```",
        "",
        "## Expected coordinated changes in phase 2",
        "- Update Python imports only if modules move.",
        "- Update YAML model/config/output paths.",
        "- Update PowerShell entrypoint and README commands.",
        "- Add migration map and path-compatibility check.",
        "- Keep `run.ps1 -Orbbec` behavior unchanged.",
    ]
    (output / "proposed_final_structure.md").write_text(
        "\n".join(structure_lines) + "\n", encoding="utf-8"
    )

    top20 = large_rows[:20]
    summary_lines = [
        "# Object Anchor repository cleanup — phase 1 summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is a read-only classification report. No project file was deleted, moved, reset, or reverted. Camera was not opened.",
        "",
        "## Inventory totals",
        f"- Files: **{len(inventory)}**",
        f"- Bytes: **{sum(int(row['size_bytes']) for row in inventory)}**",
        f"- Git tracked: **{git_counts['tracked']}**",
        f"- Git untracked: **{git_counts['untracked']}**",
        f"- Git ignored: **{git_counts['ignored']}**",
        f"- Runtime dependency/protected closure: **{len(runtime)}**",
        f"- Direct live runtime subset: **{len(direct_runtime)}**",
        "",
        "## Classification",
        *[
            f"- {category}: {category_counts[category]} files / {category_bytes[category]} bytes"
            for category in (
                "KEEP_RUNTIME", "KEEP_REPRODUCIBILITY", "ARCHIVE_RND",
                "SAFE_DELETE_CANDIDATE", "MANUAL_REVIEW",
            )
        ],
        "",
        "## Duplicate content",
        f"- Duplicate groups: **{duplicate_group_count}**",
        f"- Files participating: **{duplicate_file_count}**",
        f"- Theoretical maximum savings: **{duplicate_savings} bytes**",
        "- This is not an automatic deletion recommendation.",
        "",
        "## Largest 20 files",
        *[
            f"{index}. `{row['path']}` — {int(row['size_bytes']) / 1024 / 1024:.2f} MiB — {row['category']}"
            for index, row in enumerate(top20, start=1)
        ],
        "",
        "## Git state at snapshot",
        *[
            f"- `{rp}`: `{status}`"
            for rp, status in sorted(changed.items())
        ],
        "",
        "## Test status",
        "- Pending insertion after the separate no-camera static/regression test run.",
        "",
        "## Required approval before phase 2",
        "- Exact SAFE_DELETE rows, especially `.venv/` and logs.",
        "- Which historical out/runs directories may move to `archive_local`.",
        "- Pilot/old model provenance and retention period.",
        "- Dataset source/CVAT/legacy ownership and archive layout.",
        "- Commit/backup tag strategy for currently modified user files.",
    ]
    (output / "cleanup_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": len(inventory),
                "total_bytes": sum(int(row["size_bytes"]) for row in inventory),
                "git_counts": dict(git_counts),
                "category_counts": dict(category_counts),
                "category_bytes": dict(category_bytes),
                "runtime_count": len(runtime),
                "direct_runtime_count": len(direct_runtime),
                "duplicate_groups": duplicate_group_count,
                "duplicate_files": duplicate_file_count,
                "duplicate_savings_bytes": duplicate_savings,
                "models": len(model_rows),
                "datasets": len(dataset_rows),
                "output_runs": len(output_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
