#!/usr/bin/env python3
"""Execute the approved phase-2 archive and safe-delete operations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT = Path("cleanup_reports/object_anchor_cleanup_20260729_124549")
ARCHIVE = Path("archive_local/rnd_20260729")
ACTIVE_LOG = "Log/OrbbecSDK.log.txt"
ROTATED_LOGS = (
    "Log/OrbbecSDK.log.1.txt",
    "Log/OrbbecSDK.log.2.txt",
    "Log/OrbbecSDK.log.3.txt",
)
VALIDATION_OUTPUTS = (
    "out/tissue_box_front_only_val_predictions",
    "out/tissue_box_front_only_val_predictions_ids",
    "out/tissue_box_front_only_val_predictions_max1",
)
PROTECTED_PREFIXES = (
    ".venv/",
    "calibration/",
    "cleanup_reports/",
    "data/",
    "models/object_anchor/",
    "out/object_anchor_full99/default_orbbec_integration/20260726_151831/",
    "out/object_anchor_full99/dataset_validation/",
    "out/object_anchor_full99/offline_comparison/",
    "out/object_anchor_full99/legacy_regression/",
    "out/object_anchor_full99/live_world/",
    "out/object_anchor_full99/mvp_final_comparison/20260726_163325/",
    "out/object_anchor_full99/mvp_branch_aware_comparison/20260726_164157/",
    "out/object_anchor_full99/mvp_final_comparison/20260726_172325/",
    "out/object_anchor_full99/mvp_translation_diagnostics/20260726_164854/",
    "out/object_anchor_full99/mvp_branch_mapping_diagnostics/20260726_173332/",
    "out/object_anchor_full99/replacement_feasibility/",
    "out/object_anchor_preview/20260728_120922/",
    "out/object_anchor_world/20260722_144211/",
)
PROTECTED_EXACT = {
    ACTIVE_LOG,
    "run.ps1",
    "yolo11s.pt",
    "yolo11n-pose.pt",
    "out/object_anchor_full99/comparison_summary.json",
    "out/object_anchor_full99/split_manifest.csv",
}
OPERATING_PATHS = (
    Path("run.ps1"),
    Path("experiments/repeatability_run.py"),
    Path("configs/orbbec_gemini.yaml"),
    Path("src"),
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def enumerate_files(root: Path, *, include_archive: bool = True) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_parts = current.relative_to(root).parts
        dirnames[:] = [
            name
            for name in dirnames
            if name != ".git"
            and not (
                not include_archive
                and (
                    (rel_parts and rel_parts[0] == "archive_local")
                    or (not rel_parts and name == "archive_local")
                )
            )
        ]
        files.extend(
            current / name
            for name in filenames
            if (current / name).is_file() and not (current / name).is_symlink()
        )
    return files


def tree_metrics(root: Path) -> dict[str, Any]:
    all_files = enumerate_files(root)
    working_files = enumerate_files(root, include_archive=False)

    def area(name: str) -> dict[str, int]:
        base = root / name
        files = [path for path in base.rglob("*") if path.is_file()] if base.exists() else []
        return {
            "file_count": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
        }

    return {
        "captured_at": utc_now(),
        "total_including_archive": {
            "file_count": len(all_files),
            "size_bytes": sum(path.stat().st_size for path in all_files),
        },
        "working_area_excluding_archive": {
            "file_count": len(working_files),
            "size_bytes": sum(path.stat().st_size for path in working_files),
        },
        "areas": {
            name: area(name)
            for name in ("src", "experiments", "configs", "data", "models", "out", ".venv")
        },
    }


def operating_hashes(root: Path) -> dict[str, str]:
    files: list[Path] = []
    for item in OPERATING_PATHS:
        path = root / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix.lower() not in {".pyc", ".pyo"}
            )
    return {relative(root, path): sha256(path) for path in sorted(set(files))}


def dataset_pair_counts(root: Path) -> dict[str, Any]:
    dataset = root / "data/datasets/tissue_box_front_orbbec_full99"
    images = [
        path
        for path in (dataset / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = [
        path
        for path in (dataset / "labels").rglob("*.txt")
        if path.is_file()
    ]
    image_keys = {
        path.relative_to(dataset / "images").with_suffix("").as_posix()
        for path in images
    }
    label_keys = {
        path.relative_to(dataset / "labels").with_suffix("").as_posix()
        for path in labels
    }
    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": sorted(image_keys - label_keys),
        "orphan_labels": sorted(label_keys - image_keys),
    }


def model_fingerprints(root: Path) -> dict[str, dict[str, Any]]:
    paths = (
        "models/object_anchor/tissue_box_01_front_only/best.pt",
        "models/object_anchor/tissue_box_01_front_only_orbbec_pilot/best.pt",
        "models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt",
        "yolo11s.pt",
    )
    result: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = root / item
        result[item] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
    return result


def is_protected(path: str, protected_files: set[str]) -> bool:
    return (
        path in PROTECTED_EXACT
        or path in protected_files
        or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
    )


def archive_bucket(path: str) -> str:
    lower = path.lower()
    if path.startswith(("experiments/", "scripts/")):
        return "experiments"
    if path.startswith("configs/"):
        return "configs"
    if path.startswith("tests/"):
        return "tests"
    if path.endswith((".pt", ".pth", ".onnx", ".engine")) or path.startswith("runs/"):
        return "checkpoints"
    if any(
        token in lower
        for token in ("diagnostic", "branch", "feasibility", "mvp_", "intrinsic", "pnp")
    ):
        return "diagnostics"
    return "outputs"


def archive_files(root: Path) -> None:
    classification = read_csv(root / REPORT / "cleanup_classification.csv")
    inventory = {
        row["path"]: row
        for row in read_csv(root / REPORT / "repository_inventory.csv")
    }
    protected_files = {
        line.strip()
        for line in (root / REPORT / "protected_files.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    if (root / ARCHIVE / "archive_manifest.csv").exists():
        raise RuntimeError("Archive manifest already exists; refusing a duplicate move")

    candidates: list[tuple[dict[str, str], Path, Path]] = []
    for row in classification:
        original = row["path"]
        if row["category"] != "ARCHIVE_RND":
            continue
        if row.get("referenced_by", "").strip() or is_protected(original, protected_files):
            continue
        source = root / original
        if not source.is_file():
            continue
        inventory_row = inventory.get(original)
        if not inventory_row:
            raise RuntimeError(f"Repository inventory row missing: {original}")
        row = {**row, "sha256": inventory_row["sha256"]}
        bucket = archive_bucket(original)
        destination = root / ARCHIVE / bucket / original
        candidates.append((row, source, destination))

    # Complete all integrity checks before moving the first file.
    for row, source, destination in candidates:
        if destination.exists():
            raise RuntimeError(f"Archive destination already exists: {destination}")
        actual_hash = sha256(source)
        if row["sha256"] and actual_hash != row["sha256"]:
            raise RuntimeError(f"Hash changed since inventory: {row['path']}")

    before = tree_metrics(root)
    snapshot = {
        "captured_at": utc_now(),
        "tree": before,
        "operating_hashes": operating_hashes(root),
        "dataset": dataset_pair_counts(root),
        "models": model_fingerprints(root),
    }
    write_json(root / ARCHIVE / "manifest/before_cleanup.json", snapshot)

    archived_at = utc_now()
    manifest_rows: list[dict[str, Any]] = []
    for row, source, destination in candidates:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        manifest_rows.append(
            {
                "original_path": row["path"],
                "archived_path": relative(root, destination),
                "category": "ARCHIVE_RND",
                "size_bytes": int(row["size_bytes"]),
                "sha256": row["sha256"],
                "reason": row["reason"],
                "referenced_before_move": row.get("referenced_by", ""),
                "recoverable": row["recoverable"],
                "archived_at": archived_at,
            }
        )

    fields = [
        "original_path",
        "archived_path",
        "category",
        "size_bytes",
        "sha256",
        "reason",
        "referenced_before_move",
        "recoverable",
        "archived_at",
    ]
    write_csv(root / ARCHIVE / "archive_manifest.csv", manifest_rows, fields)
    write_csv(root / ARCHIVE / "manifest/archive_manifest.csv", manifest_rows, fields)
    readme = f"""# Object Anchor R&D archive

- Purpose: separate historical, non-runtime R&D artifacts from the active MVP workspace.
- Archived at: {archived_at}
- Files moved: {len(manifest_rows)}
- Bytes moved: {sum(int(row['size_bytes']) for row in manifest_rows)}
- This is a preservation move, not permanent deletion.
- The phase-1 reference graph classified every moved file as ARCHIVE_RND with no current reference.

## Restore

For each row in `archive_manifest.csv`, move `archived_path` back to
`original_path`, preserving the recorded SHA-256. Restore only on the cleanup
branch or after creating another backup point.
"""
    (root / ARCHIVE / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "archived_files": len(manifest_rows),
                "archived_bytes": sum(
                    int(row["size_bytes"]) for row in manifest_rows
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def collect_tree_files(path: Path) -> list[Path]:
    return [candidate for candidate in path.rglob("*") if candidate.is_file()]


def safe_delete(root: Path) -> None:
    classification = {
        row["path"]: row
        for row in read_csv(root / REPORT / "cleanup_classification.csv")
    }
    targets: dict[Path, str] = {}

    # Exact approved rotated logs; active tracked log is deliberately excluded.
    for item in ROTATED_LOGS:
        path = root / item
        if not path.is_file():
            raise RuntimeError(f"Approved rotated log missing: {item}")
        row = classification.get(item)
        if not row or row["category"] != "SAFE_DELETE_CANDIDATE":
            raise RuntimeError(f"Rotated log is not approved by classification: {item}")
        targets[path] = "approved rotated Orbbec log"

    # Validation visualization outputs must contain only approved generated files.
    for directory_name in VALIDATION_OUTPUTS:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        files = collect_tree_files(directory)
        for path in files:
            rp = relative(root, path)
            row = classification.get(rp)
            if not row or row["category"] != "SAFE_DELETE_CANDIDATE":
                raise RuntimeError(f"Validation output was not approved: {rp}")
            if row.get("referenced_by", "").strip():
                raise RuntimeError(f"Validation output is referenced: {rp}")
            targets[path] = "regenerable validation prediction visualization"

    # Approved caches and OS-generated files, including caches inside the retained venv.
    cache_dirs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if ".git" in current.relative_to(root).parts:
            continue
        for name in list(dirnames):
            if name in {
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                ".ipynb_checkpoints",
            }:
                cache_dirs.append(current / name)
                dirnames.remove(name)
        for name in filenames:
            path = current / name
            if path.suffix.lower() in {".pyc", ".pyo"}:
                targets[path] = "regenerable Python bytecode cache"
            elif name in {"Thumbs.db", ".DS_Store"}:
                targets[path] = "regenerable OS metadata"
    for directory in cache_dirs:
        for path in collect_tree_files(directory):
            targets[path] = "regenerable cache directory"

    # Preflight all protected boundaries before deleting the first file.
    for path in targets:
        rp = relative(root, path)
        if rp == ACTIVE_LOG:
            raise RuntimeError("Active tracked Orbbec log entered deletion set")
        if rp in PROTECTED_EXACT and rp not in ROTATED_LOGS:
            raise RuntimeError(f"Protected file entered deletion set: {rp}")

    deleted_at = utc_now()
    rows: list[dict[str, Any]] = []
    for path, reason in sorted(targets.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": relative(root, path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "reason": reason,
                "deleted_at": deleted_at,
            }
        )

    for path in targets:
        if path.is_file():
            path.unlink()
    for directory in sorted(cache_dirs, key=lambda item: len(item.parts), reverse=True):
        if directory.exists():
            shutil.rmtree(directory)
    for directory_name in VALIDATION_OUTPUTS:
        directory = root / directory_name
        if directory.exists():
            shutil.rmtree(directory)

    write_csv(
        root / REPORT / "phase2_deleted_files.csv",
        rows,
        ["path", "size_bytes", "sha256", "reason", "deleted_at"],
    )
    after = {
        "captured_at": utc_now(),
        "tree": tree_metrics(root),
        "operating_hashes": operating_hashes(root),
        "dataset": dataset_pair_counts(root),
        "models": model_fingerprints(root),
        "deleted_files": len(rows),
        "deleted_bytes": sum(int(row["size_bytes"]) for row in rows),
    }
    write_json(root / REPORT / "phase2_after_cleanup.json", after)
    print(
        json.dumps(
            {
                "deleted_files": len(rows),
                "deleted_bytes": sum(int(row["size_bytes"]) for row in rows),
                "rotated_logs": [
                    {
                        "path": row["path"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in rows
                    if row["path"] in ROTATED_LOGS
                ],
                "validation_outputs": {
                    "files": sum(
                        row["reason"]
                        == "regenerable validation prediction visualization"
                        for row in rows
                    ),
                    "bytes": sum(
                        int(row["size_bytes"])
                        for row in rows
                        if row["reason"]
                        == "regenerable validation prediction visualization"
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def verify_cleanup(root: Path) -> None:
    before = json.loads(
        (root / ARCHIVE / "manifest/before_cleanup.json").read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        (root / REPORT / "phase2_after_cleanup.json").read_text(encoding="utf-8")
    )
    archive_rows = read_csv(root / ARCHIVE / "archive_manifest.csv")
    referenced = json.loads(
        (root / REPORT / "referenced_files.json").read_text(encoding="utf-8")
    )

    operating_unchanged = (
        before["operating_hashes"] == after["operating_hashes"]
    )
    dataset_unchanged = before["dataset"] == after["dataset"]
    models_unchanged = before["models"] == after["models"]
    runtime_missing = [
        path
        for path in referenced["runtime_files"]
        if not (root / path).is_file()
    ]

    archive_errors: list[str] = []
    for row in archive_rows:
        original = root / row["original_path"]
        archived = root / row["archived_path"]
        if original.exists():
            archive_errors.append(f"original_still_exists:{row['original_path']}")
        if not archived.is_file():
            archive_errors.append(f"archive_missing:{row['archived_path']}")
        elif sha256(archived) != row["sha256"]:
            archive_errors.append(f"archive_hash_mismatch:{row['archived_path']}")

    scan_files: list[Path] = []
    for item in (
        "README.md",
        "run.ps1",
        "setup.ps1",
        "src",
        "experiments",
        "scripts",
        "tests",
        "configs",
        "docs",
    ):
        path = root / item
        if path.is_file():
            scan_files.append(path)
        elif path.is_dir():
            scan_files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower()
                in {".py", ".ps1", ".md", ".yaml", ".yml", ".json", ".txt"}
            )
    texts = {
        relative(root, path): path.read_text(encoding="utf-8", errors="replace")
        for path in scan_files
        if path.stat().st_size <= 5 * 1024 * 1024
    }
    stale_references: list[dict[str, str]] = []
    for row in archive_rows:
        original = row["original_path"]
        slash = original
        backslash = original.replace("/", "\\")
        for source, text in texts.items():
            if slash in text or backslash in text:
                stale_references.append(
                    {"source": source, "archived_original_path": original}
                )

    result = {
        "verified_at": utc_now(),
        "operating_code_and_config_hashes_unchanged": operating_unchanged,
        "dataset_pairing_unchanged": dataset_unchanged,
        "protected_model_hashes_unchanged": models_unchanged,
        "runtime_dependency_missing": runtime_missing,
        "archive_integrity_errors": archive_errors,
        "stale_code_config_test_doc_references": stale_references,
        "archive_file_count": len(archive_rows),
        "archive_size_bytes": sum(int(row["size_bytes"]) for row in archive_rows),
        "venv_present": (root / ".venv/Scripts/python.exe").is_file(),
    }
    write_json(root / REPORT / "phase2_verification.json", result)
    if not (
        operating_unchanged
        and dataset_unchanged
        and models_unchanged
        and not runtime_missing
        and not archive_errors
        and not stale_references
        and result["venv_present"]
    ):
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("archive", "delete", "verify"))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.action == "archive":
        archive_files(root)
    elif args.action == "delete":
        safe_delete(root)
    else:
        verify_cleanup(root)


if __name__ == "__main__":
    main()
