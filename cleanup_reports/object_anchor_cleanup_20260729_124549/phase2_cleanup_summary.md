# Object Anchor phase-2 safe cleanup

Completed at: 2026-07-29 15:08 KST  
Camera execution: not run

## Backup

- Cleanup branch: `cleanup/object-anchor-mvp`
- Pre-cleanup commit: `e9a31077342efeb2e0d27cefc993204004001208`
- Local lightweight tag: `object-anchor-pre-cleanup-20260729`
- Remote push: not performed

## Approved deletion

- Cache/temp: 8,959 files / 192,380,836 bytes
- Rotated Orbbec logs: 3 files / 314,572,612 bytes
  - `Log/OrbbecSDK.log.1.txt`: 104,857,536 bytes
  - `Log/OrbbecSDK.log.2.txt`: 104,857,599 bytes
  - `Log/OrbbecSDK.log.3.txt`: 104,857,477 bytes
- Validation prediction visualizations: 20 files / 40,978,415 bytes
- Total deleted: 8,982 files / 547,931,863 bytes
- Tracked modified `Log/OrbbecSDK.log.txt` was preserved.
- Deleted-file hashes: `phase2_deleted_files.csv`

## R&D archive

- Destination: `archive_local/rnd_20260729/`
- Moved: 1,402 files / 590,729,569 bytes
- Outputs: 1,068 files / 470,916,302 bytes
- Diagnostics: 286 files / 80,145,325 bytes
- Checkpoints/training artifacts: 43 files / 39,526,185 bytes
- Unreferenced experiment scripts: 5 files / 141,757 bytes
- Manifest: `archive_local/rnd_20260729/archive_manifest.csv`
- Archive move did not count as deleted disk space.

Tracked scripts moved to the archive:

- `scripts/analyze_apriltag_branch_diagnostics.py`
- `scripts/diagnose_apriltag_ippe_branches.py`
- `scripts/diagnose_planar_pnp_ambiguity.py`
- `scripts/evaluate_object_anchor_pilot.py`
- `scripts/prepare_object_anchor_orbbec_pilot.py`

No current runtime, test, config, or documentation reference to an archived path
was found after the move.

## Protected assets

- Active Full99: `models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt`
- Baseline: `models/object_anchor/tissue_box_01_front_only/best.pt`
- Pilot: `models/object_anchor/tissue_box_01_front_only_orbbec_pilot/best.pt`
- Cup model: `yolo11s.pt`
- Full99 dataset: 223 images / 223 labels, missing/orphan pair count 0
- Latest successful Preview: `out/object_anchor_preview/20260728_120922/`
- Median3 evidence: `out/object_anchor_full99/mvp_translation_diagnostics/20260726_164854/`
- Full99 integration: `out/object_anchor_full99/default_orbbec_integration/20260726_151831/`
- Three pytest fixture runs (`163325`, `164157`, `172325`) remain at their original paths.
- `.venv` remains present. Only regenerable bytecode caches inside it were removed.

## Capacity comparison

- Before total: 33,555 files / 2,988,049,171 bytes
- After total including archive: 24,578 files / 2,443,220,320 bytes
- After active working area excluding archive: 23,172 files / 1,851,333,421 bytes
- Active workspace separated to archive: 590,729,569 bytes
- Actual deletion saving: 547,931,863 bytes
- Direct runtime: 29 files / 25,199,856 bytes, unchanged
- Data: 1,240 files / 432,243,906 bytes, unchanged
- Models directory: 3 files / 16,876,120 bytes, unchanged
- Out: 1,897 files / 755,761,178 bytes → 521 files / 152,481,696 bytes
- `.venv`: 30,134 files / 1,320,713,956 bytes → 21,261 files / 1,129,714,426 bytes

## Verification

- In-memory Python compile: PASS (61 files)
- Full static suite: PASS (88 tests)
- Object Anchor Preview focused suite: PASS (46 tests)
- `configs/orbbec_gemini.yaml` load: PASS
- `run.ps1 -Orbbec` static path: PASS
- Runtime dependency missing: 0
- Protected model hashes changed: no
- Operating source/config hashes changed: no
- Full99 image/label pairing changed: no
- Archive hash/integrity errors: 0
- Stale references to archived paths: 0

Machine-readable verification: `phase2_verification.json`

## Scope confirmation

- No operating source or runtime config was edited.
- No large directory structure redesign was performed.
- No model or dataset source was deleted.
- No camera or live experiment was run.
