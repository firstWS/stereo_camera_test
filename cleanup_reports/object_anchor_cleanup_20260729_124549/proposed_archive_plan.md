# Proposed archive plan

No files were moved or deleted in phase 1.

- ARCHIVE_RND candidates: 1814 files
- Candidate size: 738400496 bytes

## Proposed phase-2 grouping
- `archive_local/rnd_<timestamp>/experiments/`: only entrypoints no longer referenced by runtime, setup, tests, or docs.
- `archive_local/rnd_<timestamp>/diagnostics/`: unreferenced branch/IPPE/planar-PnP/pilot scripts.
- `archive_local/rnd_<timestamp>/configs/`: only configs no longer loaded by tests or documentation.
- `archive_local/rnd_<timestamp>/results/`: failed/aborted and bulk historical outputs, kept intact initially.
- `archive_local/rnd_<timestamp>/models/`: non-active pilot/legacy checkpoints after provenance confirmation.

## Required evidence lock before moving bulk results
- `out/object_anchor_full99/comparison_summary.json`
- `out/object_anchor_full99/default_orbbec_integration/20260726_151831/`
- MVP fixtures: `mvp_final_comparison/20260726_163325`, `mvp_branch_aware_comparison/20260726_164157`, and `mvp_final_comparison/20260726_172325`
- Median3 evidence: `mvp_translation_diagnostics/20260726_164854`
- Branch mapping and feasibility summaries/config snapshots/representative images
- Latest successful Preview: `out/object_anchor_preview/20260728_120922/`
- Full99 model, dataset manifests, training args/results, and protection verification

## Candidate concentration
- `out/` contains most ARCHIVE_RND files and bulk frame/overlay images.
- `runs/` non-canonical plots and `last.pt` checkpoints may archive after canonical `best.pt`, `args.yaml`, and `results.csv` are secured.
- `out/object_anchor_full99/training/` duplicates canonical `models/` and `runs/` training artifacts.
- Baseline/pilot/Full99 canonical `models/.../best.pt` files are not archive candidates; they form the training lineage.

## Current-reference constraint
- `setup.ps1` executes the complete test suite. R&D runners/configs/fixture outputs imported by those tests are auto-protected.
- `configs/experiments/*.yaml` and `experiments/kpi_from_csv.py` are KEEP_REPRODUCIBILITY due test/doc references.
- Archiving these requires a coordinated setup/test/doc path change in a separately approved phase.

## Safety sequence
1. User approves exact rows from `cleanup_classification.csv`.
2. Create cleanup branch and backup tag/commit.
3. Copy/move to archive without compression.
4. Re-run import/config checks and all tests.
5. Hold archive for one user review cycle before any pruning.
