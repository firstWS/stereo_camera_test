# Proposed delete plan

No files were deleted in phase 1.

- SAFE_DELETE_CANDIDATE files: 30244
- Candidate size: 1677580299 bytes

## Low-risk categories after approval
- `__pycache__`, `*.pyc`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`.
- Generated SDK/application logs after confirming no investigation need.
- Empty ignored/generated files.
- Regenerable `out/tissue_box_front_only_val_predictions*` visualization directories after retaining model/summary evidence.

Current concentration:
- `.venv/`: 30,134 files / about 1.32 GB.
- Three ignored rotated Orbbec logs: about 300 MiB.
- Python/test caches outside `.venv`: about 1.3 MiB.
- Regenerable validation prediction images: about 41 MiB.

## Separate approval required
- `.venv/` is reproducible but deleting it interrupts local execution until setup is rerun.
- `Log/OrbbecSDK.log.txt` is tracked and modified, so it is MANUAL_REVIEW rather than a delete candidate.
- Exact-hash duplicates are not automatically deletable; dataset split/export copies may be intentional.
- ZIP/CVAT exports and all source images require provenance confirmation.
- Models/checkpoints require model inventory review even when another hash-identical copy exists.
