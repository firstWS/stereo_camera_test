# Object Anchor repository cleanup — phase 1 summary

Generated UTC: 2026-07-29T04:00:30.347267+00:00

This is a read-only classification report. No project file was deleted, moved, reset, or reverted. Camera was not opened.

## Inventory totals
- Files: **33538**
- Bytes: **2967560465**
- Git tracked: **99**
- Git untracked: **0**
- Git ignored: **33439**
- Runtime dependency/protected closure: **81**
- Direct live runtime subset: **29**
- Scope excludes `.git/` internals and the newly generated `cleanup_reports/` directory itself, preventing recursive self-inventory.
- The report directory is currently untracked; its CSV files match the repository-wide CSV ignore rule and need an explicit Git policy decision before committing.

## Classification
- KEEP_RUNTIME: 82 files / 26578926 bytes
- KEEP_REPRODUCIBILITY: 1397 files / 515343305 bytes
- ARCHIVE_RND: 1814 files / 738400496 bytes
- SAFE_DELETE_CANDIDATE: 30244 files / 1677580299 bytes
- MANUAL_REVIEW: 1 files / 9657439 bytes

## Duplicate content
- Duplicate groups: **571**
- Files participating: **1654**
- Theoretical maximum savings: **359274401 bytes**
- This is not an automatic deletion recommendation.
- Excluding groups wholly inside `.venv`, 417 duplicate groups / 1,197 files remain. Most large groups are intentional copies across source, pilot, and Full99 dataset layouts.

## Key findings
- Active runtime models: `yolo11s.pt` (Cup) and `models/object_anchor/tissue_box_01_front_only_orbbec_full99/best.pt` (Object Anchor).
- Baseline, Orbbec pilot, and Full99 canonical `models/.../best.pt` files are KEEP_REPRODUCIBILITY because Full99 was fine-tuned through that lineage.
- Full99 active `best.pt` SHA-256 is `2512be192c87242633b5aec4ceba332937a51c82e3853bfbde83cdd8300dd198`; exact copies exist under `runs/.../weights/best.pt` and `out/.../training/weights/best.pt`.
- Full99 final dataset at `data/datasets/tissue_box_front_orbbec_full99` has 223 images and 223 labels with zero internal train/val exact-hash overlap.
- Data caveats: Orbbec train/val are from the same capture session, and pilot validation images are included in Full99 training. These affect metric interpretation, not cleanup eligibility.
- Full99 reproduction also requires CVAT v2/v3 sources, 100 negative captures, legacy 24 images, valid99 source, pilot initialization, dataset YAML/manifests, training args/results, and preparation/verification scripts.
- Latest chronologically created Preview run (`20260729_123354`) is incomplete; latest successful 30-sample demo evidence is `out/object_anchor_preview/20260728_120922`.
- Median3 evidence is `out/object_anchor_full99/mvp_translation_diagnostics/20260726_164854` (B_median3 verdict A).
- Three MVP output directories are current pytest fixtures and therefore KEEP_RUNTIME, despite being historical live/diagnostic results.
- `.venv/` accounts for 30,134 SAFE_DELETE files and about 1.32 GB, but removing it interrupts execution until `setup.ps1` recreates it.
- Three ignored rotated Orbbec logs are low-risk candidates (about 300 MiB). Tracked modified `Log/OrbbecSDK.log.txt` remains MANUAL_REVIEW.

## Largest 20 files
1. `.venv/Lib/site-packages/torch/lib/torch_cpu.dll` — 253.51 MiB — SAFE_DELETE_CANDIDATE
2. `.venv/Lib/site-packages/_polars_runtime_32/_polars_runtime.pyd` — 175.87 MiB — SAFE_DELETE_CANDIDATE
3. `Log/OrbbecSDK.log.2.txt` — 100.00 MiB — SAFE_DELETE_CANDIDATE
4. `Log/OrbbecSDK.log.1.txt` — 100.00 MiB — SAFE_DELETE_CANDIDATE
5. `Log/OrbbecSDK.log.3.txt` — 100.00 MiB — SAFE_DELETE_CANDIDATE
6. `.venv/Lib/site-packages/cv2/cv2.pyd` — 71.35 MiB — SAFE_DELETE_CANDIDATE
7. `.venv/Lib/site-packages/torch/lib/torch_cpu.lib` — 28.29 MiB — SAFE_DELETE_CANDIDATE
8. `.venv/Lib/site-packages/cv2/opencv_videoio_ffmpeg4130_64.dll` — 27.25 MiB — SAFE_DELETE_CANDIDATE
9. `.venv/Lib/site-packages/numpy.libs/libscipy_openblas64_-63c857e738469261263c764a36be9436.dll` — 19.47 MiB — SAFE_DELETE_CANDIDATE
10. `.venv/Lib/site-packages/scipy.libs/libscipy_openblas-197ee2fc9b4d071f7e048078cac74115.dll` — 19.32 MiB — SAFE_DELETE_CANDIDATE
11. `.venv/Lib/site-packages/av.libs/libx265-668b2e71a6cd3b80e84445adafe9696a.dll` — 19.20 MiB — SAFE_DELETE_CANDIDATE
12. `yolo11s.pt` — 18.42 MiB — KEEP_RUNTIME
13. `.venv/Lib/site-packages/torch/lib/torch_python.dll` — 17.92 MiB — SAFE_DELETE_CANDIDATE
14. `.venv/Lib/site-packages/pyorbbecsdk/examples/applications/object_detection/models/yolov5s.onnx` — 14.02 MiB — SAFE_DELETE_CANDIDATE
15. `.venv/Lib/site-packages/av.libs/avcodec-60-603dc320701874565ad4f886b49958fe.dll` — 13.60 MiB — SAFE_DELETE_CANDIDATE
16. `Log/OrbbecSDK.log.txt` — 9.21 MiB — MANUAL_REVIEW
17. `.venv/Lib/site-packages/av.libs/libaom-3c98a6db777c6c7fedeac4ae87f4314b.dll` — 8.53 MiB — SAFE_DELETE_CANDIDATE
18. `.venv/Lib/site-packages/torch/lib/sleef.lib` — 8.40 MiB — SAFE_DELETE_CANDIDATE
19. `.venv/Lib/site-packages/pyorbbecsdk/OrbbecSDK.dll` — 8.16 MiB — SAFE_DELETE_CANDIDATE
20. `.venv/Lib/site-packages/PIL/_avif.cp312-win_amd64.pyd` — 7.53 MiB — SAFE_DELETE_CANDIDATE

## Git state at snapshot
- `Log/OrbbecSDK.log.txt`: ` M`
- `cleanup_reports/audit_repository_cleanup.py`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/cleanup_summary.md`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/proposed_archive_plan.md`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/proposed_delete_plan.md`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/proposed_final_structure.md`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/protected_files.txt`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/referenced_files.json`: `??`
- `cleanup_reports/object_anchor_cleanup_20260729_124549/runtime_dependency_graph.md`: `??`

## Test status
- Python compile (`src`, `experiments`, `scripts`, `tests`): **PASS**
- Complete static regression suite: **88 passed**
- Object Anchor Preview/Object Anchor focused suite: **46 passed**
- `configs/orbbec_gemini.yaml` load and protected Preview flags: **PASS**
- Static `run.ps1 -Orbbec` path check: **PASS**
- Camera/live execution: **NOT RUN**

## Required approval before phase 2
- Exact SAFE_DELETE rows, especially `.venv/` and logs.
- Which historical out/runs directories may move to `archive_local`.
- Pilot/old model provenance and retention period.
- Dataset source/CVAT/legacy ownership and archive layout.
- Commit/backup tag strategy for currently modified user files.
