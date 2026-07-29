# Proposed final structure (proposal only)

No directory moves were performed. Moving these paths requires coordinated import/config/doc updates.

```text
src/                         # runtime and reusable modules
experiments/                 # maintained entrypoints (repeatability_run.py)
scripts/
  data/                      # preparation/sanitation
  training/                  # Full99 training/reproduction
  diagnostics/               # retained active diagnostics only
configs/
  runtime/                   # Orbbec/default runtime configs
  training/                  # dataset/training configs
  archived_experiments/      # historical MVP/branch configs
models/
  active/                    # Full99 best + active Cup detector assets
  archived/                  # pilot/legacy checkpoints
data/
  source/                    # original captures/CVAT exports
  active/                    # final Full99 training dataset
  archived/                  # legacy/pilot datasets
tests/
  runtime/
  training/
  archived_rnd/
docs/
  mvp/
  runtime/
  model_data/
out/
  latest_preview/
  retained_summaries/
archive_local/rnd_<timestamp>/
```

## Expected coordinated changes in phase 2
- Update Python imports only if modules move.
- Update YAML model/config/output paths.
- Update PowerShell entrypoint and README commands.
- Add migration map and path-compatibility check.
- Keep `run.ps1 -Orbbec` behavior unchanged.
