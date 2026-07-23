# Object Anchor Orbbec capture

The capture path is separate from the existing 300-frame repeatability run. It starts
with the first synchronized Orbbec frame and exits according to the number of images
successfully written, not the number of processed frames.

## Commands

```powershell
# Defaults: 100 images, one image per second
.\run.ps1 -Orbbec -Capture -Positive
.\run.ps1 -Orbbec -Capture -Negative

# Overrides
.\run.ps1 -Orbbec -Capture -Positive -CaptureCount 150 -CaptureInterval 0.5
.\run.ps1 -Orbbec -Capture -Negative -CaptureCount 200 -CaptureInterval 1
```

Exactly one of `-Positive` and `-Negative` is required. Capture also requires
`-Orbbec`. `-RegisterObjectAnchor` cannot be combined with capture mode.

## Output

```text
data/object_anchor_capture/
|-- positive/
|   `-- images/
|-- negative/
|   |-- images/
|   `-- labels/
`-- capture_manifest.csv
```

Positive capture writes only JPEG images for later CVAT annotation. Negative capture
also writes a zero-byte `.txt` file with the same stem. Existing files are never
overwritten, including when two sessions begin during the same second.

The JPEG is the unannotated `OrbbecFrame.bgr` array used as model input. Its original
resolution and orientation are preserved. Detection and AprilTag results are metadata
only and never gate a save.

## Controls

- `Q` or `Esc`: stop early
- Close the OpenCV window: stop early
- `Ctrl+C`: stop early

All completed images, empty labels, and manifest rows remain valid after an early stop.
