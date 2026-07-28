# Tissue Box Object Anchor Pilot

## FRONT-only MVP

The first MVP replaces the full cuboid with the four physical FRONT corners. The
box must remain fixed, keep its FRONT facing the camera, and use a pre-registered
world pose. Multi-view recognition and cuboid symmetry are intentionally deferred.

Canonical files:

- Anchor geometry: `configs/object_anchors/tissue_box_01_front_only.yaml`
- CVAT order/skeleton: `configs/cvat/tissue_box_01_front_only_skeleton.yaml`
- Ultralytics dataset: `configs/datasets/tissue_box_front_only_pose.yaml`

The YOLO row contains 17 values: class, four bbox values, then four `(x,y,v)`
keypoints. The fixed order is FRONT top-left, top-right, bottom-right, bottom-left.
Do not enable horizontal or vertical flips because they change semantic IDs.

## 20-30 image selection

Start with 24 independently selected images. Nearby frames from one capture should
not be split between train and validation.

| Group | Count |
| --- | ---: |
| Near-frontal placement/lighting variation | 6 |
| Small left perspective, FRONT still visible | 4 |
| Small right perspective, FRONT still visible | 4 |
| Distance variation | 4 |
| Exposure/background variation | 4 |
| Mild occlusion or difficult cases | 2 |

Use about 18 training and 6 validation images. Hold out an additional capture
session later for the AprilTag comparison; model accuracy is not the first gate.

Sanitize the CVAT export before arranging the train/validation folders:

```powershell
.\.venv\Scripts\python.exe scripts\sanitize_yolo_pose_dataset.py `
  --input "C:\path\to\front_only_export.zip" `
  --images-dir "C:\path\to\images" `
  --anchor-config configs\object_anchors\tissue_box_01_front_only.yaml `
  --output out\tissue_box_front_only_sanitized `
  --output-zip
```

After review and split, train the pilot:

```powershell
.\.venv\Scripts\python.exe scripts\train_object_anchor_pose.py `
  --data configs\datasets\tissue_box_front_only_pose.yaml `
  --epochs 80
```

Copy `runs/object_anchor_pose/tissue_box_01_front_only_pilot/weights/best.pt` to
`models/object_anchor/tissue_box_01_front_only/best.pt`. Then set
`object_anchor.enabled: true` in `configs/orbbec_gemini.yaml`. The runtime checks
that the model returns exactly four keypoints before calling the existing PnP,
temporal validation, visualizer, and AprilTag-reference path.

## 8-point follow-up selection

Use 40 representative images before labeling the full 8-point dataset.

| Group | Count |
| --- | ---: |
| Front | 5 |
| Left oblique | 10 |
| Right oblique | 10 |
| Top oblique | 5 |
| Distance variation | 5 |
| Occlusion / difficult | 5 |

Do not randomly mix adjacent video frames across train and validation. Keep capture
sessions separated. For only 40 pilot images, use approximately 30 train and 10
validation images; keep the later 210/45/45 split for the complete 300-image set.

## Label rules

The canonical ID/name/XYZ order is `configs/object_anchors/tissue_box_01.yaml`.

- Use visibility 2 only when the physical corner is clearly visible.
- Use visibility 1 when partially occluded but its location is still reliable.
- Use visibility 0 when hidden or uncertain. Do not estimate a rear corner by eye.
- Check FRONT using the upright English Kleenex logo and TOP using the opening.
- Do not reorder left/right when labeling the BACK face. IDs use the fixed object frame.
- Review labels with the configured skeleton before training.

The pilot is for checking ID consistency, FRONT/BACK orientation, visible-corner
detection, PnP axis direction, and live Pose availability. It is not an accuracy
benchmark.

## 24-image FRONT-only pilot result

The first pilot used 18 training images and 6 validation images for 80 epochs.
Ultralytics selected epoch 75 as `best.pt` using the sum of box and pose
mAP50-95. Validation results for that checkpoint were:

| Metric | Bbox | Keypoint pose |
| --- | ---: | ---: |
| Precision | 0.852 | 0.852 |
| Recall | 1.000 | 1.000 |
| mAP50 | 0.972 | 0.972 |
| mAP50-95 | 0.592 | 0.762 |

These metrics are highly uncertain because validation contains only six related
images. A direct keypoint-to-label comparison averaged 259.5 pixels, or 11.4% of
the labeled bbox diagonal, and the most oblique image averaged 20.5%. The model
therefore demonstrates detector/runtime connectivity but is not yet accurate
enough to claim stable PnP or AprilTag replacement. Add independent sessions and
hard oblique views, then gate acceptance on pixel/reprojection and pose error in
addition to Ultralytics mAP.

The ID overlay also shows a crossed predicted skeleton on `tissue_FTR_005` and
large corner offsets on `tissue_FTL_005`. The reviewed ground-truth overlays do
not have those errors, so this is treated as a model/data-volume limitation rather
than an automatic label-remapping problem.

The deployed pilot weight is
`models/object_anchor/tissue_box_01_front_only/best.pt`. Object Anchor remains
an AprilTag comparison-only input and never replaces the existing world pipeline.

## Live AprilTag world validation and registration

Normal comparison run:

```powershell
.\run.ps1 -Orbbec
```

Collect 100 valid simultaneous AprilTag + Object Anchor frames and save a robust
world registration:

```powershell
.\run.ps1 -Orbbec -RegisterObjectAnchor
```

The transform convention is printed and logged exactly as:

```text
T_world_object = T_world_camera @ T_camera_object
T_world_camera = T_world_tag @ inverse(T_camera_tag)
```

`camera_pose_only: true` means the Object Anchor result remains comparison-only.
The existing AprilTag world result is not replaced or corrected by this path.

Outputs:

- Registration: `out/object_anchor_calibration/tissue_box_01_world_pose.yaml`
- Per-run frame CSV: `out/object_anchor_world/<timestamp>/object_anchor_world_frames.csv`
- 10-second and final statistics JSON: `out/object_anchor_world/<timestamp>/object_anchor_world_summary*.json`
- Failed raw/overlay frames: `out/object_anchor_world/<timestamp>/failures/`

Registration accepts only four confident, non-crossing keypoints with a valid PnP,
four inliers, positive depth, bounded reprojection error, and bounded temporal
translation/rotation changes. Position uses a median and rotation uses a quaternion
average. The registration YAML stores position, `(x,y,z,w)` quaternion, rotation
matrix, full homogeneous transform, used frame count, and excluded reasons.

## Sanitize CVAT export

Run the sanitizer before training. `--input` accepts either a CVAT export ZIP or a
directory containing YOLO Pose label files. Images may be inside the input dataset
or supplied separately with `--images-dir`.

```powershell
.\.venv\Scripts\python.exe scripts\sanitize_yolo_pose_dataset.py `
  --input "C:\path\to\cvat_export.zip" `
  --images-dir "C:\path\to\images" `
  --anchor-config configs\object_anchors\tissue_box_01.yaml `
  --output "out\tissue_box_pose_sanitized" `
  --output-zip
```

The sanitizer requires 29 values per row, validates normalized bbox and visible
keypoint coordinates, and rewrites every visibility-0 keypoint to `0.0 0.0 0`.
It also checks image/label stems, the eight-keypoint Object Anchor definition,
named-view visible IDs, and crossing skeleton edges. Verification overlays and
`sanitation_report.json` are written under the output directory. The ZIP is created
only when `training_ready` is true; fix every reported label error before training.

## Safe augmentation

`scripts/train_object_anchor_pose.py` disables horizontal flip, vertical flip,
perspective, shear, mosaic, mixup, and copy-paste. It allows only small rotation,
small translation/scale, and moderate brightness/saturation variation. Add weak
blur/noise only after the clean pilot succeeds; keep a portion of validation images
unaugmented.

## Dataset layout

```text
data/tissue_box_pose/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

Train after the 40 labels have been reviewed:

```powershell
.\.venv\Scripts\python.exe scripts\train_object_anchor_pose.py `
  --data configs\datasets\tissue_box_pose.yaml `
  --name tissue_box_01_cuboid_8point_pilot `
  --epochs 100
```

Copy the resulting `best.pt` to
`models/object_anchor/tissue_box_01/best.pt`, set `object_anchor.enabled: true`,
and run `run.ps1 -Orbbec`. The first live check remains camera-frame Pose only.
