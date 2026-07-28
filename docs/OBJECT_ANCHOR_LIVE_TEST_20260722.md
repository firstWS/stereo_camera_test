# Object Anchor live test (2026-07-22)

## Scope

- Orbbec RGB-D live input
- Existing AprilTag world localization kept active
- FRONT-only tissue-box YOLO-Pose model executed in parallel
- Object Anchor world pose used only for comparison/registration diagnostics
- 300-frame non-registration run

## Result

| Metric | Result |
| --- | ---: |
| Frames | 300 |
| Duration | 105.95 s |
| Effective throughput | 2.83 FPS |
| AprilTag detection success | 77.0% |
| Object Anchor detection success | 3.33% (10/300) |
| PnP success | 0% |
| Final world-pose success | 0% |
| Skeleton crossings | 1 |
| Saved failure cases | 9 raw/overlay pairs |

The runtime, transformation, statistics, and failure-capture paths completed without
interrupting the existing AprilTag pipeline. Registration was intentionally not run:
there were no valid PnP poses to register.

Visual inspection of saved failure overlays shows that the pilot model frequently
localized the office chair or background as the tissue box. The actual anchor also
occupies very few pixels in the tested view. The current `best.pt` therefore is not
ready to create a trustworthy world-pose calibration file.

## Rejection summary

- `no_object_anchor_detection`: 290
- `insufficient_correspondences:2<4`: 3
- `insufficient_correspondences:1<4`: 1
- `skeleton_crossing`: 1
- reprojection error over 5 px: 5

Observed reprojection error (six computable frames): mean 12.19 px, maximum 21.59 px.

## Artifacts

- Session CSV: `out/object_anchor_world/20260722_144211/object_anchor_world_frames.csv`
- Final summary: `out/object_anchor_world/20260722_144211/object_anchor_world_summary_final.json`
- Failure images: `out/object_anchor_world/20260722_144211/failures/`
- Existing AprilTag/cup CSV: `out/repeatability_orbbec.csv`

## Next acceptance gate

Before running registration, collect and label real Orbbec frames from the fixed
installation. Include the chair/background as negative examples and make the tissue
box larger in-frame if the installation permits. A replacement model should first
produce four consistent keypoints, no skeleton crossing, at least four PnP inliers,
and reprojection error at or below 5 px for a sustained live sequence.
