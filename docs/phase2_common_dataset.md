# Gemini 335L Phase 2 공통 Recording Dataset

## 1. 목적

Phase 2는 Orbbec Gemini 335L에서 **공통 Recording Dataset**을 수집·검증·재생하기 위한
격리된 도구입니다. VIO, SLAM, Object Anchor 런타임 통합은 범위가 아닙니다.

| 항목 | 내용 |
|------|------|
| Raw authoritative | `streams/`, `calibration/`, metadata JSON/CSV |
| Derived | `derived/apriltag/`, `derived/cups/` — `derive` CLI만 생성 |
| 출력 루트 | `out/datasets/gemini335l/<session_id>/` |
| Scenario | `scenario_a`, `scenario_b` |

## 2. 변경 격리

Phase 2 코드는 다음에만 존재합니다.

- `src/dataset_recorder/`
- `scripts/dataset/record_gemini335l.py`
- `configs/dataset/`
- `tests/dataset_recorder/`
- `docs/phase2_common_dataset.md`

Phase 1 (`src/sensor_validation/`)은 **import만** 재사용하며 동작을 변경하지 않습니다.
MVP 런타임 경로(`run.ps1`, `orbbec_rgbd_capture.py`, Object Anchor 등)는 수정하지 않습니다.

## 3. Dataset 디렉터리 구조

```
out/datasets/gemini335l/<session_id>/
  session.json
  scenario.json
  device_info.json
  selected_profiles.json
  recording_state.json
  integrity.json
  events.csv
  calibration/
    intrinsics.json
    extrinsics.json
    camera_imu.json
  streams/
    rgb/{index.csv,frames/}
    depth/{index.csv,frames/}
    left_ir/{index.csv,frames/}
    right_ir/{index.csv,frames/}
    accel.csv
    gyro.csv
  derived/                 # derive CLI만 생성
    manifest.json
    apriltag/observations.csv
    cups/
      detections.csv
      tracks.csv
      observations.csv
      track_summary.json
    annotations/
      objects.json           # optional, session-level semantic mapping
```

## 4. Raw / Derived 분리

- **Raw**: 센서 프레임·IMU·캘리브레이션·메타데이터. authoritative source.
- **Derived**: AprilTag 관측, Cup 검출·추적·semantic observation. 오프라인 `derive`만 생성하며 recording hot path에 포함되지 않습니다.

### Cup derived 계층 (detection / track / semantic)

| 파일 | 개념 | 설명 |
|------|------|------|
| `cups/detections.csv` | Detection | frame-local YOLO 결과. `detection_index`는 confidence 내림차순 등 deterministic order |
| `cups/tracks.csv` | Track | offline association 결과. `track_id`는 session-local |
| `cups/observations.csv` | Semantic observation | `track_id` + `semantic_id` (`cup1`/`cup2`/`unknown`) |
| `cups/track_summary.json` | Hint only | track 요약, MOT diagnostic (`mot` aggregate), `cup1_candidate`/`cup2_candidate` hint (자동 truth 아님) |
| `annotations/objects.json` | Annotation | 사용자 확인용 `track_id` ↔ `cup1`/`cup2` 매핑 (optional) |

`detection_index`, `track_id`, `semantic_id`는 서로 다른 개념입니다. Scenario YAML의 `planned_motion_windows`는 semantic 결정에 사용하지 않습니다.

Cup tracking은 lightweight MOT 구조를 사용합니다.

- Hungarian global assignment (`scipy.optimize.linear_sum_assignment`)
- Track lifecycle: `TENTATIVE` → `CONFIRMED` → `LOST` → `DELETED`
- Lost track reactivation within `derive.cup_mot.max_lost_frames`
- Optional world-position cue when AprilTag pose + depth are available (image-only fallback otherwise)

Cup depth는 native Depth를 RGB bbox에 **geometric projection**으로 샘플합니다 (`derive` config: `max_rgb_depth_delta_us`, `depth_is_millimeters`). RGB `frame_number`와 동일한 Depth `frame_number`를 가정하지 않으며, `device_timestamp_us` nearest matching을 사용합니다. `detections.csv` / `observations.csv`에 `depth_frame_number`, `depth_device_timestamp_us`, `rgb_depth_delta_us` provenance가 포함됩니다.

## 5. Schema 요약

### `session.json` (schema v1)

- `recorder.tool` / `recorder.version`
- `device`, `sdk`, `profiles`
- `recording`, `scenario` 요약
- `status`, `integrity_status`

### `scenario.json` (schema v1)

- `planned_*` 필드만 포함 (ground truth 아님)
- `scenario_slug`: `scenario_a` | `scenario_b`

| Scenario | motion | yaw | translation | cup2 |
|----------|--------|-----|-------------|------|
| A | rightward_yaw (handheld pan) | ~25° planned (20–30°) | null (incidental only) | initially_hidden |
| B | translation_yaw | 15–30° | planned_translation_m set | initially_hidden |

Scenario A timing (planned operator window, not ground truth):

- 0–3s: hold (AprilTag + Cup1 visible, Cup2 hidden)
- 3–5s: rightward yaw/pan (~20–30°)
- 5–15s: final hold (Cup2 visible)

Recorder prints Scenario A motion cues at t≈0 / 2 / 3 / 5. It must not ask for a 0.30 m translation.

## 6. 저장 형식

Timestamp 필드는 Phase 1과 동일합니다.

- `device_timestamp_us`, `system_timestamp_us`, `global_timestamp_us`
- `host_monotonic_ns`, `host_wall_time_ns`, `frame_number`

| Stream | 형식 |
|--------|------|
| RGB / IR | PNG (lossless) |
| Depth | uint16 PNG, resize 없음 |
| IMU | `accel.csv`, `gyro.csv` 분리 |

## 7. CLI

```powershell
python scripts/dataset/record_gemini335l.py --help
python scripts/dataset/record_gemini335l.py record --help
python scripts/dataset/record_gemini335l.py validate --help
python scripts/dataset/record_gemini335l.py derive --help
```

### record (실제 카메라 필요)

```powershell
python scripts/dataset/record_gemini335l.py record `
  --i-confirm-device-access `
  --scenario scenario_a
```

### validate

```powershell
python scripts/dataset/record_gemini335l.py validate `
  --session out/datasets/gemini335l/<session_id>
```

### derive (오프라인)

```powershell
python scripts/dataset/record_gemini335l.py derive `
  --session out/datasets/gemini335l/<session_id>
```

`derive.apriltag_world.enabled=false`이면 CLI가 경고를 출력하고 AprilTag derived row는 생성되지 않습니다.

derive 후 semantic annotation 예시 (`derived/annotations/objects.json`):

```json
{
  "schema_version": 1,
  "objects": {
    "cup1": {"description": "red mug", "visible_initially": true, "track_id": "track_0001"},
    "cup2": {"description": "transparent cup", "visible_initially": false, "track_id": "track_0002"}
  }
}
```

annotation 작성 후 derive를 다시 실행하면 `observations.csv`의 `semantic_id`가 반영됩니다.

## 8. Integrity 판정

`validate_dataset_session()`은 Phase 1 `timestamp_analysis`를 재사용하고,
index 행 수와 frame 파일 수 일치 여부를 검사합니다.

| 판정 | 조건 |
|------|------|
| VALID | blocker/warning 없음 |
| WARNING | blocker 없음, warning 있음 |
| INVALID | 필수 파일 누락, timestamp 오류, frame 수 불일치, recorder 오류 등 |

## 9. DatasetReader

`DatasetReader`는 오프라인 재생용 API입니다.

- `iterate_rgb()`, `iterate_depth()`, `iterate_left_ir()`, `iterate_right_ir()`
- `iterate_accel()`, `iterate_gyro()`
- `stream_counts()`, `derived_manifest()`

## 10. Git / 출력 정책

- Serial number, session raw data는 Git에 포함하지 않습니다.
- `out/` 디렉터리는 `.gitignore` 대상을 유지합니다.
