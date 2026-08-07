# Gemini 335L Phase 1 센서 검증

## 1. 목적과 최종 상태

이 도구는 Orbbec Gemini 335L의 RGB, Depth, Left IR, Right IR, Accelerometer,
Gyroscope가 이후 VIO / Visual-Inertial SLAM의 **입력 후보**로 쓸 수 있는지
확인하는 Phase 1 계측입니다.

Phase 1은 **센서 입력 준비 상태**를 검증합니다. VIO 정확도, 궤적 오차, SLAM
맵 품질은 범위가 아닙니다.

| 항목 | 결과 |
|------|------|
| Phase 1 종합 판정 | **PASS_WITH_WARNINGS** |
| Phase 2 진입 | **가능** (센서 입력 준비 관점) |

경고의 요지는 Translation-Yaw 초반 취급 충격으로 인한 IMU 물리 threshold 초과이며,
스트림·Timestamp·Pairing·Drop에는 영향이 없었습니다.

## 2. 변경 격리와 비범위

도구는 다음에만 존재합니다.

- `src/sensor_validation/`
- `scripts/sensor_validation/`
- `configs/sensor_validation/`
- `tests/sensor_validation/`
- `docs/phase1_gemini335l_sensor_validation.md`

다음 기존 MVP 경로는 사용·수정하지 않습니다.

- `run.ps1`
- `configs/orbbec_gemini.yaml`
- `experiments/repeatability_run.py`
- `src/orbbec_rgbd_capture.py`
- Object Anchor / AprilTag / Cup Detection / Preview 실행 경로

Phase 1에서 구현하지 않는 항목: VIO, SLAM, RGB-D Odometry, AprilTag dropout,
Object Anchor 변경, 센서 bias 추정, Allan variance, SDK native `.bag` 기록.

## 3. 실제 검증 환경

로컬 장치 검증 시점의 환경입니다. Device Serial Number는 문서에 기록하지 않습니다.

| 항목 | 값 |
|------|-----|
| 장치 | Orbbec Gemini 335L |
| Firmware | 1.4.60 |
| Hardware | 0.1 |
| USB | USB3.2 |
| OS | Windows 10.0.26200 |
| Python | 3.12.10 |
| 배포 패키지 | `pyorbbecsdk2==2.1.1` |
| import 모듈 | `pyorbbecsdk` |
| SDK Runtime | 2.8.6 |

## 4. Windows UVC Metadata 선행조건 (필수)

Windows에서 영상 `get_timestamp_us()` / `get_global_timestamp_us()`가 유효하려면
pyorbbecsdk2 공식 UVC Metadata 등록이 필요합니다.

### 4.1 등록 전 증상

- 영상 Device Timestamp 전부 0
- 영상 Global Timestamp 전부 0
- 영상 Device Timestamp Pairing 불가능
- 자동 판정 `BLOCKED` 가능

참고 Session (공식 평가 제외):
`out/sensor_validation/20260807_073216_record_static`

### 4.2 등록 명령

관리자 PowerShell에서 가상환경 Python으로 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\.venv\Lib\site-packages\pyorbbecsdk\shared\setup_env.py
```

Gemini 335L(VID `2BC5` / PID `0804`) 등록 대상 예:

- `MI_00`
- `MI_00` multipin pin 1
- `MI_00` multipin pin 2
- `MI_04`

### 4.3 등록 후 필수 절차

1. 장치 USB를 분리했다가 다시 연결합니다.
2. 짧은 Static 기록으로 Device/Global Timestamp가 0이 아닌지 확인합니다.

Metadata 효과 확인용 참고 Session:
`out/sensor_validation/20260807_080333_record_static`

### 4.4 `setup_env.py --check` 한계

Windows에서 `--check`는 **실제 Registry 등록 여부를 검증하지 않고**,
등록 스크립트 존재 여부 정도만 확인할 수 있습니다. Timestamp가 다시 0이면
등록·USB 재연결을 다시 점검하십시오.

## 5. 사용 Profile

`configs/sensor_validation/gemini335l_phase1.yaml` 기준이며 fallback은 사용하지
않았습니다.

| Stream | Profile |
|--------|---------|
| RGB | 1280×800 @ 30 Hz, RGB |
| Depth | 848×480 @ 30 Hz, Y16 |
| Left IR | 848×480 @ 30 Hz, Y8 |
| Right IR | 848×480 @ 30 Hz, Y8 |
| Accelerometer | 200 Hz / 4g |
| Gyroscope | 200 Hz / 1000 dps |

실측 rate (공식 Session 기준):

- 영상 ≈ **29.964 Hz**
- IMU ≈ **197.876 ~ 197.878 Hz**

## 6. CLI와 권장 실행 순서

```powershell
.\.venv\Scripts\python.exe scripts\sensor_validation\validate_gemini335l.py --help
```

`inspect` / `record`는 `--i-confirm-device-access`가 필요합니다. `analyze`는
카메라를 열지 않습니다.

### 6.1 Inspect

```powershell
.\.venv\Scripts\python.exe scripts\sensor_validation\validate_gemini335l.py inspect --i-confirm-device-access --device-index 0
```

### 6.2 Static 60초

```powershell
.\.venv\Scripts\python.exe scripts\sensor_validation\validate_gemini335l.py record --mode static --duration 60 --i-confirm-device-access --device-index 0 --no-preview
```

### 6.3 Translation 15초

```powershell
.\.venv\Scripts\python.exe scripts\sensor_validation\validate_gemini335l.py record --mode translation --duration 15 --i-confirm-device-access --device-index 0 --no-preview
```

동작: 약 3초 정지 → 오른쪽 약 0.5 m 횡이동 → 종료까지 정지.
터미널 강조 문구와 Windows 비프(짧은 비프=카운트다운, 긴 비프 3회=이동 시작,
중간 비프 2회=정지)를 따릅니다.

### 6.4 Translation-Yaw 15초

```powershell
.\.venv\Scripts\python.exe scripts\sensor_validation\validate_gemini335l.py record --mode translation-yaw --duration 15 --i-confirm-device-access --device-index 0 --no-preview
```

동작: 약 3초 정지 → 횡이동과 누적 Yaw 약 15~30° → 종료까지 정지.
Pitch/Roll·급회전·90° 전환은 피합니다.

## 7. 이동 안내 기능

`sensor_recorder.py`의 안내 기능은 **부가 UX**입니다.

- Static: 이동 비프 없음
- Translation / Translation-Yaw만 적용
- `winsound` lazy import, 실패 시 terminal bell fallback
- 예외는 Recorder로 전파하지 않음
- 메인 루프에서만 실행 (callback / writer와 분리)
- `duration` deadline(`perf_counter` 기준)을 변경하지 않음
- Ctrl+C 안전 종료 유지

## 8. 공식 Session (로컬 검증 식별용)

아래 경로는 Git에 포함되지 않는 로컬 `out/` Session입니다. Serial 등 민감 정보는
문서화하지 않습니다.

| 구분 | 로컬 Session 경로 |
|------|-------------------|
| Inspect | `out/sensor_validation/20260807_072621_inspect` |
| Static 60초 | `out/sensor_validation/20260807_081447_record_static` |
| Translation | `out/sensor_validation/20260807_085908_record_translation` |
| Translation-Yaw | `out/sensor_validation/20260807_091131_record_translation-yaw` |

### 8.1 공식 평가 제외 Session

| Session | 제외 사유 |
|---------|-----------|
| `out/sensor_validation/20260807_073216_record_static` | UVC Metadata 등록 전. 영상 Device Timestamp 전부 0. 환경 문제 확인용 |
| `out/sensor_validation/20260807_085455_record_translation` | 사용자가 이동 시점을 인지하지 못해 실제 이동 미실시 |
| `out/sensor_validation/20260807_080333_record_static` | Metadata 등록 후 효과 확인용 참고(10초). 공식 Static은 60초 Session |

## 9. Inspect 결과 요약

- 장치·profile 열거와 Factory calibration 조회 성공
- Camera–IMU extrinsic: **AVAILABLE** (identity 대체 없음)
- 제한된 probe 조합 성공
- Distortion model 이름은 SDK가 노출하지 않을 수 있음

## 10. Static 60초 결과 요약

Session: `20260807_081447_record_static`

- RGB/Depth/Left IR/Right IR 각 1,784 frame
- Accel/Gyro 각 11,840 sample
- 영상·IMU Device/Global/System Timestamp 0건 없음
- Timestamp 역행·중복 0, Frame/Sample 누락 0
- Queue Overflow 0, Writer/Callback/Stop 오류 0
- 영상 ≈ 29.964 Hz, IMU ≈ 197.876 Hz
- RGB–Depth Device Pairing median ≈ 19 µs
- Left–Right IR Device Pairing median ≈ 0 µs
- 영상–IMU Device Pairing median ≈ 1.26 ms 수준
- 물리적 Gyro/Accel Spike 0 (robust outlier만 소수)
- 자동 판정 PASS, 엔지니어링 PASS

## 11. Translation 결과 요약

Session: `20260807_085908_record_translation`

- 6스트림 정상 수신, Timestamp/Drop/Overflow/오류 Static과 동등
- 이동 구간 Gyro Norm이 정지 대비 상승 후 최종 정지에서 복귀
- 추적 불가급 Motion Blur 없음
- 자동·엔지니어링 PASS

## 12. Translation-Yaw 결과 요약

Session: `20260807_091131_record_translation-yaw`

- 스트림·Timestamp·Pairing·Drop은 Static/Translation과 동등하게 안정
- 이동+Yaw 구간 Gyro Norm이 Translation 단독보다 명확히 큼
- 최종 정지에서 IMU가 정지 수준으로 복귀
- **경고**: 초기 준비 구간(~1.6~3.1 s) 취급/조기 움직임으로
  Gyro 물리 threshold(1.0 rad/s) 초과 8건, Accel(20 m/s²) 초과 2건
- 센서 고장 패턴이 아니라 물리적 취급 충격으로 판단
- 스트림·Pairing·Drop에는 영향 없음
- 자동 `validation_summary` PASS, 엔지니어링 **PASS_WITH_WARNINGS**

## 13. Timestamp·Pairing·IMU 요약

| 항목 | 결과 |
|------|------|
| Device Timestamp | Metadata 등록 후 정상 |
| 역행 / 중복 / Drop | 공식 Session에서 0 |
| RGB–Depth Device median | ≈ 19 µs |
| Left–Right IR Device median | ≈ 0 µs |
| 영상–IMU Device median | ≈ 1.25 ~ 1.28 ms |
| Queue Overflow | 0 |
| Camera–IMU Extrinsic | AVAILABLE |
| Static IMU | 정지 안정, 물리 Spike 0 |
| Translation IMU | 횡이동 반응 후 복귀 |
| Translation-Yaw IMU | 회전 반응 명확, 초반 취급 경고 |

Device clock과 Host clock은 혼합 분석하지 않습니다. 자동 summary의
Video–IMU device pairing은 별도 pipeline 호환성 미검증 시 `UNAVAILABLE`로
남을 수 있으나, 동일 Device Timestamp 도메인에서의 nearest pairing은 실측상
약 1.25~1.28 ms median으로 안정적이었습니다.

## 14. 기록·분석 구조 (구현 요약)

- 영상 / IMU **별도** callback pipeline
- Host: `host_monotonic_ns` (`perf_counter_ns`), `host_wall_time_ns` (`time_ns`)
- Device/System/Global timestamp 원본 저장
- Bounded queue + writer thread
- 기본은 대표 frame만 저장 (`--save-all-frames`는 대용량)

출력 루트 `out/sensor_validation/`는 `.gitignore`의 `out/`으로 Git에서 제외됩니다.
`Log/`도 제외 대상이며 `Log/OrbbecSDK.log.txt`는 커밋하지 않습니다.

## 15. 알려진 한계

1. Phase 1은 VIO/SLAM 정확도를 검증하지 않습니다.
2. Windows UVC Metadata 미등록 시 영상 Device Timestamp가 0이 됩니다.
3. `setup_env.py --check`만으로 Registry 등록을 보증할 수 없습니다.
4. 이동 비프는 메인 루프를 짧게 동기 차단할 수 있으나 deadline·callback 경로를
   바꾸지 않습니다.
5. Host Timestamp는 batching으로 interval이 불규칙해 보일 수 있습니다.
   Device Timestamp를 권위 시계로 사용합니다.
6. Depth 대표 PNG는 raw uint16라 시각적으로 어둡게 보일 수 있습니다.
7. Distortion model 이름 등 SDK 미노출 값은 `null` / `NOT_EXPOSED`로 남깁니다.
8. Translation-Yaw 초반 취급 충격은 PASS_WITH_WARNINGS 사유입니다.

## 16. Phase 2 진입과 다음 단계

Phase 1 종합 **PASS_WITH_WARNINGS**이므로 **Phase 2 진입은 가능**합니다.
권장 다음 단계는 별도 승인 후 진행합니다.

- VIO / Visual-Inertial 파이프라인 설계 (기존 MVP와 격리 유지)
- Camera–IMU extrinsic을 입력으로 쓰는 추정기 연동
- 장시간·반복 운동 시나리오와 궤적 평가 (Phase 1 범위 밖)

Camera–IMU extrinsic이 `AVAILABLE`이므로 extrinsic 부재로 Phase 2를 막지 않습니다.

## 17. 비카메라 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest tests\sensor_validation -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

합성 timestamp/IMU, fake pipeline, queue overflow, CLI confirmation, 이동 안내
비프 안전성 등을 검사합니다. live 명령은 confirmation 없이 SDK를 열지 않습니다.
