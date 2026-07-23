# stereo_camera_test



USB 스테레오(SBS) 캡처 → 보정·정류 → SGBM 시차 → 검출(YOLO) → 3D 추정 **또는** Orbbec SDK RGB-D 정렬 깊이 → YOLO → median depth 역투영 실험 코드입니다.


## 확정 실행 환경

Windows x64에서 아래 조합을 동일한 `.venv`로 검증합니다. Python 3.12는
Orbbec 공식 Windows wheel 지원 범위(3.8~3.13)에 포함되며, 3.12.10은 Python
3.12 계열의 마지막 Windows 전체 설치본입니다.

| Component | Version |
| --- | --- |
| Python | 3.12.10 (64-bit) |
| OpenCV GUI | 4.13.0.92 |
| Ultralytics | 8.4.46 |
| Orbbec binding (`pyorbbecsdk2`) | 2.1.1 |
| NumPy | 2.4.4 |
| PyYAML | 6.0.3 |
| pytest | 8.4.2 |
| PyTorch / torchvision | 2.11.0+cpu / 0.26.0 |

`requirements.txt`는 이 버전을 고정합니다. OpenCV 미리보기가 필요하므로
`opencv-python-headless`가 아니라 `opencv-python`을 사용합니다.

환경을 처음 만들거나 깨진 `.venv`를 복구할 때:

```powershell
$env:STEREO_POC_PYTHON = "C:\Users\<user>\AppData\Local\Programs\Python\Python312\python.exe"
.\setup.ps1
```

`setup.ps1`은 Python 3.12 확인, 패키지 설치, 기존 smoke test, 환경 import,
pytest, Object Anchor 합성 PnP 테스트를 순서대로 실행합니다. 개별 확인 명령은
다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\synthetic_object_anchor_test.py
.\.venv\Scripts\python.exe scripts\orbbec_smoke_test.py --frames 3 --timeout-s 20
```



## 빠른 시작 (Windows)



저장소 루트에서 PowerShell로 실행합니다.



```powershell

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # 한 번만, 스크립트 허용 시

.\run.ps1

```



- `.\run.ps1` — 데모(`configs/demo.yaml`, YOLO 검출·미리보기 켜짐)

- `.\run.ps1 -Setup` — 설치(`setup.ps1`) 다시 한 뒤 실행

- `.\run.ps1 -Full` — 전체 설정(`configs/default.yaml`, YOLO)

- `.\run.ps1 -ImageFolder` — 좌·우 이미지 폴더(`configs/image_folder.yaml`)

- `.\run.ps1 -Orbbec` — Orbbec RGB-D (`configs/orbbec_gemini.yaml`; `pip install pyorbbecsdk2` 및 호스트 SDK 필요)
- `.\run.ps1 -Orbbec -Capture -Positive` — Object Anchor 학습용 원본 RGB 100장 자동 수집(기본 1초 간격)
- `.\run.ps1 -Orbbec -Capture -Negative` — Negative 원본 RGB와 동일 stem의 빈 YOLO 라벨 100장 자동 수집

수집 장수와 간격은 `-CaptureCount`, `-CaptureInterval`로 바꿀 수 있습니다.

```powershell
.\run.ps1 -Orbbec -Capture -Positive -CaptureCount 150 -CaptureInterval 0.5
.\run.ps1 -Orbbec -Capture -Negative -CaptureCount 200 -CaptureInterval 1
```

수집 이미지는 `data/object_anchor_capture/{positive|negative}/images`, Negative 빈 라벨은
`data/object_anchor_capture/negative/labels`, 통합 기록은
`data/object_anchor_capture/capture_manifest.csv`에 저장됩니다. 저장되는 JPEG는 오버레이가 없는
Orbbec 원본 BGR 프레임이며 화면에만 진행 상태가 표시됩니다.



수동 분리: `.\setup.ps1` → `.\run_demo.ps1`



`setup.ps1`: Python 찾기(`STEREO_POC_PYTHON` 가능) → `.venv`·`pip` → 합성 보정·rectify 맵 → `smoke_test`



### 미리보기 / 키



데모는 `preview.enabled: true`. OpenCV **창을 클릭**한 뒤 `Q` 종료, **`S`** 는 한 번에 **`image_folder` 재분석용** 좌·우 한 쌍만 저장합니다(`out/snapshots/<시간>/left/*.png`, `.../right/` 동일 파일명, 정류 전·캘리브 크롭 해상도).

`configs/image_folder.yaml`에는 **`preview.image_folder_hold_until_quit: true`** 가 기본입니다. 한 장만 넣어도 창이 바로 닫히지 않고, **종료는 `Q`**, 다음 쌍으로 넘어가려면 **`Space`** 또는 **`n`** 입니다.

창 구성: **2개** — `windows.combined` 이 켜지면 기본적으로 정류 **좌안 RGB 한 장**(YOLO와 같은 뷰), `windows.disparity` 가 켜지면 **시차 맵** 한 창(둘 다 켜는 것이 기본). 예전처럼 **L|R 가로 합성** 을 보이려면 `preview.combined_side_by_side_stereo: true`. 예전 4창에서 **좌·우 단독 창만** 코드에 주석으로 남기고 비활성화했습니다. 한 창에 세로로 붙이려면 `preview.stack_disparity_below: true`. 미리보기 창 크기는 `preview.scale`(기본 `0.5`면 가로·세로 절반, `1`이면 표시 해상도 원본), `window_autosize` 참고. 시차 깜빡임 완화는 `disparity_percentile_smooth_alpha` 또는 `disparity_vis_min` / `max`.



## Orbbec RGB-D (Gemini 등)


`input.type: orbbec` 일 때 스테레오 YAML·정류 맵 없이 장치 기본 intrinsics 및 **depth→color 정렬**(소프트웨어 `AlignFilter`)을 사용합니다.

1. [Orbbec pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk) 문서대로 호스트 드라이버/SDK 후 PyPI에서 `pip install pyorbbecsdk2`(코드에서는 `import pyorbbecsdk`).
2. 예시 설정: [`configs/orbbec_gemini.yaml`](configs/orbbec_gemini.yaml).
3. `orbbec` 블록에서 해상도·FPS·시리얼·`depth_is_millimeters`·`depth_scale_additional` 등을 장치에 맞게 조정하세요.

AprilTag: RGB-D 모드에서는 **거리 검증 전용**(측정 거리 vs `known_spacing_m`, `apriltag_scale` 기본값으로 `scale` 을 비워 둠). `apply_scale_to_depth: false` 권장. 스테레오 모드에서는 기존대로 `Q`·disparity 로 스케일을 구할 수 있습니다.

미리보기 두 번째 창은 **깊이(m) 컬러맵**(Turbo); 옵션 키 `depth_vis_*` / `depth_percentile_*` 는 [`experiments/repeatability_run.py`](experiments/repeatability_run.py) 의 `_depth_m_colormap_bgr` 참고.



```powershell

.\.venv\Scripts\python.exe experiments\repeatability_run.py --config configs\orbbec_gemini.yaml

```



## 그 밖에 직접 실행



```powershell

.\.venv\Scripts\python.exe experiments\repeatability_run.py --config configs\default.yaml

```



## 체스판 보정



```powershell

.\.venv\Scripts\python.exe scripts\calibrate_from_images.py --left_dir "<좌>" --right_dir "<우>" --board 9,6 --square_mm 25 --out_yaml calibration\stereo_calib.yaml --maps_prefix calibration\rectify_maps

```



## AprilTag 간격으로 깊이 스케일 (선택)

정류된 좌안 그레이에서 AprilTag를 검출한 뒤, 두 태그 중심의 `Q` 기반 3D 거리를 **`apriltag_scale.known_spacing_m`**(예: 실측 1 m)에 맞추는 배율을 구합니다. **`apply_scale_to_depth: true`** 이면 Track A/B의 `X,Y,Z`에 그 배율을 곱합니다. **`dictionary`** 는 실물 태그 패밀리와 같아야 합니다(`APRILTAG_36H11`, `APRILTAG_25H9` 등).  
`tag_id_a` / `tag_id_b` 를 비우면 검출된 태그 중 ID가 가장 작은 두 개를 사용합니다.

`configs/default.yaml` 의 `apriltag_scale` 블록을 참고하고, 사용 시 **`enabled: true`** 로 바꿉니다.



## 비디오 입력



YAML 예:



```yaml

input:

  type: video

  video_path: samples/my_recording.mp4

```



## 이미지 폴더 입력



`left`·`right`에 **같은 파일명** 쌍. 배치 규칙은 [samples/stereo_pairs/README.md](samples/stereo_pairs/README.md).



```yaml

input:

  type: image_folder

  image_folder:

    left_dir: samples/stereo_pairs/left

    right_dir: samples/stereo_pairs/right

```



실행: `.\run.ps1 -ImageFolder`



## CSV 요약

한 프레임에 검출이 여러 개면 **행이 여러 줄** 생깁니다. 열 **`det_idx`** 는 해당 프레임 안에서 신뢰도 순(0이 최고), **`class_id`** / **`label`** 은 YOLO 클래스입니다. 예전처럼 **대표 박스 한 줄만** 남기려면 YAML에서 **`repeatability.log_all_boxes: false`** 로 두면 됩니다.

```powershell

.\.venv\Scripts\python.exe experiments\kpi_from_csv.py out\repeatability_demo.csv

```

`kpi_from_csv.py` 통계는 **`det_idx == 0`(대표 검출)** 행만 사용합니다.


## Object Anchor 합성 Pose 테스트

현재 MVP는 FRONT 면의 4개 keypoint를 사용하는
`configs/object_anchors/tissue_box_01_front_only.yaml`이 기본입니다. 기존 8점 정의
`configs/object_anchors/tissue_box_01.yaml`도 후속 전 방향 고도화를 위해 유지합니다.
사용자 라벨, YOLO-Pose 출력, PnP 입력은 선택한 config의 ID 순서를 동일하게 사용해야
합니다.

YOLO-Pose 가중치가 없어도 합성 2D keypoint로 `solvePnPRansac` 복원을 검증할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts\synthetic_object_anchor_test.py
```

결과 행렬과 translation, roll/pitch/yaw, inlier 수, reprojection error가 출력되며,
keypoint 순서와 축 시각화는 `out/object_anchor/synthetic_front_only_pose.png`에 저장됩니다.


## Object Anchor 실시간 디버그

실제 YOLO-Pose 모델이 준비되면 `configs/orbbec_gemini.yaml`의
`object_anchor.model_path`에 `best.pt` 경로를 넣고 `enabled: true`로 변경합니다.
모델 경로가 비어 있거나 파일이 없으면 Object Anchor만 비활성화되고 기존
AprilTag·컵 처리는 계속 실행됩니다.

이 단계는 `camera_pose_only: true`이며 월드 좌표를 계산하지 않습니다. RGB 화면에
keypoint ID/confidence/effective visibility, FRONT skeleton, X/Y/Z 축, translation,
roll/pitch/yaw, reprojection error, PnP inlier 수를 표시합니다. 유효 keypoint와
inlier가 각각 4개 미만이거나, depth가 음수이거나, reprojection error 또는 프레임 간
위치·회전 변화가 임계값을 넘으면 Pose를 무효 처리합니다.

FRONT-only 20~30장 파일럿과 후속 8점 40장 절차, 라벨 규칙 및 학습 명령은
`docs/OBJECT_ANCHOR_PILOT.md`를 참고합니다. 향후 AprilTag와 동일 고정판에서 얻은
`T_world_object`는 `object_anchor_registration.py`의 저장/로드 인터페이스로 별도
YAML에 기록합니다.


## 수동 설치 (스크립트 없이)



```powershell

python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts\create_placeholder_calibration.py

.\.venv\Scripts\python.exe scripts\smoke_test.py

```



Store 스텁 문제: [python.org](https://www.python.org/downloads/) 설치본 또는 `STEREO_POC_PYTHON` 후 `.\setup.ps1`



```powershell

$env:STEREO_POC_PYTHON = "C:\Users\<사용자>\AppData\Local\Programs\Python\Python312\python.exe"

.\setup.ps1

```



Python 설치 예: `winget install Python.Python.3.12` (설치 후 **새 터미널**에서 스크립트 실행)

