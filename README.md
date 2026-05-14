# stereo_camera_test



USB 스테레오(SBS) 캡처 → 보정·정류 → SGBM 시차 → 검출(YOLO 또는 더미) → 3D 추정 실험 코드입니다.



## 빠른 시작 (Windows)



저장소 루트에서 PowerShell로 실행합니다.



```powershell

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # 한 번만, 스크립트 허용 시

.\run.ps1

```



- `.\run.ps1` — 데모(`configs/demo.yaml`, 더미 검출, 미리보기 켜짐)

- `.\run.ps1 -Setup` — 설치(`setup.ps1`) 다시 한 뒤 실행

- `.\run.ps1 -Full` — 전체 설정(`configs/default.yaml`, YOLO)

- `.\run.ps1 -ImageFolder` — 좌·우 이미지 폴더(`configs/image_folder.yaml`)

- `.\run.ps1 -Config configs\default.yaml` — 임의 설정 파일



수동 분리: `.\setup.ps1` → `.\run_demo.ps1`



`setup.ps1`: Python 찾기(`STEREO_POC_PYTHON` 가능) → `.venv`·`pip` → 합성 보정·rectify 맵 → `smoke_test`



### 미리보기 / 키



데모는 `preview.enabled: true`. OpenCV **창을 클릭**한 뒤 `Q` 종료, **`S`** 는 한 번에 **`image_folder` 재분석용** 좌·우 한 쌍만 저장합니다(`out/snapshots/<시간>/left/*.png`, `.../right/` 동일 파일명, 정류 전·캘리브 크롭 해상도).

`configs/image_folder.yaml`에는 **`preview.image_folder_hold_until_quit: true`** 가 기본입니다. 한 장만 넣어도 창이 바로 닫히지 않고, **종료는 `Q`**, 다음 쌍으로 넘어가려면 **`Space`** 또는 **`n`** 입니다.

창 구성: **2개** — `windows.combined` 이 켜지면 기본적으로 정류 **좌안 RGB 한 장**(YOLO와 같은 뷰), `windows.disparity` 가 켜지면 **시차 맵** 한 창(둘 다 켜는 것이 기본). 예전처럼 **L|R 가로 합성** 을 보이려면 `preview.combined_side_by_side_stereo: true`. 예전 4창에서 **좌·우 단독 창만** 코드에 주석으로 남기고 비활성화했습니다. 한 창에 세로로 붙이려면 `preview.stack_disparity_below: true`. 미리보기 창 크기는 `preview.scale`(기본 `0.5`면 가로·세로 절반, `1`이면 표시 해상도 원본), `window_autosize` 참고. 시차 깜빡임 완화는 `disparity_percentile_smooth_alpha` 또는 `disparity_vis_min` / `max`.



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

