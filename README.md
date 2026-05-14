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

창 종류·해상도는 YAML `preview.windows`, `preview.scale` 참고.



## 그 밖에 직접 실행



```powershell

.\.venv\Scripts\python.exe experiments\repeatability_run.py --config configs\default.yaml

```



## 체스판 보정



```powershell

.\.venv\Scripts\python.exe scripts\calibrate_from_images.py --left_dir "<좌>" --right_dir "<우>" --board 9,6 --square_mm 25 --out_yaml calibration\stereo_calib.yaml --maps_prefix calibration\rectify_maps

```



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



```powershell

.\.venv\Scripts\python.exe experiments\kpi_from_csv.py out\repeatability_demo.csv

```



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

