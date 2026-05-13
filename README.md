# stereo-3d-poc

USB 스테레오(SBS) 캡처 → 보정·정류 → SGBM 시차 → YOLO(또는 더미 검출) → 3D 추정 실험 코드입니다.

## 빠른 시작 (Windows)

PowerShell에서 저장소 루트에서 실행합니다.

**가장 간단한 방법** — PATH가 IDE 터미널에서 비어 있어도 레지스트리 기준으로 다시 맞춘 뒤, 가상환경이 없으면 설치하고 데모 설정으로 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # 한 번만, 스크립트 실행 허용 시
.\run.ps1
```

- `.\run.ps1 -Setup` — 설치 단계를 다시 실행한 뒤 실행  
- `.\run.ps1 -Full` — `configs/default.yaml`(YOLO 포함)로 실행  
- `.\run.ps1 -Config configs\default.yaml` — 설정 파일 지정  

수동으로 나누려면:

```powershell
.\setup.ps1
.\run_demo.ps1
```

`setup.ps1`은 다음을 수행합니다.

- `py` 없이도 동작하도록 `python.org` 설치본 등에서 Python 검색 (`STEREO_POC_PYTHON` 환경변수로 경로 지정 가능)
- `.venv` 생성 및 `pip install -r requirements.txt`
- 합성 보정 파일 `calibration/stereo_calib.yaml` + rectify 맵 생성 (**실측 보정 전 연습용**, 실제 깊이 스케일은 체스판 캘리브로 교체)
- 카메라 없이 파이프라인 확인용 `scripts/smoke_test.py` 실행

연습용 라이브 실행(더미 검출, `configs/demo.yaml`): `.\run.ps1` 또는 `.\run_demo.ps1`

### 라이브 미리보기 창

`configs/demo.yaml`에는 기본으로 **`preview.enabled: true`** 가 들어 있습니다. 실행 시 정류된 좌안 영상(검출 박스·깊이 텍스트)과, 옵션으로 시차 컬러맵 창이 뜹니다. **작업 표시줄에서 OpenCV 창을 선택(또는 Alt+Tab)** 한 뒤 **영문 `Q`** 키를 누르면 조기 종료합니다(대문자 `Q`도 동일). Cursor 터미널에 포커스가 있으면 키 입력이 터미널로만 가므로 **반드시 미리보기 창을 한 번 클릭**해야 합니다.  
`preview.wait_key_ms`(데모 기본 33ms)로 매 프레임마다 잠시 멈추므로, 예전처럼 80프레임만에 즉시 종료되어 창이 안 보이는 문제를 줄입니다. GUI 없는 환경이면 `preview.enabled: false` 로 끄세요.

전체 설정(YOLO 포함)은 `configs/default.yaml`을 사용합니다.

```powershell
.\.venv\Scripts\python.exe experiments\repeatability_run.py --config configs\default.yaml
```

## 실제 보정 (체스판)

합성 보정 대신 실사용 시:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_from_images.py --left_dir "<좌 이미지 폴더>" --right_dir "<우 이미지 폴더>" --board 9,6 --square_mm 25 --out_yaml calibration\stereo_calib.yaml --maps_prefix calibration\rectify_maps
```

## 비디오 파일로 실행

설정 YAML에서:

```yaml
input:
  type: video
  video_path: samples/my_recording.mp4
```

`video_path`는 저장소 루트 기준 상대 경로입니다.

## 결과 CSV 요약

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

`python`이 Microsoft Store 스텁만 가리키면 [python.org](https://www.python.org/downloads/) 설치본 경로를 사용하거나 `STEREO_POC_PYTHON`을 설정하세요.

`setup.ps1`이 Python을 못 찾으면: 설치 후 다시 실행하거나, 아래처럼 경로를 지정합니다.

```powershell
$env:STEREO_POC_PYTHON = "C:\Users\<사용자>\AppData\Local\Programs\Python\Python312\python.exe"
.\setup.ps1
```

winget 사용 예: `winget install Python.Python.3.12` (설치 후 **새 터미널**에서 `setup.ps1` 실행).
