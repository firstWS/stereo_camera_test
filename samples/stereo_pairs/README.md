# 스테레오 이미지 테스트용

`configs/image_folder.yaml` 은 이 폴더를 사용합니다.

## 배치 규칙

- `left/` 와 `right/` 안에 **파일 이름이 동일한** BGR 이미지 쌍을 넣습니다. (예: `frame_001.png` 양쪽 모두)
- 지원 확장자는 설정의 `patterns` 기본값: `*.png`, `*.jpg`, `*.jpeg`
- 이름은 대소문자 구분 정렬로 순서가 정해집니다.

## 실행 예

```powershell
.\.venv\Scripts\python.exe experiments\repeatability_run.py --config configs\image_folder.yaml
```

캘리브레이션 해상도와 맞지 않으면 `crop_to_calib_size`가 리사이즈합니다. 가능하면 촬영 시와 같은 해상도를 권장합니다.

## 라이브 실행 중 스냅 (`S` 키)

미리보기가 켜진 상태에서 `S`를 누르면 세션 폴더 아래 `left/`, `right/`에 같은 이름의 PNG가 저장됩니다. 그 폴더를 그대로 `input.image_folder.left_dir` / `right_dir`에 지정하면 동일 파이프라인으로 다시 분석할 수 있습니다.

## 한 장만 두고 디버깅할 때 (`hold`)

`preview.image_folder_hold_until_quit: true`(예제 설정 기본)이면 각 쌍을 표시한 뒤 **바로 종료하지 않습니다.** OpenCV 창 포커스에서 **`Q`** 로 프로그램 종료, **`Space`** 또는 **`n`** 으로 다음 파일 쌍으로 진행합니다.
