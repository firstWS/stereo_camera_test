# 실험 리포트 템플릿 (Track A vs B, 병목 기록)

## 0. 실행 명령 (저장소 루트에서)

```text
python scripts/calibrate_from_images.py --left_dir <L> [--right_dir <R>] --board 9,6 --square_mm 25 --out_yaml calibration/stereo_calib.yaml --maps_prefix calibration/rectify_maps
python experiments/repeatability_run.py --config configs/default.yaml
python experiments/kpi_from_csv.py out/repeatability.csv
```

(Python 경로에 `src`가 잡히도록 스크립트에서 처리함; 루트에서 실행.)

## 1. 환경

- OS / GPU / 드라이버 버전:
- Python / OpenCV / PyTorch / ultralytics 버전:
- 카메라 모델 / 펌웨어 / UVC 해상도·FPS:
- 캘리브: `stereo_calib.yaml` 생성 일시, 체커보드 그리드 (cols, rows), 격자 간격(mm)

## 2. 실험 목적

- 검출 이후 **깊이/3D 반복 안정성** 검증 (Track A: SGBM ROI 중앙값, Track B: sparse NCC)

## 3. 절차

- 대상 물체 / 거리 대역 (m):
- 고정 조건: 조명, 삼각대, 노출(AEC) 고정 여부
- `experiments/repeatability_run.py` 실행 인자 / `configs/default.yaml` 변경점
- 프레임 수, 워밍업 프레임

## 4. 결과 CSV

- 경로:
- `python experiments/kpi_from_csv.py <csv>` 출력 붙여넣기

## 5. Track A vs B 요약

| 항목 | Track A (dense+median) | Track B (sparse NCC) |
|------|-------------------------|----------------------|
| valid 비율 |  |  |
| Z 표준편차 (m) |  |  |
| Z IQR (m) |  |  |
| 연속 프레임 Z 점프 평균 (m) |  |  |
| 실패 원인 비고 | 무텍스처 / 검출 튐 / 시차 마스크 | NCC 낮음 / 탐색 범위 |

## 6. 성능 병목

- 평균 `capture_ms`, `disp_ms`, `det_ms` (CSV에서 산출):

## 7. 실패 모드 체크리스트

- [ ] 캘리브 품질 불량 (에피폴라 정합 큰 dv)
- [ ] SGBM `numDisparities` 부족 / 과다
- [ ] 노출 변화로 양 눈 밝기 불일치
- [ ] 검출 박스가 배경 위주 / 경계에 대표점
- [ ] 무텍스처·반사로 시차 붕괴
- [ ] USB 대역/CPU MJPEG 디코딩 병목

## 8. 다음 액션

- 파라미터 튜닝 후보:
- 소프트웨어 변경(ROI, 시간 필터 등):
