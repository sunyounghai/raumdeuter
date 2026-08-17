# Raumdeuter — Vision AI 기반 공간 침투 분석 시스템

Vision AI(객체 검출, 트래킹, 호모그래피)를 활용해 일반 방송 축구 영상만으로 선수의 오프더볼 공간 침투를 정량화하는 프로젝트입니다.

## 진행 상태
- [x] 프로젝트 설계
- [x] 데이터 구축 (1차 파일럿: train 104장 / val 26장)
- [x] 검출 모델 파인튜닝 (coco/roboflow/h250 3개 모델 baseline/finetuned 평가 완료 → [결과](./docs/detection/experiment_results.md))
- [ ] 트래킹
- [x] 호모그래피 (PnLCalib 기반 wrapper 구현, 391프레임 정량 검증 완료 → [결과](./docs/calibration/spotcheck_results.md))
- [ ] 지표 계산

## 현재 작업: GT 기준 calibration 정량 검증
391프레임 스팟체크(h_success 79.5%, keypoint 임계값 분석)를 완료했고, 그 결과를 GT(화면에서 직접 위치를 확인해 라벨링한 기준점)로 검증하는 단계입니다. n_keypoints 구간별 층화 샘플링(20장)과 Label Studio 라벨링까지 완료했고, 다음으로 재투영 오차(미터 단위) 계산 및 ByteTrack 연동을 진행합니다.

## 한계
- 검출 모델은 단일 경기 영상으로 학습됨. 다른 방송사/경기장/카메라 셋업에 대한 일반화 성능은 검증되지 않음 (향후 과제).
- calibration(호모그래피) 정확도는 현재 정성적 확인(라인 오버레이, 선수 좌표 스팟체크)까지만 완료. GT 기준 정량 오차(reprojection error)는 아직 측정되지 않음.

## 기술 스택
- 객체 검출/트래킹: YOLOv8/v11, ByteTrack, PyTorch
- 카메라 보정: [PnLCalib](https://github.com/mguti97/PnLCalib) (Points-and-Lines Calibration), OpenCV
- 공간 분석: SciPy(Voronoi), Pitch Control, NumPy/Pandas
- 시각화: mplsoccer, Matplotlib

## 프로젝트 구조
```
raumdeuter/
├── docs/
│   ├── calibration/
│   │   └── spotcheck_results.md       # 391프레임 calibration 정량 검증 결과
│   ├── detection/
│   │   └── experiment_results.md      # 3개 모델 baseline/finetuned 평가 결과
│   │   └── labeling_guideline.md
│   └── data_extraction_log.md
├── external/
│   └── PnLCalib/                       # git submodule (GPLv2)
├── label_studio/
│   └── config.xml                      # 라벨링 설정 (player 단일 클래스)
│   └── config_gt_points.xml            # GT keypoint 라벨링 설정
├── results/
│   └── detection/
│       ├── coco_baseline_metrics.txt
│       ├── coco_finetuned_metrics.txt
│       ├── roboflow_baseline_metrics.txt
│       ├── roboflow_finetuned_metrics.txt
│       ├── h250_baseline_metrics.txt
│       ├── h250_finetuned_metrics.txt
│       └── unified_comparison.txt
├── src/
│   ├── __init__.py
│   ├── common/                          # 공통 경로 설정
│   ├── detection/                       # YOLO 검출 모델 학습/평가
│   ├── calibration/                     # PnLCalib wrapper (호모그래피 산출)
│   └── diagnostics/                     # 정성적 검증 도구 (calibration/detection/tracking
│                                         # 스팟체크, GT 재투영 오차 측정)
├── .gitignore
├── .gitmodules
├── COPYING                              # GPLv2 원문
├── LICENSE                              # 저작권 고지
└── README.md
```

## 라이선스

이 프로젝트는 [PnLCalib](https://github.com/mguti97/PnLCalib)(GPLv2)을 기반으로 하며,
GNU General Public License v2.0에 따라 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE)를 참고하세요.