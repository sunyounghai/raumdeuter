# Raumdeuter — Vision AI 기반 공간 침투 분석 시스템

Vision AI(객체 검출, 트래킹, 호모그래피)를 활용해 일반 방송 축구 영상만으로 선수의 오프더볼 공간 침투를 정량화하는 프로젝트입니다.

## 진행 상태
- [x] 프로젝트 설계
- [x] 데이터 구축 (1차 파일럿: train 104장 / val 26장)
- [x] 검출 모델 파인튜닝 (coco/roboflow/h250 3개 모델 baseline/finetuned 평가 완료 → [결과](./docs/detection/experiment_results.md))
- [ ] 트래킹 (ByteTrack vs BoT-SORT ablation 설계 및 실행 중)
- [x] 호모그래피 (PnLCalib 기반 wrapper 구현, 391프레임 정량 검증 완료 → [결과](./docs/calibration/spotcheck_results.md))
- [ ] 지표 계산

## 현재 작업: 트래킹 ablation 설계 및 실행
ID 쪼개짐 원인을 A(밀집/교차)·B(컷 전환)·C(프레임 이탈·재진입)·D(카메라 흔들림) 네 유형으로 분해하고, 유형별로 필요한 대응(외형 매칭 vs 카메라 모션 보정)을 구분했습니다. 자체 데이터 라벨링 전에 SoccerNet tracking-2023 공개 데이터셋으로 먼저 검증하는 단계이며, `boxmot` 기반으로 ByteTrack(baseline)과 BoT-SORT(OSNet ReID + ECC 기반 CMC)를 같은 코드 경로에서 비교하는 ablation을 진행 중입니다. 평가는 `sn-trackeval`(HOTA/IDF1/AssA/DetA)로 정량화하며, 이후 GTA-Link / AFLink+GSI 등 후처리(트랙 스티칭) 단계 추가 여부를 결정할 예정입니다.


## 한계
- 검출 모델은 단일 경기 영상으로 학습됨. 다른 방송사/경기장/카메라 셋업에 대한 일반화 성능은 검증되지 않음 (향후 과제).
- calibration(호모그래피) 정확도는 현재 정성적 확인(라인 오버레이, 선수 좌표 스팟체크)까지만 완료. GT 기준 정량 오차(reprojection error)는 아직 측정되지 않음.

## 기술 스택
- 객체 검출: YOLOv8/v11, PyTorch
- 트래킹: [boxmot](https://github.com/mikel-brostrom/boxmot) (ByteTrack, BoT-SORT+OSNet), [sn-trackeval](https://github.com/SoccerNet/sn-trackeval) (HOTA/IDF1/AssA/DetA 평가)
- 후처리(검토 중): [GTA-Link](https://github.com/sjc042/gta-link), StrongSORT의 AFLink/GSI
- 카메라 보정: [PnLCalib](https://github.com/mguti97/PnLCalib) (Points-and-Lines Calibration), OpenCV
- 공간 분석: SciPy(Voronoi), Pitch Control, NumPy/Pandas
- 시각화: mplsoccer, Matplotlib
- 검증 데이터: 자체 라벨링 데이터 + [SoccerNet](https://www.soccer-net.org/) tracking-2023 (도메인 일반화 검증용 공개 데이터셋)


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
├── requirements.txt                     # pip 의존성
└── README.md
```

## 라이선스

이 프로젝트는 [PnLCalib](https://github.com/mguti97/PnLCalib)(GPLv2)을 기반으로 하며,
GNU General Public License v2.0에 따라 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE)를 참고하세요.