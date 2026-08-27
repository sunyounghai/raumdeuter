# Raumdeuter — Vision AI 기반 공간 침투 분석 시스템

Vision AI(객체 검출, 트래킹, 호모그래피)를 활용해 일반 방송 축구 영상만으로 선수의 오프더볼 공간 침투를 정량화하는 프로젝트입니다.

## 진행 상태
- [x] 프로젝트 설계
- [x] 데이터 구축 (1차 파일럿: train 104장 / val 26장)
- [x] 검출 모델 파인튜닝 (coco/roboflow/h250 3개 모델 baseline/finetuned 평가 완료 → [결과](./docs/detection/experiment_results.md))
- [x] 트래킹 (검출기x트래커 12개 조합 비교 완료, h250_baseline + BoT-SORT 최종 채택 → [결과](./docs/tracking/experiment_results.md))
- [x] 호모그래피 (PnLCalib 기반 wrapper 구현, 391프레임 정량 검증 완료 → [결과](./docs/calibration/spotcheck_results.md))
- [ ] 지표 계산

## 현재 작업: 트래킹 파이프라인 확정, 다음 단계(지표 계산) 준비
검출기(coco/roboflow/h250 baseline + h250_finetuned) x 트래커(ByteTrack/BoT-SORT/StrongSORT) 조합을 SoccerNet SNMOT-116 클립에서 비교한 결과, **h250_baseline(파인튜닝 없이) + BoT-SORT** 조합을 최종 채택했습니다. 검출기 선택이 트래커 선택보다 성능에 훨씬 큰 영향을 줬습니다. 자체 파인튜닝은 새로운 도메인에서 baseline 성능을 넘지 못해, 과적합 가능성을 실제로 확인했습니다. 상세 근거는 [트래킹 실험 결과](./docs/tracking/experiment_results.md) 참고.

다음 단계로는 확정된 트래킹 파이프라인을 calibration(호모그래피)과 연동해 실제 경기장 좌표계 위에서 선수 궤적을 그리는 지표 계산 단계로 넘어갈 예정입니다.

## 한계
- 검출 모델은 단일 경기 영상으로 학습됨. 다른 방송사/경기장/카메라 셋업에 대한 일반화 성능은 검증되지 않음 (향후 과제).
- calibration(호모그래피)은 GT 기준 재투영 오차 측정까지 완료함(20프레임, n_keypoints 구간별 평균 0.71~2.27m). 다만 표본이 작고(구간당 4~7개) 결과가 이상치 1건에 크게 좌우되어 결론을 확정하기엔 이름 — 표본 확대 필요. 상세는 [spotcheck_results.md](./docs/calibration/spotcheck_results.md) 참고.
- 트래킹 결과는 단일 클립(SNMOT-116, 코너킥 상황) 기준. 다른 성격의 클립 검증 및 후처리(트랙 스티칭) 단계는 미완료 — 상세는 [트래킹 실험 결과의 한계 섹션](./docs/tracking/experiment_results.md#한계) 참고.

## 기술 스택
- 객체 검출: YOLOv8/v11, PyTorch
- 트래킹: [boxmot](https://github.com/mikel-brostrom/boxmot) (ByteTrack, BoT-SORT+OSNet, StrongSORT), [sn-trackeval](https://github.com/SoccerNet/sn-trackeval) (HOTA/IDF1/AssA/DetA 평가)
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
│   ├── tracking/
│   │   └── experiment_results.md      # 검출기x트래커 12개 조합 비교
│   └── data_extraction_log.md
├── external/
│   └── PnLCalib/                       # git submodule (GPLv2)
├── label_studio/
│   └── config.xml                      # 라벨링 설정 (player 단일 클래스)
│   └── config_gt_points.xml            # GT keypoint 라벨링 설정
├── results/
│   ├── detection/
│   │   ├── coco_baseline_metrics.txt
│   │   ├── coco_finetuned_metrics.txt
│   │   ├── roboflow_baseline_metrics.txt
│   │   ├── roboflow_finetuned_metrics.txt
│   │   ├── h250_baseline_metrics.txt
│   │   ├── h250_finetuned_metrics.txt
│   │   └── unified_comparison.txt
│   └── tracking/
│       ├── unified_comparison.txt               # 27개 GT(선수+골키퍼+심판+공) 기준 비교
│       └── unified_comparison_players_only.txt  # 24개 GT(선수+골키퍼) 기준 최종 비교
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