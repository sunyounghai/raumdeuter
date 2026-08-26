# 실험 결과: 트래킹 (검출기 x 트래커 조합 비교)

## 실행 조건
- **평가 클립**: SoccerNet tracking-2023 test, SNMOT-116 (30초, `actionClass=Corner`, 750프레임)
- **평가 방식**: `src.tracking.run_eval` (sn-trackeval, `trackeval.datasets.MotChallenge2DBox` 기반)
    - 지표: HOTA, DetA, AssA, IDF1, IDs(예측 트랙 수) vs GT_IDs(정답 트랙 수)
- **GT 규모**: 27개 트랙(선수·골키퍼·심판·공) 중 검출기 간 클래스 지원 범위가 달라 공정 비교를 위해 player+goalkeeper만 남긴 24개 트랙 기준(`SoccerNet-players-test` 벤치마크, `filter_gt_by_role.py`로 생성)으로 최종 비교
- **검출 조건**: coco_baseline(yolov8n.pt) / roboflow_baseline(football-player-detection.pt) / h250_baseline(yolov8n_soccernetv3h250_pretrained.pt) / h250_finetuned (자체 라벨링 104장으로 파인튜닝) 4종
- **트래커 조건**: ByteTrack(모션만) / BoT-SORT(모션+OSNet 외형+ECC 기반 CMC) / StrongSORT(모션+ReID+NSA Kalman) 3종, `boxmot` 22.0.0 기반
- **실행**: `python -m src.tracking.tracker_wrapper` → `python -m src.detection.infer_to_mot` → `python -m src.tracking.run_eval --benchmark SoccerNet-players`

## 결과 (player+goalkeeper 기준 GT, 24개 트랙)

| condition | HOTA | DetA | AssA | IDF1 | Dets | IDs (GT=24) |
|---|---|---|---|---|---|---|
| strongsort_gt | 82.958 | 88.585 | 77.689 | 80.399 | 11078 | 72 |
| botsort_gt | 74.230 | 86.167 | 63.964 | 72.483 | 11236 | 76 |
| bytetrack_gt | 71.161 | 77.237 | 65.618 | 76.295 | 11182 | 130 |
| strongsort_h250_baseline | **44.943** | **61.796** | 33.039 | 51.144 | 9871 | 91 |
| botsort_h250_baseline | 44.108 | 60.363 | 32.544 | **53.535** | 9369 | **57** |
| bytetrack_h250_baseline | 44.088 | 59.068 | **33.219** | 52.526 | 9268 | 77 |
| botsort_finetuned | 43.463 | 57.711 | 33.027 | 53.078 | 8919 | 73 |
| bytetrack_finetuned | 40.336 | 55.521 | 29.619 | 48.028 | 8813 | 102 |
| botsort_roboflow_baseline | 37.328 | 44.061 | 31.993 | 44.403 | 7510 | 157 |
| botsort_coco_baseline | 37.220 | 45.442 | 30.828 | 46.627 | 6895 | 63 |
| bytetrack_coco_baseline | 34.303 | 46.359 | 25.848 | 42.159 | 7098 | 97 |
| bytetrack_roboflow_baseline | 31.749 | 42.700 | 24.021 | 37.670 | 7396 | 203 |

*(상위 3개 `_gt` 조건은 gt.txt를 검출값으로 사용한 이론적 상한선이며, 나머지 9개가 실제 검출기 기반 조건임)*

## 관찰

**1. SoccerNet 공개 `det.txt`는 실제 검출 오차가 아니라 GT와 좌표가 동일했음**
- 초기 ablation(조건 A~D)에서 `gt.txt`와 `det.txt` 입력 간 HOTA가 소수점까지 완전히 일치
- 두 파일의 박스 좌표가 100% 동일(순서만 다름) - `det.txt`가 실제로는 GT를 그대로 재사용해 배포된 것으로 판단됨
- 이로 인해 "검출 오차가 트래킹에 주는 영향"을 보려면 자체 YOLO 모델을 SoccerNet 프레임에 직접 추론해야 했음(`infer_to_mot.py` 신규 작성)

**2. 검출기(도메인 적합성)가 트래커 선택보다 훨씬 지배적인 변수였음**
- 검출기간 HOTA 격차: 최대 약 13점(h250_baseline 44.1 vs roboflow_baseline 31.7)
- 트래커 간 HOTA 격차: 최대 약 5.6점(동일 검출기 기준 BoT-SORT vs ByteTrack)
- GT(gt.txt, 정답 위치)를 검출값으로 쓴 초기 조건에서도 트래커만으로 ByteTrack 130개 vs BoT-SORT 76개로 트랙 수 차이가 났으나, 실제 검출 오차가 개입하면 그 격차보다 검출기 선택의 영향이 훨씬 컸음
- 특히 모델 크기와 성능이 비례하지 않았음 - roboflow_baseline(6800만 파라미터, h250 대비 약 22배)이 h250_baseline(300만 파라미터)에게 전 지표에서 뒤처졌으며, 이는 h250 계열은 SoccerNet 계열 데이터로 사전학습되어 도메인 적합성이 모델 크기보다 더 크게 성능을 좌우했기 때문으로 판단됨

**3. 자체 파인튜닝(h250_finetuned)이 baseline을 넘지 못함 - 과적합 가능성 실증**
- `docs/detection/experiment_results.md`의 "한계 2번"(과적합 가능성, 다른 영상에서 재검증
  필요)에서 우려했던 내용이 실제로 확인됨
- 자체 검증셋(같은 영상)에서는 h250_finetuned(mAP50-95 0.81)가 h250_baseline(0.66)보다 뚜렷이 우수했으나, 새로운 도메인(SoccerNet)에서는 HOTA 기준 baseline과 동등하거나 근소하게 낮음(BoT-SORT 기준 44.108 vs 43.463, -0.65)
- 104장 규모의 학습 데이터로는 도메인 일반화 개선까지 이어지지 못했을 가능성이 높음

**4. 트래커 간에는 지표별로 우열이 갈림 - 목적에 따라 선택 기준이 다름**
- StrongSORT: HOTA/DetA 1위 (검출 재현력이 가장 높음 - 놓치지 않고 잘 잡아냄)
- BoT-SORT: IDF1/트랙 안정성 1위 (IDs 57개로 정답 24개에 가장 근접 - 궤적이 가장 안 끊김)
- 본 프로젝트의 목적(공간 침투 지수 = 궤적의 연속성이 핵심)에는 IDF1·트랙 안정성이 더 직접적인 지표이므로, **h250_baseline + BoT-SORT 조합을 최종 채택**

**5. GT 채점 기준 불일치 문제를 발견하고 수정함**
- 검출기마다 지원 클래스가 다름(예: h250은 `{ball, person}`만, football-player-detection은 `{ball, goalkeeper, player, referee}` 지원)인데, 반면 원본 GT(27개)는 4종류를 모두 포함
- 이를 그대로 채점하면 다중 클래스 미지원 모델이 구조적으로 불리해짐(h250_finetuned 최초 평가 시 HOTA가 27개 기준 39.7까지 떨어졌으나, 24개 기준 재평가 시 40.3~43.5로 회복)
- `filter_gt_by_role.py`로 `gameinfo.ini`의 역할 정보를 이용해 player+goalkeeper만 남긴 GT(`SoccerNet-players-test` 벤치마크)를 별도로 만들어 공정 비교 확보

**6. 이론적 상한선(gt.txt 입력) 대비 실제 성능 격차 확인**
- 24개 GT 기준 이론적 최대(strongsort_gt) HOTA 82.958 대비 실제 최고 성능(strongsort_h250_baseline) HOTA 44.943 (약 38점, 54% 수준)으로 격차 존재
- gt.txt 입력에서는 StrongSORT(AssA 77.689)가 ByteTrack(65.618)·BoT-SORT(63.964)를 크게 앞섬 - 다만 이것이 "검출 품질이 좋아질수록 점진적으로 StrongSORT가 유리해진다"는 일반적 경향인지, 아니면 gt.txt라는 특수 조건에서만 나타나는 패턴인지는 두 지점(gt·h250_baseline)만으로는 판단할 수 없음. 

## 한계

**1. 단일 클립(SNMOT-116) 기준 결과임**
- `actionClass=Corner`(코너킥) 상황으로, 선수 밀집도가 높은 편에 속하는 클립
- 다른 성격의 클립(예: Goal/Yellow card는 카메라 컷이, Clearance는 프레임 이탈이 잦을 것으로 예상됨)에서는 검출기·트래커 우열이 달라질 수 있음 - 일반화 전 추가 클립 검증 필요

**2. 후처리 단계는 미완료**
- StrongSORT까지는 트래킹 완료했으나 `boxmot` 내장 GTA 후처리(`generate`/`associate`)는 `torchreid` 패키지의 하위 모듈 구조 불일치(`torchreid.utils` 부재)로 재현 불가 확인, 이번 실험에서는 보류
- 최종 IDs가 가장 좋은 조합(BoT-SORT, 57개)도 여전히 정답(24개)의 2배를 넘어 개선 여지가 남아있으나, 원인(선수 밀집 혼동 vs 검출 불안정 vs 기타)은 이번에 분석하지 않음

**3. player+goalkeeper 기준 GT의 가정**
- 골키퍼가 시각적으로 선수와 유사해 person 단일 클래스 모델도 검출 가능하다는 전제로 player+goalkeeper를 합쳐 24개로 필터링함 - 실제 검출 결과에서 골키퍼가 얼마나 정확히 잡혔는지는 역할별로 따로 확인하지 않음
- referee 및 ball 트래킹 품질은 이번 비교에서 평가하지 않음

**4. conf 임계값 미탐색**
- 모든 검출 조건에서 `--conf 0.25`로 고정, 임계값 조정에 따른 재현율-정밀도 트레이드오프는 탐색하지 않음 - DetA 개선 여지가 남아있을 가능성 있음