# 데이터 사용 현황 

각 파이프라인 단계에서 어떤 데이터를 사용하는지 정리

## 단계별 데이터

| 단계 | 데이터 소스 | 범위 | 결정 근거 |
|---|---|---|---|
| 선수 검출 평가 | SoccerNet-Tracking (`SNMOT-*/gt.txt`) | 5개 시퀀스 (아래 표) | 기존 `yolo_dataset` 대신 tracking GT와 공통 기준으로 통일 |
| 선수 트래킹 평가 | SoccerNet-Tracking (`SNMOT-*/gt.txt`) | 5개 시퀀스 | 검출 평가와 동일 영상이어야 검출 문제 vs 트래킹 문제 구분 가능 |
| 트랙 오매칭 진단 | SoccerNet-Tracking | **SNMOT-116/144/122/119/124** (5개) | actionClass 분류: 혼잡군(Corner/Penalty/Direct free-kick) vs 분산군(Clearance/Kick-off) |
| 공 검출 baseline 확인 | SoccerNet-Tracking (`gt.txt`의 ball tracklet) | SNMOT-116 (1개, 확장 예정) | gt.txt에 이미 ball bbox 포함 - 새 라벨링 불필요 |
| 공 검출 파인튜닝 | SoccerNet-Tracking | 확장 필요 (여러 시퀀스 풀링) | SNMOT-116 한 개(750프레임)로는 학습 표본 부족 |
| 침투 지표 계산 | SoccerNet-Tracking | 15~20개 시퀀스 (아직 미확정) | actionClass 다양성 확보해 지표의 타당성 검증 |
| 이벤트 인식 | SoccerNet Action Spotting | 공개 baseline 평가부터 시작, 필요 시 파인튜닝 | 85개(SNMOT-116~200)는 클래스당 평균 7개뿐이라 학습에 부족 |
| 호모그래피 검증 | SoccerNet-Tracking | keypoint 개수 기준 (actionClass 무관) | 카메라 각도/라인 가시성이 중요, 경기 상황과 무관 |

## 5개 시퀀스 상세

| 시퀀스 | actionClass | 그룹 |
|---|---|---|
| SNMOT-116 | Corner | 혼잡 |
| SNMOT-144 | Penalty | 혼잡 |
| SNMOT-122 | Direct free-kick | 혼잡 |
| SNMOT-119 | Clearance | 분산 |
| SNMOT-124 | Kick-off | 분산 |
