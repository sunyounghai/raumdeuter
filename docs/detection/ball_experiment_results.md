# 실험 결과: COCO / Roboflow / H250 공 검출 베이스라인 비교

## 실행 조건

- **평가 대상**: coco / roboflow / h250 3개 모델의 baseline

- **GT 규모**: SoccerNet-Tracking SNMOT-116 (1개 시퀀스, 750프레임 중 공이 등장하는 724프레임)
    - GT 소스: 자체 라벨링이 아니라 `gt.txt`(MOT 포맷)의 ball tracklet(트랙 ID)을 `gt_to_ball_detection_labels.py`로 YOLO 포맷 변환

- **클래스 매핑**: baseline 3개 모델 각자의 원래 클래스 체계에서 ball 인덱스만 추출
(`BALL_SOURCE_CLASS`: h250=0, roboflow=0, coco=32(표준 COCO 'sports ball'))
- **재현**: 단일 실행 (seed 고정 없이 1회)

## 결과

| 구분 | mAP50 | mAP50-95 | mAP75 | Precision(iou=0.5) | Recall(iou=0.5) | F1 | Pred_count/GT_count |
|---|---|---|---|---|---|---|---|
| h250_baseline | 0.0757 | 0.0283 | 0.0136 | 0.4539 | 0.0953 | 0.1575 | 152/724 |
| coco_baseline | 0.0468 | 0.0200 | 0.0135 | 0.4375 | 0.0580 | 0.1024 | 96/724 |
| roboflow_baseline | 0.0383 | 0.0110 | - | 0.3037 | 0.0566 | 0.0955 | -/724 |