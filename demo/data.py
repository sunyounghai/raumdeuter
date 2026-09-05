"""
Raumdeuter 데모 앱 — 실측 데이터 단일 소스(single source of truth)

이 파일에 있는 숫자는 전부 실제 실험 결과입니다.
- Detection: docs/detection/experiment_results.md
- Tracking:  docs/tracking/experiment_results.md
- Calibration: docs/calibration/spotcheck_results.md

새 실험을 돌려서 값이 바뀌면 이 파일 하나만 고치면 모든 페이지에 반영됩니다.
"""

# ── Detection: 검출기 3종(COCO/Roboflow/H250) baseline vs finetuned mAP50-95 ──
DETECTION_MODELS = ["COCO", "Roboflow", "H250"]
DETECTION_BASELINE = [0.5514, 0.3857, 0.6602]
DETECTION_FINETUNED = [0.7856, 0.8337, 0.8122]
DETECTION_ARCH = {
    "COCO": "YOLOv8n · COCO 사전학습 (person→player)",
    "Roboflow": "YOLOv8 · DFL Bundesliga 방송영상 학습 (goalkeeper 별도→리매핑)",
    "H250": "YOLOv8n · SoccerNet v3 학술 데이터셋 (Ball/Person 2-class)",
}

# ── Tracking: 검출기별 HOTA (BoT-SORT 고정) ──
TRACKING_DETECTORS = ["H250\nbaseline", "H250\nfinetuned", "Roboflow\nbaseline", "COCO\nbaseline"]
TRACKING_HOTA = [44.108, 43.463, 37.328, 37.220]

# ── Tracking: 트래커별 예측 트랙 수 (GT=24) ──
TRACKING_GT_IDS = 24
TRACKING_TRACKER_IDS = {"StrongSORT": 91, "BoT-SORT": 57, "ByteTrack": 77}
TRACKING_FINAL_CHOICE = "h250_baseline + BoT-SORT"
TRACKING_FINAL_STATS = {"HOTA": 44.1, "IDF1": 53.5, "IDs": 57}

# ── Calibration: keypoint 구간별 GT 실측 오차(m) ──
CALIB_BUCKETS = ["keypoint 4개", "keypoint 5개", "keypoint 6개", "keypoint 7개+"]
CALIB_GT_ERROR_M = [2.27, 1.00, 0.99, 0.71]
CALIB_SUCCESS_RATE = {"0~3개": 0, "4개+": 100}
CALIB_TOTAL_FRAMES = 391
CALIB_SUCCESS_FRAMES = 311

# ── 파이프라인 진행 상태 ──
PIPELINE_STATUS = [
    {"name": "01 Dataset", "status": "done", "detail": "Label Studio 라벨링 (Player/Ball/Pitch Point)"},
    {"name": "02 Detection", "status": "partial", "detail": "선수 검출 완료 · 공 검출 진행중"},
    {"name": "03 Tracking", "status": "done", "detail": "h250_baseline + BoT-SORT 채택"},
    {"name": "04 Homography", "status": "done", "detail": "391프레임 스팟체크 + GT 실측 검증"},
    {"name": "05 Spatial Analytics", "status": "partial", "detail": "공간 침투 지수 산출 진행중"},
]