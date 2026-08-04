"""
파인튜닝 전 YOLO 모델의 val set 성능(베이스라인)을 측정합니다.

전제:
- COCO로 사전학습된 YOLO는 'person' 클래스(80개 클래스 중 인덱스 0)를 이미 알고 있음
- 우리 GT는 'player' 단일 클래스(인덱스 0)
- 그래서 YOLO 모델이 예측하는 'person' 클래스를 'player'와 동일한 것으로 간주해서 비교

사용법:
    python evaluate_baseline.py
"""

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
DATASET_YAML = BASE / "dataset.yaml"

# COCO 사전학습 가중치. 처음 실행 시 자동 다운로드됨 (yolov8n = 가장 가벼운 버전)
BASELINE_WEIGHTS = "yolov8n.pt"


def main():
    print(f"베이스라인 모델 로드: {BASELINE_WEIGHTS}")
    model = YOLO(BASELINE_WEIGHTS)

    # COCO 클래스 중 'person'만 우리 'player'와 비교 대상으로 사용
    # val() 실행 시 GT 클래스가 1개(player)뿐이므로,
    # 모델이 예측한 person(class 0)과 자동으로 매칭되어 계산됨
    print("val set에 대해 평가 실행 중...")
    metrics = model.val(
        data=str(DATASET_YAML),
        split="val",
        classes=[0],       # COCO의 'person' 클래스만 예측 대상으로 제한
        save_json=False,
        plots=True,
    )

    print("\n===== 베이스라인 결과 =====")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    # 결과를 파일로도 저장 (나중에 파인튜닝 후와 비교하기 위함)
    result_path = BASE / "baseline_metrics.txt"
    with open(result_path, "w") as f:
        f.write("베이스라인 (YOLO, 파인튜닝 전)\n")
        f.write(f"mAP50    : {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95 : {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall   : {metrics.box.mr:.4f}\n")
    print(f"\n결과 저장됨: {result_path}")


if __name__ == "__main__":
    main()
