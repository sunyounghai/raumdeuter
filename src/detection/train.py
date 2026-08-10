"""
COCO / Roboflow / H250 세 모델의 파인튜닝을 하나로 통합

사용법:
    python train.py --model coco
    python train.py --model roboflow
    python train.py --model h250
    python trian.py --model h250 --epochs 80 --batch 16
"""

import argparse

from ultralytics import YOLO

from src.detection.config import TrainConfig
from src.detection.model_registry import get_model_config, finetuned_weights_path

from src.detection.paths import DATASET_YAML
from src.common.paths import RUNS_DIR

def main():
    cfg = TrainConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["coco", "roboflow", "h250"])
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--imgsz", type=int, default=cfg.imgsz)
    parser.add_argument("--batch", type=int, default=cfg.batch)
    args = parser.parse_args()

    model_cfg = get_model_config(args.model)
    weights = model_cfg["weights_path"]
    run_name = model_cfg["run_name"]

    print(f"[{args.model}] 파인튜닝 시작 (기반 모델: {weights})")
    model = YOLO(weights)

    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
    )

    best_weights = finetuned_weights_path(args.model)
    print(f"\n[{args.model}] 학습 완료, 가중치 저장됨: {best_weights}")

if __name__ == "__main__":
    main()