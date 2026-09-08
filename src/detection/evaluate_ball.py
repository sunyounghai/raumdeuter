import argparse
from pathlib import Path

from src.detection.config import EvalConfig
from src.detection.model_registry import get_model_config
from src.detection.eval_common import compute_map, compute_fixed_threshold_metrics, write_metrics_file
from src.common.paths import RESULTS_DIR, RAW_DATA_DIR

BALL_SOURCE_CLASS = {
    "h250": 0,
    "roboflow": 0,
    "coco": 32,
}

VAL_IMAGES_BALL = RAW_DATA_DIR / "SoccerNet/tracking-2023/test/SNMOT-116/img1"
VAL_LABELS_BALL = RAW_DATA_DIR / "yolo_dataset_ball/val/labels"

def evaluate_ball_baseline(model_name: str, cfg: EvalConfig) -> dict:
    if model_name not in BALL_SOURCE_CLASS:
        raise ValueError(
            f"{model_name}의 ball 클래스 인덱스가 정의되어 있지 않습니다. "
            f"BALL_SOURCE_CLASS에 추가하세요"
        )

    model_cfg = get_model_config(model_name)
    weights = model_cfg["weights_path"]
    ball_idx = BALL_SOURCE_CLASS[model_name]
    ball_class_map = {ball_idx: 0}

    print(f"]{model_name} / baseline / ball 평가 중 "
          f"(weights={weights}, pred_class={ball_idx})")

    map_metrics = compute_map(
        Path(weights), VAL_IMAGES_BALL, VAL_LABELS_BALL, ball_class_map,
        map_conf_floor = cfg.map_conf_floor,
    )
    fixed_metrics = compute_fixed_threshold_metrics(
        Path(weights), VAL_IMAGES_BALL, VAL_LABELS_BALL, ball_class_map,
        conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{model_name}_baseline_ball_metrics.txt"
    write_metrics_file(
        result_path,
        header=f"[{model_name} / baseline / ball]",
        map_metrics=map_metrics,
        fixed_metrics=fixed_metrics,
    )
    print(f"  mAP50={map_metrics['mAP50']:.4f}  mAP50-95={map_metrics['mAP50-95']:.4f}  "
          f"P={fixed_metrics['Precision']:.4f}  R={fixed_metrics['Recall']:.4f}  "
          f"F1={fixed_metrics['F1']:.4f}")
    print(f"  저장됨: {result_path}")

    return {**map_metrics, **fixed_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(BALL_SOURCE_CLASS.keys()))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    cfg = EvalConfig(conf_thres=args.conf, iou_thres=args.iou)

    evaluate_ball_baseline(args.model, cfg)

if __name__ == "__main__":
    main()