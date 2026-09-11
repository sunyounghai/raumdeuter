import argparse
from pathlib import Path

from src.detection.config import EvalConfig
from src.detection.model_registry import get_model_config, MODELS
from src.detection.eval_common import compute_map, compute_fixed_threshold_metrics, write_metrics_file
from src.detection.paths import VAL_IMAGES_BALL, VAL_LABELS_BALL
from src.common.paths import RESULTS_DIR


def evaluate_ball_baseline(model_name: str, cfg: EvalConfig) -> dict:
    model_cfg = get_model_config(model_name)
    if "ball_class" not in model_cfg:
        raise ValueError(
            f"{model_name}의 ball 클래스 인덱스가 정의되어 있지 않습니다. "
            f"model_registry.py의 MODELS[\"{model_name}\"]에 'ball_class'를 추가하세요"
        )

    weights = model_cfg["weights_path"]
    ball_idx = model_cfg["ball_class"]
    ball_class_map = {ball_idx: 0}

    print(f"{model_name} / baseline / ball 평가 중 "
          f"(weights={weights}, pred_class={ball_idx})")

    map_metrics = compute_map(
        Path(weights), VAL_IMAGES_BALL, VAL_LABELS_BALL, ball_class_map,
        map_conf_floor = cfg.map_conf_floor,
    )
    fixed_metrics = compute_fixed_threshold_metrics(
        Path(weights), VAL_IMAGES_BALL, VAL_LABELS_BALL, ball_class_map,
        conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres,
    )

    result_path = RESULTS_DIR / "detection" / f"{model_name}_baseline_ball_metrics.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    cfg = EvalConfig(conf_thres=args.conf, iou_thres=args.iou)

    evaluate_ball_baseline(args.model, cfg)

if __name__ == "__main__":
    main()