"""
COCO / Roboflow / H250 세 모델의 baseline, finetuned 평가를 하나로 통합

사용법:
    python evaluate.py --model coco --stage baseline
    python evaluate.py --model roboflow --stage finetuned
    python evaluate.py --model h250 --stage finetuned --conf 0.3 --iout 0.4
    python evaluate.py -- all # baseline_finetuned 6개 조합 전부 실행, 통합 비교표 출력
"""

import argparse
from pathlib import Path

from config import EvalConfig
from model_registry import MODELS, get_model_config, finetuned_weights_path
from eval_common import compute_map, compute_fixed_threshold_metrics, write_metrics_file
from paths import VAL_IMAGES, VAL_LABELS, RESULTS_DIR

def evaluate_one(model_name: str, stage: str, cfg: EvalConfig) -> dict:
    model_cfg = get_model_config(model_name)

    if stage == "baseline":
        weights = model_cfg["weights_path"]
        class_map = model_cfg["pretrained_class_map"]
    elif stage == "finetuned":
        weights = finetuned_weights_path(model_name)
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"파인튜닝된 가중치가 없습니다: {weights}\n"
                f"먼저 'python train.py --model {model_name}'을 실행하세요."
            )
        class_map = None # 파인튜닝 후에는 단일 클래스(player=0)
    else:
        raise ValueError("stage는 'baseline' 또는 'finetuned'이어야 합니다.")

    print(f"[{model_name} / {stage} 평가 중... (weights={weights})]")

    map_metrics = compute_map(
        Path(weights), VAL_IMAGES, VAL_LABELS, class_map,
        map_conf_floor=cfg.map_conf_floor,
    )
    fixed_metrics = compute_fixed_threshold_metrics(
        Path(weights), VAL_IMAGES, VAL_LABELS, class_map,
        conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{model_name}_{stage}_metrics.txt"
    write_metrics_file(
        result_path,
        header=f"[{model_name} / {stage}]",
        map_metrics=map_metrics,
        fixed_metrics=fixed_metrics,
    )
    print(f"  mAP50={map_metrics['mAP50']:.4f}  mAP50-95={map_metrics['mAP50-95']:.4f}  "
          f"P={fixed_metrics['Precision']:.4f}. R={fixed_metrics['Recall']:.4f}  "
          f"F1={fixed_metrics['F1']:.4f}")
    print(f"    결좌 저장됨: {result_path}")

    return {**map_metrics, **fixed_metrics}


def main():
    default_cfg = EvalConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--stage", choices=["baseline", "finetuned"])
    parser.add_argument("--conf", type=float, default=default_cfg.conf_thres)
    parser.add_argument("--iou", type=float, default=default_cfg.iou_thres)
    parser.add_argument("--all", action="store_true",
                        help="6개 조합(모델 3개 x baseline/finetuned) 평가 후 통합 비교표 출력")
    args = parser.parse_args()

    cfg = EvalConfig(conf_thres=args.conf, iou_thres=args.iou)

    if args.all:
        results = {}
        for model_name in MODELS:
            for stage in ("baseline", "finetuned"):
                try:
                    results[f"{model_name}_{stage}"] = evaluate_one(model_name, stage, cfg)
                except FileNotFoundError as e:
                    print(f"[건너뜀] {model_name}/{stage}: {e}")

        if results:
            print(f"\n===== 통합 비교표 (conf={cfg.conf_thres}, iou={cfg.iou_thres}, "
                  f"동일 기준) =====")
            header = f"{'구분':<20}" + "".join(f"{k:>12}" for k in
                     ["mAP50", "mAP50-95", "Precision", "Recall", "F1"])
            print(header)
            for name, m in results.items():
                row = f"{name:<20}"
                for k in ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]:
                    row += f"{m[k]:>12.4f}"
                print(row)

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            summary_path = RESULTS_DIR / "unified_comparision.txt"
            with open(summary_path, "w") as f:
                f.write(f"통합 비교표 (conf={cfg.conf_thres}, iou={cfg.iou_thres}), "
                        f"모든 모델 동일 기준)\n\n")
                f.write(header + "\n")
                for name, m in results.items():
                    row = f"{name:<20}"
                    for k in ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]:
                        row += f"{m[k]:12.4f}"
                    f.write(row + "\n")
            print(f"\n결과 저장됨: {summary_path}")
        return

    if not args.model or not args.stage:
        parser.error("--model과 --stage를 지정하거나 전체 비교는 --all을 사용하세요.")

    evaluate_one(args.model, args.stage, cfg)


if __name__ == "__main__":
    main()