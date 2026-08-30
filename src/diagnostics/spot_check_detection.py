"""
Detection 결과를 GT와 매칭해서 TP/FP/FN을 색깔로 구분해 프레임에 그려 저장
  - GT 매칭됨(TP)   : 초록 실선
  - GT 놓침(FN)     : 주황 실선 + "FN" 라벨
  - 예측 매칭됨(TP) : 파랑 점선 
  - 예측 오탐(FP)   : 빨강 실선 + "FP" 라벨

사용 예:
  python -m src.diagnostics.spot_check_detection \
      --model roboflow --stage baseline \
      --out docs/detection/spotcheck/roboflow_baseline

  python -m src.diagnostics.spot_check_detection \
      --model roboflow --stage finetuned \
      --out docs/detection/spotcheck/roboflow_finetuned

  # 빠르게 5장만 미리보기
  python -m src.diagnostics.spot_check_detection \
      --model roboflow --stage baseline \
      --out docs/detection/spotcheck/preview --limit 5
"""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


from src.detection.config import EvalConfig
from src.detection.model_registry import get_model_config, finetuned_weights_path
from src.detection.paths import VAL_IMAGES, VAL_LABELS
from src.detection.eval_common import load_gt_boxes, iou, collect_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Detection TP/FP/FN을 프레임에 오버레이해서 저장")
    parser.add_argument("--model", required=True, help="coco / roboflow / h250")
    parser.add_argument("--stage", required=True, choices=["baseline", "finetuned"])
    default_cfg = EvalConfig()
    parser.add_argument("--conf", type=float, default=default_cfg.conf_thres)
    parser.add_argument("--iou", type=float, default=default_cfg.iou_thres)
    parser.add_argument("--limit", type=int, default=None, help="확인할 이미지 수 제한 (기본: val 전체)")
    parser.add_argument("--out", type=Path, required=True, help="결과 저장 폴더")
    return parser.parse_args()


def match_tp_fp_fn(pred_boxes: list, gt_boxes: list, iou_thres: float):
    """
    confidence 내림차순 그리디 매칭, eval_common.compute_fixed_threshold_metrics와 동일한 방식
    반환: matched_pred_idx(TP인 예측 idx 집합), matched_gt_idx(TP로 매칭된 GT idx 집합)
    """
    matched_gt, matched_pred = set(), set()
    for i in range(len(pred_boxes)): # 이미 confidence 내림차순으로 들어온다고 가정
        best_iou, best_idx = 0.0, -1
        for j, gbox in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            val = iou(pred_boxes[i], gbox)
            if val > best_iou:
                best_iou, best_idx = val, j
        if best_iou >= iou_thres:
            matched_gt.add(best_idx)
            matched_pred.add(i)
    return matched_pred, matched_gt
            

def draw_frame(img_path: Path, label_path: Path, model: YOLO, class_map, conf: float,
               iou_thres: float, out_dir: Path) -> None:
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)

    pred_boxes, scores, _labels, img_w, img_h = collect_predictions(model, img_path, conf, class_map)
    gt_boxes = load_gt_boxes(label_path, img_w, img_h)

    # confidence 내림차순 정렬
    order = sorted(range(len(pred_boxes)), key=lambda i: scores[i], reverse=True)
    pred_boxes_sorted = [pred_boxes[i] for i in order]
    matched_pred, matched_gt = match_tp_fp_fn(pred_boxes_sorted, gt_boxes, iou_thres)

    fn_count = fp_count = 0

    # GT: 매칭됨=초록, 놓침(FN)=주황
    for j, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = box
        if j in matched_gt:
            color = "lime"
        else:
            color = "orange"
            fn_count += 1
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        if j not in matched_gt:
            ax.text(x1, y1 - 5, "FN", color="white", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="orange", alpha=0.85, pad=1))


    # 예측: 매칭됨(TP)=파랑 점선, 오탐(FP)=빨강
    for i, box in enumerate(pred_boxes_sorted):
        x1, y1, x2, y2 = box
        if i in matched_pred:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 -y1, linewidth=1.5,
                                     edgecolor="deepskyblue", facecolor="none", linestyle="--")
        else:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                     edgecolor="red", facecolor="none")
            ax.text(x1, y1 - 5, "FP", color="white", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="red", alpha=0.85, pad=1))
            fp_count += 1
        ax.add_patch(rect)

    ax.set_title(f"{img_path.name}  |  GT:{len(gt_boxes)}  Pred:{len(pred_boxes)}  FN:{fn_count}  FP:{fp_count}")
    ax.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / img_path.name, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)    


def main():
    args = parse_args()
    model_cfg = get_model_config(args.model)

    if args.stage == "baseline":
        weights = model_cfg["weights_path"]
        class_map = model_cfg["pretrained_class_map"]
    else:
        weights = finetuned_weights_path(args.model)
        if not Path(weights).exists():
            raise FileNotFoundError(f"파인튜닝된 가중치가 없습니다: {weights}")
        class_map = None

    model = YOLO(str(weights))
    images = sorted(Path(VAL_IMAGES).glob("*.jpg"))
    if args.limit:
        images = images[:args.limit]

    print(f"[{args.model} / {args.stage}] 확인할 이미지: {len(images)}개 (weights={weights})")

    for img_path in images:
        label_path = Path(VAL_LABELS) / f"{img_path.stem}.txt"
        draw_frame(img_path, label_path, model, class_map, args.conf, args.iou, args.out)

    print(f"\n전체 저장 완료 -> {args.out}")


if __name__ == "__main__":
    main()