"""
Roboflow football-player-detection 모델 평가 (수정판).

문제였던 지점: 이 모델의 클래스 번호(0=ball,1=goalkeeper,2=player,3=referee)가
우리 GT의 클래스 번호(0=player)와 다르게 매겨져 있어서, ultralytics의 val()이
자동으로는 올바르게 매칭하지 못했습니다 ("no labels found" 경고 발생).

해결: model.val()의 자동 채점에 의존하지 않고, 예측 결과(goalkeeper+player)를
직접 받아서 GT와 IoU 기준으로 수동 매칭해 Precision/Recall/F1을 계산합니다.

사용법:
    python evaluate_roboflow_baseline.py
"""

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
VAL_IMAGES = BASE / "val" / "images"
VAL_LABELS = BASE / "val" / "labels"
ROBOFLOW_WEIGHTS = BASE / "football-player-detection.pt"

TARGET_CLASSES = [1, 2]  # goalkeeper, player -> 전부 우리 'player'로 취급
CONF_THRES = 0.25
IOU_THRES_FOR_MATCH = 0.5  # mAP50과 같은 기준(IoU 0.5)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate(model_path: str, target_classes):
    model = YOLO(model_path)
    image_files = sorted(VAL_IMAGES.glob("*.jpg"))

    total_gt = 0
    total_pred = 0
    total_tp = 0

    for img_path in image_files:
        result = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            classes=target_classes if target_classes else None,
            verbose=False,
        )[0]

        img_h, img_w = result.orig_shape
        gt_boxes = load_gt_boxes(VAL_LABELS / f"{img_path.stem}.txt", img_w, img_h)
        pred_boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) else []

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        matched_gt = set()
        for pbox in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, gbox in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                val = iou(pbox, gbox)
                if val > best_iou:
                    best_iou = val
                    best_idx = i
            if best_iou >= IOU_THRES_FOR_MATCH:
                matched_gt.add(best_idx)
                total_tp += 1

    precision = total_tp / total_pred if total_pred else 0
    recall = total_tp / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "GT_count": total_gt,
        "Pred_count": total_pred,
        "TP": total_tp,
    }


def main():
    print(f"Roboflow 모델 로드: {ROBOFLOW_WEIGHTS}")
    print("'player'로 취급할 클래스: goalkeeper(1) + player(2)\n")

    metrics = evaluate(str(ROBOFLOW_WEIGHTS), TARGET_CLASSES)

    print("===== Roboflow 베이스라인 결과 (파인튜닝 전, IoU 50% 기준) =====")
    print(f"GT 총 개수   : {metrics['GT_count']}")
    print(f"예측 총 개수 : {metrics['Pred_count']}")
    print(f"맞은 개수(TP): {metrics['TP']}")
    print(f"Precision   : {metrics['Precision']:.4f}")
    print(f"Recall      : {metrics['Recall']:.4f}")
    print(f"F1          : {metrics['F1']:.4f}")

    result_path = BASE / "roboflow_baseline_metrics.txt"
    with open(result_path, "w") as f:
        f.write("Roboflow 모델 베이스라인 (파인튜닝 전, IoU 50% 기준 자체 계산)\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall   : {metrics['Recall']:.4f}\n")
        f.write(f"F1       : {metrics['F1']:.4f}\n")
    print(f"\n결과 저장됨: {result_path}")
    print("\n참고: 이 방식은 mAP(여러 IoU/confidence 임계값 평균)가 아니라")
    print("고정 IoU 50%, confidence 0.25 기준의 단순 Precision/Recall/F1입니다.")
    print("파인튜닝 후 모델과 비교할 땐 반드시 같은 방식(evaluate_finetuned_generic.py)으로 재야 공정합니다.")


if __name__ == "__main__":
    main()
