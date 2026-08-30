"""
평가 관련 공통 로직

1. 주 지표 (mAP50, mAP50-95) - torchmetrics 기반
2. 보조 지표 (conf=0.25 고정 Precision/Recall/F1)
"""

from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO

def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """YOLO 포맷(class cx cy w h, 정규화) 라벨을 xyxy 픽셀 좌표로 변환"""
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


def iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def collect_predictions(model: YOLO, img_path: Path, conf: float,
                         class_map: dict[int, int] | None):
    """예측 박스를 뽑고, class_map이 주어지면 클래스를 GT 인덱스로 리매핑
    class_map이 None이면 파인튜닝 후 모델(이미 단일 클래스)로 간주해 필터링 없이 사용"""
    result = model.predict(source=str(img_path), conf=conf, verbose=False)[0]
    img_h, img_w = result.orig_shape

    boxes, scores, labels = [], [], []
    if len(result.boxes):
        raw_boxes = result.boxes.xyxy.cpu().numpy().tolist()
        raw_confs = result.boxes.conf.cpu().numpy().tolist()
        raw_cls = result.boxes.cls.cpu().numpy().tolist()
        for box, sc, cls in zip(raw_boxes, raw_confs, raw_cls):
            cls = int(cls)
            if class_map is None:
                boxes.append(box)
                scores.append(sc)
                labels.append(0)
            elif cls in class_map:
                boxes.append(box)
                scores.append(sc)
                labels.append(class_map[cls])

    return boxes, scores, labels, img_w, img_h


def compute_map(model_path: Path, val_images: Path, val_labels: Path,
                class_map: dict[int, int] | None, map_conf_floor: float = 0.001) -> dict:
    """torchmetrics 기반 mAP50 / mAP50-95"""
    model = YOLO(str(model_path))
    metric = MeanAveragePrecision(iou_type="bbox")

    for img_path in sorted(Path(val_images).glob("*.jpg")):
        boxes, scores, labels, img_w, img_h = collect_predictions(
            model, img_path, map_conf_floor, class_map
        )
        gt_boxes = load_gt_boxes(Path(val_labels) / f"{img_path.stem}.txt", img_w, img_h)

        pred = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes
                        else torch.zeros((0, 4)),
            "scores": torch.tensor(scores, dtype=torch.float32) if scores
                        else torch.zeros((0,)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels
                        else torch.zeros((0,), dtype=torch.int64),
        }
        target = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes
                        else torch.zeros((0, 4)),
            "labels": torch.zeros((len(gt_boxes),), dtype=torch.int64),
        }
        metric.update(preds=[pred], target=[target])

    result = metric.compute()
    return {
        "mAP50": result["map_50"].item(),
        "mAP50-95": result["map"].item(),
        "mAP75": result["map_75"].item(),
    }

def compute_fixed_threshold_metrics(model_path: Path, val_images: Path, val_labels: Path,
                                    class_map: dict[int, int] | None,
                                    conf_thres: float, iou_thres: float) -> dict:
    """"conf 고정, confidence 내림차순 그리디 매칭
    정렬을 항상 적용해 파일마다 매칭 순서를 일치"""
    model = YOLO(str(model_path))

    total_gt = total_pred = total_tp = 0

    for img_path in sorted(Path(val_images).glob("*.jpg")):
        boxes, scores, labels, img_w, img_h = collect_predictions(
            model, img_path, conf_thres, class_map
        )
        gt_boxes = load_gt_boxes(Path(val_labels) / f"{img_path.stem}.txt", img_w, img_h)

        total_gt += len(gt_boxes)
        total_pred += len(boxes)

        order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        matched_gt = set()
        for i in order:
            best_iou, best_idx = 0.0, -1
            for j, gbox in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                val = iou(boxes[i], gbox)
                if val > best_iou:
                    best_iou, best_idx = val, j
            if best_iou >= iou_thres:
                matched_gt.add(best_idx)
                total_tp += 1

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "Precision": precision, "Recall": recall, "F1": f1,
        "GT_count": total_gt, "Pred_count": total_pred, "TP": total_tp,
        "conf_thres": conf_thres, "iou_thres": iou_thres,
    }


def write_metrics_file(path: Path, header:str, map_metrics:dict, fixed_metrics: dict):
    """"mAP와 conf 고정 P/R/F1, 실제 쓰인 conf/iou 값을 기록"""
    lines = [header, ""]
    lines.append("[주 지표: mAP, conf 전체 스윕]")
    for k in ("mAP50", "mAP50-95", "mAP75"):
        lines.append(f"{k}: {map_metrics[k]:.4f}")
    lines.append("")
    lines.append(f"[보조 지표: conf={fixed_metrics['conf_thres']}, "
                  f"iou={fixed_metrics['iou_thres']} 고정]")
    for k in ("Precision", "Recall", "F1"):
        lines.append(f"{k}: {fixed_metrics[k]:.4f}")
    lines.append(f"GT_count: {fixed_metrics['GT_count']}")
    lines.append(f"Pred_count: {fixed_metrics['Pred_count']}")
    lines.append(f"TP: {fixed_metrics['TP']}")
    path.write_text("\n".join(lines) + "\n")