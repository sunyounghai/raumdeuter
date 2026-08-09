from dataclasses import dataclass

@dataclass
class TrainConfig:
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8

@dataclass
class EvalConfig:
    conf_thres: float = 0.25 # P/R/F1 confidence
    iou_thres: float = 0.5 # 매칭 및 mAP50 기준 IoU
    imgsz: int = 640
    map_conf_floor: float = 0.001 # mAP 계산 시 PR curve 전체를 위한 낮은 conf