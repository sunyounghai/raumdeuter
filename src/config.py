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