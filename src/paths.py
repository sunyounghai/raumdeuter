import os
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = Path(os.environ.get("RAUMDEUTER_ROOT", _SRC_DIR.parent)).resolve()

RAW_DATA_DIR = PROJECT_ROOT / "data"            # raw_video, frames_final 등
DATASET_DIR = RAW_DATA_DIR / "yolo_dataset"     # 학습에 실제로 쓰이는 라벨링 데이터셋
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"

DATASET_YAML = DATASET_DIR / "dataset.yaml"
VAL_IMAGES = DATASET_DIR / "val" / "images"
VAL_LABELS = DATASET_DIR / "val" / "labels"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

RESULTS_DIR = PROJECT_ROOT / "results"

