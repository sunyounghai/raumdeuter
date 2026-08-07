import os
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = Path(os.environ.get("RAUMDEUTER_ROOT", _SRC_DIR.parent)).resolve()

DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"

DATASET_YAML = DATA_DIR / "dataset.yaml"
VAL_IMAGES = DATA_DIR / "val" / "images"
VAL_LABELS = DATA_DIR / "val" / "labels"
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_LABELS = DATA_DIR / "train" / "labels"

RESULTS_DIR = PROJECT_ROOT / "results"