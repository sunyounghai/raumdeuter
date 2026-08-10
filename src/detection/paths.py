from src.common.paths import RAW_DATA_DIR

DATASET_DIR = RAW_DATA_DIR / "yolo_dataset"     # 학습에 실제로 쓰이는 라벨링 데이터셋
DATASET_YAML = DATASET_DIR / "dataset.yaml"
VAL_IMAGES = DATASET_DIR / "val" / "images"
VAL_LABELS = DATASET_DIR / "val" / "labels"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

