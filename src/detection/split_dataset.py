"""
data/yolo_dataset/images, labels를 train/val로 나눠서
Ultralytics YOLO가 기대하는 구조로 재배치합니다.

실행 전 구조:
    data/yolo_dataset/
        images/*.jpg
        labels/*.txt
        classes.txt

실행 후 구조:
    data/yolo_dataset/
        train/
            images/*.jpg
            labels/*.txt
        val/
            images/*.jpg
            labels/*.txt
        classes.txt
        dataset.yaml
"""

import random
import shutil


from src.detection.paths import (
    DATASET_DIR,
    DATASET_YAML,
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VAL_IMAGES,
    VAL_LABELS,
)

# ── 설정 ──
SEED = 42          # 재현성을 위해 고정 (항상 같은 분리 결과)
VAL_RATIO = 0.2     # val 비율 20%

SRC_IMAGES = DATASET_DIR / "images"
SRC_LABELS = DATASET_DIR / "labels"
CLASSES_FILE = DATASET_DIR / "classes.txt"


def main():
    random.seed(SEED)

    image_files = sorted(SRC_IMAGES.glob("*.jpg"))
    if not image_files:
        print(f"이미지가 없습니다: {SRC_IMAGES}")
        return

    # 라벨이 있는 이미지만 대상으로 함 (혹시 매칭 안 된 게 있으면 제외)
    paired = []
    for img in image_files:
        label = SRC_LABELS / f"{img.stem}.txt"
        if label.exists():
            paired.append((img, label))
        else:
            print(f"라벨 없음, 제외: {img.name}")

    random.shuffle(paired)

    n_val = round(len(paired) * VAL_RATIO)
    val_set = paired[:n_val]
    train_set = paired[n_val:]

    print(f"전체: {len(paired)}장 -> train: {len(train_set)}장, val: {len(val_set)}장")

    split_targets = [
        (train_set, TRAIN_IMAGES, TRAIN_LABELS),
        (val_set, VAL_IMAGES, VAL_LABELS),
    ]
    
    for split_data, img_dir, lbl_dir in split_targets:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, label in split_data:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(label, lbl_dir / label.name)

    # dataset.yaml 생성 (ultralytics가 학습 시 참조하는 설정 파일)
    classes = CLASSES_FILE.read_text().strip().splitlines()
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))

    yaml_content = f"""# 자동 생성됨 (split_dataset.py)
path: {DATASET_DIR.resolve()}
train: train/images
val: val/images

names:
{names_block}
"""
    DATASET_YAML.write_text(yaml_content)

    print("dataset.yaml 생성 완료")
    print(f"경로: {DATASET_YAML}")


if __name__ == "__main__":
    main()
