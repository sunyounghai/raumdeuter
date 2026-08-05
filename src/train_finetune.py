"""
data/yolo_dataset의 GT(player 클래스)로 YOLO를 파인튜닝하고,
같은 val set에서 평가한 뒤 베이스라인 결과와 비교합니다.

전제:
- split_dataset.py 실행 완료 (train/val 분리됨)
- evaluate_baseline.py 실행 완료 (baseline_metrics.txt 존재)

사용법:
    python train_finetune.py
"""

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
DATASET_YAML = BASE / "dataset.yaml"
BASELINE_WEIGHTS = "yolov8n.pt"   # 베이스라인과 동일한 모델로 시작 (공정한 비교를 위해)

# ── 파인튜닝 설정 ──
# 데이터가 104장으로 적은 편이라, epoch을 너무 크게 잡으면 과적합 위험.
# 우선 가볍게 돌려서 파이프라인이 도는지부터 확인하는 게 목적.
EPOCHS = 50
IMGSZ = 640
BATCH = 8


def parse_metrics_file(path: Path) -> dict:
    """이전에 저장된 metrics txt 파일을 읽어서 dict로 반환"""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                result[key] = float(val)
            except ValueError:
                pass
    return result


def main():
    print(f"파인튜닝 시작 (기반 모델: {BASELINE_WEIGHTS})")
    model = YOLO(BASELINE_WEIGHTS)

    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(BASE / "runs"),
        name="finetune",
        exist_ok=True,
    )

    # 학습 완료 후 가장 성능 좋은 가중치로 val 평가
    best_weights = BASE / "runs" / "finetune" / "weights" / "best.pt"
    print(f"\n파인튜닝된 가중치로 재평가: {best_weights}")

    finetuned_model = YOLO(str(best_weights))
    metrics = finetuned_model.val(
        data=str(DATASET_YAML),
        split="val",
        plots=True,
    )

    print("\n===== 파인튜닝 후 결과 =====")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    result_path = BASE / "finetuned_metrics.txt"
    with open(result_path, "w") as f:
        f.write("파인튜닝 후 (GT 104장으로 학습)\n")
        f.write(f"mAP50    : {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95 : {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall   : {metrics.box.mr:.4f}\n")
    print(f"\n결과 저장됨: {result_path}")

    # 베이스라인과 비교표 출력
    baseline = parse_metrics_file(BASE / "baseline_metrics.txt")
    finetuned = parse_metrics_file(result_path)

    if baseline:
        print("\n===== 베이스라인 vs 파인튜닝 비교 =====")
        print(f"{'지표':<10} {'베이스라인':>10} {'파인튜닝후':>10} {'변화':>10}")
        for key in ["mAP50", "mAP50-95", "Precision", "Recall"]:
            b = baseline.get(key, 0)
            ft = finetuned.get(key, 0)
            diff = ft - b
            sign = "+" if diff >= 0 else ""
            print(f"{key:<10} {b:>10.4f} {ft:>10.4f} {sign}{diff:>9.4f}")
    else:
        print("\n(baseline_metrics.txt를 찾을 수 없어 비교표는 생략합니다)")


if __name__ == "__main__":
    main()
