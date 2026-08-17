"""
n_keypoints 구간별로 층화 샘플링해서 GT 라벨링 대상 프레임을 뽑고,
Label Studio 업로드용 폴더로 복사함
"""

import csv
import random
import shutil

from src.common.paths import RAW_DATA_DIR, RUNS_DIR

random.seed(42) # 재현 가능

FRAMES_DIR = RAW_DATA_DIR / "frames_final"

SPOT_CHECK_CSV = RUNS_DIR / "spot_check" / "spot_check_diagnostics.csv"

OUTPUT_DIR = RUNS_DIR / "gt_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "gt_sample_frames.csv"

UPLOAD_DIR = OUTPUT_DIR / "label_studio_upload"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

N_PER_BUCKET = 5


def bucket(row):
    kp = int(row["n_keypoints"])
    return kp if kp <= 6 else "7+"


def main():
    rows = list(csv.DictReader(open(SPOT_CHECK_CSV)))
    succ = [r for r in rows if r["h_success"] == "True"]

    buckets = {}
    for r in succ:
        buckets.setdefault(bucket(r), []).append(r["frame"])

    sample = []
    for b, frames in buckets.items():
        picked = random.sample(frames, min(N_PER_BUCKET, len(frames)))
        sample += [(b, f) for f in picked]

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "frame"])
        w.writerows(sample)

    print(f"샘플링 완료: {len(sample)}장 -> {OUTPUT_CSV}")
    for b in buckets:
        n_picked = sum(1 for x in sample if x[0] == b)
        print(f"  구간 {b}: 전체 {len(buckets[b])} 장 중 {n_picked}장 선택")

    copied = 0
    for _, fname in sample:
        src = FRAMES_DIR / fname
        dst = UPLOAD_DIR / fname
        if not src.exists():
            print(f"경고: {src} 없음, 건너뜀")
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"{copied}/{len(sample)}장 복사 완료 -> {UPLOAD_DIR}")
    print("이 폴더를 Label Studio Data Import에 드래그하면 됩니다.")


if __name__ == "__main__":
    main()
