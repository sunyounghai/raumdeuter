"""
Label Studio에서 export한 raw JSON을 gt_labels.csv로 변환
"""

import csv
import json
import re

from src.common.paths import RUNS_DIR

INPUT_JSON = RUNS_DIR / "gt_check" / "gt_labels_raw_export.json"
OUTPUT_CSV = RUNS_DIR / "gt_check" / "gt_labels.csv"

# label_studio config와 동일한 실제 피치 좌표 (코너-원점, m)
KNOWN_POINTS_M = {
    "corner_00": (0, 0),
    "corner_105_0": (105, 0),
    "corner_0_68": (0, 68),
    "corner_105_68": (105, 68),
    "center_circle": (52.5, 34),
    "penalty_spot_left": (11, 34),
    "penalty_spot_right": (94, 34),
}

HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-(.+)$")


def strip_hash_prefix(image_path: str) -> str:
    """'/data/upload/2/012c46fb-seg2_frame_0060.jpg' -> 'seg2_frame_0060.jpg'"""
    filename = image_path.rsplit("/", 1)[-1]
    m = HASH_PREFIX_RE.match(filename)
    if not m:
        # 예상한 해시 패턴이 아님 -> 원본 그대로 두고 알림
        print(f"주의: 해시 접두어 패턴이 예상과 다름 ({filename}), 원본 그대로 사용")
        return filename
    return m.group(1)


def main():
    tasks = json.load(open(INPUT_JSON))

    rows = []
    skipped_empty = 0
    unknown_labels = set()

    for t in tasks:
        frame = strip_hash_prefix(t["data"]["image"])

        if not t["annotations"] or not t["annotations"][0]["result"]:
            skipped_empty += 1
            continue

        for point in t["annotations"][0]["result"]:
            value = point["value"]
            label = value["keypointlabels"][0]

            if label not in KNOWN_POINTS_M:
                unknown_labels.add(label)
                continue

            width = point["original_width"]
            height = point["original_height"]
            px = value["x"] / 100 * width
            py = value["y"] / 100 * height

            rx, ry = KNOWN_POINTS_M[label]
            rows.append((frame, label, px, py, rx, ry))

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label", "pixel_x", "pixel_y", "real_x_m", "real_y_m"])
        w.writerows(rows)

    print(f"변환 완료: {len(rows)}개 포인트 -> {OUTPUT_CSV}")
    print(f"라벨 0개라 건너뛴 프레임: {skipped_empty}개 (정상 - 화면에 지점이 안 보였던 경우)")
    if unknown_labels:
        print(f"주의: KNOWN_POINTS_M에 없는 라벨 발견, 건너뜀: {unknown_labels}")


if __name__ == "__main__":
    main()