"""
라벨링된 GT 포인트를 H로 변화해서 실제 미터 오차를 계산

1. pnlcalib_wrapper.py에서 계산된 H는 "피치 -> 이미지" 방향,
   원하는 값인 "이미지(픽셀) -> 피치"는 H의 역행렬로 계산해야 함
2. H_inv로 나오는 좌표는 "피치 중심이 원점"인 좌표계(-52.~52.5, -34~34),
   코너-원점 좌표계(0~105, 0~68)로 맞추려면 (x: +52.5, y: 부호 반대 +34)를 더해야 함
   (PnLCalib 내부 y축 방향이 코너-원점 좌표계와 반대였음, 오차가 60~70m -> 0~6m로 줄어듦)
3. get_homography_matrix()는 frame_bgr(cv2 배열)과 models(로드된 모델 번들)를 같이 받음
"""

import csv
import math
import cv2
import numpy as np
from collections import defaultdict

from src.common.paths import RAW_DATA_DIR, WEIGHTS_DIR, RUNS_DIR
from src.calibration.pnlcalib_wrapper import load_models, get_homography_matrix

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

FRAMES_DIR = RAW_DATA_DIR / "frames_final"
OUTPUT_DIR = RUNS_DIR / "gt_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS_CSV = OUTPUT_DIR / "gt_labels.csv"
RESULTS_CSV = OUTPUT_DIR / "gt_error_results.csv"

SPOT_CHECK_CSV = RUNS_DIR / "spot_check" / "spot_check_diagnostics.csv"

WEIGHTS_KP = WEIGHTS_DIR / "SV_kp"
WEIGHTS_LINE = WEIGHTS_DIR / "SV_lines"


def project_pixel_to_pitch(H_pitch_to_image: np.ndarray, px: float, py: float):
    """
    픽셀 좌표(px, py) -> 피치 좌표(m, 코너-원점 0~105/0~68 기준)
    """
    H_image_to_pitch = np.linalg.inv(H_pitch_to_image)
    vec = np.array([px, py, 1.0])
    out = H_image_to_pitch @ vec
    w = out[2]
    if abs(w) < 1e-8:
        return None # w가 0에 가까움 -> 투영 불안정
    x_center_origin = out[0] / w
    y_center_origin = out[1] / w

    # y축 부호가 반대
    return x_center_origin + PITCH_LENGTH / 2, -y_center_origin + PITCH_WIDTH / 2


def main():
    rows = list(csv.DictReader(open(LABELS_CSV)))
    spot_check = {r["frame"]: r for r in csv.DictReader(open(SPOT_CHECK_CSV))}

    models = load_models(str(WEIGHTS_KP), str(WEIGHTS_LINE))

    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)

    results = []
    for frame, points in by_frame.items():
        frame_bgr = cv2.imread(str(FRAMES_DIR / frame))
        if frame_bgr is None:
            print(f"경고: {frame} 못 읽음, 건너뜀")
            continue

        diag = {}
        H = get_homography_matrix(frame_bgr, models, diagnostics=diag)
        if H is None:
            continue

        n_kp_from_wrapper = diag["n_keypoints"]
        n_kp_from_csv = int(spot_check[frame]["n_keypoints"]) if frame in spot_check else None
        if n_kp_from_csv is not None and n_kp_from_wrapper != n_kp_from_csv:
            print(f"주의: {frame} n_keypoints 불일치 (wrapper={n_kp_from_wrapper}, "
                  f"csv={n_kp_from_csv}) - 모델/threshold 버전 차이 확인 필요")

        for r in points:
            px, py = float(r["pixel_x"]), float(r["pixel_y"])
            rx, ry = float(r["real_x_m"]), float(r["real_y_m"])

            proj = project_pixel_to_pitch(H, px, py)
            error_m = math.hypot(proj[0] - rx, proj[1] - ry) if proj else float("inf")

            results.append((frame, r["label"], error_m, n_kp_from_wrapper))

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label", "error_m", "n_keypoints"])
        w.writerows(results)

    buckets = defaultdict(list)
    for frame, label, err, n_kp in results:
        if n_kp is None or err == float("inf"):
            continue
        b = n_kp if n_kp <= 6 else "7+"
        buckets[b].append(err)

    print(f"{'구간':<6}{'표본':<6}{'평균오차(m)':<12}{'중앙값(m)':<10}{'최대(m)'}")
    for b in sorted(buckets, key=lambda x: (isinstance(x, str), x)):
        vals = buckets[b]
        print(f"{b!s:<6}{len(vals):<6}{sum(vals)/len(vals):<12.2f}"
              f"{sorted(vals)[len(vals)//2]:<10.2f}{max(vals):.2f}")



if __name__ == "__main__":
    main()