"""
project_tracks_to_pitch.py가 만든 pitch 좌표(frame_id, track_id, pitch_x_m, pitch_y_m)로부터
같은 트랙 안에서 연속된 두 관측점 사이의 이동 속도(m/s)를 계산함

프레임 간격이 항상 1이 아닐 수 있으므로 실제 프레임 차이를 그대로 반영해 속도를 계산함:
    speed = 거리(m) / (프레임차이 / fps)

사용 예:
  python -m src.metrics.compute_speed \
      --pitch_coords results/tracking/pitch_coords_gt/SNMOT-116_gt.txt \
      --fps 25 \
      --out results/metrics/speed_SNMOT-116_gt.txt
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

MAX_SPEED_MPS = 12.0


def parse_args():
    parser = argparse.ArgumentParser(description="pitch 좌표로부터 이동 속도(m/s) 계산")
    parser.add_argument("--pitch_coords", type=Path, required=True, help="project_tracks_to_pitch.py 결과 csv")
    parser.add_argument("--fps", type=float, default=25.0, help="원본 영상 프레임레이트")
    parser.add_argument(
        "--max_speed", type=float, default=MAX_SPEED_MPS,
        help="이 값을 초과하면 비현실적 속도로 펴시(기본 12 m/s)",
    )
    parser.add_argument(
        "--stride", type=int, default=5,
        help="속도를 이 프레임 간격으로 계산(기본 5, 0.2초@25fps) "
             "매 프레임(stride=1)으로 계산하면 calibration이 프레임마다 독립적으로 "
             "재계산되면서 생기는 미세한 흔들림이 아주 짧은 시간(1프레임=0.04초@25fps)에 "
             "나뉘어 속도로 크게 증폭됨"
    )
    parser.add_argument("--out", type=Path, required=True, help="결과 저장 경로")
    return parser.parse_args()


def load_pitch_coords(path: Path) -> dict[int, list]:
    """track_id -> [(frame_id, x, y), ...] (프레임 순서대로 정렬됨)"""
    by_track = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            track_id = int(row["track_id"])
            x = float(row["pitch_x_m"])
            y = float(row["pitch_y_m"])
            by_track[track_id].append((frame_id, x, y))

    for track_id in by_track:
        by_track[track_id].sort(key=lambda p: p[0])
    return by_track


def compute_speed(f1: int, x1: float, y1: float, f2: int, x2: float, y2: float, fps: float) -> float | None:
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    time = (f2 - f1) / fps
    if time <= 0:
        return None
    return distance / time
    

def main():
    args = parse_args()
    by_track = load_pitch_coords(args.pitch_coords)

    rows = []
    n_unrealistic = 0

    for track_id, points in by_track.items():
        for i in range(len(points) - args.stride):
            f1, x1, y1 = points[i]
            f2, x2, y2 = points[i + args.stride]
            speed = compute_speed(f1, x1, y1, f2, x2, y2, args.fps)
            if speed is None:
                continue            

            is_unrealistic = speed > args.max_speed
            if is_unrealistic:
                n_unrealistic += 1

            rows.append([f2, track_id, round(speed, 3), f2 - f1, is_unrealistic])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "track_id", "speed_mps", "frame_gap", "unrealistic"])
        for row in sorted(rows, key=lambda r: (r[1], r[0])):
            writer.writerow(row)

    speeds = [r[2] for r in rows if not r[4]] # 비현실적인 값 제외한 정상 속도들
    print(f"완료 -> {args.out}")
    print(f"총 {len(rows)}개 속도 계산 (트랙 {len(by_track)}개)")
    if speeds:
        print(f"정상 속도 범위: 평균 {sum(speeds)/len(speeds):.2f} m/s, "
              f"최대 {max(speeds):.2f} m/s, 최소 {min(speeds):.2f} m/s")
    print(f"비현실적 속도(>{args.max_speed} m/s): {n_unrealistic}개 "
          f"({n_unrealistic/len(rows)*100:.1f}%) - 트랙 오매칭 등 파이프라인 오류 의심")


if __name__ == "__main__":
    main()