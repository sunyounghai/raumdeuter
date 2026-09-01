"""
트래킹 결과(픽셀 bbox)를 calibration의 homography(H)로 실제 경기장 좌표(미터)로 변환함
프레임마다 H를 새로 계산하므로(카메라가 계속 움직이는 방송 영상이라 프레임별 H가 다름), 
트랙 파일의 프레임 개수만큼 calibration을 반복 실행함

사용 예:
  python -m src.calibration.project_tracks_to_pitch \
      --tracks results/tracking/botsort_h250_baseline_own/tracks_seg1.txt \
      --frames_dir data/frames_final_ordered/img1_seg1 \
      --weights_kp weights/SV_kp \
      --weights_line weights/SV_lines \
      --out results/tracking/pitch_coords_own/seg1.txt
"""


import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.calibration.pnlcalib_wrapper import(
    get_homography_matrix,
    get_player_anchor_point_xyxy,
    image_point_to_pitch,
    load_models,
)

def parse_args():
    parser = argparse.ArgumentParser(description="트랙 좌표를 calibration H로 피치 좌표로 변환")
    parser.add_argument("--tracks", type=Path, required=True, help="트래킹 결과 MOT 포맷 csv")
    parser.add_argument("--frames_dir", type=Path, required=True, help="프레임 이미지 폴더")
    parser.add_argument("--weights_kp", type=str, required=True, help="PnLCalib keypoint 가중치")
    parser.add_argument("--weights_line", type=str, required=True, help="PnLCalib line 가중치")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_keypoints", type=int, default=5,help="이 개수 미만이면 H가 나와도 실패로 취급")
    parser.add_argument("--max_interp_gap", type=int, default=3, help="같은 트랙에서 이 프레임 수 이하로 끊긴 구간만 선형보간으로 채움")
    parser.add_argument("--out", type=Path, required=True, help="결과 저장 경로")
    return parser.parse_args()


def load_tracks(path: Path) -> dict[int, list]:
    """frame_id -> [(track_id, x, y, w, h), ...]"""
    tracks_by_frame = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame_id = int(row[0])
            track_id = int(row[1])
            x, y, w, h = map(float, row[2:6])
            tracks_by_frame[frame_id].append((track_id, x, y, w, h))
    return tracks_by_frame


def bbox_to_foot_point(x: float, y: float, w: float, h: float) -> tuple[float, float]:
    x1, y1, x2, y2 = x, y, x + w, y + h
    return get_player_anchor_point_xyxy((x1, y1, x2, y2))


def pixel_to_pitch(px: float, py: float, H) -> tuple[float, float] | None:
    if H is None:
        return None
    return image_point_to_pitch(np.array([px, py]), H)


def interpolate_short_gaps(rows: list, max_gap: int) -> list:
    """
    같은 트랙(track_id) 안에서 max_gap 프레임 이하로 끊긴 구간만 선형보간으로 채움
    """
    by_track = defaultdict(list)
    for frame_id, track_id, x, y in rows:
        by_track[track_id].append((frame_id, x, y))

    filled_rows = list(rows)
    for track_id, points in by_track.items():
        points.sort(key=lambda p: p[0])
        for i in range(len(points) - 1):
            f1, x1, y1 = points[i]
            f2, x2, y2 = points[i + 1]
            gap = f2 - f1
            if 1 < gap <= max_gap + 1: # 사이에 1~max_gap 프레임이 비어있는 경우
                for step in range(1, gap):
                    ratio = step / gap
                    fx = f1 + step
                    ix = x1 + (x2 - x1) * ratio
                    iy = y1 + (y2 - y1) * ratio
                    filled_rows.append([fx, track_id, ix, iy])

    return filled_rows


def main():
    args = parse_args()

    tracks_by_frame = load_tracks(args.tracks)
    frame_ids = sorted(tracks_by_frame.keys())

    print(f"모델 로드 중...")
    models = load_models(args.weights_kp, args.weights_line, device=args.device)

    rows = []
    failed_frames = []

    for frame_id in frame_ids:
        img_path = args.frames_dir / f"{frame_id:06d}.jpg"
        if not img_path.exists():
            print(f"  [건너뜀] 프레임 이미지 없음: {img_path}")
            continue

        frame_bgr = cv2.imread(str(img_path))
        diag = {}
        H = get_homography_matrix(frame_bgr, models, diagnostics=diag)

        if H is None or diag.get("n_keypoints", 0) < args.min_keypoints:
            failed_frames.append(frame_id)
            continue

        for track_id, x, y, w, h in tracks_by_frame[frame_id]:
            foot_x, foot_y = bbox_to_foot_point(x, y, w, h)
            pitch_coords = pixel_to_pitch(foot_x, foot_y, H)
            if pitch_coords is None:
                continue
            pitch_x, pitch_y = pitch_coords
            rows.append([frame_id, track_id, pitch_x, pitch_y])

        if frame_id % 50 == 0:
            print(f"  frame {frame_id} 처리 중... (누적 {len(rows)}행, calibration 실패 {len(failed_frames)}건)")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    filled_rows = interpolate_short_gaps(rows, args.max_interp_gap)
    n_interpolated = len(filled_rows) - len(rows)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "track_id", "pitch_x_m", "pitch_y_m", "interpolated"])
        real_keys = {(r[0], r[1]) for r in rows}
        for row in sorted(filled_rows, key=lambda r: (r[1], r[0])):
            is_interp = (row[0], row[1]) not in real_keys
            writer.writerow(row + [is_interp])

    print(f"\n완료 -> {args.out}")
    print(f"실측 {len(rows)}행 + 선형보간 {n_interpolated}행 (max_interp_gap={args.max_interp_gap})")
    print(f"총 {len(frame_ids)}프레임 중 calibration 실패(keypoint<{args.min_keypoints}포함) "
          f"{len(failed_frames)}개 ({len(failed_frames)/len(frame_ids)*100:.1f}%)")
    if failed_frames:
        print(f"실패 프레임 예시(최대 10개): {failed_frames[:10]}")


if __name__ == "__main__":
    main()