"""
MOT 포맷 트래킹 결과를 실제 프레임 이미지 위에 박스+ID로 그려서 저장
트랙이 실제로 잘 이어지는지 확인하는 용도

기본적으로 지정한 프레임 구간(--frames) 전체를 그리며
--around_boundary를 쓰면 특정 지점(예: 세그먼트 경계) 전후 몇 프레임만 뽑아 그림

사용 예:
  # 경계(frame_id=207) 전후 5프레임씩 확인
  python -m src.diagnostics.spot_check_tracking \
      --tracks results/tracking/botsort_h250_baseline_own/tracks.txt \
      --frames_dir data/frames_final_ordered/img1 \
      --around_boundary 207 --window 5 \
      --out docs/tracking/spotcheck_own_footage/baseline_boundary

  # 처음부터 20프레임 확인
  python -m src.diagnostics.spot_check_tracking \
      --tracks results/tracking/botsort_h250_baseline_own/tracks.txt \
      --frames_dir data/frames_final_ordered/img1 \
      --frames 1-20 \
      --out docs/tracking/spotcheck_own_footage/baseline_start
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="트래킹 결과를 프레임에 오버레이해서 저장")
    parser.add_argument("--tracks", type=Path, required=True, help="트래킹 결과 MOT 포맷 csv")
    parser.add_argument("--frames_dir", type=Path, required=True, help="프레임 이미지 폴더 (img1/)")
    parser.add_argument("--frames", type=str, default=None, help="확인할 프레임 범위, 예: 1-20")
    parser.add_argument("--around_boundary", type=int, default=None, help="이 frame_id 전후를 확인")
    parser.add_argument("--window", type=int, default=5, help="--around_boundary 사용 시 전후 프레임")
    parser.add_argument("--out", type=Path, required=True, help="결과 이미지 저장 폴더")
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


def resolve_frame_ids(args) -> list[int]:
    if args.around_boundary is not None:
        start = max(1, args.around_boundary - args.window)
        end = args.around_boundary + args.window
        return list(range(start, end + 1))
    if args.frames is not None:
        start, end = map(int, args.frames.split("-"))
        return list(range(start, end + 1))
    raise ValueError("--frames 또는 --around_boundary 중 하나는 지정해야 함")


def draw_frame(frame_id: int, tracks: list, frames_dir: Path, out_dir: Path) -> None:
    img_path = frames_dir / f"{frame_id:06d}.jpg"
    if not img_path.exists():
        print(f"   [건너뜀] 프레임 없음: {img_path}")
        return

    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)

    for track_id, x, y, w, h in tracks:
        # 트랙 ID에 따라 색 다르게
        color = plt.cm.tab20(track_id % 20)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x, y - 5, str(track_id),
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=1),
        )
    ax.set_title(f"frame {frame_id}")
    ax.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_{frame_id:06d}.jpg"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    args = parse_args()
    tracks_by_frame = load_tracks(args.tracks)
    frame_ids = resolve_frame_ids(args)

    print(f"확인할 프레임: {frame_ids[0]} ~ {frame_ids[-1]} ({len(frame_ids)}개)")

    for frame_id in frame_ids:
        tracks = tracks_by_frame.get(frame_id, [])
        draw_frame(frame_id, tracks, args.frames_dir, args.out)

    print(f"\n전체 저장 완료 -> {args.out}")


if __name__ == "__main__":
    main()