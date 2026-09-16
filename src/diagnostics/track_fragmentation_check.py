"""
트랙 결과(MOT 포맷)에서 새로운 트랙 ID가 프레임 축을 따라 
'균등하게 계속 생성'되는지 '특정 구간에 몰려서 급증'하는지 확인함
이 결과를 통해 트래커 파라미터 vs 특정 구간 원인을 파악

사용법:
    python -m src.diagnostics.track_fragmentation_check --tracks <트랙결과.txt>
"""


import argparse
from collections import defaultdict
from pathlib import Path
import csv


def load_tracks(path: Path) -> dict[int, list]:
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


def main(track_path: Path):
    tracks_by_frame = load_tracks(track_path)

    first_seen_frame = {} # track_id -> 최초 등장 프레임
    for frame_id, tracks in tracks_by_frame.items():
        for track_id, *_ in tracks:
            if track_id not in first_seen_frame or frame_id < first_seen_frame[track_id]:
                first_seen_frame[track_id] = frame_id

    total_tracks = len(first_seen_frame)
    max_frame = max(first_seen_frame.values())

    # 10프레임 단위 버킷으로 이 구간에 새로 생긴 트랙 수 집계
    bucket_size = 10
    new_tracks_per_bucket = defaultdict(int)
    for frame in first_seen_frame.values():
        bucket = frame // bucket_size
        new_tracks_per_bucket[bucket] += 1

    print(f"전체 고유 트랙 수: {total_tracks}")
    print(f"프레임 범위: 1 ~ {max_frame}")
    print(f"\n구간별(10프레임 단위) 신규 트랙 생성 수:")
    print(f"{'구간(frame)':<20}{'신규 트랙 수':>12}")

    avg = total_tracks / (max_frame / bucket_size)
    hotspots = []
    for bucket in sorted(new_tracks_per_bucket):
        start = bucket * bucket_size + 1
        end = start + bucket_size - 1
        count = new_tracks_per_bucket[bucket]
        marker = "  <-- 평균 대비 급증" if count > avg * 2 else ""
        print(f"{start}-{end:<15}{count:>12}{marker}")
        if count > avg * 2:
            hotspots.append((start, end, count))

    print(f"\n구간당 평균 신규 트랙 수: {avg:.1f}")
    if hotspots:
        print(f"\n[급증 구간 발견] {len(hotspots)}개 구간이 평균의 2배 이상:")
        for start, end, count in hotspots:
            print(f"  - 프레임 {start}~{end}: {count}개 신규 트랙 "
                  f"-> 이 구간부터 spot_check_tracking.py로 시각 확인")
    else:
        print(f"\n[균등 분포] 특정 구간 몰림 없음 -> 트래커 파라미터"
              f"(appearance_thresh, match_thresh) 문제일 가능성이 높음")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="트랙 결과에서 신규 트랙 생성이 균등분포인지 특정 구간 몰림인지 진단"
    )
    parser.add_argument("--tracks", type=Path, required=True, help="트래킹 결과 MOT 포맷 파일")
    args = parser.parse_args()
    main(args.tracks)