"""
data/frames_final/의 seg1_frame_NNNN.jpg, seg2_frame_NNNN.jpg 형식 프레임을
000001.jpg, 000002.jpg... 순차 번호로 재정렬해 img1/ 폴더에 심볼릭 링크로 배치함

기존 tracker_wrapper.py/infer_to_mot.py가 SoccerNet과 같은 순수 숫자 파일명을
가정하므로 코드를 건드리지 않고 입력 쪽에서 형식을 맞추는 방식

seg1 -> seg2 순서로 이어붙이며, 그 경계(리플레이로 잘려나간 지점)의 새 프레임 번호를 별도로 기록해서
나중에 그 지점에서 트랙이 실제로 끊기는지 확인할 때 사용함

사용 예:
  python -m src.tracking.prepare_own_footage_frames \
      --src_dir data/frames_final \
      --out_dir data/frames_final_ordered
"""

import argparse
import csv
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="자체 영상 프레임을 표준 형식으로 재정렬")
    parser.add_argument("--src_dir", type=Path, required=True, help="seg1_frame_*.jpg 등이 있는 폴더")
    parser.add_argument("--out_dir", type=Path, required=True, help="img1/ 및 매핑 파일을 저장할 폴더")
    return parser.parse_args()

def main():
    args = parse_args()

    frame_files = sorted(args.src_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"프레임을 찾을 수 없음: {args.src_dir}")

    img1_dir = args.out_dir / "img1"
    img1_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    prev_segment = None
    boundary_frame_ids = [] # segment가 바뀌는 지점의 새 frame_id

    for new_id, src_path in enumerate(frame_files, start=1):
        segment = src_path.stem.split("_")[0] # "seg1" 또는 "seg2"
        if prev_segment is not None and segment != prev_segment:
            boundary_frame_ids.append(new_id)
        prev_segment = segment

        new_name = f"{new_id:06d}.jpg"
        link_path = img1_dir / new_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(src_path.resolve())

        mapping_rows.append([new_id, new_name, src_path.name, segment])

    mapping_path = args.out_dir / "frame_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_frame_id", "new_filename", "original_filename", "segment"])
        writer.writerows(mapping_rows)

    print(f"총 {len(frame_files)}개 프레임을 {img1_dir}에 재정렬 완료")
    print(f"매핑 파일 저장 -> {mapping_path}")
    if boundary_frame_ids:
        print(f"세그먼트 경계(리플레이로 잘린 지점) 새 frame_id: {boundary_frame_ids}")
        print("  -> 이 지점 전후로 트랙 ID가 끊기는지 나중에 확인할 것")
    else:
        print("세그먼트가 하나뿐이거나 경계를 못 찾음 - src_dir 내용 확인 필요")


if __name__ == "__main__":
    main()
