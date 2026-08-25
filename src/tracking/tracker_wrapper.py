"""
boxmot 기반 트래커 실행 wrapper
SoccerNet MOT 포맷 gt.txt(또는 자체 검출 결과 csv)를 입력 검출값으로 사용해 트래킹을 돌리고, 
결과를 MOT Challenge 표준 csv로 저장함.

사용 예시:
  # 조건 A: ByteTrack baseline
  python tracker_wrapper.py \
      --detections data/SoccerNet/tracking/test/clip01/gt/gt.txt \
      --frames_dir data/SoccerNet/tracking/test/clip01/img1 \
      --tracker bytetrack \
      --out results/tracking/bytetrack/clip01.txt

  # 조건 B: BoT-SORT + OSNet ReID
  python tracker_wrapper.py \
      --detections data/SoccerNet/tracking/test/clip01/gt/gt.txt \
      --frames_dir data/SoccerNet/tracking/test/clip01/img1 \
      --tracker botsort \
      --reid_weights weights/osnet_x0_25_msmt17.pt \
      --device cuda:0 \
      --out results/tracking/botsort/clip01.txt
"""


import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from boxmot.trackers.bbox.botsort import BotSort
from boxmot.trackers.bbox.bytetrack import ByteTrack
from boxmot.trackers.bbox.strongsort import StrongSort


def load_mot_detections(csv_path: Path) -> dict[int, np.ndarray]:
    """
    MOT 포맷 csv(frame,id,bb_left,bb_top,bb_w,bb_h,conf,x,y,z)를 읽어
    프레임 번호 -> [[x1,y1,x2,y2,conf,cls], ...] 딕셔너리로 변환
    """
    detections_by_frame: dict[int, list] = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame = int(row[0])
            x, y, w, h = map(float, row[2:6])
            conf = float(row[6]) if len(row) > 6 and row[6] not in ("", "-1") else 1.0
            x1, y1, x2, y2 = x, y, x + w, y + h
            cls = 0 # player 단일 클래스 (label_studio config.xml과 동일한 가정)
            detections_by_frame[frame].append([x1, y1, x2, y2, conf, cls])

    return{
        frame: np.array(dets, dtype=np.float32)
        for frame, dets in detections_by_frame.items()
    }


def build_tracker(tracker_type: str, reid_weights: Path | None, device: str):
    """
    tracker_type:
        "bytetrack" - 모션(IoU)만 사용
        "botsort" - 모션 + 외형 임베딩(OSNet) + 카메라 모션 보정(ECC) 사용
    """
    if tracker_type == "bytetrack":
        return ByteTrack()

    if tracker_type == "botsort":
        if reid_weights is None:
            raise ValueError(
                "BoT-SORT는 ReID 가중치가 필요합니다. "
                "--reid_weights weights/osnet_x0_25_msmt17.pt 형태로 지정하세요."
            )

        from boxmot.reid.core import ReID
        reid = ReID(weights=reid_weights, device=device, half=False)
        return BotSort(
            reid_model=reid.model,
            with_reid=True,
        )

    if tracker_type == "strongsort":
        if reid_weights is None:
            raise ValueError(
                "StrongSORT는 ReID 가중치가 필요합니다. "
                "--reid_weights weights/osnet_x0_25_msmt17.pt 형태로 지정하세요."
            )
        from boxmot.reid.core import ReID

        reid = ReID(weights=reid_weights, device=device, half=False)
        return StrongSort(reid_model=reid.model)
        
    raise ValueError(f"알 수 없는 tracker_type: {tracker_type}")


def run_tracking(
    tracker,
    detections_by_frame: dict[int, np.ndarray],
    frames_dir: Path,
    frame_name_fmt: str = "{:06d}.jpg",
) -> list[list]:
    """
    프레임 순서대로 tracker.update()를 호출해 트랙 ID를 부여하고
    MOT 포맷 결과 행 리스트를 반환

    BoT-SORT처럼 외형 임베딩을 쓰는 트래커는 원본 이미지(img)가 필요함
    """

    results: list[list] = []
    frames_ids = sorted(detections_by_frame.keys())

    for frame_id in frames_ids:
        img_path = frames_dir / frame_name_fmt.format(frame_id)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"프레임 이미지를 찾을 수 없음: {img_path}")

        dets = detections_by_frame[frame_id] # [x1,y1,x2,y2,conf,cls]
        # tracks: [[x1,y1,x2,y2,track_id,conf,cls,ind], ...]
        tracks = tracker.update(dets, img)

        for t in tracks:
            x1, y1, x2, y2, track_id = t[0], t[1], t[2], t[3], int(t[4])
            conf = float(t[5]) if len(t) > 5 else 1.0
            w, h = x2 - x1, y2 - y1
            results.append([frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1])

    return results

def save_mot_results(results: list[list], out_path: Path) -> None:
    """MOT Challenge 표준 csv로 저장 (sn-trackeval이 그대로 읽는 입력 포맷)"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in sorted(results, key=lambda r: (r[0], r[1])):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SoccerNet 클립에 대해 ByteTrack / Bot-SORT 트래킹 실행"
    )
    parser.add_argument("--detections", type=Path, required=True, help="입력 검출값 csv (MOT 포맷)")
    parser.add_argument("--frames_dir", type=Path, required=True, help="프레임 이미지 폴더")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort", "strongsort"], default="bytetrack")
    parser.add_argument(
        "--reid_weights",
        type=Path,
        default=None,
        help="BoT-SORT 사용 시 ReID 가중치 경로 (예: osnet_x0_25_msmt17.pt)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True, help="결과 csv 저장 경로")
    args = parser.parse_args()

    detections_by_frame = load_mot_detections(args.detections)
    tracker = build_tracker(args.tracker, args.reid_weights, args.device)
    results = run_tracking(tracker, detections_by_frame, args.frames_dir)
    save_mot_results(results, args.out)

    print(f"[{args.tracker}] {len(results)}개 트랙 결과 저장 완료 -> {args.out}")


if __name__ == "__main__":
    main()