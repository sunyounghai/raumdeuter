"""
파인튜닝된 YOLO 검출 모델을 클립(프레임 이미지 폴더)에 돌려서
결과를 MOT 포맷 csv(frame,-1,x,y,w,h,conf,-1,-1,-1)로 저장함

tracker_wrapper.py의 --detections의 인자에 사용 (det.txt/gt.txt와 동일한 포맷)
트랙 ID는 전부 -1이며, conf 값은 다양하게 나옴

사용 예시:
  python -m src.detection.infer_to_mot \
      --weights weights/roboflow_finetuned.pt \
      --frames_dir data/SoccerNet/tracking-2023/test/SNMOT-116/img1 \
      --conf 0.25 \
      --out results/tracking/detection_on_soccernet/roboflow_finetuned_SNMOT-116.txt

그 뒤 트래킹까지 이어가려면:
  python -m src.tracking.tracker_wrapper \
      --detections results/tracking/detection_on_soccernet/roboflow_finetuned_SNMOT-116.txt \
      --frames_dir data/SoccerNet/tracking-2023/test/SNMOT-116/img1 \
      --tracker bytetrack \
      --out results/tracking/bytetrack_finetuned/SNMOT-116.txt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="파인튜닝 검출 모델을 클립에 돌려 MOT 포맷으로 저장"
    )
    parser.add_argument("--weights", type=Path, required=True, help="검출 모델 가중치 경로")
    parser.add_argument("--frames_dir", type=Path, required=True, help="프레임 이미지 폴더 (img1/)")
    parser.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계값")
    parser.add_argument(
        "--frame_name_fmt",
        default="{:06d}.jpg",
        help="프레임 파일명 형식 (SoccerNet 기본: 000001.jpg)"
    )
    parser.add_argument("--out", type=Path, required=True, help="결과 csv 저장 경로")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(args.weights))

    frame_files = sorted(args.frames_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"프레임 이미지를 찾을 수 없음: {args.frames_dir}")

    rows = []
    for frame_path in frame_files:
        # 파일명(예: 000123.jpg)에서 프레임 번호 추출
        frame_id = int(frame_path.stem)

        results = model.predict(str(frame_path), conf=args.conf, verbose=False)
        r = results[0]

        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            w, h = x2 - x1, y2 - y1
            # 트랙 ID는 검출 단계라 항상 -1 (나중에 tracker가 정함)
            rows.append([frame_id, -1, x1, y1, w, h, c, -1, -1, -1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        for row in sorted(rows, key=lambda r: r[0]):
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")

    print(f"[infer_to_mot] {len(frame_files)}개 프레임, {len(rows)}개 검출 저장 완료 -> {args.out}")



if __name__ == "__main__":
    main()