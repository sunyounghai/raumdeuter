"""
SoccerNet-Tracking의 gt.txt(MOT format) + gameinfo.ini에서
ball 트랙만 골라내 detection 평가(mAP 계열)용 GT로 변환함
"""

import configparser
import csv
from pathlib import Path

from src.common.paths import RAW_DATA_DIR

_SEQ_DIR = RAW_DATA_DIR / "SoccerNet/tracking-2023/test/SNMOT-116"
GAMEINFO_PATH = _SEQ_DIR / "gameinfo.ini"
GT_TXT_PATH = _SEQ_DIR / "gt" / "gt.txt"
OUTPUT_DIR = RAW_DATA_DIR / "yolo_dataset_ball/val/labels"

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def find_ball_tracklet_ids(gameinfo_path: Path) -> list[int]:
    """gameinfo.ini를 파싱해 'ball;1`로 표시된 trackletID_N에서 N을 찾음
    공이 중간에 가려지거나 화면 밖으로 나가면 같은 공인데도 ball;1, ball;2 처럼 트랙이 여러 개로
    나뉘어 기록될 수 있어 리스트로 반환
    """
    config = configparser.ConfigParser()
    config.read(gameinfo_path)
    seq = config["Sequence"]
    ball_ids = []
    for key, value in seq.items():
        if key.startswith("trackletid_") and value.strip().lower().startswith("ball"):
            ball_ids.append(int(key.split("_")[1]))
    if not ball_ids:
        raise ValueError("gameinfo.ini에서 ball 트랙릿을 찾지 못했습니다. "
                          "trackletID_N=ball;... 형식인지 확인하세요.")
    return ball_ids


def load_ball_boxes(gt_txt_path: Path, ball_ids: list[int]):
    """gt.txt에서 ball_ids에 해당하는 행만 프레임별로 추출함
    공 트랙이 여러 개(ball;1, ball;2 등)로 나뉘어 있어도 전부 하나로 합침"""
    boxes_by_frame = {}
    with open(gt_txt_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            frame, obj_id, x, y, w, h = int(row[0]), int(row[1]), *map(float, row[2:6])
            if obj_id in ball_ids:
                if frame in boxes_by_frame:
                    print(f"[경고] 프레임 {frame}에 공 트랙이 두 개 이상 겹칩니다. "
                          f"(id={obj_id} 포함) - 데이터 확인 필요")
                boxes_by_frame[frame] = (x, y, w, h)
    return boxes_by_frame


def write_yolo_labels(boxes_by_frame, img_w, img_h, output_dir: Path, class_id: int = 0):
    """프레임별 YOLO 포맷(class cx cy w h, 0~1 정규화) txt 파일 생성
    프레임에 공이 없으면 빈 txt를 만들어 해당 프레임엔 공이 없음을 명시함"""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = range(1, max(boxes_by_frame.keys()) + 1)

    for frame in all_frames:
        label_path = output_dir / f"{frame:06d}.txt"
        if frame in boxes_by_frame:
            x, y, w, h = boxes_by_frame[frame]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            label_path.write_text("") # 공이 안 보이는 프레임


def main():
    
    ball_ids = find_ball_tracklet_ids(GAMEINFO_PATH)
    print(f"ball tracklet ID = {ball_ids}")
    if len(ball_ids) > 1:
        print(f"  [참고] 공 트랙이 {len(ball_ids)}개로 나뉘어 있습니다. "
              f"(occlusion 등으로 GT 자체에서 트랙이 끊겼을 가능성) - 전부 합쳐서 사용합니다.")

    boxes = load_ball_boxes(GT_TXT_PATH, ball_ids)
    print(f"공이 등장하는 프레임 수: {len(boxes)}")

    write_yolo_labels(boxes, IMG_WIDTH, IMG_HEIGHT, OUTPUT_DIR)
    print(f"YOLO 포맷 라벨 생성 완료: {OUTPUT_DIR}/")

    
if __name__ == "__main__":
    main()