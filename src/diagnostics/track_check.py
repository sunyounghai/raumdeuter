"""
ByteTrack이 detection과 결합했을 때 ID가 안정적으로 유지되는지 확인
"""

from ultralytics import YOLO

VIDEO_PATH = "data/raw_video/track_check_clip.mp4"

DETECTION_WEIGHTS = "runs/h250_finetune/weights/best.pt"

def main():
    model = YOLO(DETECTION_WEIGHTS)

    # ultralytics에 내장된 ByteTrack 사용 (tracker="bytetrack.yaml")
    # persist=True: 프레임 간 트래커 상태를 유지 (매 프레임 새로 시작하지 않음)
    # save=True: bbox+ID가 그려진 결과 영상을 자동 저장
    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        persist=True,
        save=True,
        name="track_check",
        conf=0.25,
        verbose=False,
    )

    # ID가 얼마나 자주 바뀌는지(=track이 끊기는지) 확인하는 최소한의 지표
    prev_ids = set()
    total_frames = 0
    id_appearances = {} # id -> 등장한 프레임 수 (연속성이 좋으면 값이 큼)
    save_dir = None

    for i, r in enumerate(results):
        if save_dir is None:
            save_dir = r.save_dir
        total_frames += 1
        if r.boxes.id is None:
            print(f"frame {i}: 트래킹 ID 없음 (검출 자체가 없거나 매칭 실패)")
            continue
        ids = set(r.boxes.id.cpu().numpy().tolist())
        new_ids = ids - prev_ids
        lost_ids = prev_ids - ids
        if new_ids or lost_ids:
            print(f"frame {i}: 새 ID={new_ids}, 사라진 ID={lost_ids}, 현재={len(ids)}명")
        for id_ in ids:
            id_appearances[id_] = id_appearances.get(id_, 0) + 1
        prev_ids = ids

    print(f"\n총 프레임: {total_frames}")
    print(f"등장한 고유 ID 개수: {len(id_appearances)}")
    print(f"ID별 평균 등장 프레임 수: {sum(id_appearances.values())/len(id_appearances):.1f}"
          if id_appearances else "ID 없음")
    print(f"\n결과 영상 확인: {save_dir}")
    print("확인할 것: 같은 선수에게 계속 같은 ID 번호가 붙어있는지 (색깔/번호가 안 바뀌는지)") 


if __name__ == "__main__":
    main()
