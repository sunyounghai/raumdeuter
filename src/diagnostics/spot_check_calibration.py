"""
GT 라벨링 전 calibration(H) + detection(bbox)을 결합했을 때 선수 피치좌표를 눈으로 확인
이 스크립트의 결과는 깨지지 않은 상태만 확인함 (정량 검증은 GT 확보 후 진행)

+ 프레임마다 진단 지표(keypoint/line 개수, 블러 정도)를 CSV로 남겨서
전체 데이터에 대한 통계적 분석(성공/실패와 지표 간 상관관계 등)을 할 수 있음

+ 검출된 keypoint/line의 실제 좌표도 원본 프레임 위에 시각화해서 개수와 정확한 위치를 확인함
"""

import csv
import sys
from pathlib import Path


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from ultralytics import YOLO

# macOS 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False 

sys.path.insert(0, "src")
from calibration.pnlcalib_wrapper import load_models, get_homography_matrix, PITCH_LENGTH, PITCH_WIDTH

FRAMES_DIR = Path("data/frames_final")
OUTPUT_DIR = Path("runs/spot_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "spot_check_diagnostics.csv"

DETECTION_WEIGHTS = "runs/h250_finetune/weights/best.pt"

def get_player_anchor_point(bbox):
    """
    bbox: (x1, y1, x2, y2), 픽셀 좌표
    선수의 '발이 닿는 지점' = bbox 하단 중심(bottom-center)으로 근사
    카메라 각도가 심하면(줌인/측면) 이 근사 자체가 깨질 수 있음
    """
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y2])


def compute_blur_score(frame_bgr: np.ndarray) -> float:
    """
    라플라시안 분산으로 이미지의 선명도를 측정
    값이 낮을수록 흐릿함(모션 블러, 포커스 아웃 등)
    절대적인 기준값은 없고, 다른 프레임들과 상대 비교하는 용도
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def image_to_pitch_corner_origin(image_pt: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    이미지 픽셀 좌표 -> 피치 좌표(코너 원점, 0~105, 0~68)로 변환

    get_homography_matrix()가 주는 H는 pitch(중심원점, -52.5~52.5) -> image 방향이므로
    역방향(image -> pitch)은 H의 역행렬을 씀
    그 결과도 중심원점 기준이라 mplsoccer 기본 좌표계(코너원점)에 맞추려면
    PITCH_LENGTH/2, PITCH_WIDTH/2 만큼 다시 평행이동해야 함
    """
    H_inv = np.linalg.inv(H)
    p = np.array([image_pt[0], image_pt[1], 1.0])
    pitch_p = H_inv @ p
    pitch_p /= pitch_p[2]

    x_centered, y_centered = pitch_p[0], pitch_p[1]
    x_corner = x_centered + PITCH_LENGTH / 2
    y_corner = y_centered + PITCH_WIDTH / 2
    return np.array([x_corner, y_corner])


def draw_keypoints_and_lines(ax, kp_dict, lines_dict):
    """
    검출된 keypoint(노란 점)와 line(시안 색 선)을 원본 프레임 좌표계 위에 그림
    """
    for kp_id, kp in kp_dict.items():
        px, py = kp["x"], kp["y"]  # 곱하지 않고 그대로 시도
        ax.plot(px, py, "o", color="yellow", markersize=7, markeredgecolor="black")
        ax.annotate(str(kp_id), (px, py), color="yellow", fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    for line_id, ln in lines_dict.items():
        x1, y1 = ln["x_1"], ln["y_1"]
        x2, y2 = ln["x_2"], ln["y_2"]
        ax.plot([x1, x2], [y1, y2], "-", color="cyan", linewidth=2)    


def spot_check_frame(frame_path: Path, calib_models, detector, csv_writer):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"[스킵] 이미지를 못 읽었습니다: {frame_path}")
        return

    blur_score = compute_blur_score(frame)
    diag = {}

    H = get_homography_matrix(frame, calib_models, diagnostics=diag)

    n_keypoints = diag.get("n_keypoints", -1)
    n_lines = diag.get("n_lines", -1)
    kp_dict = diag.get("kp_dict", {})
    lines_dict = diag.get("lines_dict", {})
    camera_position_m = diag.get("camera_position_m")
    rep_err = diag.get("rep_err")

    if H is not None and camera_position_m is not None:
        print(f"{frame_path.name}: 카메라 위치(m, 피치중심원점 기준) = {camera_position_m}, "
              f"내부 재투영 오차(rep_err) = {rep_err}")

    if H is None:
        print(f"[체크 불가] {frame_path.name}: calibration 실패 (H=None) "
              f"(keypoints={n_keypoints}, lines={n_lines}, blur={blur_score:.1f})")
        csv_writer.writerow([frame_path.name, False, n_keypoints, n_lines,
                             round(blur_score, 1), "", ""])

        # H=None이어도 검출된 keypoint/line은 있을 수 있으므로 원인 확인용으로 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"[H=None] keypoints/lines: {frame_path.name}")
        ax.axis("off")
        draw_keypoints_and_lines(ax, kp_dict, lines_dict)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"kpcheck_{frame_path.stem}.png", dpi=120)
        plt.close(fig)
        return

    results = detector(frame, verbose=False)[0]
    bboxes = results.boxes.xyxy.cpu().numpy()
    if len(bboxes) == 0:
        print(f"[체크 불가] {frame_path.name}: 검출된 선수 없음")
        csv_writer.writerow([frame_path.name, True, n_keypoints, n_lines,
                             round(blur_score, 1), 0, 0, round(rep_err, 3)])
        return

    anchor_points_px = [get_player_anchor_point(b) for b in bboxes]
    pitch_coords = [image_to_pitch_corner_origin(p, H) for p in anchor_points_px]

    # 육안 체크 1: 좌표 범위 - 피치 밖으로 튀어나간 점이 있는가 (코너원점 0~105, 0~68 기준)
    out_of_bounds = [
        (x, y) for x, y in pitch_coords
        if not (0 <= x <= PITCH_LENGTH and 0 <= y <= PITCH_WIDTH)        
    ]

    csv_writer.writerow([frame_path.name, True, n_keypoints, n_lines,
                         round(blur_score, 1), len(bboxes), len(out_of_bounds),
                         round(rep_err, 3)])

    if out_of_bounds:
        print(f"[경고] {frame_path.name}: 피치 밖 좌표 {len(out_of_bounds)}개 발견 "
              f"-> calibration 오차 또는 앵커 포인트 근사 실패 가능성")

    # 육안 체크 2: 원본 프레임(bbox + keypoint/line 표시) vs 변환된 피치 좌표 나란히 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"원본 + bbox + keypoint/line: {frame_path.name}")
    axes[0].axis("off")
    for idx, ((x1, y1, x2, y2), (ax, ay)) in enumerate(zip(bboxes, anchor_points_px)):
        axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2- y1,
                                        fill=False, edgecolor="lime", linewidth=1.5))
        axes[0].plot(ax, ay, "ro", markersize=5) # 앵커 포인트(발 위치)
        axes[0].annotate(str(idx), (ax, ay), color="red", fontsize=11, fontweight="bold",
                         xytext=(5, -10), textcoords="offset points")
    draw_keypoints_and_lines(axes[0], kp_dict, lines_dict)

    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH,
                  pitch_width=PITCH_WIDTH, line_color="black")
    pitch.draw(ax=axes[1])
    for idx, (x, y) in enumerate(pitch_coords):
        axes[1].plot(x, y, "go", markersize=8)
        axes[1].annotate(str(idx), (x, y), color="darkgreen", fontsize=11, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points")
    axes[1].set_title("변환된 피치 좌표 (원본과 같은 번호 = 같은 선수)")

    # Y축 반전: 이 영상의 카메라가 Y가 큰 쪽에 위치에 있어(diagnostis의 camera_position_m으로 확인)
    #         반전 없이 그리면 "화면에서 카메라에 가까운(아래쪽) 선수"가 그림에서는 위쪽에 그려짐
    #         시각화만 뒤집는 것이며 pitch_coords 자체(CSV/지표 계산에 쓰이는 원본 데이터)는 건드리지 않음
    axes[1].invert_yaxis()
    
    out_path = OUTPUT_DIR / f"spotcheck_{frame_path.stem}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[완료] {out_path} (검출 {len(bboxes)}명, 피치 밖 {len(out_of_bounds)}개, "
          f"keypoints={n_keypoints}, lines={n_lines}, blur={blur_score:.1f})")

    
def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg")) + sorted(FRAMES_DIR.glob("*.png"))
    if not frame_paths:
        print(f"{FRAMES_DIR} 폴더에 이미지가 없습니다. 정면/측면/줌인 프레임을 몇 장 넣어주세요.")
        return

    calib_models = load_models("weights/SV_kp", "weights/SV_lines", device="cpu")
    detector = YOLO(DETECTION_WEIGHTS)

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "h_success", "n_keypoints", "n_lines",
                         "blur_score", "n_detected", "n_out_of_bounds", "rep_err"])

        print(f"{len(frame_paths)}개 프레임 스팟체크 시작\n")
        for fp in frame_paths:
            spot_check_frame(fp, calib_models, detector, writer)

    print(f"\n진단 데이터 저장 완료: {CSV_PATH}")
    print(f"\n확인할 것 (체크리스트):")
    print(f" 좌표축이 뒤집혀 있지 않은가 (x/y 스왑, 좌우 반전)")
    print(f" 선수가 피치 경계 밖으로 튀어 나가지 않는가")
    print(f" 골키퍼가 골대 근처에 있는가")
    print(f" 정면/측면/줌인 프레임 간 오차 패턴이 눈에 띄게 달라지는가")
    print(f" bbox 하단 중심(빨간 점)이 실제로 발 위치와 가까운가")
    print(f" keypoint(노란 점)가 실제 피치 특징점(모서리, 교차점)과 일치하는가")


if __name__ == "__main__":
    main()