"""
get_homography_matrix()로 얻은 H를 실제로 시각화하여 검증하는 스크립트
알고 있는 실제 피치 라인 좌표를 H로 이미지에 투영해서
원본 프레임의 실제 라인과 겹치는지 그려서 확인
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, "src")
from calibration.pnlcalib_wrapper import load_models, get_homography_matrix, PITCH_LENGTH, PITCH_WIDTH

# inference.py에 하드코딩되어 있던 실제 피치 라인 좌표 (코너 원점, 0~105, 0~68)
# 각 항목: [시작점(X, Y, Z), 끝점(X, Y, Z)]
LINES_COORDS = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[52.5, 0., 0.], [52.5, 68, 0.]],
    [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]],
    [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]],
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]],
]

def draw_pitch_overlay(frame_bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """H를 이용해 실제 피치 라인(직선, 아크)을 프레임 위에 빨간 선으로 오버레이"""
    overlay = frame_bgr.copy()

    def to_image(x_corner, y_corner):
        """코너 원점(0~105, 0~68) 좌표 하나를 이미지 픽셀 좌표로 변환"""
        # 1) 코너 원점(0~105, 0~68) -> H에서의 중심 원점(-52.5~52.5, -34~34)으로 평행이동
        # 2) 뒤에 1.0을 붙여 homogeneous 좌표 [X, Y, 1]로 만듦
        #    (H는 3x3이라 3개짜리 벡터와만 곱셈 가능 + 평행이동을 행렬곱으로 표현하기 위해)
        p = np.array([x_corner - PITCH_LENGTH / 2, y_corner - PITCH_WIDTH / 2, 1.0])
        img_p = H @ p
        # H @ p 결과는 [u, v, w] 형태 (아직 원근 반영 안 된 상태)
        # w로 나눠야 실제 픽셀 좌표 [u/w, v/w]가 나옴 (원근 투영 처리 방식)
        img_p /= img_p[2]
        return int(img_p[0]), int(img_p[1])

    # 1) 직선 라인들
    for (x1, y1, _), (x2, y2, _) in LINES_COORDS:
        pt1 = to_image(x1, y1)
        pt2 = to_image(x2, y2)
        cv2.line(overlay, pt1, pt2, (0, 0, 255), 3) # BGR

    # 2) 아크 3개 (왼쪽 페널티 아크, 오른쪽 페널티 아크, 센터서클)
    # inference.py의 project() 함수 사용 - 반지름 9.15m 원 위의 점들을
    # 촘촘히 샘플링해서 각각 이미지로 투영한 뒤 폴리라인으로 이어 그림
    r = 9.15
    arcs = [
        (11, 34, np.linspace(37, 143, 50)),     # 왼쪽 페널티 아크
        (94, 34, np.linspace(217, 323, 200)),   # 오른쪽 페널티 아크
        (52.5, 34, np.linspace(0, 360, 500)),   # 센터서클
    ]

    for center_x, center_y, angles in arcs:
        pts = []
        for ang_deg in angles:
            ang = np.deg2rad(ang_deg)
            x = center_x + r * np.sin(ang)
            y = center_y + r * np.cos(ang)
            pts.append(to_image(x, y))
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)

    return overlay



if __name__ == "__main__":
    models = load_models("weights/SV_kp", "weights/SV_lines", device="cpu")
    frame = cv2.imread("data/frames_final/seg1_frame_0001.jpg")

    H = get_homography_matrix(frame, models)
    if H is None:
        print("H 검출 실패")
        sys.exit(1)

    overlay = draw_pitch_overlay(frame, H)
    cv2.imwrite("calibration_overlay_check.jpg", overlay)
    print("저장 완료: calibration_overlay_check.jpg")