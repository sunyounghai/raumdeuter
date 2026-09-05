"""Calibration Validation 결과 페이지"""

import plotly.graph_objects as go
import streamlit as st

from data import CALIB_BUCKETS, CALIB_GT_ERROR_M, CALIB_TOTAL_FRAMES, CALIB_SUCCESS_FRAMES
from utils import inject_base_style, image_or_placeholder, get_theme_colors, style_chart, GREEN

st.set_page_config(page_title="Calibration · Raumdeuter", page_icon="📐", layout="wide")
inject_base_style()

st.title("Homography — Calibration Validation")
st.markdown(
    '<p class="rd-lead" style="font-style:italic;">'
    'keypoint 개수 기준 이중 검증</p>',
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("keypoint 구간별 GT 실측 오차 (m)")
    st.markdown(
        '<p class="rd-caption">FIFA 규격 기준점 7개를 직접 라벨링 → H 역행렬로 역산한 좌표와 '
        "실제 좌표(m) 비교</p>",
        unsafe_allow_html=True,
    )
    theme_colors = get_theme_colors()
    bar_colors = ["#C1443D", "#6D4E9E", "#6D4E9E", "#6D4E9E"]
    fig = go.Figure()
    fig.add_bar(x=CALIB_BUCKETS, y=CALIB_GT_ERROR_M, marker_color=bar_colors,
                text=[f"{v:.2f}m" for v in CALIB_GT_ERROR_M], textposition="outside")
    fig.update_layout(height=380, showlegend=False)
    style_chart(fig, theme_colors, y_title="오차 (m)")
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        "결론: 필터링 기준을 \u201Ckeypoint ≥ 4\u201D에서 \u201Ckeypoint ≥ 5\u201D로 조정 "
        "(4개 구간만 유독 오차가 큼, n=4로 표본은 작아 확대 필요)"
    )
    st.metric("391프레임 스팟체크 성공률", f"{CALIB_SUCCESS_FRAMES/CALIB_TOTAL_FRAMES:.1%}",
              f"{CALIB_SUCCESS_FRAMES}/{CALIB_TOTAL_FRAMES}장")

with col2:
    st.subheader("seg2_frame_0061 — 계산된 피치 좌표")
    # image_or_placeholder(
    #     "assets/images/seg2_frame_0061_pitch.jpg",
    #     "seg2_frame_0061 — 계산된 피치 좌표",
    #     "검출 13명 중 9명(69%)이 경계 밖으로 계산됨",
    # )




st.divider()
st.subheader("원본 방송 프레임 → 탑뷰 좌표 변환")
# c1, c2 = st.columns(2)
# with c1:
#     image_or_placeholder("assets/images/homography_original.jpg", "원본 방송 프레임",
#                           "bbox + keypoint/line 표시")
# with c2:
#     image_or_placeholder("assets/images/homography_topview.jpg", "탑뷰 좌표 변환",
#                           "mplsoccer 피치 다이어그램")


