"""Tracking 결과 페이지"""

import plotly.graph_objects as go
import streamlit as st

from data import (
    TRACKING_DETECTORS, TRACKING_HOTA, TRACKING_GT_IDS, TRACKING_TRACKER_IDS,
    TRACKING_FINAL_CHOICE, TRACKING_FINAL_STATS,
)
from utils import inject_base_style, image_or_placeholder, get_theme_colors, style_chart, GREEN, AMBER

st.set_page_config(page_title="Tracking · Raumdeuter", page_icon="🔗", layout="wide")
inject_base_style()

st.title("Tracking — Detector × Tracker Validation")
st.markdown(
    '<p class="rd-lead">검출기 4종 × 트래커 3종(ByteTrack/BoT-SORT/StrongSORT) 조합 비교 — '
    "SoccerNet SNMOT-116(코너킥, 750프레임)</p>",
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("검출기별 HOTA (BoT-SORT 고정)")
    st.markdown(
        '<p class="rd-caption">트래커를 동일하게 고정하고 검출기만 바꿔 비교 — 검출기 격차(13점)가 '
        "트래커 간 격차(5.6점)보다 훨씬 큼</p>",
        unsafe_allow_html=True,
    )
    theme_colors = get_theme_colors()
    bar_colors = [GREEN, AMBER, "#6D4E9E", "#6D4E9E"]
    fig = go.Figure()
    fig.add_bar(x=[d.replace("\n", "<br>") for d in TRACKING_DETECTORS], y=TRACKING_HOTA,
                marker_color=bar_colors, text=[f"{v:.1f}" for v in TRACKING_HOTA], textposition="outside")
    fig.update_layout(height=420, showlegend=False)
    style_chart(fig, theme_colors, y_title="HOTA", y_range=[0, 55])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "H250 계열(SoccerNet 사전학습)이 Roboflow(6800만 파라미터, h250 대비 22배 큼)보다도 "
        "우수 — **모델 크기보다 도메인 적합성이 지배적**"
    )

with col2:
    st.subheader("트래커별 예측 트랙 수 (정답 24개 대비)")
    for name, ids in TRACKING_TRACKER_IDS.items():
        delta = ids - TRACKING_GT_IDS
        st.metric(f"{name} · IDs", ids, f"{'+' if delta > 0 else ''}{delta} (GT={TRACKING_GT_IDS})",
                   delta_color="inverse")
    st.markdown(
        f'<p class="rd-caption">정답(GT) 트랙 수 = {TRACKING_GT_IDS}개 — 이 숫자에 가장 가까울수록 '
        "궤적이 안 끊기고 이어졌다는 뜻</p>",
        unsafe_allow_html=True,
    )

st.divider()

st.subheader(f"최종 채택: {TRACKING_FINAL_CHOICE}")
c1, c2, c3 = st.columns(3)
c1.metric("HOTA", TRACKING_FINAL_STATS["HOTA"])
c2.metric("IDF1 (트랙 안정성)", TRACKING_FINAL_STATS["IDF1"])
c3.metric("IDs (정답 24개 대비)", TRACKING_FINAL_STATS["IDs"])

st.markdown(
    """
- IDF1·트랙 연속성 기준 최종 h250_baseline + BoT-SORT 채택 — 본 프로젝트(공간 침투 지수)는
  궤적의 연속성이 핵심이라 IDs·IDF1이 HOTA보다 직접적인 기준
- 자체 파인튜닝(h250_finetuned)이 새 도메인(SoccerNet)에서 baseline을 넘지 못함
  (HOTA 43.5 vs 44.1) — 이전에 예상했던 과적합 위험이 완전히 새로운 데이터에서 실제로 확인됨
- StrongSORT는 HOTA/검출 재현력 1위지만 IDs가 GT의 약 4배로 트랙이 자주 끊김 —
  목적에 따라 최적 트래커가 다름
    """
)