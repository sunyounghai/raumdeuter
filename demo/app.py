"""
Raumdeuter 데모 앱 — Home
실행: streamlit run demo/app.py
"""

import streamlit as st
from data import PIPELINE_STATUS
from utils import inject_base_style, status_badge

st.set_page_config(page_title="Raumdeuter", page_icon="⚽", layout="wide")
inject_base_style()

st.title("Raumdeuter")
st.markdown(
    '<p class="rd-lead">Vision AI 기반 축구 오프더볼 공간 분석 — 방송 영상만으로 '
    "전용 트래킹 장비 없이 선수의 공간 활용을 정량화합니다.</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── 파이프라인 다이어그램 (5단계, 상태 배지 포함) ──
st.subheader("파이프라인 진행 상태")
cols = st.columns(len(PIPELINE_STATUS))
for col, stage in zip(cols, PIPELINE_STATUS):
    with col:
        st.markdown(
            f"""
            <div class="rd-card" style="text-align:center; min-height:150px;">
                <div style="font-weight:700; margin-bottom:8px;">{stage['name']}</div>
                {status_badge(stage['status'])}
                <div style="font-size:0.78rem; opacity:0.75; margin-top:10px;">{stage['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    f'<p class="rd-caption">완료(초록) / 진행중(주황) / 예정(회색) — 상태가 바뀔 때마다 업데이트됩니다.</p>',
    unsafe_allow_html=True,
)

st.divider()

# ── 핵심 요약 3개 ──
st.subheader("지금까지 검증된 것")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Detection mAP50-95 (파인튜닝)", "0.78~0.83", "세 모델 수렴")
with c2:
    st.metric("Tracking HOTA (최종 조합)", "44.1", "h250_baseline + BoT-SORT")
with c3:
    st.metric("Calibration 성공률", "79.5%", "391프레임 기준")

st.divider()

st.subheader("페이지 안내")
st.markdown(
    """
- **Detection** — 사전학습 체크포인트 3종(COCO/Roboflow/H250) 비교, 오탐·미탐 분석
- **Tracking** — 검출기 × 트래커 조합 비교, 분석
- **Calibration** — 호모그래피 계산 신뢰도 검증
- **Roadmap** — 프로젝트 파이프라인 
    """
)

st.divider()
st.markdown(
    f'<p class="rd-caption">전체 개발 코드 및 실험 스크립트: '
    f'<a href="https://github.com/sunyounghai/raumdeuter" target="_blank">github.com/sunyounghai/raumdeuter</a></p>',
    unsafe_allow_html=True,
)
