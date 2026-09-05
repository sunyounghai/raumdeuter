"""Roadmap 페이지 — 파이프라인 진행 상황 표시"""

import streamlit as st
from data import PIPELINE_STATUS
from utils import inject_base_style, status_badge

st.set_page_config(page_title="Roadmap · Raumdeuter", page_icon="🚧", layout="wide")
inject_base_style()

st.title("Roadmap")

st.divider()

for stage in PIPELINE_STATUS:
    st.markdown(
        f"""
        <div class="rd-card">
            <span style="font-weight:700; font-size:1.05rem;">{stage['name']}</span>
            {status_badge(stage['status'])}
            <div style="opacity:0.75; margin-top:8px;">{stage['detail']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("다음 단계")
st.markdown(
    """
    1. **공 검출 완료** — 소형 객체 특화 데이터 증강 및 학습
    2. **공간 침투 지수 산출** — 검출·트래킹·호모그래피로 확보한 좌표 데이터를 바탕으로
       4개 축(공간 인식/침투/활용/타이밍)의 지수 계산
    3. **포지션별 분포 검증** — 공격형 MF↑ · 센터백↓ 구조적 타당성 확인
    4. **외부 기준 보조 검증** — \u201C공간 활용 우수\u201D 선수 사전 목록과 지수 상위권 일치 여부 확인
    """
)

