"""Detection 결과 페이지"""

import plotly.graph_objects as go
import streamlit as st

from data import DETECTION_MODELS, DETECTION_BASELINE, DETECTION_FINETUNED, DETECTION_ARCH
from utils import inject_base_style, image_or_placeholder, get_theme_colors, style_chart, GREEN, GRAY

st.set_page_config(page_title="Detection · Raumdeuter", page_icon="🎯", layout="wide")
inject_base_style()

st.title("Object Detection")
st.markdown(
    '<p class="rd-lead">YOLOv8 계열 사전학습 체크포인트 3종(COCO/Roboflow/H250)을 '
    "파인튜닝하여 선수 검출 모델 구축</p>",
    unsafe_allow_html=True,
)

st.divider()

st.subheader("선수 검출 mAP50-95 · 베이스라인 vs 파인튜닝")
colors = get_theme_colors()
fig = go.Figure()
fig.add_bar(name="Baseline", x=DETECTION_MODELS, y=DETECTION_BASELINE, marker_color=GRAY,
            text=[f"{v:.2f}" for v in DETECTION_BASELINE], textposition="outside")
fig.add_bar(name="Fine-tuned", x=DETECTION_MODELS, y=DETECTION_FINETUNED, marker_color=GREEN,
            text=[f"{v:.2f}" for v in DETECTION_FINETUNED], textposition="outside")
fig.update_layout(barmode="group", height=420, legend=dict(orientation="h", y=1.1, font=dict(color=colors["text"])))
style_chart(fig, colors, y_title="mAP50-95", y_range=[0, 1.0])
st.plotly_chart(fig, use_container_width=True)

for m in DETECTION_MODELS:
    st.markdown(f"**{m}** — {DETECTION_ARCH[m]}")


st.divider()

st.subheader("오탐 · 미탐 분석")
st.markdown(
    f'<p class="rd-caption">mAP 수치 개선의 원인을 실제 검출 사례로 검증 — 베이스라인 vs 파인튜닝 비교</p>',
    unsafe_allow_html=True,
)

# t1, t2, t3 = st.columns(3)
# with t1:
#     st.markdown("**미탐(FN) 개선**")
#     image_or_placeholder("assets/images/fn_baseline.jpg", "베이스라인 — 놓친 선수")
#     image_or_placeholder("assets/images/fn_finetuned.jpg", "파인튜닝 — 정상 검출")
# with t2:
#     st.markdown("**오탐(FP) 감소**")
#     image_or_placeholder("assets/images/fp_baseline.jpg", "베이스라인 — 오탐 사례")
#     image_or_placeholder("assets/images/fp_finetuned.jpg", "파인튜닝 — 오탐 제거")
# with t3:
#     st.markdown("**박스 정밀도 개선**")
#     image_or_placeholder("assets/images/precision_baseline.jpg", "베이스라인 — 느슨한 박스")
#     image_or_placeholder("assets/images/precision_finetuned.jpg", "파인튜닝 — 정밀한 박스")
