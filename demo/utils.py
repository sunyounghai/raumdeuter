"""
공통 테마 + 재사용 UI 헬퍼

색상 자체는 .streamlit/config.toml의 [theme.light] / [theme.dark]에서 정의하고,
여기 커스텀 CSS는 Streamlit이 제공하는 테마 변수(var(--text-color) 등)를 그대로
참조해서, 시스템 설정이 바뀌거나 앱 안에서 수동으로 테마를 바꿔도 자동으로 맞춰진다.
"""

from pathlib import Path
import streamlit as st

# ── 브랜드 강조색 (테마와 무관하게 항상 같은 의미를 가지는 색) ──
GREEN = "#00B37E"
GREEN_DARK = "#059669"
AMBER = "#E8A33D"
GRAY = "#94A3B8"

STATUS_COLOR = {
    "done": (GREEN, "완료"),
    "partial": (AMBER, "진행중"),
    "planned": (GRAY, "예정"),
}


def inject_base_style():
    """전체 앱에 공통 폰트/카드 스타일 주입. 각 페이지 상단에서 한 번만 호출.
    배경/글자색은 여기서 고정하지 않고 Streamlit 테마 변수를 그대로 따른다."""
    st.markdown(
        """
        <style>
        h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; }

        .rd-badge {
            display: inline-block; padding: 4px 14px; border-radius: 14px;
            font-size: 0.8rem; font-weight: 700; color: white; margin-left: 8px;
        }
        .rd-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 10px; padding: 18px 20px; margin-bottom: 14px;
            color: var(--text-color);
        }
        .rd-lead {
            font-size: 1.05rem; color: var(--text-color); opacity: 0.85;
        }
        .rd-caption {
            color: var(--text-color); opacity: 0.62;
            font-size: 0.85rem; font-style: italic;
        }
        .rd-placeholder {
            border: 2px dashed rgba(128,128,128,0.45); border-radius: 10px;
            background-color: var(--secondary-background-color);
            padding: 40px 20px; text-align: center; color: var(--text-color);
        }
        .rd-placeholder-title { font-weight: 700; opacity: 0.9; }
        .rd-placeholder-sub { font-size: 0.8rem; margin-top: 4px; opacity: 0.65; }
        .rd-placeholder-hint { font-size: 0.72rem; margin-top: 10px; opacity: 0.45; }
        .rd-placeholder-tag {
            display: inline-block; background-color: #1C7293; color: white;
            font-size: 0.7rem; font-weight: 700; padding: 3px 10px; border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_theme_colors() -> dict:
    """현재 활성 테마(라이트/다크)를 감지해서 Plotly 차트에 맞는 색상을 반환.
    Streamlit 위젯/카드는 var(--text-color)로 자동 대응되지만, Plotly는
    별도로 렌더링되는 컴포넌트라 이렇게 명시적으로 넘겨줘야 다크모드에서도
    글자가 흐릿하게 묻히지 않는다. 감지 실패 시(구버전 클라이언트 등)
    라이트 모드 색상으로 안전하게 대체한다."""
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:
        is_dark = False
    if is_dark:
        return {"bg": "#13273F", "text": "#E8EEF4", "grid": "rgba(232,238,244,0.15)"}
    return {"bg": "#FFFFFF", "text": "#16202B", "grid": "rgba(22,32,43,0.12)"}


def style_chart(fig, colors: dict, y_title: str = None, y_range: list = None):
    """Plotly figure에 테마 색상을 일괄 적용. 배경, 축 글자, 그리드, 막대 위 숫자 라벨까지 포함."""
    fig.update_layout(
        plot_bgcolor=colors["bg"], paper_bgcolor=colors["bg"],
        font=dict(color=colors["text"]),
        margin=dict(t=20, b=20),
    )
    fig.update_xaxes(tickfont=dict(color=colors["text"], size=12), gridcolor=colors["grid"])
    fig.update_yaxes(
        tickfont=dict(color=colors["text"]), gridcolor=colors["grid"],
        title=y_title, range=y_range,
    )
    fig.update_traces(textfont=dict(color=colors["text"], size=13))
    return fig


def status_badge(status: str) -> str:
    color, label = STATUS_COLOR.get(status, (GRAY, status))
    return f'<span class="rd-badge" style="background-color:{color};">{label}</span>'


def image_or_placeholder(path: str, caption: str, sub: str = "", kind: str = "IMAGE"):
    """
    path에 실제 이미지 파일이 있으면 그대로 보여주고,
    없으면 pptx 덱과 같은 스타일의 '자리 표시' 박스를 대신 보여준다.
    나중에 같은 경로에 이미지 파일만 넣으면 자동으로 실제 이미지로 바뀐다.
    """
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div class="rd-placeholder">
                <div class="rd-placeholder-tag">{kind}</div>
                <div class="rd-placeholder-title">{caption}</div>
                <div class="rd-placeholder-sub">{sub}</div>
                <div class="rd-placeholder-hint">
                    demo/assets/images/{p.name} 에 이미지를 넣으면 자동으로 표시됩니다
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
