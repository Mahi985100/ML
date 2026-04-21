from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from backend import get_about_charts, get_runtime, refresh_runtime

st.set_page_config(page_title="Home | AgriPredict", page_icon="🌾", layout="wide")

PALETTE = {
    "bg": "#F6F5EF",
    "card": "#FFFFFF",
    "soft": "#EEF4EA",
    "green": "#356B3B",
    "green_2": "#7FA37D",
    "gold": "#D88A2A",
    "blue": "#5B97D0",
    "text": "#1F3422",
    "muted": "#66756B",
    "line": "#DCE5D8",
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {PALETTE['bg']} 0%, #FBFBF8 100%);
        color: {PALETTE['text']};
    }}

    [data-testid="stSidebar"] {{display:none;}}
    [data-testid="collapsedControl"] {{display:none;}}

    /* 🔥 FIX NAVBAR CUT */
    .block-container {{
        max-width: 1180px;
        padding-top: 130px !important;   /* increased */
        padding-bottom: 3rem;
    }}

    .nav-shell {{
        background: rgba(255,255,255,.94);
        border: 1px solid {PALETTE['line']};
        border-radius: 22px;
        padding: 1.15rem 1.25rem;
        min-height: 90px;                /* important */
        display: flex;
        align-items: center;
        box-shadow: 0 10px 28px rgba(31,52,34,.06);
        margin-bottom: 1.25rem;
    }}

    .brand {{
        display:flex;
        align-items:center;
        gap:.65rem;
        font-weight:700;
        font-size:2rem;
        color:{PALETTE['green']};
    }}

    .brand-badge {{
        width:38px;
        height:38px;
        border-radius:12px;
        background: linear-gradient(135deg, {PALETTE['green']} 0%, {PALETTE['green_2']} 100%);
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:1.05rem;
        box-shadow: 0 8px 18px rgba(53,107,59,.22);
    }}

    .hero-wrap {{
        background: radial-gradient(circle at top right, rgba(127,163,125,.12), transparent 35%),
                    linear-gradient(135deg, #FBFBF7 0%, #F4F7EF 100%);
        border: 1px solid {PALETTE['line']};
        border-radius: 28px;
        padding: 1.65rem;
        box-shadow: 0 16px 38px rgba(31,52,34,.06);
        margin-bottom: 1.2rem;
    }}

    .badge {{
        display:inline-block;
        padding:.42rem .9rem;
        border-radius:999px;
        font-size:.88rem;
        font-weight:600;
        background:#F8F0D8;
        color:{PALETTE['gold']};
        border:1px solid rgba(216,138,42,.42);
    }}

    .hero-title {{
        font-size:4rem;
        line-height:1.04;
        color:{PALETTE['text']};
        font-weight:800;
        margin:.95rem 0 .8rem 0;
    }}

    .hero-copy {{
        font-size:1.16rem;
        color:{PALETTE['muted']};
        line-height:1.8;
        max-width:95%;
    }}

    /* NAV LINKS */
    div[data-testid="stPageLink"] {{
        display:flex;
        justify-content:center;
    }}

    div[data-testid="stPageLink"] a {{
        padding: .58rem 1rem;
        border-radius: 14px;
        text-decoration:none;
        font-weight:700;
        color: {PALETTE['muted']};
        border: 1px solid transparent;
    }}

    div[data-testid="stPageLink"] a:hover {{
        color: {PALETTE['green']};
        background:#F5F8F2;
        border-color:{PALETTE['line']};
    }}

    div[data-testid="stPageLink"] a[aria-current="page"] {{
        background:{PALETTE['green']};
        color:white !important;
        border-color:{PALETTE['green']};
        box-shadow: 0 8px 18px rgba(53,107,59,.22);
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


def render_top_nav() -> None:
    left, mid, right = st.columns([1.2, 0.8, 1.4], vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="brand"><div class="brand-badge">🌾</div><div>AgriPredict</div></div>',
            unsafe_allow_html=True,
        )
    with mid:
        st.page_link("app.py", label="Home")
    with right:
        nav1, nav2 = st.columns(2)
        with nav1:
            st.page_link("pages/1_About.py", label="About")
        with nav2:
            st.page_link("pages/2_Predict.py", label="🔮 Predict Yield")


@st.cache_resource(show_spinner=False)
def load_runtime():
    return get_runtime()


@st.cache_data(show_spinner=False)
def load_about_charts():
    return get_about_charts()


runtime = load_runtime()
charts = load_about_charts()
model = runtime.pipeline.named_steps["model"]
feature_count = len(runtime.feature_ranges)


def style_plotly(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=13),
        title=dict(font=dict(size=20, color=PALETTE["text"]), x=0.02),
        margin=dict(l=18, r=18, t=52, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#E8EEE6", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#E8EEE6", zeroline=False),
    )
    return fig


comparison_fig = style_plotly(charts["comparison"], height=300)
comparison_fig.update_traces(marker_line_width=0)
scatter_fig = style_plotly(charts["actual_vs_predicted"], height=330)
scatter_fig.update_traces(marker=dict(size=9, color=PALETTE["green_2"], opacity=0.72))

with st.container(border=True):
    render_top_nav()

with st.container(border=True):
    hero_left, hero_right = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    with hero_left:
        st.markdown('<span class="badge">🌾 ML-Powered Agriculture • Random Forest Regressor</span>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Predict Crop Yield<br>with Precision.</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="hero-copy">Enter soil nutrients, fertilizer dosage, temperature, rainfall, humidity, crop season, and crop name to instantly estimate crop yield — powered by your existing backend-connected prediction pipeline with <strong>{runtime.metrics["r2"] * 100:.1f}% R² accuracy</strong>.</div>',
            unsafe_allow_html=True,
        )
        cta1, cta2, cta3 = st.columns([1.25, 1.05, 0.9])
        with cta1:
            st.page_link("pages/2_Predict.py", label="🔮 Try Prediction", use_container_width=True)
        with cta2:
            st.page_link("pages/1_About.py", label="Learn More →", use_container_width=True)
        with cta3:
            if st.button("Refresh", use_container_width=True):
                refresh_runtime()
                load_runtime.clear()
                load_about_charts.clear()
                st.rerun()

    with hero_right:
        top_a, top_b = st.columns([1, 1])
        with top_a:
            st.markdown('<span class="small-note"><strong>📊 MODEL PERFORMANCE DASHBOARD</strong></span>', unsafe_allow_html=True)
        with top_b:
            st.markdown('<div style="text-align:right;"><span class="hero-chip">🌡️ Temp-aware</span></div>', unsafe_allow_html=True)
        st.plotly_chart(comparison_fig, use_container_width=True, config={"displayModeBar": False})
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='info-card' style='padding:.95rem 1rem; text-align:center;'><div style='font-size:1.95rem; font-weight:800; color:{PALETTE['text']};'>{runtime.metrics['r2']*100:.1f}%</div><div class='small-note'>R² Score</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='info-card' style='padding:.95rem 1rem; text-align:center;'><div style='font-size:1.95rem; font-weight:800; color:{PALETTE['text']};'>{runtime.metrics['rows_raw']:,}</div><div class='small-note'>Dataset Rows</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='info-card' style='padding:.95rem 1rem; text-align:center;'><div style='font-size:1.95rem; font-weight:800; color:{PALETTE['text']};'>{feature_count}</div><div class='small-note'>Features</div></div>", unsafe_allow_html=True)


metrics_html = f"""
<div class="metric-band">
  <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap: .5rem;">
    <div class="metric-item"><div style="font-size:1.6rem;">📦</div><div class="metric-value">{runtime.metrics['rows_raw']:,}</div><div class="metric-label">Training Samples</div></div>
    <div class="metric-item"><div style="font-size:1.6rem;">🎯</div><div class="metric-value">{runtime.metrics['r2']*100:.1f}%</div><div class="metric-label">R² Accuracy</div></div>
    <div class="metric-item"><div style="font-size:1.6rem;">📉</div><div class="metric-value">{runtime.metrics['mae']:.3f}</div><div class="metric-label">Mean Abs. Error</div></div>
    <div class="metric-item"><div style="font-size:1.6rem;">🧬</div><div class="metric-value">{feature_count}</div><div class="metric-label">Input Features</div></div>
    <div class="metric-item"><div style="font-size:1.6rem;">🌲</div><div class="metric-value">{getattr(model, 'n_estimators', '—')}</div><div class="metric-label">Forest Trees</div></div>
  </div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)

st.markdown('<div style="text-align:center; margin-top:1rem;"><span class="section-kicker">📁 Dataset Overview</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Understanding the Data</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-copy">The dataset contains <strong>{runtime.metrics["rows_raw"]:,}</strong> agricultural records spanning soil nutrients, temperature, rainfall, humidity, crop season, and crop type. The home page now reflects the expanded feature set while keeping the overall experience clean and fast.</div>',
    unsafe_allow_html=True,
)


def render_feature_distribution_card(df: pd.DataFrame) -> str:
    rows = [
        ("Fertilizer", "fertilizer", "kg/ha"),
        ("Temperature", "temp", "°C"),
        ("Nitrogen (N)", "n", "kg/ha"),
        ("Phosphorus (P)", "p", "kg/ha"),
        ("Potassium (K)", "k", "kg/ha"),
        ("Rainfall", "rainfall", "mm"),
        ("Humidity", "humidity", "%"),
        ("Yield", "yield", "tons/ha"),
    ]
    html = [
        "<div class='section-card'><h3 style='margin:.1rem 0 1.2rem 0; color:#1F3422; font-size:1.85rem;'>📊 Feature Distributions</h3>"
    ]
    for label, col, unit in rows:
        mean_val = float(df[col].mean())
        max_val = float(df[col].max()) if float(df[col].max()) != 0 else 1.0
        percent = max(min((mean_val / max_val) * 100, 100), 0)
        html.append(
            f"<div class='feature-row'>"
            f"<div class='feature-head'><span>{label}</span><span>Mean: <strong>{mean_val:.1f} {unit}</strong></span></div>"
            f"<div class='feature-track'><div class='feature-fill' style='width:{percent:.1f}%'></div></div>"
            f"<div class='feature-foot'><span>0</span><span>{max_val:.0f} {unit}</span></div>"
            f"</div>"
        )
    html.append("</div>")
    return "".join(html)

def donut_figure(title: str, value: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            values=[value, max(1 - value, 0)],
            hole=0.7,
            sort=False,
            marker=dict(colors=[color, "#E7EEE5"], line=dict(width=0)),
            textinfo="none",
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=18, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{value*100:.1f}%</b><br>R²", showarrow=False, font=dict(size=16, color=PALETTE['text']))],
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color=PALETTE["text"])),
        showlegend=False,
    )
    return fig

left_card, right_card = st.columns(2, gap="large")
with left_card:
    st.markdown(render_feature_distribution_card(runtime.processed_df), unsafe_allow_html=True)
with right_card:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:.1rem 0 .2rem 0; color:#1F3422; font-size:1.85rem;'>🎯 Model Accuracy Comparison</h3><div class='small-note' style='margin-bottom:1rem;'>R² score on test data — a quick visual comparison between the final model and the baseline model.</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    final_r2 = runtime.model_comparison["Final Model"]["R2"]
    base_r2 = runtime.model_comparison["Baseline"]["R2"]
    with d1:
        st.plotly_chart(donut_figure("Final Model", final_r2, PALETTE["green_2"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='info-card' style='padding:.85rem 1rem; margin-top:.35rem;'><div class='small-note'>Train R²</div><div style='font-weight:700; color:{PALETTE['green']};'>{runtime.metrics['train_r2']:.3f}</div><div class='small-note'>MAE: {runtime.metrics['mae']:.3f} &nbsp; • &nbsp; RMSE: {runtime.metrics['rmse']:.3f}</div></div>", unsafe_allow_html=True)
    with d2:
        st.plotly_chart(donut_figure("Baseline Model", base_r2, PALETTE["blue"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='info-card' style='padding:.85rem 1rem; margin-top:.35rem;'><div class='small-note'>Test R²</div><div style='font-weight:700; color:{PALETTE['blue']};'>{base_r2:.3f}</div><div class='small-note'>MAE: {runtime.model_comparison['Baseline']['MAE']:.3f} &nbsp; • &nbsp; RMSE: {runtime.model_comparison['Baseline']['RMSE']:.3f}</div></div>", unsafe_allow_html=True)
    delta = (final_r2 - base_r2) * 100
    st.markdown(f"<div class='info-card' style='padding:1rem 1.05rem; margin-top:1rem; background:#F2FAF0; border-color:#CFE2CF;'><strong>🏆 Final model improves test performance by {delta:.1f}% R²</strong> while keeping prediction error lower for a cleaner user experience.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.write("")
st.markdown(
    f"""
    <div style="text-align:center; margin-top:2rem; margin-bottom:1.2rem;">
        <div class="section-kicker">⚙️ Prediction Flow</div>
        <div class="section-title" style="margin-top:.7rem;">How It Works</div>
       
    </div>
    """,
    unsafe_allow_html=True,
)

steps = [
    ("01", "🌱", "Enter Field Data", "Provide fertilizer, temperature, soil nutrients, rainfall, humidity, season, and crop details."),
    ("02", "⚙️", "Model Processing", "The backend pipeline cleans the inputs and runs the trained Random Forest model."),
    ("03", "📈", "Get Yield Estimate", "View the predicted crop yield with a clean result card and helpful visual feedback."),
]

step_cols = st.columns(3, gap="large")

for col, (idx, emoji, title, copy) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-index">{idx}</div>
                <span class="step-emoji">{emoji}</span>
                <div class="step-title">{title}</div>
                <div class="step-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
cta_left, cta_mid, cta_right = st.columns([1, 1.2, 1])

with cta_mid:
    st.page_link("pages/2_Predict.py", label="🔮 Start Predicting", use_container_width=True)
bottom_left, bottom_right = st.columns([0.7, 1.3], gap="large")

