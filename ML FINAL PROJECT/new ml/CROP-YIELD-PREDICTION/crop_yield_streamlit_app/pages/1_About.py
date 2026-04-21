from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend import get_about_charts, get_runtime

st.set_page_config(page_title="About | AgriPredict", page_icon="📘", layout="wide")

PALETTE = {
    "bg": "#F6F5EF",
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

    [data-testid="stSidebar"] {{
        display: none;
    }}

    [data-testid="collapsedControl"] {{
        display: none;
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 100px !important;
        padding-bottom: 3rem;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: .65rem;
        font-weight: 700;
        font-size: 2rem;
        color: {PALETTE['green']};
    }}

    .brand-badge {{
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: linear-gradient(135deg, {PALETTE['green']}, {PALETTE['green_2']});
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        box-shadow: 0 8px 18px rgba(53,107,59,.22);
    }}

    .nav-shell {{
        background: rgba(255,255,255,.94);
        border: 1px solid {PALETTE['line']};
        border-radius: 22px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 28px rgba(31,52,34,.06);
        margin-bottom: 1.25rem;
    }}

    .kicker {{
        display: inline-block;
        padding: .38rem .8rem;
        border-radius: 999px;
        font-size: .82rem;
        font-weight: 700;
        color: {PALETTE['blue']};
        border: 1px solid rgba(91,151,208,.42);
        background: #F4F9FF;
    }}

    .hero-title {{
        font-size: 3rem;
        line-height: 1.14;
        font-weight: 800;
        margin: .9rem 0 .75rem;
        color: {PALETTE['text']};
        text-align: center;
    }}

    .hero-copy {{
        font-size: 1.04rem;
        color: {PALETTE['muted']};
        line-height: 1.85;
        max-width: 760px;
        margin: 0 auto;
        text-align: center;
    }}

    .small-note {{
        font-size: .92rem;
        color: {PALETTE['muted']};
    }}

    div[data-testid="stPageLink"] {{
        display: flex;
        justify-content: center;
    }}

    div[data-testid="stPageLink"] a {{
        padding: .58rem 1rem;
        border-radius: 14px;
        text-decoration: none;
        font-weight: 700;
        color: {PALETTE['muted']};
        border: 1px solid transparent;
    }}

    div[data-testid="stPageLink"] a:hover {{
        color: {PALETTE['green']};
        background: #F5F8F2;
        border-color: {PALETTE['line']};
    }}

    div[data-testid="stPageLink"] a[aria-current="page"] {{
        background: {PALETTE['green']};
        color: white !important;
        border-color: {PALETTE['green']};
        box-shadow: 0 8px 18px rgba(53,107,59,.22);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_top_nav() -> None:
    left, mid, right = st.columns([1.2, 0.8, 1.4], vertical_alignment="center")
    with left:
        st.markdown('<div class="brand"><div class="brand-badge">🌾</div><div>AgriPredict</div></div>', unsafe_allow_html=True)
    with mid:
        st.page_link("app.py", label="Home")
    with right:
        a, b = st.columns(2)
        with a:
            st.page_link("pages/1_About.py", label="About")
        with b:
            st.page_link("pages/2_Predict.py", label="🔮 Predict Yield")


@st.cache_resource(show_spinner=False)
def load_runtime():
    return get_runtime()


@st.cache_data(show_spinner=False)
def load_charts():
    return get_about_charts()


runtime = load_runtime()
charts = load_charts()


def style_fig(fig, height=330):
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

comparison_fig = style_fig(charts["comparison"], height=320)
comparison_fig.update_traces(marker_line_width=0)
actual_pred_fig = style_fig(charts["actual_vs_predicted"], height=320)
actual_pred_fig.update_traces(marker=dict(size=9, color=PALETTE["green_2"], opacity=0.72))
relationship_fig = style_fig(charts["feature_relationship"], height=340)
relationship_fig.update_traces(marker_color=PALETTE["green_2"])

with st.container(border=True):
    render_top_nav()

st.markdown('<div style="text-align:center;"><span class="kicker">📘 About This Project</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Crop Yield Prediction<br>Using Machine Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">This page keeps the backend connection active while presenting the project in a cleaner layout. It now reflects the expanded dataset with rainfall, humidity, crop season, and crop name.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    with st.container(border=True):
        st.markdown("<span class='kicker' style='color:#D88A2A; border-color:rgba(216,138,42,.42); background:#F8F0D8;'>🗂️ Dataset Details</span>", unsafe_allow_html=True)
        st.markdown("<h2 style='margin:.75rem 0 1rem 0; color:#1F3422;'>The Crop Yield Dataset</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='small-note' style='line-height:1.95;'>The dataset contains <strong>{runtime.metrics['rows_raw']:,}</strong> rows after loading. The cleaned training data retains <strong>{runtime.metrics['rows_processed']:,}</strong> rows after preprocessing and outlier handling. The target variable is yield, predicted from fertilizer, temperature, NPK, rainfall, humidity, crop season, and crop name values.</div>",
            unsafe_allow_html=True,
        )
        for label, value in [
            ("Total Samples", f"{runtime.metrics['rows_raw']:,}"),
            ("Processed Rows", f"{runtime.metrics['rows_processed']:,}"),
            ("Target Variable", "yield (tons per hectare)"),
            ("Missing Values", "Imputed using column medians"),
            ("Negative Values", "Corrected using abs()"),
            ("Train / Test Split", "80% / 20%"),
        ]:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:.65rem 0; border-bottom:1px solid {PALETTE['line']};'><span class='small-note'>{label}</span><span style='font-weight:700; color:{PALETTE['text']};'>{value}</span></div>",
                unsafe_allow_html=True,
            )
with right:
    features = [
        ("🧪", "Fertilizer", "Total fertilizer applied to the field per hectare."),
        ("🌡️", "Temperature", "Average field temperature during the growing season."),
        ("🧬", "Nitrogen (N)", "Soil nitrogen content important for leaf and shoot growth."),
        ("🧂", "Phosphorus (P)", "Soil phosphorus that supports root development."),
        ("🪴", "Potassium (K)", "Soil potassium linked to water balance and resilience."),
        ("🌧️", "Rainfall", "Rainfall captured in the updated dataset for moisture conditions."),
        ("💧", "Humidity", "Humidity used as an added environmental input."),
        ("🗓️", "Crop Season", "Season feature included in the updated dataset and prediction flow."),
        ("🗂️", "Crop Name", "Categorical crop type feature added to the updated dataset."),
        ("🌾", "Yield", "The final crop output the model tries to estimate."),
    ]
    for emoji, title, desc in features:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-weight:700; color:{PALETTE['text']}; margin-bottom:.25rem;'>{emoji} {title}</div><div class='small-note'>{desc}</div>",
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<span class='kicker' style='color:#D88A2A; border-color:rgba(216,138,42,.42); background:#F8F0D8;'>📊 Model Evaluation</span>", unsafe_allow_html=True)
st.markdown("<h2 style='margin:.75rem 0 .4rem 0; color:#1F3422;'>Performance Metrics</h2>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4, gap="large")
for col, (emoji, label, value, color) in zip(
    [m1, m2, m3, m4],
    [
        ("🎯", "R² Score", f"{runtime.metrics['r2']*100:.1f}%", PALETTE["green"]),
        ("📉", "MAE", f"{runtime.metrics['mae']:.3f}", PALETTE["gold"]),
        ("📏", "RMSE", f"{runtime.metrics['rmse']:.3f}", PALETTE["blue"]),
        ("🏋️", "Train R²", f"{runtime.metrics['train_r2']*100:.1f}%", PALETTE["green_2"]),
    ],
):
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:1.4rem;'>{emoji}</div><div style='font-size:2rem; font-weight:800; color:{color};'>{value}</div><div class='small-note'>{label}</div>",
                unsafe_allow_html=True,
            )

c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(border=True):
        st.plotly_chart(comparison_fig, use_container_width=True, config={"displayModeBar": False})
with c2:
    with st.container(border=True):
        st.plotly_chart(actual_pred_fig, use_container_width=True, config={"displayModeBar": False})

with st.container(border=True):
    st.plotly_chart(relationship_fig, use_container_width=True, config={"displayModeBar": False})
