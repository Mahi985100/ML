from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend import get_about_charts, get_feature_ranges, get_runtime, predict_yield

st.set_page_config(page_title="Predict | AgriPredict", page_icon="🌾", layout="wide")

PALETTE = {
    "bg": "#F6F5EF",
    "green": "#356B3B",
    "green_2": "#7FA37D",
    "gold": "#D88A2A",
    "text": "#1F3422",
    "muted": "#66756B",
    "line": "#DCE5D8",
}

FEATURE_LABELS = {
    "fertilizer": "Fertilizer",
    "temp": "Temperature",
    "n": "Nitrogen (N)",
    "p": "Phosphorus (P)",
    "k": "Potassium (K)",
    "rainfall": "Rainfall",
    "humidity": "Humidity",
}
FEATURE_UNITS = {
    "fertilizer": "kg/ha",
    "temp": "°C",
    "n": "kg/ha",
    "p": "kg/ha",
    "k": "kg/ha",
    "rainfall": "mm",
    "humidity": "%",
}
SEASON_OPTIONS = ["Kharif", "Rabi"]
CROP_OPTIONS = ["Cotton", "Maize", "Rice", "Wheat"]
SEASON_MAP = {"Kharif": 0, "Rabi": 1}
CROP_MAP = {"Cotton": 0, "Maize": 1, "Rice": 2, "Wheat": 3}

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

    .nav-shell {{
        background: rgba(255,255,255,.94);
        border: 1px solid {PALETTE['line']};
        border-radius: 22px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 28px rgba(31,52,34,.06);
        margin-bottom: 1.25rem;
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

    .hero {{
        background: linear-gradient(135deg, #F8FBF6 0%, #EDF5E9 100%);
        border: 1px solid {PALETTE['line']};
        border-radius: 28px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 12px 28px rgba(31,52,34,.05);
        margin-bottom: 1rem;
    }}

    .chip {{
        display: inline-block;
        padding: .45rem .9rem;
        border-radius: 999px;
        background: #F8F0D8;
        color: {PALETTE['gold']};
        border: 1px solid rgba(216,138,42,.35);
        font-weight: 700;
        font-size: .86rem;
    }}

    .hero-title {{
        font-size: 2.8rem;
        line-height: 1.08;
        font-weight: 800;
        color: {PALETTE['text']};
        margin: .8rem 0 .4rem 0;
    }}

    .hero-copy {{
        font-size: 1rem;
        color: {PALETTE['muted']};
        line-height: 1.8;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .75rem;
        margin-top: 1rem;
    }}

    .metric {{
        background: #fff;
        border: 1px solid {PALETTE['line']};
        border-radius: 18px;
        padding: .85rem 1rem;
    }}

    .metric .label {{
        font-size: .85rem;
        color: {PALETTE['muted']};
    }}

    .metric .value {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {PALETTE['green']};
        margin-top: .15rem;
    }}

    .section-title {{
        font-size: 1.18rem;
        font-weight: 800;
        color: {PALETTE['text']};
        margin-bottom: .2rem;
    }}

    .section-copy {{
        font-size: .94rem;
        color: {PALETTE['muted']};
        line-height: 1.7;
        margin-bottom: .8rem;
    }}

    .snapshot-row {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid {PALETTE['line']};
        padding: .46rem 0;
        font-size: .95rem;
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

    .stButton>button,
    .stFormSubmitButton>button {{
        border-radius: 16px !important;
        font-weight: 800 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_ranges():
    return get_feature_ranges()

@st.cache_data(show_spinner=False)
def load_charts():
    return get_about_charts()

@st.cache_resource(show_spinner=False)
def load_runtime():
    return get_runtime()

runtime = load_runtime()
ranges = load_ranges()
charts = load_charts()
keys = ["fertilizer", "temp", "n", "p", "k", "rainfall", "humidity"]

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "input_values" not in st.session_state:
    st.session_state.input_values = {k: float(ranges[k]["default"]) for k in keys}
    st.session_state.input_values["crop_season"] = "Kharif"
    st.session_state.input_values["crop_name"] = "Wheat"


def reset_inputs():
    st.session_state.input_values = {k: float(ranges[k]["default"]) for k in keys}
    st.session_state.input_values["crop_season"] = "Kharif"
    st.session_state.input_values["crop_name"] = "Wheat"
    st.session_state.prediction_result = None


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=13),
        margin=dict(l=18, r=18, t=48, b=18),
        xaxis=dict(showgrid=True, gridcolor="#E8EEE6", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#E8EEE6", zeroline=False),
    )
    return fig

relationship_fig = style_fig(charts["feature_relationship"])

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

with st.container(border=True):
    render_top_nav()

st.markdown(
    f"""
    <div class='hero'>
      <span class='chip'>🌿 Crop-based Prediction UI • Random Forest Model</span>
      <div class='hero-title'>Predict Crop Yield with richer field inputs</div>
      <div class='hero-copy'>This page now includes fertilizer, temperature, NPK, rainfall, humidity, crop season, and crop name so the frontend matches the updated dataset and backend pipeline.</div>
      <div class='metric-grid'>
        <div class='metric'><div class='label'>Processed rows</div><div class='value'>{runtime.metrics['rows_processed']:,}</div></div>
        <div class='metric'><div class='label'>Test R²</div><div class='value'>{runtime.metrics['r2']:.3f}</div></div>
        <div class='metric'><div class='label'>Input features</div><div class='value'>9</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.02, .98], gap="large")

with left:
    with st.container(border=True):
        st.markdown("<div class='section-title'>🪴 Field Parameters</div><div class='section-copy'>Adjust the values to match field conditions, then generate a live backend-powered crop yield prediction.</div>", unsafe_allow_html=True)
        current = st.session_state.input_values
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            new_inputs = {}
            for idx, feature in enumerate(keys):
                col = c1 if idx % 2 == 0 else c2
                with col:
                    meta = ranges[feature]
                    new_inputs[feature] = st.slider(
                        f"{FEATURE_LABELS[feature]} ({FEATURE_UNITS[feature]})",
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(current.get(feature, meta["default"])),
                        step=float(meta["step"]),
                    )
            c3, c4 = st.columns(2)
            with c3:
                new_inputs["crop_season"] = st.selectbox("Crop Season", SEASON_OPTIONS, index=SEASON_OPTIONS.index(current.get("crop_season", "Kharif")))
            with c4:
                new_inputs["crop_name"] = st.selectbox("Crop Name", CROP_OPTIONS, index=CROP_OPTIONS.index(current.get("crop_name", "Wheat")))
            b1, b2 = st.columns(2)
            predict_clicked = b1.form_submit_button("🔮 Predict Yield", use_container_width=True)
            reset_clicked = b2.form_submit_button("♻ Reset Form", use_container_width=True)

        if predict_clicked:
            st.session_state.input_values = new_inputs.copy()
            st.session_state.prediction_result = predict_yield(
                fertilizer=new_inputs["fertilizer"],
                temp=new_inputs["temp"],
                n=new_inputs["n"],
                p=new_inputs["p"],
                k=new_inputs["k"],
                crop_season=SEASON_MAP[new_inputs["crop_season"]],
                rainfall=new_inputs["rainfall"],
                humidity=new_inputs["humidity"],
                crop_name=CROP_MAP[new_inputs["crop_name"]],
            )
        if reset_clicked:
            reset_inputs()
            st.rerun()

        snapshot = st.session_state.input_values
        rows = []
        for feature in keys:
            rows.append(f"<div class='snapshot-row'><span>{FEATURE_LABELS[feature]}</span><strong>{snapshot[feature]:.0f} {FEATURE_UNITS[feature]}</strong></div>")
        rows.append(f"<div class='snapshot-row'><span>Crop Season</span><strong>{snapshot['crop_season']}</strong></div>")
        rows.append(f"<div class='snapshot-row'><span>Crop Name</span><strong>{snapshot['crop_name']}</strong></div>")
        st.markdown(f"<div style='margin-top:1rem;'><div class='section-title' style='font-size:1rem;'>Current Input Snapshot</div>{''.join(rows)}</div>", unsafe_allow_html=True)

with right:
    with st.container(border=True):
        pred = st.session_state.prediction_result
        if pred is None:
            defaults = st.session_state.input_values
            pred = predict_yield(
                fertilizer=defaults["fertilizer"],
                temp=defaults["temp"],
                n=defaults["n"],
                p=defaults["p"],
                k=defaults["k"],
                crop_season=SEASON_MAP[defaults["crop_season"]],
                rainfall=defaults["rainfall"],
                humidity=defaults["humidity"],
                crop_name=CROP_MAP[defaults["crop_name"]],
            )
        max_yield = max(float(runtime.processed_df["yield"].max()), 1.0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"font": {"size": 34, "color": PALETTE["green"]}},
            gauge={
                "axis": {"range": [0, max_yield]},
                "bar": {"color": PALETTE["green_2"]},
                "steps": [
                    {"range": [0, max_yield*0.33], "color": "#E8EEE6"},
                    {"range": [max_yield*0.33, max_yield*0.66], "color": "#DCE9D7"},
                    {"range": [max_yield*0.66, max_yield], "color": "#CFE2CF"},
                ],
            },
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.markdown("<div class='section-title'>🌾 Predicted Crop Yield</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:3.4rem; font-weight:800; color:{PALETTE['green']};'>{pred:.2f}</div>", unsafe_allow_html=True)
        st.caption("tons / hectare")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown("<div class='section-title'>📊 Feature Influence</div><div class='section-copy'>This chart now includes the added dataset inputs instead of showing only the old five-column view.</div>", unsafe_allow_html=True)
        st.plotly_chart(relationship_fig, use_container_width=True, config={"displayModeBar": False})
