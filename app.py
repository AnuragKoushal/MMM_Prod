"""
Enterprise MMM Decision Engine – main entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from config import get_settings
from ui.components import (
    render_sidebar,
    render_data_upload,
    render_model_training,
    render_insights,
    render_report_tab,
    render_scenario_planner,
    render_optimizer,
)
from utils import get_logger, reset_app_state

log = get_logger(__name__)
cfg = get_settings().app


# ── SESSION STATE INITIALIZATION (CRITICAL) ────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True

    # Data
    st.session_state["model_df"] = None
    st.session_state["channel_cols"] = None

    # Model artifacts
    st.session_state["mmm"] = None
    st.session_state["idata"] = None
    st.session_state["eval_report"] = None

    # Control flags
    st.session_state["data_locked"] = False


# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title=cfg.title,
    page_icon=cfg.page_icon,
    layout=cfg.layout,
    initial_sidebar_state="expanded",
)


# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0 20px;
            border-radius: 6px 6px 0 0;
        }
        div[data-testid="metric-container"] > label {
            font-size: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────
render_sidebar()


# ── Page title ─────────────────────────────────────────────────────────────
st.title(f"{cfg.page_icon} {cfg.title}")
st.caption("Bayesian Media Mix Modelling · Budget Optimisation · Scenario Planning")
st.divider()


# ── Global status bar (UX upgrade) ─────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.get("mmm") is not None:
        st.success("✅ Model trained and ready")
    else:
        st.warning("⚠️ No trained model")

with col2:
    if st.button("🔄 Reset App"):
        reset_app_state()


# ── Tabs ───────────────────────────────────────────────────────────────────
TABS = [
    "📂 Data",
    "🚀 Train",
    "📊 Insights",
    "📝 Report",
    "🔮 Scenarios",
    "💰 Optimise",
]

tab_data, tab_train, tab_insights, tab_report, tab_scenario, tab_opt = st.tabs(TABS)


# ── DATA TAB (Protected) ───────────────────────────────────────────────────
with tab_data:
    render_data_upload()


# ── TRAIN TAB ──────────────────────────────────────────────────────────────
with tab_train:
    render_model_training()


# ── INSIGHTS TAB ───────────────────────────────────────────────────────────
with tab_insights:
    render_insights()
with tab_report:
    render_report_tab()


# ── SCENARIO TAB ───────────────────────────────────────────────────────────
with tab_scenario:
    render_scenario_planner()


# ── OPTIMIZER TAB ──────────────────────────────────────────────────────────
with tab_opt:
    render_optimizer()
