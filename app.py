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
    render_scenario_planner,
    render_optimizer,
)
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings().app

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=cfg.title,
    page_icon=cfg.page_icon,
    layout=cfg.layout,
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
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

# ── Sidebar (always rendered) ─────────────────────────────────────────────────
render_sidebar()

# ── Page title ────────────────────────────────────────────────────────────────
st.title(f"{cfg.page_icon} {cfg.title}")
st.caption("Bayesian Media Mix Modelling · Budget Optimisation · Scenario Planning")
st.divider()

# ── Tab navigation ────────────────────────────────────────────────────────────
TABS = [
    "📂 Data",
    "🚀 Train",
    "📊 Insights",
    "🔮 Scenarios",
    "💰 Optimise",
]

tab_data, tab_train, tab_insights, tab_scenario, tab_opt = st.tabs(TABS)

with tab_data:
    render_data_upload()

with tab_train:
    render_model_training()

with tab_insights:
    render_insights()

with tab_scenario:
    render_scenario_planner()

with tab_opt:
    render_optimizer()
