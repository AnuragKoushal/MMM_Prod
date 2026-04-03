"""
Sidebar component: app-wide controls and session state display.
"""
from __future__ import annotations

import streamlit as st
from config import get_settings

cfg = get_settings()


def render_sidebar() -> None:
    """Render the application sidebar with global controls."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
        st.title("MMM Engine")
        st.caption("Enterprise Marketing Mix Model")

        st.divider()

        st.subheader("⚙️ Model Settings")
        draws = st.number_input(
            "MCMC Draws",
            min_value=200,
            max_value=5000,
            value=cfg.model.draws,
            step=100,
            help="More draws → better estimates, slower training.",
        )
        chains = st.number_input(
            "MCMC Chains",
            min_value=1,
            max_value=8,
            value=cfg.model.chains,
            step=1,
        )
        adstock_lag = st.slider(
            "Adstock Max Lag (weeks)",
            min_value=1,
            max_value=12,
            value=cfg.model.adstock_max_lag,
        )

        st.divider()

        st.subheader("📁 Session State")
        has_data = st.session_state.get("model_df") is not None
        has_model = st.session_state.get("mmm") is not None

        st.markdown(
            f"- Data loaded: {'✅' if has_data else '❌'}\n"
            f"- Model trained: {'✅' if has_model else '❌'}"
        )

        if st.button("🗑️ Reset Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.caption("Built with PyMC-Marketing · Streamlit")

    # Persist sidebar settings into session_state for downstream use
    st.session_state["mcmc_draws"] = draws
    st.session_state["mcmc_chains"] = chains
    st.session_state["adstock_lag"] = adstock_lag
