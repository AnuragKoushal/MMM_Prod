"""
Insights component — channel contributions, ROAS, response curves,
posterior diagnostics.
"""
from __future__ import annotations

import streamlit as st

from src.inference import (
    get_channel_contributions,
    plot_channel_contributions,
    plot_contribution_over_time,
    plot_response_curves,
    plot_posterior_diagnostics,
    plot_posterior_predictive_check,
    compute_roas,
)
from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings()


def render_insights() -> None:
    st.header("📊 Model Insights")

    # 🔥 DEBUG (remove later)
    # st.write("Session keys:", list(st.session_state.keys()))

    # ✅ SAFE ACCESS
    if "mmm" not in st.session_state or st.session_state["mmm"] is None:
        st.warning("⚠️ Train a model first (Train tab).")
        return

    if "model_df" not in st.session_state or "channel_cols" not in st.session_state:
        st.warning("⚠️ Missing data context.")
        return

    mmm          = st.session_state["mmm"]
    idata        = st.session_state.get("idata")
    model_df     = st.session_state["model_df"]
    channel_cols = st.session_state.get("channel_cols", [])

    # ── Channel Contributions ─────────────────────────────────────────────
    st.subheader("📌 Channel Contributions")
    with st.spinner("Computing contributions…"):
        try:
            contrib_df = get_channel_contributions(mmm, model_df, channel_cols)

            tab1, tab2 = st.tabs(["Bar Chart (Total)", "Over Time"])

            with tab1:
                st.pyplot(
                    plot_channel_contributions(contrib_df),
                    use_container_width=True
                )

            with tab2:
                st.pyplot(
                    plot_contribution_over_time(contrib_df),
                    use_container_width=True
                )

        except Exception as exc:
            st.error(f"Contribution plot failed: {exc}")
            log.exception(exc)

    # ── ROAS Table ────────────────────────────────────────────────────────
    st.subheader("💹 ROAS by Channel")
    if model_df is not None and channel_cols:
        with st.spinner("Computing ROAS…"):
            try:
                roas_df = compute_roas(mmm, model_df, channel_cols)

                st.dataframe(
                    roas_df.style.format({
                        "total_spend":        "{:,.0f}",
                        "total_contribution": "{:,.0f}",
                        "roas":               "{:.3f}",
                    }),
                    use_container_width=True,
                )

            except Exception as exc:
                st.warning(f"ROAS computation unavailable: {exc}")

    # ── Response Curves ───────────────────────────────────────────────────
    st.subheader("📈 Saturation / Response Curves")
    with st.spinner("Plotting response curves…"):
        try:
            fig = plot_response_curves(mmm, model_df, channel_cols)
            st.pyplot(fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Response curves unavailable: {exc}")

    # ── Posterior Diagnostics ─────────────────────────────────────────────
    if idata is not None:
        st.subheader("🔬 Posterior Diagnostics")

        with st.expander("Trace Plots", expanded=False):
            with st.spinner("Rendering trace plots…"):
                try:
                    fig = plot_posterior_diagnostics(idata)
                    st.pyplot(fig, use_container_width=True)

                except Exception as exc:
                    st.warning(f"Trace plots unavailable: {exc}")

        with st.expander("Posterior Predictive Check", expanded=False):
            with st.spinner("Running PPC…"):
                try:
                    fig = plot_posterior_predictive_check(
                        mmm, idata, cfg.data.target_col, model_df, channel_cols
                    )
                    st.pyplot(fig, use_container_width=True)

                except Exception as exc:
                    st.warning(f"PPC unavailable: {exc}")
