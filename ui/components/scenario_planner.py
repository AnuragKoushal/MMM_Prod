"""
Scenario planner UI: interactive what-if budget multipliers.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.scenario import generate_scenario, simulate_scenario, build_scenario_summary
from src.risk import summarize_with_uncertainty, plot_prediction_distribution
from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings()


def render_scenario_planner() -> None:
    """Render the multi-period scenario planning tool."""
    st.header("🔮 Scenario Planner")

    mmm = st.session_state.get("mmm")
    model_df = st.session_state.get("model_df")
    channel_cols = st.session_state.get("channel_cols", [])

    if mmm is None or model_df is None:
        st.warning("⚠️ Train a model first.")
        return

    st.markdown(
        "Adjust spend **multipliers** per channel (1.0 = baseline). "
        "The model will predict the target metric for the specified number of future periods."
    )

    # --- Multiplier inputs ---
    st.subheader("Channel Spend Multipliers")
    n_cols = min(len(channel_cols), 3)
    cols = st.columns(n_cols)
    multipliers: dict[str, float] = {}

    for i, ch in enumerate(channel_cols):
        with cols[i % n_cols]:
            multipliers[ch] = st.slider(
                ch,
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key=f"scenario_mult_{ch}",
            )

    periods = st.number_input(
        "Forecast Periods (months)",
        min_value=1,
        max_value=24,
        value=cfg.optimization.n_scenarios,
    )

    if st.button("▶️ Run Scenario", type="primary"):
        _run_scenario(mmm, model_df, channel_cols, multipliers, int(periods))


def _run_scenario(mmm, model_df, channel_cols, multipliers, periods):
    with st.spinner("Simulating scenario…"):
        try:
            scenario_df = generate_scenario(model_df, multipliers, periods=periods)
            preds = simulate_scenario(mmm, scenario_df)
            summary_df = build_scenario_summary(
                scenario_df, preds, cfg.data.target_col, channel_cols
            )
            uncertainty = summarize_with_uncertainty(preds)

            st.subheader("📊 Scenario Forecast")
            st.dataframe(
                summary_df.tail(periods).style.format(
                    {c: "{:,.0f}" for c in summary_df.select_dtypes("number").columns}
                ),
                use_container_width=True,
            )

            st.subheader("📉 Prediction Distribution")
            fig = plot_prediction_distribution(preds)
            st.pyplot(fig, use_container_width=True)

            st.subheader("🎯 Uncertainty Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Mean", f"{uncertainty['mean']:,.0f}")
            col2.metric("P5", f"{uncertainty['p5']:,.0f}")
            col3.metric("P25", f"{uncertainty['p25']:,.0f}")
            col4.metric("P75", f"{uncertainty['p75']:,.0f}")
            col5.metric("P95", f"{uncertainty['p95']:,.0f}")

        except Exception as exc:
            st.error(f"Scenario simulation failed: {exc}")
            log.exception(exc)
