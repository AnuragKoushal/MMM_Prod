"""
Budget optimization UI: constraint editor, optimizer controls, result charts,
and export functionality.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.optimization import build_constraints, run_optimization
from src.risk import summarize_with_uncertainty, build_risk_report
from src.export import export_json, export_csv
from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings()


def render_optimizer() -> None:
    """Render the budget optimization panel."""
    st.header("💰 Budget Optimizer")

    mmm = st.session_state.get("mmm")
    model_df = st.session_state.get("model_df")
    channel_cols = st.session_state.get("channel_cols", [])

    if mmm is None or model_df is None:
        st.warning("⚠️ Train a model first.")
        return

    # --- Budget input ---
    st.subheader("Total Budget")
    total_budget = st.number_input(
        "Total budget to allocate",
        min_value=1_000.0,
        max_value=100_000_000.0,
        value=float(cfg.optimization.default_budget),
        step=10_000.0,
        format="%.0f",
    )

    # --- Per-channel constraints ---
    st.subheader("Channel Constraints (% of total budget)")
    st.caption("Set the minimum and maximum share of budget each channel can receive.")

    overrides: dict = {}
    n_cols = min(len(channel_cols), 2)
    cols = st.columns(n_cols)

    for i, ch in enumerate(channel_cols):
        with cols[i % n_cols]:
            with st.expander(f"🔧 {ch}", expanded=False):
                lo = st.slider(
                    "Min share",
                    0.0, 1.0,
                    cfg.optimization.min_channel_share,
                    0.01,
                    key=f"opt_lo_{ch}",
                )
                hi = st.slider(
                    "Max share",
                    0.0, 1.0,
                    cfg.optimization.max_channel_share,
                    0.01,
                    key=f"opt_hi_{ch}",
                )
                overrides[ch] = (lo, hi)

    # --- Run optimization ---
    if st.button("⚡ Optimize Budget", type="primary", use_container_width=True):
        _run_optimizer(mmm, model_df, total_budget, channel_cols, overrides)


def _run_optimizer(mmm, model_df, total_budget, channel_cols, overrides):
    with st.spinner("Running optimization…"):
        try:
            bounds = build_constraints(channel_cols, overrides)
        except ValueError as exc:
            st.error(f"❌ Constraint error: {exc}")
            return

        try:
            allocation = run_optimization(mmm, total_budget, bounds)
        except RuntimeError as exc:
            st.error(f"❌ Optimization failed: {exc}")
            return

    # --- Results ---
    st.subheader("📊 Optimized Allocation")
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.dataframe(
            allocation.style.format(
                {"optimized_spend": "{:,.0f}", "share": "{:.1%}"}
            ),
            use_container_width=True,
        )

    with col_right:
        st.bar_chart(
            allocation.set_index("channel")["optimized_spend"],
            use_container_width=True,
        )

    # --- Uncertainty ---
    st.subheader("🎯 Predicted Outcome (Uncertainty)")
    with st.spinner("Predicting outcomes…"):
        try:
            preds = mmm.predict(model_df)
            risk = build_risk_report(preds)
            summary = risk["summary"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{summary['mean']:,.0f}")
            col2.metric("P5", f"{summary['p5']:,.0f}")
            col3.metric("P95", f"{summary['p95']:,.0f}")
            col4.metric("VaR (5%)", f"{risk['var_5pct']:,.0f}")

        except Exception as exc:
            st.warning(f"Prediction unavailable: {exc}")
            risk = {}

    # --- Export ---
    st.subheader("📥 Export Results")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("💾 Export JSON", use_container_width=True):
            try:
                path = export_json({"allocation": allocation, "risk": risk})
                st.success(f"Saved to `{path}`")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

    with col_b:
        if st.button("📄 Export CSV", use_container_width=True):
            try:
                path = export_csv(allocation, "allocation.csv")
                st.success(f"Saved to `{path}`")
            except Exception as exc:
                st.error(f"Export failed: {exc}")
