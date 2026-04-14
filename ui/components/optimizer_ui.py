"""
Budget optimizer UI — compatible with pymc-marketing >= 0.19.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.optimization import build_constraints, run_optimization
from src.risk import build_risk_report
from src.export import export_json, export_csv
from config import get_settings
from utils import get_logger, predict_array

log = get_logger(__name__)
cfg = get_settings()


def render_optimizer() -> None:
    st.header("💰 Budget Optimizer")

    mmm          = st.session_state.get("mmm")
    model_df     = st.session_state.get("model_df")
    channel_cols = st.session_state.get("channel_cols", [])

    if mmm is None or model_df is None:
        st.warning("⚠️ Train a model first (Train tab).")
        return

    st.subheader("Total Budget")
    total_budget = st.number_input(
        "Total budget to allocate",
        min_value=1_000.0, max_value=100_000_000.0,
        value=float(cfg.optimization.default_budget),
        step=10_000.0, format="%.0f",
    )

    st.subheader("Channel Constraints (share of total budget)")
    st.caption("Set min/max share for each channel. Leave at defaults for unconstrained optimisation.")

    overrides: dict = {}
    n_cols = min(len(channel_cols), 2)
    cols = st.columns(n_cols or 1)

    for i, ch in enumerate(channel_cols):
        with cols[i % n_cols]:
            with st.expander(f"🔧 {ch}", expanded=False):
                lo = st.slider(
                    "Min share", 0.0, 1.0,
                    cfg.optimization.min_channel_share, 0.01,
                    key=f"opt_lo_{ch}",
                )
                hi = st.slider(
                    "Max share", 0.0, 1.0,
                    cfg.optimization.max_channel_share, 0.01,
                    key=f"opt_hi_{ch}",
                )
                overrides[ch] = (lo, hi)

    if st.button("⚡ Optimise Budget", type="primary", use_container_width=True):
        _run_optimizer(mmm, model_df, total_budget, channel_cols, overrides)


def _run_optimizer(mmm, model_df, total_budget, channel_cols, overrides):
    with st.spinner("Running optimisation…"):
        try:
            bounds = build_constraints(channel_cols, overrides)
        except ValueError as exc:
            st.error(f"❌ Constraint error: {exc}")
            return

        try:
            allocation = run_optimization(mmm, total_budget, bounds)
        except RuntimeError as exc:
            st.error(f"❌ Optimisation failed: {exc}")
            log.error("Optimisation error: %s", exc)
            return

    st.subheader("📊 Optimised Allocation")
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.dataframe(
            allocation.style.format({
                "optimized_spend": "{:,.0f}",
                "share":           "{:.1%}",
            }),
            use_container_width=True,
        )
    with col_right:
        st.bar_chart(
            allocation.set_index("channel")["optimized_spend"],
            use_container_width=True,
        )

    # Uncertainty for the optimized plan
    st.subheader("🎯 Optimized Plan Risk")
    with st.spinner("Estimating risk…"):
        try:
            X = _build_optimized_plan_frame(model_df, allocation, channel_cols)
            preds = predict_array(mmm, X)

            risk = build_risk_report(preds)
            summary = risk["summary"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean",     f"{summary['mean']:,.0f}")
            c2.metric("P5",       f"{summary['p5']:,.0f}")
            c3.metric("P95",      f"{summary['p95']:,.0f}")
            c4.metric("VaR (5%)", f"{risk['var_5pct']:,.0f}")

        except Exception as exc:
            st.warning(f"Risk computation unavailable: {exc}")
            risk = {}

    # Export
    st.subheader("📥 Export Results")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💾 Export JSON", use_container_width=True):
            try:
                path = export_json({"allocation": allocation, "risk": risk})
                st.success(f"Saved → `{path}`")
            except Exception as exc:
                st.error(f"Export failed: {exc}")
    with col_b:
        if st.button("📄 Export CSV", use_container_width=True):
            try:
                path = export_csv(allocation, "allocation.csv")
                st.success(f"Saved → `{path}`")
            except Exception as exc:
                st.error(f"Export failed: {exc}")


def _build_optimized_plan_frame(model_df, allocation, channel_cols):
    """Create a one-period future frame from the optimized allocation."""
    last_row = model_df.iloc[-1].copy()
    last_row["date"] = pd.to_datetime(last_row["date"]) + pd.DateOffset(months=1)

    allocation_map = allocation.set_index("channel")["optimized_spend"].to_dict()
    for ch in channel_cols:
        last_row[ch] = float(allocation_map.get(ch, 0.0))

    return pd.DataFrame([last_row])[["date"] + channel_cols]
