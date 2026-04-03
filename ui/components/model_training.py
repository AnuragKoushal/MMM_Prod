"""
Model training component – triggers MMM fitting and persists results.
"""
from __future__ import annotations

import streamlit as st

from src.model import train_mmm, full_evaluation_report
from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings()


def render_model_training() -> None:
    """Render model training controls and display training metrics."""
    st.header("🚀 Model Training")

    model_df = st.session_state.get("model_df")
    channel_cols = st.session_state.get("channel_cols")

    if model_df is None or channel_cols is None:
        st.warning("⚠️ Please upload and process data first.")
        return

    # Summary of what will be trained on
    st.info(
        f"**Ready to train** on **{len(model_df)} time periods** "
        f"and **{len(channel_cols)} channels**: `{', '.join(channel_cols)}`"
    )

    # Pull sidebar settings (set by sidebar component)
    draws = st.session_state.get("mcmc_draws", cfg.model.draws)
    chains = st.session_state.get("mcmc_chains", cfg.model.chains)
    adstock_lag = st.session_state.get("adstock_lag", cfg.model.adstock_max_lag)

    with st.expander("🛠️ Active Training Configuration"):
        col1, col2, col3 = st.columns(3)
        col1.metric("MCMC Draws", draws)
        col2.metric("MCMC Chains", chains)
        col3.metric("Adstock Lag", adstock_lag)

    if st.button("🚀 Train MMM", type="primary", use_container_width=True):
        _run_training(model_df, channel_cols, draws, chains, adstock_lag)

    # Show evaluation if model already trained
    if st.session_state.get("mmm") and st.session_state.get("eval_report"):
        _render_eval_report(st.session_state["eval_report"])


def _run_training(model_df, channel_cols, draws, chains, adstock_lag):
    """Execute model training with a live progress placeholder."""
    cfg_override = get_settings()
    cfg_override.model.draws = draws
    cfg_override.model.chains = chains
    cfg_override.model.adstock_max_lag = adstock_lag

    progress = st.progress(0, text="Initialising model…")
    try:
        progress.progress(20, text="Building PyMC graph…")
        mmm, idata = train_mmm(model_df, channel_cols)

        progress.progress(80, text="Computing evaluation metrics…")
        eval_report = full_evaluation_report(
            mmm, idata, model_df, cfg_override.data.target_col
        )

        progress.progress(100, text="Done!")

        # Persist
        st.session_state["mmm"] = mmm
        st.session_state["idata"] = idata
        st.session_state["eval_report"] = eval_report

        st.success("✅ Model trained successfully!")
        _render_eval_report(eval_report)

    except RuntimeError as exc:
        progress.empty()
        st.error(f"❌ Training failed: {exc}")
        log.error("Training error: %s", exc)


def _render_eval_report(report: dict) -> None:
    st.subheader("📋 Model Evaluation")
    in_sample = report.get("in_sample", {})
    waic = report.get("waic", {})
    loo = report.get("loo", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{in_sample.get('r2', 0):.3f}")
    col2.metric("MAPE", f"{in_sample.get('mape', 0):.1%}")
    col3.metric("RMSE", f"{in_sample.get('rmse', 0):,.0f}")

    if waic:
        col4, col5 = st.columns(2)
        col4.metric("WAIC", f"{waic.get('waic', 0):.1f}")
        col5.metric("LOO ELPD", f"{loo.get('elpd_loo', 0):.1f}")
