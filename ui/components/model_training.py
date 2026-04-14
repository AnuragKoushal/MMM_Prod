"""
Model training component — passes MCMC overrides directly to train_mmm()
instead of mutating the cached Settings singleton.
"""
from __future__ import annotations

import streamlit as st

from src.model import train_mmm, full_evaluation_report
from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings()


def render_model_training() -> None:
    st.header("🚀 Model Training")

    model_df     = st.session_state.get("model_df")
    channel_cols = st.session_state.get("channel_cols")

    if model_df is None or channel_cols is None:
        st.warning("⚠️ Please upload and process data first (Data tab).")
        return

    st.info(
        f"**Ready to train** on **{len(model_df)} time periods** "
        f"and **{len(channel_cols)} channels**: `{', '.join(channel_cols)}`"
    )

    # Sidebar values
    draws       = int(st.session_state.get("mcmc_draws",   cfg.model.draws))
    chains      = int(st.session_state.get("mcmc_chains",  cfg.model.chains))
    adstock_lag = int(st.session_state.get("adstock_lag",  cfg.model.adstock_max_lag))

    with st.expander("🛠️ Active Training Configuration"):
        c1, c2, c3 = st.columns(3)
        c1.metric("MCMC Draws",  draws)
        c2.metric("MCMC Chains", chains)
        c3.metric("Adstock Lag", adstock_lag)

    st.caption(
        "💡 Tip: Use Draws = 200 for a quick smoke-test; "
        "use 1000+ for production-quality results."
    )

    # 🚀 TRAIN BUTTON
    if st.button("🚀 Train MMM", type="primary", use_container_width=True):
        _run_training(model_df, channel_cols, draws, chains, adstock_lag)

    # ✅ Show result if already trained
    if (
        "mmm" in st.session_state
        and st.session_state["mmm"] is not None
        and "eval_report" in st.session_state
    ):
        st.success("✅ Model already trained")
        _render_eval_report(st.session_state["eval_report"])

    if st.session_state.get("training_just_completed"):
        st.info("The model finished training successfully. You can review metrics here or move to Insights, Scenarios, or Optimise.")
        st.session_state["training_just_completed"] = False


def _run_training(model_df, channel_cols, draws, chains, adstock_lag):
    progress = st.progress(0, text="Initialising model…")
    status   = st.empty()

    try:
        progress.progress(15, text="Building MMM model…")

        mmm, idata = train_mmm(
            model_df,
            channel_cols,
            draws=draws,
            chains=chains,
            adstock_max_lag=adstock_lag,
        )

        progress.progress(80, text="Computing evaluation metrics…")
        status.info("Computing model metrics…")

        eval_report = full_evaluation_report(mmm, idata, model_df, channel_cols)

        progress.progress(100, text="Done!")
        status.empty()

        # ✅ SAVE TO SESSION STATE (CRITICAL)
        st.session_state["mmm"]         = mmm
        st.session_state["idata"]       = idata
        st.session_state["eval_report"] = eval_report
        st.session_state["data_locked"] = True
        st.session_state["training_just_completed"] = True

        st.success("✅ Model trained successfully!")

    except RuntimeError as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Training failed: {exc}")
        log.error("Training error: %s", exc)

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Unexpected error: {exc}")
        log.exception(exc)


def _render_eval_report(report: dict) -> None:
    st.subheader("📋 Model Evaluation")

    in_sample = report.get("in_sample", {})
    waic      = report.get("waic", {})
    loo       = report.get("loo", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("R²",   f"{in_sample.get('r2',   float('nan')):.3f}")
    c2.metric("MAPE", f"{in_sample.get('mape', float('nan')):.1%}")
    c3.metric("RMSE", f"{in_sample.get('rmse', float('nan')):,.0f}")

    if waic:
        c4, c5 = st.columns(2)
        c4.metric("WAIC",     f"{waic.get('waic',    float('nan')):.1f}")
        c5.metric("LOO ELPD", f"{loo.get('elpd_loo', float('nan')):.1f}")
