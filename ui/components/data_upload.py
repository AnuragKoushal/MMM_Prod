"""
Data upload and exploration component.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data import load_file, prepare_data, DataValidationError
from utils import clear_model_artifacts, get_logger

log = get_logger(__name__)


def render_data_upload() -> None:
    """Render the data upload section and populate session_state."""
    st.header("📂 Data Upload & Preview")

    # 🔥 Initialize flag
    if "data_locked" not in st.session_state:
        st.session_state["data_locked"] = False

    uploaded = st.file_uploader(
        "Upload your media spend CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Required columns: date, channel, spend, and target (clicks/revenue).",
    )

    if uploaded is None:
        st.info(
            "👆 Upload a CSV or Excel file to begin. "
            "Expected columns: `Cal_Month`, `site`, `media_spend`, `clicks`."
        )
        return

    try:
        with st.spinner("Loading file…"):
            raw_df = load_file(uploaded)
    except Exception as exc:
        st.error(f"❌ Failed to load file: {exc}")
        log.error("File load error: %s", exc)
        return

    # --- Raw data preview ---
    st.subheader("Raw Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(raw_df):,}")
    col2.metric("Columns", f"{len(raw_df.columns):,}")
    col3.metric("Missing Values", f"{raw_df.isnull().sum().sum():,}")

    with st.expander("Preview (first 20 rows)", expanded=True):
        st.dataframe(raw_df.head(20), use_container_width=True)

    with st.expander("Column Summary"):
        st.dataframe(raw_df.describe(include="all").T, use_container_width=True)

    # --- Preprocessing ---
    try:
        with st.spinner("Preprocessing…"):
            model_df, channel_cols = prepare_data(raw_df)
    except DataValidationError as exc:
        st.error(f"❌ Validation error: {exc}")
        return
    except Exception as exc:
        st.error(f"❌ Preprocessing failed: {exc}")
        log.exception("Preprocessing error: %s", exc)
        return

    st.subheader("Model-Ready Data (Wide Format)")
    col1, col2 = st.columns(2)
    col1.metric("Time Periods", f"{len(model_df):,}")
    col2.metric("Media Channels", f"{len(channel_cols):,}")
    st.dataframe(model_df.head(20), use_container_width=True)

    _sync_uploaded_data(raw_df, model_df, channel_cols)

    st.success(f"✅ Data ready: {len(model_df)} rows × {len(channel_cols)} channels")

    st.caption("Use `Reset App` in the header or sidebar if you want to clear the current upload and trained model.")


def _sync_uploaded_data(raw_df, model_df, channel_cols) -> None:
    """
    Keep uploaded data and trained model state consistent.

    If the uploaded dataset changes after training, invalidate model artifacts so
    downstream tabs cannot use a stale model with new data.
    """
    previous_model_df = st.session_state.get("model_df")
    previous_channels = st.session_state.get("channel_cols")
    had_model = st.session_state.get("mmm") is not None

    data_changed = False
    if previous_model_df is None or previous_channels is None:
        data_changed = True
    else:
        same_channels = list(previous_channels) == list(channel_cols)
        same_data = previous_model_df.equals(model_df)
        data_changed = not (same_channels and same_data)

    if data_changed and had_model:
        clear_model_artifacts()
        st.warning("Uploaded data changed, so the previous trained model was cleared. Retrain to refresh Insights, Scenarios, and Optimise.")

    st.session_state["raw_df"] = raw_df
    st.session_state["model_df"] = model_df
    st.session_state["channel_cols"] = channel_cols
