"""
Helpers for consistent Streamlit session-state management.
"""
from __future__ import annotations

import streamlit as st


def clear_model_artifacts() -> None:
    """Remove trained-model artifacts while keeping uploaded data intact."""
    st.session_state["mmm"] = None
    st.session_state["idata"] = None
    st.session_state["eval_report"] = None
    st.session_state["data_locked"] = False
    st.session_state["training_just_completed"] = False


def reset_app_state() -> None:
    """Clear the whole session and rerun the app."""
    st.session_state.clear()
    st.rerun()
