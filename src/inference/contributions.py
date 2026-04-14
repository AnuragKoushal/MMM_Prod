"""
Channel contribution extraction and visualisation — pymc-marketing >= 0.19.
"""
from __future__ import annotations

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_logger

log = get_logger(__name__)


def get_channel_contributions(mmm, df: pd.DataFrame, channel_cols: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame of posterior-mean channel contributions per time period.
    Rows = dates, Columns = channels.

    Uses the installed MMM API first, then falls back to a forward pass.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            contrib_df = mmm.compute_mean_contributions_over_time()
        result = _normalise_contribution_frame(contrib_df, df, channel_cols)
        log.info("Channel contributions computed. Shape: %s", result.shape)
        return result
    except Exception as exc:
        log.warning("compute_mean_contributions_over_time failed (%s), using forward pass.", exc)
        return _forward_pass_contributions(mmm, df, channel_cols)


def _normalise_contribution_frame(
    contrib_df: pd.DataFrame,
    df: pd.DataFrame,
    channel_cols: list[str],
) -> pd.DataFrame:
    """Convert contribution output to a predictable date + channel DataFrame."""
    result = pd.DataFrame(contrib_df).copy()

    if "date" not in result.columns:
        result.insert(0, "date", pd.to_datetime(df["date"]).reset_index(drop=True))
    else:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")

    for ch in channel_cols:
        if ch not in result.columns:
            result[ch] = 0.0

    ordered = result[["date"] + channel_cols].copy()
    ordered[channel_cols] = ordered[channel_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return ordered


def _forward_pass_contributions(mmm, df: pd.DataFrame, channel_cols: list[str]) -> pd.DataFrame:
    """Fallback for environments where contribution summaries are unavailable."""
    try:
        channel_data = df[channel_cols].to_numpy(dtype="float64", copy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mmm.channel_contribution_forward_pass(channel_data)
        mean_arr = np.array(raw)
        if mean_arr.ndim >= 4:
            mean_arr = mean_arr.mean(axis=(0, 1))
        elif mean_arr.ndim == 3:
            mean_arr = mean_arr.mean(axis=0)

        result = pd.DataFrame(mean_arr, columns=channel_cols)
        result.insert(0, "date", pd.to_datetime(df["date"]).reset_index(drop=True))
        return result
    except Exception as exc2:
        log.error("Forward pass contributions also failed: %s", exc2)
        fallback = pd.DataFrame(np.zeros((len(df), len(channel_cols))), columns=channel_cols)
        fallback.insert(0, "date", pd.to_datetime(df["date"]).reset_index(drop=True))
        return fallback


def plot_channel_contributions(
    contrib_df: pd.DataFrame,
    title: str = "Channel Contributions (Posterior Mean)",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart of total posterior-mean contribution per channel."""
    value_cols = [c for c in contrib_df.columns if c != "date"]
    totals = contrib_df[value_cols].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    totals.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Contribution")
    ax.set_xlabel("Channel")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_contribution_over_time(
    contrib_df: pd.DataFrame,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Stacked area chart of channel contributions over time."""
    plot_df = contrib_df.copy()
    value_cols = [c for c in plot_df.columns if c != "date"]
    if "date" in plot_df.columns:
        plot_df = plot_df.set_index("date")

    fig, ax = plt.subplots(figsize=figsize)
    plot_df[value_cols].plot.area(ax=ax, alpha=0.75, linewidth=0)
    ax.set_title("Channel Contributions Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contribution")
    ax.set_xlabel("Period")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig
