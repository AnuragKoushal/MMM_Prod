"""
Channel contribution extraction and visualisation.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM

from utils import get_logger

log = get_logger(__name__)


def get_channel_contributions(mmm: MMM) -> xr.DataArray:
    """
    Compute per-channel posterior contributions.

    Returns:
        xr.DataArray with dims (chain, draw, date, channel).
    """
    contrib = mmm.compute_channel_contribution()
    log.info("Channel contributions computed. Shape: %s", dict(contrib.sizes))
    return contrib


def contributions_to_dataframe(contrib: xr.DataArray) -> pd.DataFrame:
    """
    Collapse posterior to a mean-per-channel DataFrame indexed by date.

    Returns:
        pd.DataFrame with one row per date, one column per channel.
    """
    mean_contrib = contrib.mean(dim=["chain", "draw"])
    df = mean_contrib.to_pandas()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def plot_channel_contributions(
    contrib: xr.DataArray,
    title: str = "Channel Contributions (Posterior Mean)",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart of mean posterior channel contributions."""
    df = contributions_to_dataframe(contrib)
    channel_totals = df.sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    channel_totals.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Contribution")
    ax.set_xlabel("Channel")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_contribution_over_time(
    contrib: xr.DataArray,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Stacked area chart of channel contributions over time."""
    df = contributions_to_dataframe(contrib)

    fig, ax = plt.subplots(figsize=figsize)
    df.plot.area(ax=ax, alpha=0.75, linewidth=0)
    ax.set_title("Channel Contributions Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contribution")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig
