"""
Posterior diagnostics: trace plots, response curves, predictive checks.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import arviz as az
from pymc_marketing.mmm import MMM

from utils import get_logger

log = get_logger(__name__)


def plot_posterior_diagnostics(idata, var_names: list[str] | None = None) -> plt.Figure:
    """
    Render ArviZ trace plots for key model parameters.

    Args:
        idata: ArviZ InferenceData from MMM.fit().
        var_names: Optional list of variable names to trace. Defaults to all.

    Returns:
        matplotlib Figure.
    """
    try:
        axes = az.plot_trace(idata, var_names=var_names, compact=True)
        fig = axes.ravel()[0].get_figure()
        fig.suptitle("Posterior Trace Plots", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig
    except Exception as exc:
        log.warning("Trace plot failed: %s", exc)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Trace plot unavailable:\n{exc}", ha="center", va="center")
        return fig


def plot_response_curves(mmm: MMM, figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot saturation / response curves for each media channel.

    Returns:
        matplotlib Figure produced by pymc-marketing.
    """
    try:
        fig = mmm.plot_response_curves()
        if fig is None:
            raise ValueError("plot_response_curves() returned None")
        fig.set_size_inches(*figsize)
        fig.tight_layout()
        return fig
    except Exception as exc:
        log.warning("Response curve plot failed: %s", exc)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Response curves unavailable:\n{exc}", ha="center", va="center")
        return fig


def plot_posterior_predictive_check(
    mmm: MMM,
    idata,
    target_col: str,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Overlay posterior predictive samples against observed target.

    Returns:
        matplotlib Figure.
    """
    try:
        axes = az.plot_ppc(idata, observed_rug=True)
        fig = axes.ravel()[0].get_figure()
        fig.suptitle("Posterior Predictive Check", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig
    except Exception as exc:
        log.warning("PPC plot failed: %s", exc)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"PPC unavailable:\n{exc}", ha="center", va="center")
        return fig
