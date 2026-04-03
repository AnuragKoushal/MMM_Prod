"""
Uncertainty quantification and risk metrics for MMM predictions.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_logger

log = get_logger(__name__)


def summarize_with_uncertainty(
    predictions,
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> Dict[str, float]:
    """
    Compute summary statistics over posterior prediction samples.

    Args:
        predictions: Array-like of shape (chains, draws, time) or flat array.
        percentiles: Which percentile bands to report.

    Returns:
        Dict with keys: mean, std, and p{x} for each requested percentile.
    """
    arr = np.asarray(predictions).ravel()

    summary: Dict[str, float] = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
    for p in percentiles:
        summary[f"p{p}"] = float(np.percentile(arr, p))

    log.info("Uncertainty summary: %s", {k: f"{v:.2f}" for k, v in summary.items()})
    return summary


def value_at_risk(
    predictions,
    confidence: float = 0.05,
) -> float:
    """
    Compute the Value-at-Risk at *confidence* level (lower tail).

    Args:
        predictions: Array of predictions.
        confidence: Left-tail probability (e.g. 0.05 → 5th percentile).

    Returns:
        VaR value (the threshold below which *confidence* of outcomes fall).
    """
    arr = np.asarray(predictions).ravel()
    var = float(np.percentile(arr, confidence * 100))
    log.info("VaR @ %.0f%%: %.4f", confidence * 100, var)
    return var


def conditional_value_at_risk(
    predictions,
    confidence: float = 0.05,
) -> float:
    """
    Expected Shortfall (CVaR): mean of outcomes below VaR.

    Args:
        predictions: Array of predictions.
        confidence: Left-tail probability.

    Returns:
        CVaR value.
    """
    arr = np.asarray(predictions).ravel()
    var = np.percentile(arr, confidence * 100)
    cvar = float(arr[arr <= var].mean()) if (arr <= var).any() else var
    log.info("CVaR @ %.0f%%: %.4f", confidence * 100, cvar)
    return cvar


def plot_prediction_distribution(
    predictions,
    title: str = "Posterior Prediction Distribution",
    figsize: tuple = (9, 4),
) -> plt.Figure:
    """Histogram with percentile bands overlay."""
    arr = np.asarray(predictions).ravel()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(arr, bins=50, color="steelblue", edgecolor="white", alpha=0.85)

    for p, color, label in [
        (5, "red", "P5"),
        (50, "orange", "Median"),
        (95, "green", "P95"),
    ]:
        val = np.percentile(arr, p)
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{label}: {val:,.0f}")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def build_risk_report(predictions) -> Dict[str, Any]:
    """Aggregate all risk metrics into a single report dict."""
    return {
        "summary": summarize_with_uncertainty(predictions),
        "var_5pct": value_at_risk(predictions, 0.05),
        "cvar_5pct": conditional_value_at_risk(predictions, 0.05),
    }
