"""
Posterior diagnostics: trace plots, response curves, predictive checks — v0.19+.
"""
from __future__ import annotations

import warnings
import matplotlib.pyplot as plt
import arviz as az
import numpy as np

from utils import get_logger

log = get_logger(__name__)


def plot_posterior_diagnostics(idata, var_names=None) -> plt.Figure:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axes = az.plot_trace(idata, var_names=var_names, compact=True)
        fig = axes.ravel()[0].get_figure()
        fig.suptitle("Posterior Trace Plots", fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig
    except Exception as exc:
        log.warning("Trace plot failed: %s", exc)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Trace plot unavailable:\n{exc}", ha="center", va="center")
        return fig


def plot_response_curves(mmm, df=None, channel_cols=None, figsize=(12, 6)) -> plt.Figure:
    """Plot response curves using the installed MMM plotting API."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = mmm.plot_direct_contribution_curves()
        if fig is None:
            raise ValueError("plot_direct_contribution_curves() returned None")
        fig.set_size_inches(*figsize)
        fig.tight_layout()
        return fig
    except Exception as exc:
        log.warning("plot_direct_contribution_curves failed (%s), trying grid.", exc)
        try:
            channels = channel_cols or list(getattr(mmm, "channel_columns", []) or [])
            if not channels:
                raise ValueError("No channel metadata available for response curve plotting.")

            if df is not None:
                channel_frame = df[channels].astype(float)
                values = channel_frame.to_numpy().ravel()
                values = values[np.isfinite(values) & (values >= 0)]
                stop = float(values.max()) if values.size else 1.0
            else:
                stop = 1.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = mmm.plot_channel_contribution_grid(start=0.0, stop=max(stop, 1.0), num=50)
            fig.set_size_inches(*figsize)
            fig.tight_layout()
            return fig
        except Exception as exc2:
            log.warning("Response curve plot also failed: %s", exc2)
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Response curves unavailable:\n{exc2}", ha="center", va="center")
            return fig


def plot_posterior_predictive_check(mmm, idata, target_col=None, df=None, channel_cols=None, figsize=(12, 4)) -> plt.Figure:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axes = az.plot_ppc(idata, observed_rug=True)
        fig = axes.ravel()[0].get_figure()
        fig.suptitle("Posterior Predictive Check", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig
    except Exception as exc:
        log.warning("PPC plot failed: %s", exc)
        try:
            if df is not None and channel_cols:
                X = df[["date"] + channel_cols].copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mmm.sample_posterior_predictive(
                        X=X,
                        extend_idata=True,
                        combined=False,
                        include_last_observations=True,
                        original_scale=True,
                    )
                refreshed_idata = getattr(mmm, "idata", None)
                if refreshed_idata is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        axes = az.plot_ppc(refreshed_idata, observed_rug=True)
                    fig = axes.ravel()[0].get_figure()
                    fig.suptitle("Posterior Predictive Check", fontsize=13, fontweight="bold")
                    fig.tight_layout()
                    return fig

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = mmm.plot_posterior_predictive()
            return fig
        except Exception as exc2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"PPC unavailable:\n{exc2}", ha="center", va="center")
            return fig
