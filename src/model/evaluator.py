"""
Model evaluation: WAIC, LOO-CV, R², MAPE, and posterior predictive checks.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
import arviz as az
from pymc_marketing.mmm import MMM

from utils import get_logger

log = get_logger(__name__)


def compute_waic(idata) -> Dict[str, float]:
    """Compute Widely Applicable Information Criterion."""
    try:
        waic = az.waic(idata)
        return {
            "waic": float(waic.waic),
            "waic_se": float(waic.waic_se),
            "p_waic": float(waic.p_waic),
        }
    except Exception as exc:
        log.warning("WAIC computation failed: %s", exc)
        return {}


def compute_loo(idata) -> Dict[str, float]:
    """Compute Leave-One-Out cross-validation score."""
    try:
        loo = az.loo(idata)
        return {
            "elpd_loo": float(loo.elpd_loo),
            "se": float(loo.se),
            "p_loo": float(loo.p_loo),
        }
    except Exception as exc:
        log.warning("LOO computation failed: %s", exc)
        return {}


def compute_in_sample_metrics(
    mmm: MMM, df: pd.DataFrame, target_col: str
) -> Dict[str, float]:
    """
    Compute R², MAPE, and RMSE on training data.

    Args:
        mmm: Fitted MMM instance.
        df: Training DataFrame.
        target_col: Name of the target column.

    Returns:
        Dict with r2, mape, rmse keys.
    """
    try:
        posterior_pred = mmm.predict(df)
        y_pred = np.mean(posterior_pred, axis=(0, 1))  # mean over chains/draws
        y_true = df[target_col].values

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        metrics = {"r2": float(r2), "mape": mape, "rmse": rmse}
        log.info("In-sample metrics: %s", metrics)
        return metrics

    except Exception as exc:
        log.warning("In-sample metrics computation failed: %s", exc)
        return {}


def full_evaluation_report(
    mmm: MMM, idata, df: pd.DataFrame, target_col: str
) -> Dict[str, Any]:
    """Return a combined evaluation report."""
    return {
        "in_sample": compute_in_sample_metrics(mmm, df, target_col),
        "waic": compute_waic(idata),
        "loo": compute_loo(idata),
    }
