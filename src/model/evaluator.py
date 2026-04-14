"""
Model evaluation — compatible with pymc-marketing >= 0.19.
R², MAPE, RMSE, WAIC, LOO-CV.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import get_settings
from utils import get_logger, posterior_mean, predict_array

log = get_logger(__name__)
cfg_data = get_settings().data


def compute_waic(idata) -> Dict[str, float]:
    try:
        import arviz as az
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
    try:
        import arviz as az
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
    mmm,
    df: pd.DataFrame,
    channel_cols: List[str],
) -> Dict[str, float]:
    """
    R², MAPE, RMSE on training data using mmm.predict().

    predict() takes X and returns posterior predictions; we extract a mean
    prediction vector that matches the training periods.
    """
    target_col = cfg_data.target_col
    try:
        X = df[["date"] + channel_cols].copy()
        y_true = df[target_col].values.astype(float)

        pred_arr = predict_array(mmm, X)
        y_pred = posterior_mean(pred_arr)

        # Align lengths (predict may return all rows incl. future)
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not valid_mask.any():
            raise ValueError("Model predictions contain no finite values for evaluation.")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        metrics = {"r2": r2, "mape": mape, "rmse": rmse}
        log.info("In-sample metrics: %s", metrics)
        return metrics

    except Exception as exc:
        log.warning("In-sample metrics failed: %s", exc)
        return {}


def full_evaluation_report(
    mmm,
    idata,
    df: pd.DataFrame,
    channel_cols: List[str],
) -> Dict[str, Any]:
    return {
        "in_sample": compute_in_sample_metrics(mmm, df, channel_cols),
        "waic": compute_waic(idata),
        "loo": compute_loo(idata),
    }
