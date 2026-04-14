"""
Scenario planner — compatible with pymc-marketing >= 0.19.
mmm.predict() now takes X (date + channel cols only, no target).
"""
from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM

from config import get_settings
from utils import get_logger, posterior_mean, predict_array, prediction_interval, timer

log = get_logger(__name__)
cfg_data = get_settings().data


def generate_scenario(
    df: pd.DataFrame,
    multipliers: Dict[str, float],
    periods: int = 3,
) -> pd.DataFrame:
    """
    Extend the historical DataFrame with *periods* future rows.

    Args:
        df: Historical wide-format DataFrame (must contain 'date' column).
        multipliers: {channel_col: multiplier}. Channels not listed are copied.
        periods: Number of future monthly periods to simulate.

    Returns:
        DataFrame containing history + simulated future rows.
    """
    if df.empty:
        raise ValueError("Cannot generate scenario from empty DataFrame.")
    if periods < 1:
        raise ValueError(f"periods must be ≥ 1, got {periods}.")

    scenario_df = df.copy()

    for _ in range(periods):
        row = scenario_df.iloc[-1].copy()
        row["date"] = row["date"] + pd.DateOffset(months=1)
        if cfg_data.target_col in row.index:
            row[cfg_data.target_col] = np.nan

        for ch, mult in multipliers.items():
            if ch in row.index:
                row[ch] = max(0.0, float(row[ch]) * mult)
            else:
                log.warning("Channel '%s' not in DataFrame; skipping.", ch)

        scenario_df = pd.concat(
            [scenario_df, pd.DataFrame([row])], ignore_index=True
        )

    log.info("Generated %d-period scenario. Shape: %s", periods, scenario_df.shape)
    return scenario_df


def simulate_scenario(
    mmm: "MMM",
    scenario_df: pd.DataFrame,
    channel_cols: List[str],
) -> np.ndarray:
    """
    Run posterior prediction on the scenario DataFrame.

    Args:
        mmm: Fitted MMM instance.
        scenario_df: Output of generate_scenario().
        channel_cols: Channel column names (used to build X).

    Returns:
        Raw prediction array from the MMM model.
    """
    X = scenario_df[["date"] + channel_cols].copy()

    with timer("Scenario simulation"):
        pred_arr = predict_array(mmm, X)
    log.info("Scenario simulation complete. pred shape: %s", pred_arr.shape)
    return pred_arr


def build_scenario_summary(
    scenario_df: pd.DataFrame,
    preds: np.ndarray,
    target_col: str,
    channel_cols: List[str],
) -> pd.DataFrame:
    """
    Align predictions with the scenario DataFrame.

    Returns DataFrame with date, channel spends, predicted_mean, p5, p95,
    and actual target (where available).
    """
    out = scenario_df[["date"] + channel_cols].copy().reset_index(drop=True)

    arr = np.asarray(preds)
    mean_1d = posterior_mean(arr)
    p5_1d, p95_1d = prediction_interval(arr, 5, 95)

    n = min(len(out), len(mean_1d))
    out["predicted_mean"] = np.nan
    out["predicted_p5"]   = np.nan
    out["predicted_p95"]  = np.nan
    out.loc[:n-1, "predicted_mean"] = mean_1d[:n]
    out.loc[:n-1, "predicted_p5"]   = p5_1d[:n]
    out.loc[:n-1, "predicted_p95"]  = p95_1d[:n]

    if target_col in scenario_df.columns:
        actual_values = pd.to_numeric(scenario_df[target_col], errors="coerce").values[:len(out)]
        out["actual"] = actual_values

    return out
