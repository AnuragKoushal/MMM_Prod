"""
Scenario planner: generate and simulate multi-period what-if scenarios.
"""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM

from utils import get_logger, timer

log = get_logger(__name__)


def generate_scenario(
    df: pd.DataFrame,
    multipliers: Dict[str, float],
    periods: int = 3,
) -> pd.DataFrame:
    """
    Extend the historical DataFrame with *periods* future rows, each
    modifying channel spends by the supplied multipliers.

    Args:
        df: Historical model DataFrame (wide format, must include 'date' col).
        multipliers: Dict of {channel_col: multiplier}. Channels not listed
                     are copied unchanged.
        periods: Number of future periods (months) to simulate.

    Returns:
        DataFrame containing history + simulated future rows.

    Raises:
        ValueError: If *periods* < 1 or df is empty.
    """
    if df.empty:
        raise ValueError("Cannot generate scenario from empty DataFrame.")
    if periods < 1:
        raise ValueError(f"periods must be ≥ 1, got {periods}.")

    scenario_df = df.copy()

    for i in range(1, periods + 1):
        row = scenario_df.iloc[-1].copy()
        row["date"] = row["date"] + pd.DateOffset(months=1)

        for ch, mult in multipliers.items():
            if ch in row.index:
                row[ch] = max(0.0, row[ch] * mult)
            else:
                log.warning("Channel '%s' not found in DataFrame; skipping.", ch)

        scenario_df = pd.concat(
            [scenario_df, pd.DataFrame([row])], ignore_index=True
        )

    log.info(
        "Generated %d-period scenario. New shape: %s",
        periods, scenario_df.shape,
    )
    return scenario_df


def simulate_scenario(mmm: "MMM", scenario_df: pd.DataFrame):
    """
    Run posterior prediction on the scenario DataFrame.

    Args:
        mmm: Fitted MMM instance.
        scenario_df: Scenario DataFrame produced by generate_scenario().

    Returns:
        Raw prediction array (shape: chains × draws × time).
    """
    with timer("Scenario simulation"):
        preds = mmm.predict(scenario_df)
    log.info("Scenario simulation complete.")
    return preds


def build_scenario_summary(
    scenario_df: pd.DataFrame,
    preds,
    target_col: str,
    channel_cols: List[str],
) -> pd.DataFrame:
    """
    Align predictions with scenario rows and return a tidy summary DataFrame.

    Returns:
        DataFrame with date, channel spends, actual/predicted target columns.
    """
    import numpy as np

    pred_mean = np.mean(preds, axis=(0, 1))  # mean over chains and draws
    pred_p5 = np.percentile(preds, 5, axis=(0, 1))
    pred_p95 = np.percentile(preds, 95, axis=(0, 1))

    out = scenario_df[["date"] + channel_cols].copy()
    out["predicted_mean"] = pred_mean
    out["predicted_p5"] = pred_p5
    out["predicted_p95"] = pred_p95

    if target_col in scenario_df.columns:
        out["actual"] = scenario_df[target_col].values

    return out
