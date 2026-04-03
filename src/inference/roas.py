"""
Return on Ad Spend (ROAS) and Marginal ROAS computation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pymc_marketing.mmm import MMM

from utils import get_logger, safe_divide

log = get_logger(__name__)


def compute_marginal_roas(mmm: MMM) -> pd.Series:
    """
    Approximate marginal ROAS as mean posterior contribution per channel.

    Returns:
        pd.Series indexed by channel name.
    """
    contrib = mmm.compute_channel_contribution()
    mean_contrib = contrib.mean(dim=["chain", "draw"])
    series = mean_contrib.to_pandas()
    if isinstance(series, pd.DataFrame):
        series = series.sum()
    log.info("Marginal ROAS computed for %d channels.", len(series))
    return series


def compute_roas(
    mmm: MMM,
    df: pd.DataFrame,
    channel_cols: list[str],
) -> pd.DataFrame:
    """
    Compute overall ROAS: total contribution / total spend per channel.

    Args:
        mmm: Fitted MMM.
        df: Model DataFrame containing channel spend columns.
        channel_cols: List of channel column names.

    Returns:
        DataFrame with columns [channel, total_spend, total_contribution, roas].
    """
    contrib = mmm.compute_channel_contribution()
    mean_contrib_by_channel = (
        contrib.mean(dim=["chain", "draw"]).to_pandas()
    )
    if isinstance(mean_contrib_by_channel, pd.Series):
        mean_contrib_by_channel = mean_contrib_by_channel.to_frame().T

    records = []
    for ch in channel_cols:
        total_spend = float(df[ch].sum()) if ch in df.columns else 0.0
        total_contribution = (
            float(mean_contrib_by_channel[ch].sum())
            if ch in mean_contrib_by_channel.columns
            else 0.0
        )
        roas = safe_divide(total_contribution, total_spend)
        records.append(
            {
                "channel": ch,
                "total_spend": total_spend,
                "total_contribution": total_contribution,
                "roas": roas,
            }
        )

    result = pd.DataFrame(records).sort_values("roas", ascending=False).reset_index(drop=True)
    log.info("ROAS table:\n%s", result.to_string(index=False))
    return result
