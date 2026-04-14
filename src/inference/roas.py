"""
ROAS computation — pymc-marketing >= 0.19.
"""
from __future__ import annotations

import pandas as pd

from src.inference.contributions import get_channel_contributions
from utils import get_logger, safe_divide

log = get_logger(__name__)


def compute_marginal_roas(mmm, df: pd.DataFrame, channel_cols: list[str]) -> pd.Series:
    """Posterior-mean total contribution per channel as a proxy for marginal ROAS."""
    contrib_df = get_channel_contributions(mmm, df, channel_cols)
    return contrib_df[channel_cols].sum().rename("marginal_roas_proxy")


def compute_roas(
    mmm,
    df: pd.DataFrame,
    channel_cols: list[str],
) -> pd.DataFrame:
    """
    ROAS table: total_contribution / total_spend per channel.
    Returns DataFrame with [channel, total_spend, total_contribution, roas].
    """
    contrib_df = get_channel_contributions(mmm, df, channel_cols)

    records = []
    for ch in channel_cols:
        total_spend = float(df[ch].sum()) if ch in df.columns else 0.0
        total_contribution = float(contrib_df[ch].sum()) if ch in contrib_df.columns else 0.0
        roas = safe_divide(total_contribution, total_spend)
        records.append({
            "channel": ch,
            "total_spend": total_spend,
            "total_contribution": total_contribution,
            "roas": roas,
        })

    result = (
        pd.DataFrame(records)
        .sort_values("roas", ascending=False)
        .reset_index(drop=True)
    )
    log.info("ROAS table computed for %d channels.", len(result))
    return result
