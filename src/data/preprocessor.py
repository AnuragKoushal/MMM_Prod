"""
Data preprocessing: pivots long-format channel data into model-ready wide format.
"""
from __future__ import annotations

import pandas as pd

from config import get_settings
from utils import get_logger, timer
from src.data.validator import validate_raw, validate_model_df

log = get_logger(__name__)
cfg = get_settings().data


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw long-format DataFrame to wide model-ready format.

    Returns:
        model_df: Wide DataFrame with one row per date, channel spend as columns.
        channel_cols: List of channel column names.
    """
    with timer("Data preparation"):
        validate_raw(df)

        df = df.copy()
        df["date"] = pd.to_datetime(df[cfg.date_col])

        agg_df = (
            df.groupby(["date", cfg.channel_col])
            .agg({cfg.spend_col: "sum", cfg.target_col: "sum"})
            .reset_index()
        )

        pivot_spend = (
            agg_df.pivot(index="date", columns=cfg.channel_col, values=cfg.spend_col)
            .fillna(0)
        )
        pivot_spend.columns.name = None  # remove MultiIndex name

        target_series = agg_df.groupby("date")[cfg.target_col].sum()

        model_df = pivot_spend.copy()
        model_df[cfg.target_col] = target_series
        model_df = model_df.reset_index().sort_values("date").reset_index(drop=True)

        channel_cols = [
            col for col in model_df.columns
            if col not in ("date", cfg.target_col)
        ]

        validate_model_df(model_df, channel_cols)

        log.info(
            "Data prepared: %d rows, %d channels: %s",
            len(model_df), len(channel_cols), channel_cols,
        )

    return model_df, channel_cols
