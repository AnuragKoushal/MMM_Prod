"""
Input data validation.
Raises descriptive ValueError / TypeError so the UI can surface clean messages.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings().data


class DataValidationError(ValueError):
    """Raised when the uploaded DataFrame fails validation."""


def validate_raw(df: pd.DataFrame) -> None:
    """Validate that required columns exist and basic sanity checks pass."""
    if df.empty:
        raise DataValidationError("Uploaded file is empty.")

    required = {cfg.date_col, cfg.spend_col, cfg.target_col, cfg.channel_col}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    # Spend / target must be numeric
    for col in [cfg.spend_col, cfg.target_col]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"Column '{col}' must be numeric. Got dtype: {df[col].dtype}"
            )

    if df[cfg.spend_col].lt(0).any():
        raise DataValidationError(f"Column '{cfg.spend_col}' contains negative values.")

    if df[cfg.target_col].lt(0).any():
        raise DataValidationError(f"Column '{cfg.target_col}' contains negative values.")

    null_counts = df[list(required)].isnull().sum()
    if null_counts.any():
        raise DataValidationError(
            f"Null values found:\n{null_counts[null_counts > 0].to_dict()}"
        )

    log.info("Raw data validation passed (%d rows, %d cols)", *df.shape)


def validate_model_df(df: pd.DataFrame, channel_cols: List[str]) -> None:
    """Validate the processed model-ready DataFrame."""
    if df.empty:
        raise DataValidationError("Model DataFrame is empty after preprocessing.")

    if "date" not in df.columns:
        raise DataValidationError("Model DataFrame is missing 'date' column.")

    if cfg.target_col not in df.columns:
        raise DataValidationError(
            f"Model DataFrame is missing target column '{cfg.target_col}'."
        )

    missing_channels = set(channel_cols) - set(df.columns)
    if missing_channels:
        raise DataValidationError(
            f"Channel columns missing from model DataFrame: {missing_channels}"
        )

    if df[channel_cols].lt(0).any().any():
        raise DataValidationError("Channel spend columns contain negative values.")

    log.info(
        "Model DataFrame validation passed (%d rows, %d channels)",
        len(df),
        len(channel_cols),
    )
