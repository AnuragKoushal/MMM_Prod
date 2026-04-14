"""
Tests for src/data/preprocessor.py and src/data/validator.py
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.data.validator import validate_raw, validate_model_df, DataValidationError
from src.data.preprocessor import prepare_data


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_raw_df() -> pd.DataFrame:
    """Minimal valid long-format DataFrame matching default config columns."""
    dates = pd.date_range("2023-01-01", periods=6, freq="MS")
    channels = ["channel_a", "channel_b"]
    rows = []
    for d in dates:
        for ch in channels:
            rows.append(
                {
                    "Cal_Month": d.strftime("%Y-%m-%d"),
                    "site": ch,
                    "media_spend": np.random.uniform(1000, 5000),
                    "clicks": np.random.randint(100, 1000),
                }
            )
    return pd.DataFrame(rows)


# ── validate_raw ──────────────────────────────────────────────────────────────

class TestValidateRaw:
    def test_passes_on_valid_df(self, valid_raw_df):
        validate_raw(valid_raw_df)  # should not raise

    def test_raises_on_empty_df(self):
        with pytest.raises(DataValidationError, match="empty"):
            validate_raw(pd.DataFrame())

    def test_raises_on_missing_column(self, valid_raw_df):
        df = valid_raw_df.drop(columns=["media_spend"])
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_raw(df)

    def test_raises_on_negative_spend(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "media_spend"] = -100.0
        with pytest.raises(DataValidationError, match="negative"):
            validate_raw(df)

    def test_raises_on_negative_target(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "clicks"] = -1
        with pytest.raises(DataValidationError, match="negative"):
            validate_raw(df)

    def test_raises_on_null_values(self, valid_raw_df):
        df = valid_raw_df.copy()
        df.loc[0, "clicks"] = None
        with pytest.raises(DataValidationError, match="Null"):
            validate_raw(df)

    def test_raises_on_non_numeric_spend(self, valid_raw_df):
        df = valid_raw_df.copy()
        df["media_spend"] = "abc"
        with pytest.raises(DataValidationError, match="numeric"):
            validate_raw(df)


# ── prepare_data ──────────────────────────────────────────────────────────────

class TestPrepareData:
    def test_returns_wide_format(self, valid_raw_df):
        model_df, channel_cols = prepare_data(valid_raw_df)
        assert "date" in model_df.columns
        assert "clicks" in model_df.columns
        assert set(channel_cols) == {"channel_a", "channel_b"}

    def test_no_negative_spend_after_pivot(self, valid_raw_df):
        model_df, channel_cols = prepare_data(valid_raw_df)
        assert model_df[channel_cols].ge(0).all().all()

    def test_sorted_by_date(self, valid_raw_df):
        model_df, _ = prepare_data(valid_raw_df)
        assert model_df["date"].is_monotonic_increasing

    def test_no_duplicate_dates(self, valid_raw_df):
        model_df, _ = prepare_data(valid_raw_df)
        assert model_df["date"].nunique() == len(model_df)

    def test_channel_count(self, valid_raw_df):
        _, channel_cols = prepare_data(valid_raw_df)
        assert len(channel_cols) == 2

    def test_target_column_preserved(self, valid_raw_df):
        model_df, _ = prepare_data(valid_raw_df)
        assert model_df["clicks"].notna().all()


# ── validate_model_df ─────────────────────────────────────────────────────────

class TestValidateModelDf:
    def test_raises_on_missing_date(self, valid_raw_df):
        model_df, channel_cols = prepare_data(valid_raw_df)
        df = model_df.drop(columns=["date"])
        with pytest.raises(DataValidationError, match="date"):
            validate_model_df(df, channel_cols)

    def test_raises_on_missing_target(self, valid_raw_df):
        model_df, channel_cols = prepare_data(valid_raw_df)
        df = model_df.drop(columns=["clicks"])
        with pytest.raises(DataValidationError, match="target"):
            validate_model_df(df, channel_cols)

    def test_raises_on_missing_channel(self, valid_raw_df):
        model_df, channel_cols = prepare_data(valid_raw_df)
        with pytest.raises(DataValidationError, match="Channel columns missing"):
            validate_model_df(model_df, channel_cols + ["nonexistent_ch"])
