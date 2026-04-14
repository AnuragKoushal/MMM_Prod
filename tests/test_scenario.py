"""
Tests for src/scenario/planner.py
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.scenario.planner import generate_scenario, build_scenario_summary


@pytest.fixture
def base_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=6, freq="MS")
    return pd.DataFrame(
        {
            "date": dates,
            "channel_a": np.linspace(1000, 2000, 6),
            "channel_b": np.linspace(500, 1500, 6),
            "clicks": np.linspace(200, 800, 6),
        }
    )


CHANNELS = ["channel_a", "channel_b"]


class TestGenerateScenario:
    def test_adds_correct_number_of_rows(self, base_df):
        result = generate_scenario(base_df, {"channel_a": 1.5}, periods=3)
        assert len(result) == len(base_df) + 3

    def test_future_dates_are_monthly(self, base_df):
        result = generate_scenario(base_df, {}, periods=3)
        future = result.tail(3)
        diffs = future["date"].diff().dropna()
        # Each step should be approximately 1 month (~28-31 days)
        for diff in diffs:
            assert 27 <= diff.days <= 32

    def test_multiplier_applied_correctly(self, base_df):
        original_last = float(base_df["channel_a"].iloc[-1])
        result = generate_scenario(base_df, {"channel_a": 2.0}, periods=1)
        new_row = result.iloc[-1]
        assert new_row["channel_a"] == pytest.approx(original_last * 2.0)

    def test_zero_multiplier_clamps_to_zero(self, base_df):
        result = generate_scenario(base_df, {"channel_a": 0.0}, periods=1)
        assert result.iloc[-1]["channel_a"] == pytest.approx(0.0)

    def test_unknown_channel_ignored(self, base_df):
        # Should not raise even if multiplier key is not a column
        result = generate_scenario(base_df, {"nonexistent": 2.0}, periods=1)
        assert len(result) == len(base_df) + 1

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="empty"):
            generate_scenario(pd.DataFrame(), {}, periods=1)

    def test_raises_on_zero_periods(self, base_df):
        with pytest.raises(ValueError, match="periods"):
            generate_scenario(base_df, {}, periods=0)

    def test_historical_rows_unchanged(self, base_df):
        result = generate_scenario(base_df, {"channel_a": 99.0}, periods=2)
        original_part = result.iloc[: len(base_df)]
        pd.testing.assert_frame_equal(
            original_part.reset_index(drop=True),
            base_df.reset_index(drop=True),
        )


class TestBuildScenarioSummary:
    def test_summary_has_predicted_columns(self, base_df):
        scenario_df = generate_scenario(base_df, {"channel_a": 1.2}, periods=3)
        fake_preds = np.random.normal(500, 50, (2, 100, len(scenario_df)))
        summary = build_scenario_summary(scenario_df, fake_preds, "clicks", CHANNELS)

        assert "predicted_mean" in summary.columns
        assert "predicted_p5" in summary.columns
        assert "predicted_p95" in summary.columns

    def test_summary_row_count_matches_scenario(self, base_df):
        scenario_df = generate_scenario(base_df, {}, periods=2)
        fake_preds = np.random.normal(500, 50, (2, 100, len(scenario_df)))
        summary = build_scenario_summary(scenario_df, fake_preds, "clicks", CHANNELS)
        assert len(summary) == len(scenario_df)

    def test_p5_leq_mean_leq_p95(self, base_df):
        scenario_df = generate_scenario(base_df, {}, periods=2)
        fake_preds = np.random.normal(500, 50, (2, 100, len(scenario_df)))
        summary = build_scenario_summary(scenario_df, fake_preds, "clicks", CHANNELS)
        assert (summary["predicted_p5"] <= summary["predicted_mean"]).all()
        assert (summary["predicted_mean"] <= summary["predicted_p95"]).all()
