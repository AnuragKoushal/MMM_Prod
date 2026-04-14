"""
Tests for src/risk/uncertainty.py
"""
from __future__ import annotations

import numpy as np
import pytest

from src.risk.uncertainty import (
    summarize_with_uncertainty,
    value_at_risk,
    conditional_value_at_risk,
    build_risk_report,
)


@pytest.fixture
def sample_preds() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=5000, scale=800, size=(2, 500, 12))  # chains × draws × time


class TestSummarizeWithUncertainty:
    def test_keys_present(self, sample_preds):
        summary = summarize_with_uncertainty(sample_preds)
        for key in ["mean", "std", "p5", "p25", "p50", "p75", "p95"]:
            assert key in summary

    def test_percentiles_ordered(self, sample_preds):
        summary = summarize_with_uncertainty(sample_preds)
        assert summary["p5"] <= summary["p25"] <= summary["p50"] <= summary["p75"] <= summary["p95"]

    def test_mean_within_range(self, sample_preds):
        summary = summarize_with_uncertainty(sample_preds)
        assert summary["p5"] <= summary["mean"] <= summary["p95"]

    def test_handles_flat_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = summarize_with_uncertainty(arr)
        assert summary["mean"] == pytest.approx(3.0)

    def test_custom_percentiles(self, sample_preds):
        summary = summarize_with_uncertainty(sample_preds, percentiles=(10, 90))
        assert "p10" in summary
        assert "p90" in summary
        assert "p5" not in summary


class TestValueAtRisk:
    def test_var_less_than_mean(self, sample_preds):
        var = value_at_risk(sample_preds, confidence=0.05)
        mean = float(np.mean(sample_preds))
        assert var < mean

    def test_var_at_50pct_is_median(self, sample_preds):
        var = value_at_risk(sample_preds, confidence=0.50)
        median = float(np.median(sample_preds))
        assert var == pytest.approx(median, rel=1e-3)


class TestConditionalValueAtRisk:
    def test_cvar_leq_var(self, sample_preds):
        var = value_at_risk(sample_preds, confidence=0.05)
        cvar = conditional_value_at_risk(sample_preds, confidence=0.05)
        assert cvar <= var

    def test_cvar_is_float(self, sample_preds):
        cvar = conditional_value_at_risk(sample_preds)
        assert isinstance(cvar, float)


class TestBuildRiskReport:
    def test_report_structure(self, sample_preds):
        report = build_risk_report(sample_preds)
        assert "summary" in report
        assert "var_5pct" in report
        assert "cvar_5pct" in report

    def test_var_leq_cvar_is_false(self, sample_preds):
        # CVaR ≤ VaR by definition
        report = build_risk_report(sample_preds)
        assert report["cvar_5pct"] <= report["var_5pct"]
