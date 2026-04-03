"""
Tests for src/optimization/constraints.py
"""
from __future__ import annotations

import pytest

from src.optimization.constraints import build_constraints


CHANNELS = ["tv", "digital", "radio", "ooh"]


class TestBuildConstraints:
    def test_default_bounds_applied(self):
        bounds = build_constraints(CHANNELS)
        for ch in CHANNELS:
            assert ch in bounds
            lo, hi = bounds[ch]
            assert 0.0 <= lo <= hi <= 1.0

    def test_all_channels_present(self):
        bounds = build_constraints(CHANNELS)
        assert set(bounds.keys()) == set(CHANNELS)

    def test_override_respected(self):
        overrides = {"tv": (0.1, 0.5)}
        bounds = build_constraints(CHANNELS, overrides=overrides)
        assert bounds["tv"] == (0.1, 0.5)

    def test_non_overridden_channels_use_defaults(self):
        overrides = {"tv": (0.1, 0.5)}
        bounds = build_constraints(CHANNELS, overrides=overrides)
        for ch in ["digital", "radio", "ooh"]:
            lo, hi = bounds[ch]
            assert lo == 0.05
            assert hi == 0.60

    def test_raises_on_min_gt_max(self):
        with pytest.raises(ValueError, match="min_share.*max_share"):
            build_constraints(["tv"], overrides={"tv": (0.6, 0.1)})

    def test_raises_on_bound_out_of_range(self):
        with pytest.raises(ValueError, match="must be in"):
            build_constraints(["tv"], overrides={"tv": (-0.1, 0.5)})

    def test_raises_when_min_sum_exceeds_one(self):
        # 5 channels each with min_share=0.25 → sum = 1.25 > 1.0
        channels = [f"ch_{i}" for i in range(5)]
        overrides = {ch: (0.25, 0.3) for ch in channels}
        with pytest.raises(ValueError, match="No feasible"):
            build_constraints(channels, overrides=overrides)

    def test_empty_channels(self):
        bounds = build_constraints([])
        assert bounds == {}

    def test_single_channel(self):
        bounds = build_constraints(["tv"])
        assert "tv" in bounds
        lo, hi = bounds["tv"]
        assert lo <= hi
