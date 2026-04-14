"""
Budget optimisation — compatible with pymc-marketing >= 0.19.

The v0.19 BudgetOptimizer API:
    BudgetOptimizer(mmm_model=mmm, num_periods=N)
    optimizer.allocate_budget(total_budget, budget_bounds={ch: (lo, hi)})
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import warnings
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM

from src.optimization.constraints import BoundsDict
from utils import get_logger, timer

log = get_logger(__name__)


def run_optimization(
    mmm: "MMM",
    total_budget: float,
    bounds: Optional[BoundsDict] = None,
    num_periods: int = 1,
) -> pd.DataFrame:
    """
    Optimise budget allocation using BudgetOptimizer.allocate_budget().

    Args:
        mmm: Fitted MMM instance.
        total_budget: Total budget to allocate.
        bounds: Per-channel (min_absolute, max_absolute) bounds in spend units.
                If None, the optimizer runs unconstrained.
        num_periods: Number of periods to optimise over (default 1).

    Returns:
        DataFrame with [channel, optimized_spend, share].
    """
    log.info("Starting budget optimisation. Budget=%.2f, periods=%d", total_budget, num_periods)

    try:
        from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
    except ImportError as exc:
        raise RuntimeError(f"BudgetOptimizer not available: {exc}") from exc

    try:
        with timer("Budget optimisation"), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            optimizer = BudgetOptimizer(
                model=mmm,
                num_periods=num_periods,
            )

            # Convert share bounds → absolute spend bounds
            abs_bounds: dict | None = None
            if bounds:
                abs_bounds = {
                    ch: (lo * total_budget, hi * total_budget)
                    for ch, (lo, hi) in bounds.items()
                }

            allocation_da, opt_result = optimizer.allocate_budget(
                total_budget=total_budget,
                budget_bounds=abs_bounds,
            )

        log.info("Optimisation complete. Status: %s", getattr(opt_result, "message", "N/A"))
        return _format_allocation(allocation_da, total_budget)

    except Exception as exc:
        log.exception("Budget optimisation failed: %s", exc)
        raise RuntimeError(f"Optimisation failed: {exc}") from exc


def _format_allocation(allocation_da, total_budget: float) -> pd.DataFrame:
    """Convert xarray DataArray allocation → tidy DataFrame."""
    import xarray as xr
    if isinstance(allocation_da, xr.DataArray):
        # Sum over periods if multiple
        vals = np.array(allocation_da).ravel()
        channels = list(allocation_da.coords[allocation_da.dims[0]].values)
        if len(vals) > len(channels):
            # multiple periods: sum across periods per channel
            arr = np.array(allocation_da)
            if arr.ndim == 2:
                vals = arr.sum(axis=1) if arr.shape[0] == len(channels) else arr.sum(axis=0)
            channels = channels[:len(vals)]
    elif isinstance(allocation_da, dict):
        channels = list(allocation_da.keys())
        vals = [float(v) for v in allocation_da.values()]
    elif isinstance(allocation_da, pd.Series):
        channels = list(allocation_da.index)
        vals = list(allocation_da.values)
    else:
        channels = []
        vals = []

    df = pd.DataFrame({
        "channel": channels,
        "optimized_spend": [float(v) for v in vals[:len(channels)]],
    })
    df["share"] = df["optimized_spend"] / total_budget
    return df.sort_values("optimized_spend", ascending=False).reset_index(drop=True)
