"""
Budget optimization: wraps pymc-marketing BudgetOptimizer with
robust error handling and result formatting.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pymc_marketing.mmm import MMM

from src.optimization.constraints import BoundsDict
from utils import get_logger, timer

log = get_logger(__name__)


def run_optimization(
    mmm: MMM,
    total_budget: float,
    bounds: Optional[BoundsDict] = None,
) -> pd.DataFrame:
    """
    Optimize budget allocation across media channels.

    Args:
        mmm: Fitted MMM instance.
        total_budget: Total budget to allocate (same currency as training data).
        bounds: Per-channel (min_share, max_share) constraints. If None, no
                constraints are applied.

    Returns:
        DataFrame with columns [channel, optimized_spend, share].

    Raises:
        RuntimeError: If the optimizer fails.
    """
    log.info(
        "Starting budget optimization. Total budget: %.2f, Bounds: %s",
        total_budget,
        bounds,
    )

    try:
        from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
        with timer("Budget optimization"):
            optimizer = BudgetOptimizer(mmm)
            result = optimizer.optimize(
                total_budget=total_budget,
                bounds=bounds,
            )

        allocation = _format_result(result, total_budget)
        log.info("Optimization complete:\n%s", allocation.to_string(index=False))
        return allocation

    except Exception as exc:
        log.exception("Budget optimization failed: %s", exc)
        raise RuntimeError(f"Optimization failed: {exc}") from exc


def _format_result(raw_result, total_budget: float) -> pd.DataFrame:
    """
    Normalise the optimizer output into a tidy DataFrame.
    Handles both dict and Series outputs from different pymc-marketing versions.
    """
    if isinstance(raw_result, dict):
        df = pd.DataFrame(
            [{"channel": k, "optimized_spend": v} for k, v in raw_result.items()]
        )
    elif isinstance(raw_result, pd.Series):
        df = raw_result.reset_index()
        df.columns = ["channel", "optimized_spend"]
    elif isinstance(raw_result, pd.DataFrame):
        df = raw_result.copy()
        if "channel" not in df.columns:
            df = df.reset_index().rename(columns={"index": "channel"})
        spend_col = [c for c in df.columns if c != "channel"][0]
        df = df.rename(columns={spend_col: "optimized_spend"})
    else:
        raise TypeError(f"Unexpected optimizer result type: {type(raw_result)}")

    df["share"] = df["optimized_spend"] / total_budget
    return df.sort_values("optimized_spend", ascending=False).reset_index(drop=True)
