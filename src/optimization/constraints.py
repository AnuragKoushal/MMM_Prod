"""
Budget constraint definitions for the optimizer.
Supports per-channel floor/ceiling as share of total budget.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings().optimization

BoundsTuple = Tuple[float, float]
BoundsDict = Dict[str, BoundsTuple]


def build_constraints(
    channel_cols: List[str],
    overrides: Optional[Dict[str, BoundsTuple]] = None,
) -> BoundsDict:
    """
    Build per-channel (min_share, max_share) bounds for the optimizer.

    Args:
        channel_cols: List of channel names.
        overrides: Optional dict of {channel: (min_share, max_share)} to
                   override the global defaults for specific channels.

    Returns:
        Dict mapping each channel to its (min_share, max_share) bounds.

    Raises:
        ValueError: If any bound is outside [0, 1] or min > max.
    """
    bounds: BoundsDict = {}
    overrides = overrides or {}

    for ch in channel_cols:
        lo, hi = overrides.get(ch, (cfg.min_channel_share, cfg.max_channel_share))
        _validate_bound(ch, lo, hi)
        bounds[ch] = (lo, hi)

    _validate_global_feasibility(bounds)
    log.info("Constraints built for %d channels.", len(bounds))
    return bounds


def _validate_bound(channel: str, lo: float, hi: float) -> None:
    if not (0.0 <= lo <= 1.0):
        raise ValueError(f"Channel '{channel}': min_share={lo} must be in [0, 1].")
    if not (0.0 <= hi <= 1.0):
        raise ValueError(f"Channel '{channel}': max_share={hi} must be in [0, 1].")
    if lo > hi:
        raise ValueError(
            f"Channel '{channel}': min_share ({lo}) > max_share ({hi})."
        )


def _validate_global_feasibility(bounds: BoundsDict) -> None:
    """Ensure the sum of minimums ≤ 1 (otherwise no feasible allocation exists)."""
    total_min = sum(lo for lo, _ in bounds.values())
    if total_min > 1.0:
        raise ValueError(
            f"Sum of all min_share values ({total_min:.2f}) exceeds 1.0. "
            "No feasible budget allocation exists."
        )
