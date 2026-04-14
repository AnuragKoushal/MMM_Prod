"""
General-purpose helpers shared across the application.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from utils.logger import get_logger

log = get_logger(__name__)


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s completed in %.2fs", label, elapsed)


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns *fallback* when denominator is zero."""
    if denominator == 0:
        return fallback
    return numerator / denominator


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dict."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
