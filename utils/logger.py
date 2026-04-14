"""
Centralised logging configuration.
All modules import get_logger() from here to ensure consistent formatting.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


@lru_cache(maxsize=128)
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger. Calling with the same *name* always returns
    the same logger object (cached)."""
    _configure_root(level)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
