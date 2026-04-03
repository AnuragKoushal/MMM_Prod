"""
Export results to JSON, CSV, and HTML report formats.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config import get_settings
from utils import get_logger

log = get_logger(__name__)
cfg = get_settings().export


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stamp(filename: str) -> str:
    """Prefix filename with ISO timestamp to keep exports unique."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return f"{ts}_{stem}{suffix}"


def export_json(
    data: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    timestamp: bool = True,
) -> Path:
    """
    Serialise *data* to a JSON file.

    Args:
        data: Dictionary to serialise. Values must be JSON-serialisable
              (DataFrames are converted to records automatically).
        filename: Target filename. Defaults to settings value.
        output_dir: Output directory. Defaults to settings value.
        timestamp: Prepend ISO timestamp to filename.

    Returns:
        Path to the written file.
    """
    output_dir = output_dir or cfg.output_dir
    filename = filename or cfg.results_filename
    if timestamp:
        filename = _stamp(filename)

    out_path = _ensure_output_dir(output_dir) / filename

    # Convert DataFrames / Series inside data to serialisable forms
    serialisable = _make_serialisable(data)

    with open(out_path, "w") as fh:
        json.dump(serialisable, fh, indent=4, default=str)

    log.info("Results exported to %s", out_path)
    return out_path


def export_csv(
    df: pd.DataFrame,
    filename: str = "results.csv",
    output_dir: Optional[str] = None,
    timestamp: bool = True,
) -> Path:
    """Serialise a DataFrame to CSV."""
    output_dir = output_dir or cfg.output_dir
    if timestamp:
        filename = _stamp(filename)
    out_path = _ensure_output_dir(output_dir) / filename
    df.to_csv(out_path, index=False)
    log.info("CSV exported to %s", out_path)
    return out_path


def export_model(
    mmm,
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    timestamp: bool = True,
) -> Path:
    """Pickle the fitted MMM instance for later reuse."""
    output_dir = output_dir or cfg.output_dir
    filename = filename or cfg.model_filename
    if timestamp:
        filename = _stamp(filename)
    out_path = _ensure_output_dir(output_dir) / filename
    with open(out_path, "wb") as fh:
        pickle.dump(mmm, fh)
    log.info("Model exported to %s", out_path)
    return out_path


def load_model(path: str):
    """Load a pickled MMM from disk."""
    with open(path, "rb") as fh:
        mmm = pickle.load(fh)
    log.info("Model loaded from %s", path)
    return mmm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serialisable(obj: Any) -> Any:
    """Recursively convert pandas / numpy types to JSON-safe Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
