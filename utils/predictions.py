"""
Helpers for working with MMM posterior prediction arrays.
"""
from __future__ import annotations

import warnings

import numpy as np


def predict_array(mmm, X) -> np.ndarray:
    """Run `mmm.predict()` and always return a NumPy array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = mmm.predict(X)
    return np.asarray(pred)


def posterior_mean(predictions) -> np.ndarray:
    """Collapse posterior predictions to a 1-D mean vector over periods."""
    arr = np.asarray(predictions)
    if arr.ndim >= 3:
        return np.nanmean(arr, axis=(0, 1)).ravel()
    if arr.ndim == 2:
        return np.nanmean(arr, axis=0).ravel()
    return arr.ravel()


def prediction_interval(predictions, lower: float = 5, upper: float = 95) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper percentile vectors over periods."""
    arr = np.asarray(predictions)
    if arr.ndim >= 3:
        low = np.nanpercentile(arr, lower, axis=(0, 1)).ravel()
        high = np.nanpercentile(arr, upper, axis=(0, 1)).ravel()
        return low, high

    mean = arr.ravel()
    return mean * 0.85, mean * 1.15


def tail_predictions(predictions, periods: int):
    """Keep only the last `periods` values from the time axis."""
    arr = np.asarray(predictions)
    if arr.ndim >= 1:
        return arr[..., -periods:]
    return arr
