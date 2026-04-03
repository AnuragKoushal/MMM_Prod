"""
MMM model trainer.
Wraps pymc-marketing MMM with configurable priors and MCMC settings.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import pymc as pm

from pymc_marketing.mmm import MMM

from config import get_settings
from utils import get_logger, timer

log = get_logger(__name__)
cfg_data = get_settings().data
cfg_model = get_settings().model


def build_model_config() -> dict:
    """Return prior distributions for the MMM."""
    return {
        "intercept": pm.Normal.dist(mu=0, sigma=1),
        "beta_channel": pm.HalfNormal.dist(sigma=1),
        "sigma": pm.HalfNormal.dist(sigma=1),
    }


def train_mmm(
    df: pd.DataFrame,
    channel_cols: List[str],
) -> Tuple[MMM, object]:
    """
    Fit a Marketing Mix Model.

    Args:
        df: Wide-format model DataFrame (date + channels + target).
        channel_cols: Column names for media channels.

    Returns:
        mmm: Fitted MMM instance.
        idata: ArviZ InferenceData (posterior samples).

    Raises:
        RuntimeError: If model fitting fails.
    """
    log.info(
        "Training MMM on %d rows with channels: %s",
        len(df), channel_cols,
    )

    try:
        with timer("MMM training"):
            mmm = MMM(
                date_column="date",
                target_column=cfg_data.target_col,
                channel_columns=channel_cols,
                adstock_max_lag=cfg_model.adstock_max_lag,
                model_config=build_model_config(),
            )

            idata = mmm.fit(
                df,
                draws=cfg_model.draws,
                tune=cfg_model.tune,
                chains=cfg_model.chains,
                target_accept=cfg_model.target_accept,
                random_seed=cfg_model.random_seed,
            )

        log.info("MMM training complete.")
        return mmm, idata

    except Exception as exc:
        log.exception("MMM training failed: %s", exc)
        raise RuntimeError(f"Model training failed: {exc}") from exc
