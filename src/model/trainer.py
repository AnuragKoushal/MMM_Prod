from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
)

from config import get_settings
from utils import get_logger, timer

log = get_logger(__name__)
cfg_data = get_settings().data
cfg_model = get_settings().model


def train_mmm(
    df: pd.DataFrame,
    channel_cols: List[str],
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
    adstock_max_lag: int | None = None,
) -> Tuple[MMM, object]:

    # ⚡ FAST MODE (respect UI min 200 if needed)
    _draws = draws or 200
    _tune = tune or 100
    _chains = chains or 1
    _target_accept = target_accept or 0.8
    _seed = random_seed or 42
    _lag = int(adstock_max_lag or cfg_model.adstock_max_lag or 3)

    log.warning("⚡ Running FAST TRAIN MODE")

    # --- CLEAN INPUT ---
    df = pd.DataFrame(df).copy(deep=True)

    X = df[["date"] + channel_cols].copy()
    y = df[cfg_data.target_col].copy()

    X["date"] = pd.to_datetime(X["date"], errors="coerce")
    X[channel_cols] = X[channel_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    X = X.dropna()
    y = y.loc[X.index]

    X[channel_cols] = X[channel_cols].astype("float64")
    y = y.astype("float64")

    try:
        with timer("MMM training (fast)"):
            mmm = MMM(
                date_column="date",
                channel_columns=channel_cols,
                adstock=GeometricAdstock(l_max=_lag),   # ✅ FIXED
                saturation=LogisticSaturation(),        # ✅ FIXED
            )

            idata = mmm.fit(
                X,
                y,
                draws=_draws,
                tune=_tune,
                chains=_chains,
                target_accept=_target_accept,
                random_seed=_seed,
                progressbar=False,
                compute_convergence_checks=False,
            )

        log.info("✅ Training complete.")
        return mmm, idata

    except Exception as exc:
        log.exception("MMM training failed: %s", exc)
        raise RuntimeError(f"Model training failed: {exc}") from exc