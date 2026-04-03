"""
Production-grade settings management using dataclasses + environment variables.
Override any default by setting the corresponding env variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


@dataclass
class ModelConfig:
    adstock_max_lag: int = int(os.getenv("ADSTOCK_MAX_LAG", "4"))
    draws: int = int(os.getenv("MCMC_DRAWS", "1000"))
    tune: int = int(os.getenv("MCMC_TUNE", "500"))
    chains: int = int(os.getenv("MCMC_CHAINS", "2"))
    target_accept: float = float(os.getenv("MCMC_TARGET_ACCEPT", "0.9"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))


@dataclass
class DataConfig:
    date_col: str = os.getenv("DATE_COL", "Cal_Month")
    target_col: str = os.getenv("TARGET_COL", "clicks")
    spend_col: str = os.getenv("SPEND_COL", "media_spend")
    channel_col: str = os.getenv("CHANNEL_COL", "site")
    date_format: str = os.getenv("DATE_FORMAT", "%Y-%m-%d")


@dataclass
class OptimizationConfig:
    default_budget: float = float(os.getenv("DEFAULT_BUDGET", "100000"))
    min_channel_share: float = float(os.getenv("MIN_CHANNEL_SHARE", "0.05"))
    max_channel_share: float = float(os.getenv("MAX_CHANNEL_SHARE", "0.60"))
    n_scenarios: int = int(os.getenv("N_SCENARIOS", "3"))


@dataclass
class ExportConfig:
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    results_filename: str = "results.json"
    model_filename: str = "mmm_model.pkl"
    report_filename: str = "report.html"


@dataclass
class AppConfig:
    title: str = "Enterprise MMM Decision Engine"
    page_icon: str = "📊"
    layout: str = "wide"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


@dataclass
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    app: AppConfig = field(default_factory=AppConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
