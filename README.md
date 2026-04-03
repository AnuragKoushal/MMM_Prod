# 📊 Enterprise MMM Decision Engine

A production-grade **Marketing Mix Modelling (MMM)** application built with
[PyMC-Marketing](https://www.pymc-marketing.io/) and [Streamlit](https://streamlit.io/).

---

## Features

| Module | Description |
|---|---|
| **Data** | CSV / Excel upload, validation, long→wide pivoting |
| **Model** | Bayesian MMM with configurable adstock & priors (NUTS sampler) |
| **Evaluation** | R², MAPE, RMSE, WAIC, LOO-CV |
| **Inference** | Channel contributions, ROAS, response curves, posterior diagnostics |
| **Scenario Planner** | Multi-period what-if simulations with uncertainty bands |
| **Optimiser** | Budget allocation with per-channel share constraints |
| **Risk** | Posterior uncertainty, Value-at-Risk, Conditional VaR |
| **Export** | JSON, CSV, pickled model |

---

## Project Structure

```
mmm_prod/
├── app.py                   # Streamlit entry point
├── config/
│   └── settings.py          # Typed settings (env-var driven)
├── src/
│   ├── data/
│   │   ├── loader.py        # CSV / Excel loading
│   │   ├── preprocessor.py  # Long→wide transform
│   │   └── validator.py     # Input validation
│   ├── model/
│   │   ├── trainer.py       # MMM fitting
│   │   └── evaluator.py     # WAIC, LOO, R², MAPE
│   ├── inference/
│   │   ├── contributions.py # Channel contribution plots
│   │   ├── diagnostics.py   # Trace plots, PPC, response curves
│   │   └── roas.py          # ROAS computation
│   ├── optimization/
│   │   ├── constraints.py   # Budget constraint builder
│   │   └── optimizer.py     # BudgetOptimizer wrapper
│   ├── scenario/
│   │   └── planner.py       # Multi-period scenario simulation
│   ├── risk/
│   │   └── uncertainty.py   # VaR, CVaR, uncertainty summaries
│   └── export/
│       └── exporter.py      # JSON, CSV, pickle export
├── ui/
│   └── components/
│       ├── sidebar.py
│       ├── data_upload.py
│       ├── model_training.py
│       ├── insights.py
│       ├── scenario_planner.py
│       └── optimizer_ui.py
├── utils/
│   ├── logger.py            # Centralised logging
│   └── helpers.py           # timer(), safe_divide(), flatten_dict()
├── tests/                   # pytest test suite
├── .env.example             # Environment variable template
├── Makefile                 # Dev task shortcuts
└── requirements.txt
```

---

## Quickstart

### 1. Clone & install

```bash
git clone <your-repo-url>
cd mmm_prod
cp .env.example .env          # edit as needed
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
# or
make run
```

### 3. Upload data

Your CSV/Excel file must contain these columns (configurable via `.env`):

| Column | Default name | Type |
|---|---|---|
| Date | `Cal_Month` | date string |
| Channel | `site` | string |
| Spend | `media_spend` | numeric |
| Target | `clicks` | numeric |

---

## Configuration

All settings are controlled via environment variables (see `.env.example`).
No hard-coded values exist in the codebase.

```bash
# Example: change target metric to revenue
TARGET_COL=revenue MCMC_DRAWS=2000 streamlit run app.py
```

---

## Running Tests

```bash
make test            # full suite with coverage
make test-fast       # fail-fast, no HTML report
```

Coverage is reported for all `src/` modules. Minimum threshold: **70%**.

---

## Key Design Decisions

- **`config/settings.py`** — single source of truth; env-var overrides at runtime  
- **`utils/logger.py`** — all modules use `get_logger(__name__)`; no bare `print()`  
- **`DataValidationError`** — clean, user-facing exceptions surfaced by the UI  
- **`timer()` context manager** — latency logged for every major operation  
- **`_make_serialisable()`** — safely converts numpy/pandas types before JSON export  
- **Layered imports** — each sub-package exposes a clean `__init__.py` public API  
- **No global state** — all mutable state flows through `st.session_state`  
