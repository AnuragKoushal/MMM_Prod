#  Enterprise MMM Decision Engine

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

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd mmm_prod
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create your local environment file

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

You can keep the defaults for the first run, or edit `.env` if your input
column names are different.

### 5. Run the test suite

Recommended quick validation:

```bash
python -m pytest -q -o addopts=''
```

If you want to use the repo shortcuts:

```bash
make test
```

### 6. Start the Streamlit app

```bash
streamlit run app.py
```

If `streamlit` is not available on your shell path, use:

```bash
python -m streamlit run app.py
```

### 7. Open the app in your browser

After the command starts, Streamlit will print a local URL such as:

```text
http://localhost:8501
```

Open that URL in your browser.

### 8. Use the app step by step

1. Go to the `Data` tab.
2. Upload your CSV or Excel file.
3. Confirm the raw preview and model-ready preview look correct.
4. Go to the `Train` tab and click `Train MMM`.
5. Review the evaluation metrics after training completes.
6. Open `Insights` to inspect contributions, ROAS, and response curves.
7. Use `Scenarios` for what-if forecasting.
8. Use `Optimise` for budget allocation suggestions.

### 9. Reset the app when switching datasets

If you want a completely fresh run, use `Reset App` in the header or sidebar.
The app also clears stale trained-model artifacts automatically when uploaded
data changes.

### 10. Stop the app

In the terminal where Streamlit is running, press:

```text
Ctrl+C
```

## Input Data Format

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

Direct Python command:

```bash
python -m pytest -q -o addopts=''
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
