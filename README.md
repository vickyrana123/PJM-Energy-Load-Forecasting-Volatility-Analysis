# PJM Energy Load Forecasting & Volatility Analysis

An end-to-end machine learning pipeline for forecasting hourly electricity load on the PJM Interconnection grid (AEP sub-grid), built with a strict leak-free temporal validation framework.

**Final Result:** LightGBM regression model — Test RMSE **173.84 MW**, MAPE **0.89%** — a **91% improvement** over a naive weekly-lag baseline (1,910.83 MW).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kOOG25MdjlKZ-wX84wtWFbvKYqewtF3z?usp=sharing)

---

## Overview

This project implements a full forecasting pipeline for hourly electricity demand, using historical consumption data from AEP (American Electric Power), a regional sub-grid within the PJM Interconnection network. The goal was to build a model that maintains strong baseline accuracy during normal consumption periods while remaining resilient during extreme peak-demand hours — without any lookahead bias or data leakage in the validation setup.

The notebook is organized into two sections:

- **Part A — Executable Code Pipeline**: data ingestion, leak-free feature engineering, chronological validation, model training, and quantitative evaluation.
- **Part B — Operational Technical Report**: model/loss justification, residual diagnostics, and documented ablation experiments.

---

## Dataset

- **Source:** [PJM Hourly Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (Kaggle)
- **Sub-grid used:** AEP (American Electric Power)
- **Range:** 2004-10-01 to 2018-08-03 (hourly resolution, ~121K records)
- **Target variable:** Hourly load in megawatts (MW)

---

## Methodology

### 1. Data & Feature Pipeline
- Automated ingestion via Google Drive + `gdown` (no manual upload required — the notebook runs end-to-end for anyone with the link).
- Deduplication and chronological sorting of timestamps.
- **Leak-free feature engineering:**
  - Calendar features: hour, day of week, month, weekend flag
  - Cyclical encodings: `sin`/`cos` transforms for hour and day-of-week
  - Lag features: 1, 24, 48, and 168 hours (capturing immediate, daily, and weekly patterns)
  - Rolling statistics (mean, std) — all computed with `shift(1)` applied *before* rolling, guaranteeing no future information leaks into any feature

### 2. Validation Framework
- Strict **chronological** train (70%) / validation (15%) / test (15%) split — no shuffling, no random sampling.
- Leak-free ordering verified programmatically with explicit assertions, not just claimed.

### 3. Model
- **LightGBM** regressor, chosen for its speed, native handling of tabular data, and built-in feature-importance interpretability.
- Trained with early stopping on a held-out validation set.

### 4. Evaluation
- Metrics: RMSE, MAE, and MAPE — computed overall, and separately for peak-demand hours (top 5% of load values) vs. normal periods.

---

## Key Finding: Loss Function Selection (Huber vs. MSE)

EDA revealed a right-skewed load distribution with a long peak-demand tail, initially motivating the use of **Huber loss** to reduce sensitivity to outliers. However, training showed abnormally slow, near-linear convergence that never plateaued, even after 5,000+ boosting rounds.

Feature importance analysis revealed the cause: Huber's gradient capping was suppressing the model's ability to exploit `lag_1` (the previous hour's load) — the single strongest predictor — because typical residuals were already large enough to trigger the cap. The model underperformed even a naive weekly-lag heuristic (RMSE 2,449 MW vs. 1,910 MW baseline).

Switching to standard **MSE loss** resolved this immediately, converging cleanly via early stopping and reducing test RMSE to **173.84 MW**.

This finding, along with a second ablation (removing `lag_1`, which tripled RMSE to 538.63 MW), is documented in full in the notebook as the required ablation/failed-hypothesis analysis.

---

## Results Summary

| Segment | RMSE (MW) | MAE (MW) | MAPE |
|---|---|---|---|
| Overall | 173.84 | 130.66 | 0.89% |
| Peak periods (top 5%) | 201.34 | 154.64 | 0.75% |
| Normal periods | 172.27 | 129.39 | 0.89% |
| Naive baseline (`lag_168`) | 1,910.83 | — | — |

Residual analysis shows no systematic bias or drift over the test period, with a mild widening of error spread at very high load levels — consistent with peak-demand hours being influenced by external factors (e.g., weather) not captured by historical load patterns alone.

---

## Repository Structure

```
├── PJM_Load_Forecasting_AEP.ipynb   # Full notebook (also available on Google Colab)
└── README.md
```

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1kOOG25MdjlKZ-wX84wtWFbvKYqewtF3z?usp=sharing) — no setup required, the dataset is fetched automatically.
2. Alternatively, clone this repo and run locally:
   ```bash
   pip install lightgbm gdown pandas numpy matplotlib scikit-learn
   jupyter notebook PJM_Load_Forecasting_AEP.ipynb
   ```

---

## Scope & Future Extensions

- No exogenous weather data was used in this version — incorporating temperature would likely reduce peak-period residual variance further, based on the error patterns observed.
- The model was trained on a single sub-grid (AEP); generalization to other PJM regions was not tested.
- A single chronological split was used rather than full walk-forward cross-validation, due to project time constraints.

---

## Tech Stack

`Python` · `LightGBM` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Google Colab`

---

## Author

**Vicky Rana**
