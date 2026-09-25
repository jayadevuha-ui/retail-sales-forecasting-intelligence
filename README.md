# Retail Sales Forecasting Intelligence

End-to-end **analytics + time-series forecasting** project built on the UCI Online Retail dataset.

## Why this project

The portfolio already contains a supervised machine-learning classification project. This second project broadens the work into:

- transaction-level business analytics
- revenue trend analysis
- time-series feature engineering
- lag and rolling-window features
- seasonal baseline forecasting
- regression-based forecasting
- tree-based forecasting
- held-out chronological evaluation
- interactive Streamlit reporting

## Dataset

**UCI Online Retail** — 541,909 transaction rows from a UK-based non-store retailer, covering December 2010 to December 2011.

Source: UCI Machine Learning Repository  
DOI: `10.24432/C5BW33`  
License: CC BY 4.0

The training script downloads the official UCI archive automatically.

## Forecasting design

The pipeline aggregates valid transactions into **daily revenue** and uses the final **42 days** as an untouched chronological test horizon.

Features include:

- day of week / month / week of year
- weekend flag
- lags: 1, 7, 14, 28 days
- rolling means: 7, 14, 28 days
- rolling standard deviations: 7, 14, 28 days

Models compared:

- 7-day seasonal naive baseline
- Ridge Regression
- Random Forest
- HistGradientBoosting

Metrics:

- MAE
- RMSE
- MAPE

## Run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Executed results are written to `outputs/` by the training workflow.
