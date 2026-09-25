# Retail Sales Forecasting Intelligence

End-to-end **analytics + time-series forecasting** project built on the UCI Online Retail dataset.

## Live demo

**[Open the Streamlit app](https://retail-sales-forecasting-intelligence-3gt7qsyapxmpho4w7rquft.streamlit.app)**

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


## Executed results

The GitHub Actions pipeline ran successfully on the full UCI dataset.

- Raw transactions: **541,909**
- Clean transactions: **530,104**
- Analysis period: **2010-12-01 to 2011-12-09**
- Chronological test horizon: **42 days**
- Clean revenue analyzed: **£10.67M**
- Best model by MAE/RMSE: **7-day Seasonal Naive**
- Test MAE: **£14,934**
- Test RMSE: **£27,933**
- Test MAPE: **26.32%**

### Model comparison

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Seasonal Naive (7-day) | £14,934 | £27,933 | 26.32% |
| Ridge Regression | £16,172 | £28,631 | **24.15%** |
| Random Forest | £17,329 | £30,368 | 28.65% |
| HistGradientBoosting | £19,220 | £32,014 | 30.96% |

A useful result from this project is that the simple weekly seasonal baseline outperformed the more complex machine-learning models on MAE and RMSE. Ridge Regression achieved the lowest MAPE. The project therefore demonstrates baseline benchmarking rather than assuming model complexity guarantees better forecasts.
