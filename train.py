from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def load_data():
    import io, zipfile, requests
    r = requests.get(DATA_URL, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    xlsx = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
    with z.open(xlsx) as f:
        df = pd.read_excel(f)
    return df

def clean_and_aggregate(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    daily = (
        df.set_index("InvoiceDate")
          .resample("D")["Revenue"]
          .sum()
          .rename("sales")
          .to_frame()
    )
    daily = daily.asfreq("D", fill_value=0.0)
    return df, daily

def add_features(daily):
    x = daily.copy()
    idx = x.index
    x["dow"] = idx.dayofweek
    x["dom"] = idx.day
    x["month"] = idx.month
    x["weekofyear"] = idx.isocalendar().week.astype(int).values
    x["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    for lag in [1, 7, 14, 28]:
        x[f"lag_{lag}"] = x["sales"].shift(lag)
    for win in [7, 14, 28]:
        x[f"roll_mean_{win}"] = x["sales"].shift(1).rolling(win).mean()
        x[f"roll_std_{win}"] = x["sales"].shift(1).rolling(win).std()
    return x.dropna()

def eval_metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAPE": mape(y, pred),
    }

def main():
    raw = load_data()
    clean, daily = clean_and_aggregate(raw)
    feat = add_features(daily)

    horizon = 42
    train = feat.iloc[:-horizon].copy()
    test = feat.iloc[-horizon:].copy()

    feature_cols = [c for c in feat.columns if c != "sales"]
    X_train, y_train = train[feature_cols], train["sales"]
    X_test, y_test = test[feature_cols], test["sales"]

    results = []
    predictions = {}

    naive = test["lag_7"].values
    predictions["SeasonalNaive7"] = naive
    results.append({"model":"SeasonalNaive7", **eval_metrics(y_test, naive)})

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    p = np.clip(ridge.predict(X_test), 0, None)
    predictions["Ridge"] = p
    results.append({"model":"Ridge", **eval_metrics(y_test, p)})

    rf = RandomForestRegressor(
        n_estimators=350, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    p = np.clip(rf.predict(X_test), 0, None)
    predictions["RandomForest"] = p
    results.append({"model":"RandomForest", **eval_metrics(y_test, p)})

    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=RANDOM_STATE
    )
    hgb.fit(X_train, y_train)
    p = np.clip(hgb.predict(X_test), 0, None)
    predictions["HistGradientBoosting"] = p
    results.append({"model":"HistGradientBoosting", **eval_metrics(y_test, p)})

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]
    best_pred = predictions[best_name]

    metrics = {
        "dataset_rows_raw": int(len(raw)),
        "dataset_rows_clean": int(len(clean)),
        "date_min": str(daily.index.min().date()),
        "date_max": str(daily.index.max().date()),
        "forecast_horizon_days": horizon,
        "best_model": best_name,
        "test_metrics": eval_metrics(y_test, best_pred),
        "total_revenue_clean_gbp": float(clean["Revenue"].sum()),
        "daily_sales_mean_gbp": float(daily["sales"].mean()),
        "daily_sales_median_gbp": float(daily["sales"].median()),
    }

    results_df.to_csv(OUT / "model_comparison.csv", index=False)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    monthly = daily.resample("MS")["sales"].sum().to_frame()
    monthly["mom_pct"] = monthly["sales"].pct_change() * 100
    monthly.to_csv(OUT / "monthly_sales.csv")

    country = clean.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(15)
    country.rename("revenue").to_csv(OUT / "top_countries.csv")

    plt.figure(figsize=(11,5))
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, best_pred, label=f"Forecast — {best_name}")
    plt.title("42-Day Retail Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Daily revenue (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "forecast_vs_actual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11,5))
    plt.plot(monthly.index, monthly["sales"].values)
    plt.title("Monthly Retail Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig(OUT / "monthly_revenue.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
