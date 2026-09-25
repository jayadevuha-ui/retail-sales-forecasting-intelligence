import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retail Sales Forecasting Intelligence", layout="wide")

OUT = Path("outputs")
REQUIRED = [
    OUT / "metrics.json",
    OUT / "model_comparison.csv",
    OUT / "monthly_sales.csv",
    OUT / "top_countries.csv",
    OUT / "forecast_predictions.csv",
]

st.title("Retail Sales Forecasting Intelligence")
st.caption("Time-series forecasting + business analytics using the UCI Online Retail dataset.")

if not all(p.exists() for p in REQUIRED):
    with st.spinner("Preparing the dataset and building forecasting artifacts for this session..."):
        from train import main
        main()

metrics = json.loads((OUT / "metrics.json").read_text())
comparison = pd.read_csv(OUT / "model_comparison.csv")
monthly = pd.read_csv(OUT / "monthly_sales.csv")
countries = pd.read_csv(OUT / "top_countries.csv")

st.subheader("Executive summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model by MAE", metrics["best_model"])
c2.metric("MAE", f"£{metrics['test_metrics']['MAE']:,.0f}")
c3.metric("RMSE", f"£{metrics['test_metrics']['RMSE']:,.0f}")
c4.metric("MAPE", f"{metrics['test_metrics']['MAPE']:.1f}%")

st.caption(
    f"Analysis period: {metrics['date_min']} to {metrics['date_max']} · "
    f"Clean transactions: {metrics['dataset_rows_clean']:,} · "
    f"Revenue analyzed: £{metrics['total_revenue_clean_gbp']:,.0f}"
)

st.subheader("42-day forecast vs actual")
forecast = pd.read_csv(OUT / "forecast_predictions.csv", parse_dates=["date"]).set_index("date")
st.line_chart(forecast[["actual", "forecast"]], use_container_width=True)
st.caption(
    "The 7-day seasonal naive baseline produced the lowest MAE and RMSE on the held-out period. "
    "That is an important forecasting result: a more complex model is not automatically better."
)

left, right = st.columns(2)
with left:
    st.subheader("Model comparison")
    st.dataframe(
        comparison.style.format({"MAE":"£{:,.0f}", "RMSE":"£{:,.0f}", "MAPE":"{:.1f}%"}),
        use_container_width=True,
        hide_index=True
    )
with right:
    st.subheader("Top countries by revenue")
    st.bar_chart(countries.set_index("Country")["revenue"])

st.subheader("Monthly revenue trend")
monthly["InvoiceDate"] = pd.to_datetime(monthly.iloc[:, 0])
monthly_chart = monthly.set_index("InvoiceDate")[["sales"]]
st.line_chart(monthly_chart, use_container_width=True)

st.subheader("What this project demonstrates")
st.write(
    "Transaction cleaning, business analytics, temporal aggregation, lag and rolling-window "
    "feature engineering, chronological validation, baseline benchmarking, machine-learning "
    "forecasting, and communicating results through an interactive dashboard."
)

st.info(
    "Educational portfolio analysis of historical retail transactions. "
    "Forecasts are illustrative and are not production inventory or financial planning advice."
)
