import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retail Sales Forecasting Intelligence", layout="wide")

OUT = Path("outputs")
st.title("Retail Sales Forecasting Intelligence")
st.caption("Time-series forecasting + business analytics portfolio project using the UCI Online Retail dataset.")

if not (OUT / "metrics.json").exists():
    st.warning("Run `python train.py` first to generate the executed metrics and charts.")
    st.stop()

metrics = json.loads((OUT / "metrics.json").read_text())
comparison = pd.read_csv(OUT / "model_comparison.csv")
monthly = pd.read_csv(OUT / "monthly_sales.csv")
countries = pd.read_csv(OUT / "top_countries.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", metrics["best_model"])
c2.metric("MAE", f"£{metrics['test_metrics']['MAE']:,.0f}")
c3.metric("RMSE", f"£{metrics['test_metrics']['RMSE']:,.0f}")
c4.metric("MAPE", f"{metrics['test_metrics']['MAPE']:.1f}%")

st.subheader("Forecast vs actual")
st.image(str(OUT / "forecast_vs_actual.png"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Model comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
with right:
    st.subheader("Top countries by revenue")
    st.bar_chart(countries.set_index("Country")["revenue"])

st.subheader("Monthly revenue trend")
st.image(str(OUT / "monthly_revenue.png"), use_container_width=True)

st.info("Educational portfolio analysis of historical retail transactions; forecasts are illustrative and not production planning advice.")
