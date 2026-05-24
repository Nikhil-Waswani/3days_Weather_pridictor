import streamlit as st
import pandas as pd
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="centered")

CSV_PATH = "aqi_features.csv"

# ── AQI LABEL HELPER ─────────────────────────────────────────────────────────
def aqi_label(aqi):
    labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return labels.get(aqi, "Unknown")

def aqi_color(aqi):
    colors = {1: "green", 2: "blue", 3: "orange", 4: "red", 5: "violet"}
    return colors.get(aqi, "gray")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
st.title("🌤️ AQI Forecast — Khairpur mirs")
st.caption("Data collected via OpenWeather API · Updates every hour")

if not os.path.exists(CSV_PATH):
    st.warning("No data yet. Run `feature_pipeline.py` first.")
    st.stop()

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

latest = df.iloc[-1]

# ── CURRENT CONDITIONS ────────────────────────────────────────────────────────
st.subheader("Current Conditions")

col1, col2, col3, col4 = st.columns(4)
col1.metric("AQI", f"{int(latest['aqi'])} — {aqi_label(int(latest['aqi']))}")
col2.metric("Temperature", f"{latest['temp']}°C")
col3.metric("Humidity", f"{latest['humidity']}%")
col4.metric("Wind Speed", f"{latest['wind_speed']} m/s")

st.divider()

# ── 3-DAY FORECAST (placeholder until model is ready) ────────────────────────
st.subheader("3-Day AQI Forecast")
st.info("⏳ Model not trained yet — collect more data by running the pipeline hourly. Forecast will appear here once the model is trained.")

col1, col2, col3 = st.columns(3)
col1.metric("Tomorrow", "—")
col2.metric("Day 2", "—")
col3.metric("Day 3", "—")

st.divider()

# ── HISTORICAL CHART ──────────────────────────────────────────────────────────
st.subheader("AQI History")

if len(df) < 2:
    st.info("Only 1 data point so far. Run the pipeline a few more times to see the trend.")
else:
    chart_df = df[["timestamp", "aqi"]].set_index("timestamp")
    st.line_chart(chart_df)

st.divider()

# ── RAW DATA TABLE ────────────────────────────────────────────────────────────
with st.expander("View raw collected data"):
    st.dataframe(df.tail(20), use_container_width=True)

st.caption(f"Total rows collected: {len(df)} · Last updated: {latest['timestamp']}")
