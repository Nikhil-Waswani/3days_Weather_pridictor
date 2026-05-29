import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="centered")

# ── INIT FIREBASE ─────────────────────────────────────────────────────────────
FIREBASE_KEY = os.getenv("FIREBASE_KEY_PATH")

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

# ── AQI HELPERS ───────────────────────────────────────────────────────────────
def aqi_label(aqi):
    aqi = round(aqi)
    labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return labels.get(aqi, "Unknown")

def aqi_color(aqi):
    aqi = round(aqi)
    colors = {1: "green", 2: "blue", 3: "orange", 4: "red", 5: "violet"}
    return colors.get(aqi, "gray")

# ── FETCH FIREBASE DATA ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)   # refresh every hour
def fetch_firestore_data():
    docs = db.collection("aqi_features") \
             .order_by("timestamp", direction=firestore.Query.DESCENDING) \
             .limit(25) \
             .stream()
    rows = [d.to_dict() for d in docs]
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp")
    return df

# ── PREDICT 3 DAYS ────────────────────────────────────────────────────────────
def predict_3_days(model_data, scaler, latest_row):
    feature_cols = model_data["feature_cols"]
    model        = model_data["model"]
    scaled       = model_data["scaled"]

    predictions = []
    current_row = latest_row.copy()

    for day in range(1, 4):
        # Adjust time features for each future day
        current_row["hour"]  = 12        # midday prediction
        current_row["day"]   = (latest_row["day"] + day) % 7
        current_row["month"] = latest_row["month"]

        # Build input for model
        X = pd.DataFrame([current_row])[feature_cols]

        if scaled:
            X = scaler.transform(X)

        pred = model.predict(X)[0]
        pred = round(np.clip(pred, 1, 5))   # AQI must be between 1-5
        predictions.append(pred)

        # Use prediction as next input
        current_row["aqi"]            = pred
        current_row["aqi_change_rate"] = pred - current_row["aqi"]

    return predictions

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
st.title("🌤️ AQI Forecast — Khairpur, PK")
st.caption("Live data from OpenWeather · Predictions by Random Forest model")

# Load model and data
try:
    model_data = load_model()
    scaler     = load_scaler()
except FileNotFoundError:
    st.error("model.pkl not found. Run train_model.py first!")
    st.stop()

try:
    df = fetch_firestore_data()
except Exception as e:
    st.error(f"Firebase connection error: {e}")
    st.stop()

if df.empty:
    st.warning("No data in Firestore yet. Run feature_pipeline.py first.")
    st.stop()

# Get latest row
latest = df.iloc[-1]

st.divider()

# ── CURRENT CONDITIONS ────────────────────────────────────────────────────────
st.subheader("📍 Current Conditions")

col1, col2, col3, col4 = st.columns(4)
col1.metric("AQI", f"{int(latest['aqi'])} — {aqi_label(int(latest['aqi']))}")
col2.metric("Temperature", f"{latest['temp']}°C")
col3.metric("Humidity", f"{latest['humidity']}%")
col4.metric("Wind Speed", f"{latest['wind_speed']} m/s")

st.divider()

# ── 3-DAY FORECAST ────────────────────────────────────────────────────────────
st.subheader("📅 3-Day AQI Forecast")

predictions = predict_3_days(model_data, scaler, latest.to_dict())

today    = datetime.now(tz=timezone.utc)
day1     = (today + timedelta(days=1)).strftime("%b %d")
day2     = (today + timedelta(days=2)).strftime("%b %d")
day3     = (today + timedelta(days=3)).strftime("%b %d")

col1, col2, col3 = st.columns(3)

col1.metric(
    f"Tomorrow · {day1}",
    f"AQI {predictions[0]}",
    aqi_label(predictions[0])
)
col2.metric(
    f"Day 2 · {day2}",
    f"AQI {predictions[1]}",
    aqi_label(predictions[1])
)
col3.metric(
    f"Day 3 · {day3}",
    f"AQI {predictions[2]}",
    aqi_label(predictions[2])
)

st.divider()

# ── AQI HISTORY CHART ─────────────────────────────────────────────────────────
st.subheader("📈 Last 24 Hours — AQI Trend")

if len(df) >= 2:
    chart_df = df[["timestamp", "aqi"]].set_index("timestamp")
    st.line_chart(chart_df)
else:
    st.info("Not enough data yet for chart. Collecting hourly via GitHub Actions.")

st.divider()

# ── MODEL INFO ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Model Info"):
    st.write(f"**Model:** {model_data['model_name']}")
    st.write(f"**RMSE:** {model_data['rmse']:.4f}")
    st.write(f"**MAE:** {model_data['mae']:.4f}")
    st.write(f"**R²:** {model_data['r2']:.4f}")
    st.write(f"**Features:** {model_data['feature_cols']}")

# ── RAW DATA ──────────────────────────────────────────────────────────────────
with st.expander("🗃️ Raw Collected Data (last 25 rows)"):
    st.dataframe(df, use_container_width=True)

st.caption(f"Last updated: {latest['timestamp']} · Total rows in Firestore: {len(df)}")
