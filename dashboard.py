import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI Predictor", page_icon="🌤️", layout="centered")

# ── INIT FIREBASE ─────────────────────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    if firebase_admin._apps:
        return firestore.client()

    try:
        firebase_secrets = st.secrets.get("firebase")
    except:
        firebase_secrets = None

    if firebase_secrets:
        firebase_dict = {
            "type":                        firebase_secrets["type"],
            "project_id":                  firebase_secrets["project_id"],
            "private_key_id":              firebase_secrets["private_key_id"],
            "private_key":                 firebase_secrets["private_key"].replace("\\n", "\n"),
            "client_email":                firebase_secrets["client_email"],
            "client_id":                   firebase_secrets["client_id"],
            "auth_uri":                    firebase_secrets["auth_uri"],
            "token_uri":                   firebase_secrets["token_uri"],
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url":        f"https://www.googleapis.com/robot/v1/metadata/x509/{firebase_secrets['client_email']}"
        }
        cred = credentials.Certificate(firebase_dict)
    else:
        FIREBASE_KEY = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
        cred = credentials.Certificate(FIREBASE_KEY)

    firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

# ── AQI HELPERS ───────────────────────────────────────────────────────────────
def aqi_label(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

def aqi_color(aqi):
    if aqi <= 50:   return "green"
    if aqi <= 100:  return "orange"
    if aqi <= 150:  return "red"
    if aqi <= 200:  return "red"
    if aqi <= 300:  return "violet"
    return "violet"

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

# ── FETCH FIREBASE DATA ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
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
        current_row["hour"]  = 12
        current_row["day"]   = (latest_row["day"] + day) % 7
        current_row["month"] = latest_row["month"]

        X = pd.DataFrame([current_row])[feature_cols]

        if scaled:
            X = scaler.transform(X)

        pred = model.predict(X)[0]
        pred = round(np.clip(pred, 0, 500))   # clip to 0-500 range
        predictions.append(pred)

        current_row["aqi"]             = pred
        current_row["aqi_change_rate"] = pred - current_row["aqi"]

    return predictions

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
st.title("🌤️ AQI Forecast — Khairpur, PK")
st.caption("Live data from OpenWeather · Standard US EPA AQI (0-500) · Predictions by Random Forest")

try:
    model_data = load_model()
    scaler     = load_scaler()
except FileNotFoundError:
    st.error("model.pkl not found!")
    st.stop()

try:
    df = fetch_firestore_data()
except Exception as e:
    st.error(f"Firebase connection error: {e}")
    st.stop()

if df.empty:
    st.warning("No data in Firestore yet.")
    st.stop()

latest = df.iloc[-1]

st.divider()

# ── AQI SCALE REFERENCE ───────────────────────────────────────────────────────
with st.expander("📊 AQI Scale Reference (US EPA Standard)"):
    col1, col2, col3 = st.columns(3)
    col1.success("0-50: Good")
    col1.warning("51-100: Moderate")
    col2.error("101-150: Unhealthy for Sensitive Groups")
    col2.error("151-200: Unhealthy")
    col3.error("201-300: Very Unhealthy")
    col3.error("301-500: Hazardous")

st.divider()

# ── CURRENT CONDITIONS ────────────────────────────────────────────────────────
st.subheader("📍 Current Conditions")

current_aqi = int(latest["aqi"])
col1, col2, col3, col4 = st.columns(4)
col1.metric("AQI (0-500)", current_aqi)
col2.metric("Temperature", f"{latest['temp']}°C")
col3.metric("Humidity", f"{latest['humidity']}%")
col4.metric("Wind Speed", f"{latest['wind_speed']} m/s")

st.info(f"**Current AQI Status:** {aqi_label(current_aqi)}")

st.divider()

# ── 3-DAY FORECAST ────────────────────────────────────────────────────────────
st.subheader("📅 3-Day AQI Forecast")

predictions = predict_3_days(model_data, scaler, latest.to_dict())
today = datetime.now(tz=timezone.utc)
day1  = (today + timedelta(days=1)).strftime("%b %d")
day2  = (today + timedelta(days=2)).strftime("%b %d")
day3  = (today + timedelta(days=3)).strftime("%b %d")

col1, col2, col3 = st.columns(3)
col1.metric(f"Tomorrow · {day1}", predictions[0], aqi_label(predictions[0]))
col2.metric(f"Day 2 · {day2}",    predictions[1], aqi_label(predictions[1]))
col3.metric(f"Day 3 · {day3}",    predictions[2], aqi_label(predictions[2]))

st.divider()

# ── AQI HISTORY CHART ─────────────────────────────────────────────────────────
st.subheader("📈 Last 24 Hours — AQI Trend")

if len(df) >= 2:
    chart_df = df[["timestamp", "aqi"]].set_index("timestamp")
    st.line_chart(chart_df)
else:
    st.info("Not enough data yet for chart.")

st.divider()

# ── MODEL INFO ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Model Info"):
    st.write(f"**Model:** {model_data['model_name']}")
    st.write(f"**RMSE:** {model_data['rmse']:.4f}")
    st.write(f"**MAE:** {model_data['mae']:.4f}")
    st.write(f"**R²:** {model_data['r2']:.4f}")
    st.write(f"**Features:** {model_data['feature_cols']}")

with st.expander("🗃️ Raw Collected Data (last 25 rows)"):
    st.dataframe(df, use_container_width=True)

st.caption(f"Last updated: {latest['timestamp']}")