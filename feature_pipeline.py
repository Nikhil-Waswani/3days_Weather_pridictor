import requests
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv(dotenv_path=r"D:\Projects\Weather_pridictor\.env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("OPENWEATHER_API_KEY")
LAT          = float(os.getenv("LAT"))
LON          = float(os.getenv("LON"))
FIREBASE_KEY = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
# ─────────────────────────────────────────────────────────────────────────────

# ── INIT FIREBASE ─────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ─────────────────────────────────────────────────────────────────────────────


def calculate_aqi(pm2_5):
    """
    Convert PM2.5 (µg/m³) to standard US EPA AQI (0-500 scale).
    Source: US EPA AQI Breakpoints
    """
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]

    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm2_5 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm2_5 - bp_lo) + aqi_lo
            return round(aqi)

    return 500  # cap at 500 for very high PM2.5


def aqi_category(aqi):
    """Return AQI category label based on standard 0-500 scale."""
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def fetch_current_air_pollution():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    entry = data["list"][0]
    return {
        "dt":    entry["dt"],
        "co":    entry["components"]["co"],
        "no":    entry["components"]["no"],
        "no2":   entry["components"]["no2"],
        "o3":    entry["components"]["o3"],
        "so2":   entry["components"]["so2"],
        "pm2_5": entry["components"]["pm2_5"],
        "pm10":  entry["components"]["pm10"],
        "nh3":   entry["components"]["nh3"],
    }


def fetch_current_weather():
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return {
        "temp":       data["main"]["temp"],
        "humidity":   data["main"]["humidity"],
        "pressure":   data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "wind_deg":   data["wind"].get("deg", 0),
        "clouds":     data["clouds"]["all"],
        "visibility": data.get("visibility", 10000),
    }


def compute_features(air_data, weather_data):
    dt = datetime.fromtimestamp(air_data["dt"], tz=timezone.utc)

    # Calculate standard AQI from PM2.5
    aqi = calculate_aqi(air_data["pm2_5"])

    row = {}
    row["timestamp"]    = dt.isoformat()
    row["hour"]         = dt.hour
    row["day"]          = dt.weekday()
    row["month"]        = dt.month
    row["aqi"]          = aqi                          # standard 0-500 AQI
    row["aqi_category"] = aqi_category(aqi)            # Good/Moderate/etc
    row["pm2_5"]        = air_data["pm2_5"]
    row["pm10"]         = air_data["pm10"]
    row["co"]           = air_data["co"]
    row["no"]           = air_data["no"]
    row["no2"]          = air_data["no2"]
    row["o3"]           = air_data["o3"]
    row["so2"]          = air_data["so2"]
    row["nh3"]          = air_data["nh3"]
    row["temp"]         = weather_data["temp"]
    row["humidity"]     = weather_data["humidity"]
    row["pressure"]     = weather_data["pressure"]
    row["wind_speed"]   = weather_data["wind_speed"]
    row["wind_deg"]     = weather_data["wind_deg"]
    row["clouds"]       = weather_data["clouds"]
    row["visibility"]   = weather_data["visibility"]
    return row


def compute_aqi_change_rate(row):
    docs = db.collection("aqi_features") \
             .order_by("timestamp", direction=firestore.Query.DESCENDING) \
             .limit(2) \
             .stream()
    rows = [d.to_dict() for d in docs]
    if len(rows) >= 2:
        row["aqi_change_rate"] = row["aqi"] - rows[1]["aqi"]
    else:
        row["aqi_change_rate"] = 0.0
    return row


def save_to_firestore(row):
    doc_id = row["timestamp"].replace(":", "-")
    db.collection("aqi_features").document(doc_id).set(row)
    print(f"[{row['timestamp']}] Saved — AQI: {row['aqi']} ({row['aqi_category']}), PM2.5: {row['pm2_5']}, Temp: {row['temp']}°C")


def run_pipeline():
    print("── Running feature pipeline ──")
    air_data     = fetch_current_air_pollution()
    weather_data = fetch_current_weather()
    row          = compute_features(air_data, weather_data)
    row          = compute_aqi_change_rate(row)
    save_to_firestore(row)
    print("── Done ──")


if __name__ == "__main__":
    run_pipeline()