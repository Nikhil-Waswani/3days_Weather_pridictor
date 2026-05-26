import requests
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY          = os.getenv("OPENWEATHER_API_KEY")
LAT              = float(os.getenv("LAT"))
LON              = float(os.getenv("LON"))
FIREBASE_KEY     = os.getenv("FIREBASE_KEY_PATH")
# ─────────────────────────────────────────────────────────────────────────────

# ── INIT FIREBASE ─────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ─────────────────────────────────────────────────────────────────────────────


def fetch_current_air_pollution():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    entry = data["list"][0]
    return {
        "dt":    entry["dt"],
        "aqi":   entry["main"]["aqi"],
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
    row = {}
    row["timestamp"]  = dt.isoformat()
    row["hour"]       = dt.hour
    row["day"]        = dt.weekday()
    row["month"]      = dt.month
    row["aqi"]        = air_data["aqi"]
    row["pm2_5"]      = air_data["pm2_5"]
    row["pm10"]       = air_data["pm10"]
    row["co"]         = air_data["co"]
    row["no"]         = air_data["no"]
    row["no2"]        = air_data["no2"]
    row["o3"]         = air_data["o3"]
    row["so2"]        = air_data["so2"]
    row["nh3"]        = air_data["nh3"]
    row["temp"]       = weather_data["temp"]
    row["humidity"]   = weather_data["humidity"]
    row["pressure"]   = weather_data["pressure"]
    row["wind_speed"] = weather_data["wind_speed"]
    row["wind_deg"]   = weather_data["wind_deg"]
    row["clouds"]     = weather_data["clouds"]
    row["visibility"] = weather_data["visibility"]
    return row


def compute_aqi_change_rate(row):
    """
    Fetch last saved AQI from Firestore and compute change rate.
    """
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
    """
    Save one row to Firestore under collection 'aqi_features'.
    Each document is named by its timestamp.
    """
    doc_id = row["timestamp"].replace(":", "-")   # Firestore doesn't allow : in doc IDs
    db.collection("aqi_features").document(doc_id).set(row)
    print(f"[{row['timestamp']}] Saved to Firestore — AQI: {row['aqi']}, PM2.5: {row['pm2_5']}, Temp: {row['temp']}°C")


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