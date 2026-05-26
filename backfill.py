import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"D:\Projects\Weather_pridictor\.env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY  = os.getenv("OPENWEATHER_API_KEY")
LAT      = float(os.getenv("LAT"))
LON      = float(os.getenv("LON"))
CSV_PATH = "historical_data.csv"

# Backfill last 1 year
END_DATE   = datetime.now(tz=timezone.utc)
START_DATE = END_DATE - timedelta(days=365)
# ─────────────────────────────────────────────────────────────────────────────


def fetch_historical_air_pollution(start_ts, end_ts):
    """
    Fetch historical AQI + pollutants for a time range.
    OpenWeather returns hourly data for the given range.
    """
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat":   LAT,
        "lon":   LON,
        "start": int(start_ts),
        "end":   int(end_ts),
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["list"]


def fetch_historical_weather(dt_timestamp):
    """
    Fetch historical weather for a specific timestamp.
    Uses OpenWeather One Call API timemachine endpoint.
    """
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {
        "lat":   LAT,
        "lon":   LON,
        "dt":    int(dt_timestamp),
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        current = data.get("data", [{}])[0]
        return {
            "temp":       current.get("temp", 0),
            "humidity":   current.get("humidity", 0),
            "pressure":   current.get("pressure", 0),
            "wind_speed": current.get("wind_speed", 0),
            "wind_deg":   current.get("wind_deg", 0),
            "clouds":     current.get("clouds", 0),
            "visibility": current.get("visibility", 10000),
        }
    return {
        "temp": 0, "humidity": 0, "pressure": 0,
        "wind_speed": 0, "wind_deg": 0, "clouds": 0, "visibility": 10000
    }


def compute_features(entry, weather):
    dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
    return {
        "timestamp":  dt.isoformat(),
        "hour":       dt.hour,
        "day":        dt.weekday(),
        "month":      dt.month,
        "aqi":        entry["main"]["aqi"],
        "pm2_5":      entry["components"]["pm2_5"],
        "pm10":       entry["components"]["pm10"],
        "co":         entry["components"]["co"],
        "no":         entry["components"]["no"],
        "no2":        entry["components"]["no2"],
        "o3":         entry["components"]["o3"],
        "so2":        entry["components"]["so2"],
        "nh3":        entry["components"]["nh3"],
        "temp":       weather["temp"],
        "humidity":   weather["humidity"],
        "pressure":   weather["pressure"],
        "wind_speed": weather["wind_speed"],
        "wind_deg":   weather["wind_deg"],
        "clouds":     weather["clouds"],
        "visibility": weather["visibility"],
    }


def run_backfill():
    print("── Starting backfill ──")
    print(f"From: {START_DATE.date()} → To: {END_DATE.date()}")

    all_rows = []

    # Fetch air pollution in 30-day chunks (API limit)
    chunk_start = START_DATE
    while chunk_start < END_DATE:
        chunk_end = min(chunk_start + timedelta(days=30), END_DATE)

        print(f"Fetching air pollution: {chunk_start.date()} → {chunk_end.date()}")
        entries = fetch_historical_air_pollution(
            chunk_start.timestamp(),
            chunk_end.timestamp()
        )

        for entry in entries:
            # Fetch weather for every 6th hour only to avoid rate limits
            if datetime.fromtimestamp(entry["dt"], tz=timezone.utc).hour % 6 == 0:
                weather = fetch_historical_weather(entry["dt"])
                time.sleep(0.5)   # avoid hitting rate limit
            else:
                weather = {
                    "temp": 0, "humidity": 0, "pressure": 0,
                    "wind_speed": 0, "wind_deg": 0, "clouds": 0, "visibility": 10000
                }

            row = compute_features(entry, weather)
            all_rows.append(row)

        chunk_start = chunk_end
        time.sleep(1)

    # Save to CSV
    df = pd.DataFrame(all_rows)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df.to_csv(CSV_PATH, index=False)

    print(f"── Done! {len(df)} rows saved to {CSV_PATH} ──")


if __name__ == "__main__":
    run_backfill()
