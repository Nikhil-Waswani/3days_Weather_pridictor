import pandas as pd

def calculate_aqi(pm2_5):
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
    return 500

df = pd.read_csv("historical_data.csv")
df["aqi"] = df["pm2_5"].apply(calculate_aqi)
df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
df.to_csv("historical_data.csv", index=False)
print(f"Done! Sample AQI values: {df['aqi'].head(10).tolist()}")