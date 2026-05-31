import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH   = "historical_data.csv"
OUTPUT_DIR = "eda_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

print("── Loading data ──")
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna()
print(f"Total rows: {len(df)}")

sns.set_theme(style="darkgrid")

# ── 1. AQI Distribution ───────────────────────────────────────────────────────
print("── Plot 1: AQI Distribution ──")
plt.figure(figsize=(8, 5))
sns.countplot(x="aqi", data=df, palette="viridis")
plt.title("AQI Distribution (1=Good to 5=Very Poor)")
plt.xlabel("AQI Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_aqi_distribution.png")
plt.close()

# ── 2. AQI by Month ───────────────────────────────────────────────────────────
print("── Plot 2: AQI by Month ──")
monthly = df.groupby("month")["aqi"].mean().reset_index()
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly["month_name"] = monthly["month"].map(month_names)

plt.figure(figsize=(10, 5))
sns.barplot(x="month_name", y="aqi", data=monthly, palette="coolwarm")
plt.title("Average AQI by Month (Seasonal Trends)")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_aqi_by_month.png")
plt.close()

# ── 3. AQI by Hour of Day ─────────────────────────────────────────────────────
print("── Plot 3: AQI by Hour ──")
hourly = df.groupby("hour")["aqi"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x="hour", y="aqi", data=hourly, marker="o", color="orange")
plt.title("Average AQI by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Average AQI")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_aqi_by_hour.png")
plt.close()

# ── 4. AQI by Day of Week ─────────────────────────────────────────────────────
print("── Plot 4: AQI by Day of Week ──")
day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
df["day_name"] = df["day"].map(day_names)
daily = df.groupby("day_name")["aqi"].mean().reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

plt.figure(figsize=(8, 5))
sns.barplot(x=daily.index, y=daily.values, palette="Blues_d")
plt.title("Average AQI by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_aqi_by_day.png")
plt.close()

# ── 5. Correlation Heatmap ────────────────────────────────────────────────────
print("── Plot 5: Correlation Heatmap ──")
corr_cols = ["aqi", "pm2_5", "pm10", "co", "no2", "o3",
             "temp", "humidity", "wind_speed", "pressure"]
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Between Features and AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_correlation_heatmap.png")
plt.close()

# ── 6. PM2.5 vs AQI ──────────────────────────────────────────────────────────
print("── Plot 6: PM2.5 vs AQI ──")
plt.figure(figsize=(8, 5))
sns.scatterplot(x="pm2_5", y="aqi", data=df.sample(1000), alpha=0.5, color="red")
plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/6_pm25_vs_aqi.png")
plt.close()

# ── 7. Temperature vs AQI ─────────────────────────────────────────────────────
print("── Plot 7: Temperature vs AQI ──")
plt.figure(figsize=(8, 5))
sns.scatterplot(x="temp", y="aqi", data=df.sample(1000), alpha=0.5, color="green")
plt.title("Temperature vs AQI")
plt.xlabel("Temperature (°C)")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/7_temp_vs_aqi.png")
plt.close()

# ── 8. AQI Trend Over Time ────────────────────────────────────────────────────
print("── Plot 8: AQI Trend Over Time ──")
df_sorted = df.sort_values("timestamp")
plt.figure(figsize=(14, 5))
plt.plot(df_sorted["timestamp"], df_sorted["aqi"], alpha=0.5, color="blue", linewidth=0.8)
plt.title("AQI Trend Over Time (1 Year)")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/8_aqi_trend_over_time.png")
plt.close()

print(f"\n── EDA Complete! All charts saved to '{OUTPUT_DIR}/' folder ──")
print("Charts generated:")
print("  1. AQI Distribution")
print("  2. AQI by Month (Seasonal Trends)")
print("  3. AQI by Hour of Day")
print("  4. AQI by Day of Week")
print("  5. Correlation Heatmap")
print("  6. PM2.5 vs AQI")
print("  7. Temperature vs AQI")
print("  8. AQI Trend Over Time")
