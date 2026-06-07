# 🌤️ AQI Forecast — Khairpur, PK

A fully automated, serverless machine learning system that predicts the **Air Quality Index (AQI)** for Khairpur, Pakistan for the next **3 days** using real-time data from OpenWeather API.

🔗 **Live Dashboard:** [https://3daysweatherpridictor.streamlit.app/](https://3daysweatherpridictor.streamlit.app/)

---

## 📌 Project Overview

This project builds an end-to-end ML pipeline that:
- Collects live weather and pollutant data **every hour** automatically
- Stores data in **Firebase Firestore** (feature store)
- Trains **3 ML models** daily and saves the best one
- Displays **3-day AQI forecasts** on a live Streamlit dashboard

The AQI values follow the **standard US EPA scale (0-500)** calculated from PM2.5 concentrations.

---

## 🏗️ Architecture

```
OpenWeather API
      ↓ (every hour via GitHub Actions)
feature_pipeline.py → Firebase Firestore (aqi_features)
                              ↓ (every day via GitHub Actions)
                       train_model.py → model.pkl → GitHub
                       ↑ also reads from aqi_historical collection
                                                        ↓
                                              dashboard.py → Streamlit Cloud
```

---

## 📁 Project Structure

```
3days_Weather_pridictor/
├── feature_pipeline.py    # Hourly data collection → Firestore
├── backfill.py            # One-time historical data collection
├── upload_historical.py   # Upload historical CSV to Firestore
├── train_model.py         # Daily model training (3 models)
├── dashboard.py           # Streamlit web dashboard
├── eda.py                 # Exploratory Data Analysis (8 charts)
├── shap_analysis.py       # SHAP feature importance (3 charts)
├── model.pkl              # Trained best model (Random Forest)
├── scaler.pkl             # Feature scaler
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignores .env and firebase_key.json
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml  # Runs every hour
│       └── train_model.yml       # Runs every day
├── eda_charts/            # 8 EDA charts
└── shap_charts/           # 3 SHAP charts
```

---

## 🗄️ Firebase Firestore Collections

| Collection | Purpose | Rows |
|------------|---------|------|
| `aqi_features` | Live hourly data (grows automatically) | ~50+ and growing |
| `aqi_historical` | 1 year backfilled historical data | 8,520 |

---

## 🤖 ML Models

Three models are trained and compared. The best one (lowest RMSE) is saved automatically:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge Regression | 21.2154 | 12.3999 | 0.8772 |
| **Random Forest** ✅ | **2.2757** | **0.1027** | **0.9986** |
| XGBoost | 6.2547 | 0.8048 | 0.9893 |

**Random Forest** consistently wins with **99.86% accuracy** on 1,704 unseen test rows.

---

## 📊 AQI Scale (US EPA Standard)

| AQI Range | Category |
|-----------|----------|
| 0 - 50 | ✅ Good |
| 51 - 100 | 🟡 Moderate |
| 101 - 150 | 🟠 Unhealthy for Sensitive Groups |
| 151 - 200 | 🔴 Unhealthy |
| 201 - 300 | 🟣 Very Unhealthy |
| 301 - 500 | ⚫ Hazardous |

---

## 🔑 SHAP Feature Importance

| Feature | Importance |
|---------|-----------|
| pm2_5 | 46.60 (dominant) |
| aqi_change_rate | 2.00 |
| pm10 | 0.07 |
| Others | < 0.01 |

PM2.5 is the dominant predictor, which aligns with the EPA AQI formula.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Nikhil-Waswani/3days_Weather_pridictor.git
cd 3days_Weather_pridictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```
OPENWEATHER_API_KEY=your_api_key
LAT=27.534138
LON=68.757283
FIREBASE_KEY_PATH=firebase_key.json
```

### 4. Add Firebase key
Place your `firebase_key.json` (downloaded from Firebase Console → Project Settings → Service Accounts) in the project folder.

### 5. Run the pipeline
```bash
# Collect live data once
python feature_pipeline.py

# Train models
python train_model.py

# Launch dashboard
streamlit run dashboard.py
```

---

## 🔄 Automated Pipelines

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `feature_pipeline.yml` | Every hour | Collect live data → Firestore `aqi_features` |
| `train_model.yml` | Every day (midnight UTC) | Retrain 3 models → Push best `model.pkl` |

### GitHub Secrets Required
- `OPENWEATHER_API_KEY`
- `LAT`
- `LON`
- `FIREBASE_KEY_JSON`

---

## 📈 EDA Charts

8 charts generated in `eda_charts/`:
1. AQI Distribution
2. AQI by Month (Seasonal Trends)
3. AQI by Hour of Day
4. AQI by Day of Week
5. Correlation Heatmap
6. PM2.5 vs AQI
7. Temperature vs AQI
8. AQI Trend Over Time

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Data source | OpenWeather API (free tier) |
| Feature store | Firebase Firestore (Spark plan, free) |
| Historical store | Firebase Firestore `aqi_historical` |
| ML models | scikit-learn (Ridge, Random Forest), XGBoost |
| Feature importance | SHAP |
| EDA | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud (free) |

---

## 📄 License

This project was developed as part of the Pearls AQI Predictor internship project.