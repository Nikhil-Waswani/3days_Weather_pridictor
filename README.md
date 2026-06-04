# 🌤️ AQI Forecast — Khairpur, PK

A fully automated, serverless machine learning system that predicts the **Air Quality Index (AQI)** for Khairpur, Pakistan for the next **3 days** using real-time data from OpenWeather API.

🔗 **Live Dashboard:** [3daysweatherpridictor.streamlit.app](https://3daysweatherpridictor.streamlit.app)

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
feature_pipeline.py → Firebase Firestore
                              ↓ (every day via GitHub Actions)
                       train_model.py → model.pkl → GitHub
                                                        ↓
                                              dashboard.py → Streamlit Cloud
```

---

## 📁 Project Structure

```
3days_Weather_pridictor/
├── feature_pipeline.py    # Hourly data collection
├── backfill.py            # One-time historical data collection
├── train_model.py         # Daily model training (3 models)
├── dashboard.py           # Streamlit web dashboard
├── eda.py                 # Exploratory Data Analysis
├── shap_analysis.py       # SHAP feature importance
├── convert_aqi.py         # Convert AQI to EPA standard
├── historical_data.csv    # 8,520 rows of backfilled data
├── model.pkl              # Trained best model
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

## 🤖 ML Models

Three models are trained and compared. The best one (lowest RMSE) is saved automatically:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge Regression | 22.4852 | 12.9172 | 0.8596 |
| **Random Forest** ✅ | **2.4789** | **0.1433** | **0.9983** |
| XGBoost | 2.7995 | 0.6302 | 0.9978 |

**Random Forest** consistently wins with **99.83% accuracy** on unseen data.

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
Place your `firebase_key.json` (downloaded from Firebase Console) in the project folder.

### 5. Run the pipeline
```bash
# Collect live data
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
| `feature_pipeline.yml` | Every hour | Collect live data → Firestore |
| `train_model.yml` | Every day (midnight UTC) | Retrain models → Push model.pkl |

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
| Data source | OpenWeather API |
| Feature store | Firebase Firestore |
| ML models | scikit-learn, XGBoost |
| Feature importance | SHAP |
| EDA | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |

---

## 📄 License

This project was developed as part of the Pearls AQI Predictor internship project.