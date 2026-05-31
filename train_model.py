import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pickle
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv(dotenv_path=r"D:\Projects\Weather_pridictor\.env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH      = "historical_data.csv" 
MODEL_PATH    = "model.pkl"
SCALER_PATH   = "scaler.pkl"
FIREBASE_KEY  = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
# ─────────────────────────────────────────────────────────────────────────────

# ── INIT FIREBASE ─────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()
# ─────────────────────────────────────────────────────────────────────────────


def load_historical_data():
    print("── Loading historical data from CSV ──")
    df = pd.read_csv(CSV_PATH)
    print(f"Historical rows: {len(df)}")
    return df


def fetch_firestore_data():
    print("── Fetching live data from Firestore ──")
    docs = db.collection("aqi_features") \
             .order_by("timestamp") \
             .stream()
    rows = [d.to_dict() for d in docs]
    df = pd.DataFrame(rows)
    print(f"Firestore rows: {len(df)}")
    return df


def combine_data(df_hist, df_live):
    print("── Combining historical + live data ──")
    
    # Keep only columns that exist in historical data
    common_cols = [c for c in df_hist.columns if c in df_live.columns]
    
    df_hist_clean = df_hist[common_cols]
    df_live_clean = df_live[common_cols] if len(df_live) > 0 else pd.DataFrame(columns=common_cols)
    
    df = pd.concat([df_hist_clean, df_live_clean], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Total combined rows: {len(df)}")
    return df


def prepare_features(df):
    print("── Preparing features ──")
    df = df.dropna()
    # Drop non-numeric columns
    df = df.drop(columns=["aqi_category"], errors="ignore")

    feature_cols = [
        "hour", "day", "month",
        "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3",
        "temp", "humidity", "pressure", "wind_speed", "wind_deg",
        "clouds", "visibility", "aqi_change_rate"
    ]
    target_col   = "aqi"
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Training samples: {len(X)}")
    return X, y, feature_cols


def evaluate_model(name, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n── {name} Results ──")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    return rmse, mae, r2


def train(X_train, y_train, X_test, y_test, scaler):
    results = {}

    # ── Model 1: Ridge Regression ─────────────────────────────────────────────
    print("\nTraining Ridge Regression...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    rmse, mae, r2 = evaluate_model("Ridge Regression", y_test, y_pred_ridge)
    results["Ridge"] = {"model": ridge, "rmse": rmse, "mae": mae, "r2": r2, "scaled": True}

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse, mae, r2 = evaluate_model("Random Forest", y_test, y_pred_rf)
    results["RandomForest"] = {"model": rf, "rmse": rmse, "mae": mae, "r2": r2, "scaled": False}

    # ── Model 3: XGBoost ──────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    rmse, mae, r2 = evaluate_model("XGBoost", y_test, y_pred_xgb)
    results["XGBoost"] = {"model": xgb, "rmse": rmse, "mae": mae, "r2": r2, "scaled": False}

    return results


def save_best_model(results, scaler, feature_cols):
    print("\n── Comparing all 3 models ──")
    for name, res in results.items():
        print(f"{name:15} → RMSE: {res['rmse']:.4f} | MAE: {res['mae']:.4f} | R²: {res['r2']:.4f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best      = results[best_name]
    print(f"\n✅ Best model: {best_name} (RMSE: {best['rmse']:.4f}, R²: {best['r2']:.4f})")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":        best["model"],
            "model_name":   best_name,
            "scaled":       best["scaled"],
            "feature_cols": feature_cols,
            "rmse":         best["rmse"],
            "mae":          best["mae"],
            "r2":           best["r2"],
        }, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


def run_training():
    print("══ Starting Training Pipeline ══")

    df_hist          = load_historical_data()
    df_live          = fetch_firestore_data()
    df               = combine_data(df_hist, df_live)
    X, y, feat_cols  = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    scaler  = StandardScaler()
    results = train(X_train, y_train, X_test, y_test, scaler)
    save_best_model(results, scaler, feat_cols)

    print("\n══ Training Complete ══")


if __name__ == "__main__":
    run_training()