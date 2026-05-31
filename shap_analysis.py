import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "shap_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

print("── Loading model and data ──")
model_data = pickle.load(open('model.pkl', 'rb'))
model      = model_data['model']
feat_cols  = model_data['feature_cols']

df = pd.read_csv('historical_data.csv').dropna()
X  = df[feat_cols].sample(500, random_state=42)   # use 500 rows for speed

print("── Computing SHAP values ──")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ── Plot 1: Summary Bar Plot ──────────────────────────────────────────────────
print("── Plot 1: Feature Importance Bar Chart ──")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_shap_feature_importance.png", bbox_inches='tight')
plt.close()

# ── Plot 2: Summary Dot Plot ──────────────────────────────────────────────────
print("── Plot 2: SHAP Summary Dot Plot ──")
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_shap_summary.png", bbox_inches='tight')
plt.close()

# ── Plot 3: Top feature dependence plot ───────────────────────────────────────
print("── Plot 3: PM2.5 Dependence Plot ──")
plt.figure()
shap.dependence_plot("pm2_5", shap_values, X, show=False)
plt.title("SHAP Dependence Plot — PM2.5")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_shap_pm25_dependence.png", bbox_inches='tight')
plt.close()

# ── Print top features ────────────────────────────────────────────────────────
print("\n── Top Features by SHAP Importance ──")
importance = pd.DataFrame({
    'feature':    feat_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print(importance.to_string(index=False))

print(f"\n── SHAP Analysis Complete! Charts saved to '{OUTPUT_DIR}/' ──")
print("Charts generated:")
print("  1. Feature Importance Bar Chart")
print("  2. SHAP Summary Dot Plot")
print("  3. PM2.5 Dependence Plot")