import pickle, pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

m      = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
df     = pd.read_csv('historical_data.csv').dropna()
sample = df.sample(50, random_state=42)
X      = sample[m['feature_cols']]
y      = sample['aqi'].values

if m['scaled']:
    X = scaler.transform(X)

pred = np.round(m['model'].predict(X)).astype(int)

print(f'RMSE: {np.sqrt(mean_squared_error(y,pred)):.4f}')
print(f'MAE : {mean_absolute_error(y,pred):.4f}')
print(f'R2  : {r2_score(y,pred):.4f}')
print()

for i,(a,p) in enumerate(zip(y,pred)):
    diff  = abs(a-p)
    match = "PASS" if diff <= 5 else "FAIL"
    print(f'Row {i+1:2}: Actual={int(a):4} Predicted={p:4} Diff={diff:3} {match}')