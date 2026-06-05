import pandas as pd
import os
import time
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv(dotenv_path=r"D:\Projects\Weather_pridictor\.env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH     = "historical_data.csv"
COLLECTION   = "aqi_historical"        # separate collection for historical data
BATCH_SIZE   = 400                     # Firestore batch limit is 500
FIREBASE_KEY = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
# ─────────────────────────────────────────────────────────────────────────────

# ── INIT FIREBASE ─────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def upload_historical():
    print("── Loading historical data ──")
    df = pd.read_csv(CSV_PATH)
    df = df.fillna(0)
    print(f"Total rows to upload: {len(df)}")

    total    = len(df)
    uploaded = 0

    # Upload in batches
    for start in range(0, total, BATCH_SIZE):
        batch = db.batch()
        chunk = df.iloc[start:start + BATCH_SIZE]

        for _, row in chunk.iterrows():
            doc_id  = str(row["timestamp"]).replace(":", "-").replace(" ", "T")
            doc_ref = db.collection(COLLECTION).document(doc_id)
            batch.set(doc_ref, row.to_dict())

        batch.commit()
        uploaded += len(chunk)
        print(f"Uploaded {uploaded}/{total} rows...")
        time.sleep(0.5)   # avoid rate limiting

    print(f"\n── Done! {uploaded} rows uploaded to Firestore collection '{COLLECTION}' ──")

if __name__ == "__main__":
    upload_historical()
