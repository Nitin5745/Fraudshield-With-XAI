import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import json
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)}  |  Fraud: {df['Class'].sum()}  |  Normal: {(df['Class']==0).sum()}")

# ── Scale Amount (drop Time — not useful for prediction) ───────────────
scaler = RobustScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
X = df.drop(['Time', 'Class'], axis=1)   # 29 features: V1-V28 + Amount
y = df['Class']

# ── Save artifacts needed by services.py ──────────────────────────────
np.save(os.path.join(MODEL_DIR, "mean_features.npy"), X.mean().values)

with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
    json.dump(list(X.columns), f)

print("Saved mean_features.npy and feature_columns.json")

# ── Train / test split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE — oversample fraud so the model learns it properly ───────────
print("Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE — Fraud: {y_train_res.sum()}  Normal: {(y_train_res==0).sum()}")

# ── Train ──────────────────────────────────────────────────────────────
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_res, y_train_res)

# ── Evaluate ───────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Evaluation ---")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ── Sanity check ───────────────────────────────────────────────────────
print("\n--- Sanity Check ---")
mean_vec  = X.mean().values.copy()

fraud_vec = mean_vec.copy()
fraud_vec[11] = -9.0   # V12
fraud_vec[13] = -9.0   # V14
fraud_vec[16] = -9.0   # V17
fraud_vec[28] = 7.0    # high scaled amount

fraud_df  = pd.DataFrame([fraud_vec], columns=X.columns)
normal_df = pd.DataFrame([mean_vec],  columns=X.columns)

fraud_proba  = model.predict_proba(fraud_df)[0][1]
normal_proba = model.predict_proba(normal_df)[0][1]

print(f"Fraud-like probability:  {fraud_proba:.4f}  → {'FRAUD ✓' if fraud_proba > 0.5 else 'MISSED ✗'}")
print(f"Normal-like probability: {normal_proba:.4f}  → {'OK ✓' if normal_proba <= 0.5 else 'FALSE POSITIVE ✗'}")

# ── Save model ─────────────────────────────────────────────────────────
joblib.dump(model,  os.path.join(MODEL_DIR, "fraud_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
print("\nModel and scaler saved successfully. ✓")