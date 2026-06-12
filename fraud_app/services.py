import numpy as np
import pandas as pd
import joblib
import json
import shap
import threading
import logging
import os

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────

_MODEL           = None
_SCALER          = None
_MEAN_FEATURES   = None
_FEATURE_COLUMNS = None


def _load_artifacts():
    global _MODEL, _SCALER, _MEAN_FEATURES, _FEATURE_COLUMNS

    if _MODEL is not None:
        return

    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, "ml_core", "models")

    _MODEL  = joblib.load(os.path.join(model_dir, "fraud_model.joblib"))
    _SCALER = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    mean_path = os.path.join(model_dir, "mean_features.npy")
    _MEAN_FEATURES = np.load(mean_path) if os.path.exists(mean_path) else np.zeros(29)

    col_path = os.path.join(model_dir, "feature_columns.json")
    with open(col_path) as f:
        _FEATURE_COLUMNS = json.load(f)

    logger.info("Fraud detection artifacts loaded from %s", model_dir)


# ─────────────────────────────────────────────
# Main prediction entry-point
# ─────────────────────────────────────────────

def predict_fraud(data):
    """
    Accepts a dict with:
        amount    – transaction amount (float)
        V1…V28   – PCA features (optional, defaults to training mean)
        vpn       – bool
        midnight  – bool
        highrisk  – bool
        country   – 'india' | 'international'
        device    – 'known' | 'new'

    Returns:
        is_fraud   – bool
        score      – fraud probability from the model (0.0–1.0)
        risk_score – combined score (0–100)
        features   – list[float]
    """

    _load_artifacts()

    amount = float(data.get('amount', 0))

    # ── 1. Build feature vector from training mean as baseline ─────────
    features = _MEAN_FEATURES.copy()

    for i in range(1, 29):
        key = f'V{i}'
        if key in data:
            features[i - 1] = float(data[key])

    # ── 2. Scale Amount the same way as training ───────────────────────
    amount_df     = pd.DataFrame([[amount]], columns=['Amount'])
    scaled_amount = _SCALER.transform(amount_df)[0][0]
    features[28]  = scaled_amount

    # ── 3. Random Forest fraud probability ────────────────────────────
    # predict_proba returns [prob_normal, prob_fraud]
    features_df  = pd.DataFrame([features], columns=_FEATURE_COLUMNS)
    fraud_proba  = float(_MODEL.predict_proba(features_df)[0][1])
    # Convert 0.0–1.0 probability to 0–70 ML points
    ml_pts = min(70.0, fraud_proba * 70.0)

    # ── 4. Rule engine (behavioural signals) ──────────────────────────
    rule_score = 0

    vpn      = bool(data.get('vpn',      False))
    midnight = bool(data.get('midnight', False))
    highrisk = bool(data.get('highrisk', False))
    intl     = data.get('country', 'india') == 'international'
    new_dev  = data.get('device',  'known') == 'new'

    if vpn:            rule_score += 20
    if midnight:       rule_score += 10
    if highrisk:       rule_score += 15
    if intl:           rule_score += 10
    if new_dev:        rule_score += 10
    if amount > 50000: rule_score += 25
    if amount > 10000: rule_score += 10

    # ── 5. Combined decision ───────────────────────────────────────────
    # ML model contributes up to 70 pts, rules up to 30 pts
    combined = ml_pts + rule_score * 0.60
    is_fraud = combined >= 50

    print(
        f"DEBUG | amount={amount:.2f} | fraud_proba={fraud_proba:.4f} | "
        f"ml_pts={ml_pts:.1f} | rule={rule_score} | combined={combined:.1f} | fraud={is_fraud}"
    )

    return {
        'is_fraud':   is_fraud,
        'score':      round(fraud_proba, 4),
        'risk_score': round(combined, 1),
        'features':   features.tolist(),
    }


# ─────────────────────────────────────────────
# SHAP  (async, best-effort)
# ─────────────────────────────────────────────

def generate_shap_async(features, audit_log_id):
    try:
        from .models import SecurityAuditLog
        _load_artifacts()

        log = SecurityAuditLog.objects.get(id=audit_log_id)

        features_df = pd.DataFrame([features], columns=_FEATURE_COLUMNS)
        explainer   = shap.TreeExplainer(_MODEL)

        # check_additivity=False skips expensive verification — much faster
        shap_values = explainer.shap_values(features_df, check_additivity=False)

        print(f"SHAP type: {type(shap_values)}, shape: {getattr(shap_values, 'shape', len(shap_values))}")

        arr = np.array(shap_values)

        # Handle every return shape shap may produce:
        #   list of 2 arrays  → [class_0, class_1]  (old API)
        #   3-D array          → (n_samples, n_features, n_classes)
        #   2-D array          → (n_samples, n_features)  — class 1 already selected
        #   1-D array          → (n_features,)
        if isinstance(shap_values, list):
            vals = np.array(shap_values[1]).flatten().tolist()
        elif arr.ndim == 3:
            vals = arr[0, :, 1].tolist()      # class 1 (fraud)
        elif arr.ndim == 2:
            vals = arr[0].tolist()
        else:
            vals = arr.flatten().tolist()

        print(f"SHAP vals computed: {len(vals)} features")

        log.shap_values = [round(float(v), 4) for v in vals]
        log.save()
        print(f"SHAP saved for log {audit_log_id}")

    except Exception as e:
        import traceback
        print(f"SHAP FAILED: {e}")
        traceback.print_exc()
def trigger_shap_background(features, audit_log_id):
    t = threading.Thread(target=generate_shap_async, args=(features, audit_log_id))
    t.daemon = True
    t.start()