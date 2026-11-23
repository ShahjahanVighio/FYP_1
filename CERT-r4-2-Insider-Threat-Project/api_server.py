"""
CERT r4.2 Insider Threat Detection - Live Risk Inference API
-------------------------------------------------------------
FastAPI micro-server for real-time insider threat risk scoring using
our trained Random Forest, XGBoost, CatBoost, and Stacking Meta-Learner.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostClassifier

app = FastAPI(
    title="CERT r4.2 Insider Threat Inference API",
    description="Real-time multi-model risk assessment microservice",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load metadata
with open("prepared_data/selected_features.json", "r") as f:
    SELECTED_FEATURES = json.load(f)

# Global model containers
MODELS = {}

@app.on_event("startup")
def load_models():
    print("Loading models and preprocessors for live inference...")
    MODELS["rf"] = joblib.load("models/rf_best.joblib")
    MODELS["xgb"] = joblib.load("models/xgb_best.joblib")
    
    try:
        MODELS["cb"] = joblib.load("models/cb_best.joblib")
    except Exception:
        cb = CatBoostClassifier()
        cb.load_model("models/cb_best.cbm")
        MODELS["cb"] = cb
        
    MODELS["ensemble"] = joblib.load("models/stacking_ensemble.joblib")
    
    # Load default feature medians for padding
    X_train_raw = pd.read_parquet("prepared_data/X_train_raw.parquet")
    MODELS["defaults"] = X_train_raw[SELECTED_FEATURES].median().to_dict()
    print("Models and feature medians loaded successfully!")


class SessionPayload(BaseModel):
    features: Dict[str, float]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "dataset": "CMU CERT r4.2",
        "models_loaded": list(MODELS.keys())
    }


@app.post("/predict")
def predict_session_risk(payload: SessionPayload):
    if "rf" not in MODELS:
        load_models()
        
    user_feats = payload.features
    
    # Assemble complete feature row
    row_dict = {}
    for feat in SELECTED_FEATURES:
        row_dict[feat] = user_feats.get(feat, MODELS["defaults"].get(feat, 0.0))
        
    df_single = pd.DataFrame([row_dict])
    
    # Predict with individual base models
    p_rf = float(MODELS["rf"].predict_proba(df_single)[0, 1])
    p_xgb = float(MODELS["xgb"].predict_proba(df_single)[0, 1])
    p_cb = float(MODELS["cb"].predict_proba(df_single)[0, 1])
    
    # Stacking ensemble soft voting
    w = MODELS["ensemble"]["winner_info"]["weights"]
    p_ensemble = float(w["rf"] * p_rf + w["xgb"] * p_xgb + w["cb"] * p_cb)
    opt_threshold = float(MODELS["ensemble"]["winner_info"]["opt_thresh"])
    
    # Verdict logic
    is_threat = bool(p_ensemble >= opt_threshold)
    
    if p_ensemble >= 0.70:
        risk_level = "CRITICAL"
        risk_color = "#EF4444" # red
        recommendation = "Immediate SOC intervention: Restrict workstation access and isolate active session credentials."
    elif p_ensemble >= opt_threshold:
        risk_level = "HIGH"
        risk_color = "#F97316" # orange
        recommendation = "Optimal threshold exceeded (T* = 0.34). Flag session for tier-2 security analyst audit."
    elif p_ensemble >= 0.15:
        risk_level = "ELEVATED"
        risk_color = "#F59E0B" # amber
        recommendation = "Minor behavioral drift detected. Continue passive endpoint monitoring."
    else:
        risk_level = "LOW / BENIGN"
        risk_color = "#10B981" # emerald
        recommendation = "Normal user behavior consistent with baseline organizational profile."
        
    # Contributing factors
    key_signals = []
    if row_dict.get("n_usb", 0) > 0:
        key_signals.append(f"Removable media connected ({int(row_dict['n_usb'])} USB events)")
    if row_dict.get("http_n_leakf", 0) > 0:
        key_signals.append(f"Cloud storage / data leak browsing ({int(row_dict['http_n_leakf'])} leak requests)")
    if row_dict.get("http_n_jobf", 0) > 0:
        key_signals.append(f"Job seeking activity ({int(row_dict['http_n_jobf'])} job searches)")
    if row_dict.get("isworkhour", 1) == 0:
        key_signals.append("Off-hours or weekend activity")
    if row_dict.get("n_file", 0) > 10:
        key_signals.append(f"Elevated file operations ({int(row_dict['n_file'])} files modified/copied)")
        
    return {
        "ensemble_threat_probability": round(p_ensemble, 4),
        "optimal_threshold": opt_threshold,
        "is_threat": is_threat,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "base_model_probabilities": {
            "random_forest": round(p_rf, 4),
            "xgboost": round(p_xgb, 4),
            "catboost": round(p_cb, 4)
        },
        "model_weights": {
            "random_forest": w["rf"],
            "xgboost": w["xgb"],
            "catboost": w["cb"]
        },
        "key_signals": key_signals if key_signals else ["Standard enterprise activity stream"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
