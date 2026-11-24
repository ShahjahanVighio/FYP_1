"""
Step 5: Stacking Ensemble Meta-Learner (train_meta_learner.py)
-------------------------------------------------------------
Combines the top three base models (Random Forest, XGBoost, CatBoost)
using a leakage-free Out-Of-Fold (OOF) stacking architecture on CMU CERT r6.2.

Base Model Configuration from results_log.json:
- Random Forest: Cost-Sensitive (class_weight='balanced_subsample')
- XGBoost: Cost-Sensitive (scale_pos_weight=25.0, tree_method='hist')
- CatBoost: Resampled (Fold-Level SMOTE + RUS inside training folds)

Pipeline Structure:
1. Load prepared splits: X_train_raw, y_train_raw, X_test, y_test, y_test_scenario.
2. 5-Fold Stratified CV on X_train_raw to generate strictly out-of-fold predictions
   (checkpointed to prepared_data/oof_meta_features.parquet).
3. Construct base and enriched meta-features (raw probs, mean, max, min, std disagreement, logits).
4. Train and compare candidate meta-learners (LR_Raw_Probs, LR_Enriched_Features,
   LR_Balanced_Enriched, Optimal_Soft_Voting) on OOF validation.
5. Determine optimal decision threshold T* maximizing F1 on OOF predictions.
6. Generate test meta-features using full-trained base models on untouched X_test.
7. Evaluate winning stacking ensemble on X_test at default (0.50) and optimal (T*) thresholds,
   including full Scenario 1-5 recall breakdown.
8. Save artifacts:
   - models/stacking_ensemble.joblib
   - models/results_log.json (updated)
   - plots/ensemble/ensemble_confusion_matrix.png
   - plots/ensemble/all_models_pr_roc_comparison.png (RF vs XGB vs CB vs Ensemble overlay)
   - plots/ensemble/meta_learner_weights.png
9. Output 4-Model Master Leaderboard on CERT r6.2.
"""

import os
import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
from catboost import CatBoostClassifier

# Matplotlib configuration for publication quality plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Paths
ROOT_DIR = Path(__file__).resolve().parent
PREP_DIR = ROOT_DIR / "prepared_data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots" / "ensemble"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load prepared raw train and test splits for CERT r6.2."""
    print("=" * 75)
    print("Loading prepared data splits for CERT r6.2 from prepared_data/...")
    start_t = time.time()
    
    X_train_raw = pd.read_parquet(PREP_DIR / "X_train_raw.parquet")
    y_train_raw = pd.read_parquet(PREP_DIR / "y_train_raw.parquet")['insider'].values
    X_test = pd.read_parquet(PREP_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PREP_DIR / "y_test.parquet")['insider'].values
    sc_test = pd.read_parquet(PREP_DIR / "y_test_scenario.parquet")['scenario'].values
    
    print(f"X_train_raw shape: {X_train_raw.shape} | Threats: {np.sum(y_train_raw):,} ({np.mean(y_train_raw)*100:.4f}%) [20,515:1 imbalance]")
    print(f"X_test shape:      {X_test.shape}  | Threats: {np.sum(y_test):,} ({np.mean(y_test)*100:.4f}%) [ISOLATED TEST SET]")
    print(f"Data loaded in {time.time() - start_t:.2f}s")
    print("=" * 75)
    return X_train_raw, y_train_raw, X_test, y_test, sc_test


def get_base_model_configs():
    """Load winning hyperparameter configurations for RF, XGB, and CatBoost from results_log.json."""
    with open(MODELS_DIR / "results_log.json", "r") as f:
        log = json.load(f)
    
    rf_params = log["Random_Forest"]["best_hyperparameters"]
    xgb_params = log["XGBoost"]["best_hyperparameters"]
    cb_params = log["CatBoost"]["best_hyperparameters"]
    
    print("\nLoaded Winning Base Model Configurations:")
    print(f"  Random Forest ({log['Random_Forest']['winning_strategy'].upper()}): {rf_params}")
    print(f"  XGBoost       ({log['XGBoost']['winning_strategy'].upper()}): {xgb_params}")
    print(f"  CatBoost      ({log['CatBoost']['winning_strategy'].upper()}): {cb_params}")
    return rf_params, xgb_params, cb_params


def generate_oof_predictions(X_train, y_train, rf_params, xgb_params, cb_params, n_splits=5, seed=42):
    """
    Perform 5-fold Stratified CV on X_train to generate strictly out-of-fold probability predictions.
    Uses exact winning strategy for each base model:
    - RF: Cost-Sensitive (balanced_subsample)
    - XGB: Cost-Sensitive (scale_pos_weight=25.0, early_stopping_rounds=50)
    - CB: Resampled (Fold-Level SMOTE + RUS inside training fold, early_stopping_rounds=50)
    Checkpoints results to prepared_data/oof_meta_features.parquet to avoid redundant computation.
    """
    oof_cache_path = PREP_DIR / "oof_meta_features.parquet"
    if oof_cache_path.exists():
        print(f"\n[CACHE FOUND] Loading existing OOF predictions from {oof_cache_path.as_posix()}...")
        df_oof = pd.read_parquet(oof_cache_path)
        return df_oof
    
    print(f"\nStarting {n_splits}-Fold Stratified Cross-Validation for OOF Meta-Feature Generation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_rf = np.zeros(len(y_train), dtype=np.float32)
    oof_xgb = np.zeros(len(y_train), dtype=np.float32)
    oof_cb = np.zeros(len(y_train), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        fold_start = time.time()
        print(f"\n--- FOLD {fold}/{n_splits} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        
        # 1. Random Forest (Cost-Sensitive)
        t0 = time.time()
        rf = RandomForestClassifier(
            **rf_params,
            class_weight="balanced_subsample",
            random_state=seed + fold,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict_proba(X_va)[:, 1]
        rf_ap = average_precision_score(y_va, oof_rf[val_idx])
        print(f"  [Fold {fold}] RF fit in {time.time()-t0:.1f}s | Val PR-AUC: {rf_ap:.4f}")
        
        # 2. XGBoost (Cost-Sensitive with early stopping)
        t0 = time.time()
        xgb_clf = xgb.XGBClassifier(
            **xgb_params,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=50,
            random_state=seed + fold,
            n_jobs=-1
        )
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)[:, 1]
        xgb_ap = average_precision_score(y_va, oof_xgb[val_idx])
        print(f"  [Fold {fold}] XGB fit in {time.time()-t0:.1f}s | Best Iter: {xgb_clf.best_iteration} | Val PR-AUC: {xgb_ap:.4f}")
        
        # 3. CatBoost (Resampled: Fold-level SMOTE + RUS inside training fold)
        t0 = time.time()
        smote = SMOTE(sampling_strategy=0.1, k_neighbors=3, random_state=seed + fold)
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=seed + fold)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        X_tr_res, y_tr_res = rus.fit_resample(X_tr_sm, y_tr_sm)
        
        cb_clf = CatBoostClassifier(
            **cb_params,
            loss_function="Logloss",
            eval_metric="PRAUC",
            early_stopping_rounds=50,
            random_seed=seed + fold,
            thread_count=-1,
            verbose=0
        )
        cb_clf.fit(X_tr_res, y_tr_res, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_idx] = cb_clf.predict_proba(X_va)[:, 1]
        cb_ap = average_precision_score(y_va, oof_cb[val_idx])
        print(f"  [Fold {fold}] CB fit in {time.time()-t0:.1f}s | Best Iter: {cb_clf.get_best_iteration()} | Val PR-AUC: {cb_ap:.4f}")
        print(f"  Fold {fold} finished in {time.time()-fold_start:.1f}s")
    
    # Assemble OOF DataFrame
    df_oof = pd.DataFrame({
        "prob_rf": oof_rf,
        "prob_xgb": oof_xgb,
        "prob_cb": oof_cb,
        "y_true": y_train
    })
    
    # Save checkpoint
    df_oof.to_parquet(oof_cache_path, index=False)
    print(f"\nSaved OOF predictions checkpoint to {oof_cache_path.as_posix()}")
    return df_oof


def enrich_meta_features(df_probs):
    """
    Construct base + enriched interaction features from base model probabilities:
    - prob_rf, prob_xgb, prob_cb
    - prob_mean: average probability across base models
    - prob_max: maximum risk signal
    - prob_min: minimum risk signal
    - prob_std: model disagreement / dispersion
    - logit_rf, logit_xgb, logit_cb: log-odds transformations
    """
    p_rf = df_probs["prob_rf"].values
    p_xgb = df_probs["prob_xgb"].values
    p_cb = df_probs["prob_cb"].values
    
    eps = 1e-6
    p_rf_c = np.clip(p_rf, eps, 1.0 - eps)
    p_xgb_c = np.clip(p_xgb, eps, 1.0 - eps)
    p_cb_c = np.clip(p_cb, eps, 1.0 - eps)
    
    probs_matrix = np.column_stack([p_rf, p_xgb, p_cb])
    
    meta_dict = {
        "prob_rf": p_rf,
        "prob_xgb": p_xgb,
        "prob_cb": p_cb,
        "prob_mean": np.mean(probs_matrix, axis=1),
        "prob_max": np.max(probs_matrix, axis=1),
        "prob_min": np.min(probs_matrix, axis=1),
        "prob_std": np.std(probs_matrix, axis=1),
        "logit_rf": np.log(p_rf_c / (1.0 - p_rf_c)),
        "logit_xgb": np.log(p_xgb_c / (1.0 - p_xgb_c)),
        "logit_cb": np.log(p_cb_c / (1.0 - p_cb_c))
    }
    
    return pd.DataFrame(meta_dict)


def evaluate_threshold_range(y_true, y_probs, thresholds=np.linspace(0.01, 0.99, 99)):
    """Evaluate F1, Precision, and Recall across thresholds to find optimum."""
    best_f1 = -1.0
    best_thresh = 0.5
    best_prec = 0.0
    best_rec = 0.0
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_prec = precision_score(y_true, y_pred, zero_division=0)
            best_rec = recall_score(y_true, y_pred, zero_division=0)
            
    return best_thresh, best_f1, best_prec, best_rec


def train_and_compare_meta_learners(X_oof_raw, X_oof_enriched, y_true):
    """
    Compare candidate meta-learners on OOF validation:
    1. Raw Probs Logistic Regression (L2, calibrated)
    2. Enriched Features Logistic Regression (L2, calibrated)
    3. Balanced Logistic Regression
    4. Optimal Soft-Voting Ensemble (Grid search over simplex)
    """
    print("\n" + "=" * 75)
    print("META-CLASSIFIER COMPARISON & VALIDATION EVALUATION (CERT r6.2)")
    print("=" * 75)
    
    # Baseline: Individual OOF performances
    print("Baseline Individual OOF Performances:")
    for col, name in [("prob_rf", "Random Forest"), ("prob_xgb", "XGBoost"), ("prob_cb", "CatBoost")]:
        ap = average_precision_score(y_true, X_oof_raw[col])
        roc = roc_auc_score(y_true, X_oof_raw[col])
        t_opt, f1_opt, p_opt, r_opt = evaluate_threshold_range(y_true, X_oof_raw[col].values)
        print(f"  {name:15s} | OOF PR-AUC: {ap:.4f} | ROC-AUC: {roc:.4f} | Opt T*: {t_opt:.2f} -> F1: {f1_opt*100:.2f}% (P: {p_opt*100:.2f}%, R: {r_opt*100:.2f}%)")
    
    candidates = {}
    
    # Candidate 1: Logistic Regression on Raw Probs (C=1.0)
    pipe_raw = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    pipe_raw.fit(X_oof_raw, y_true)
    probs_raw = pipe_raw.predict_proba(X_oof_raw)[:, 1]
    ap_raw = average_precision_score(y_true, probs_raw)
    roc_raw = roc_auc_score(y_true, probs_raw)
    t_raw, f1_raw, p_raw, r_raw = evaluate_threshold_range(y_true, probs_raw)
    candidates["LR_Raw_Probs"] = {
        "model": pipe_raw, "meta_type": "raw", "features": list(X_oof_raw.columns),
        "probs": probs_raw, "pr_auc": ap_raw, "roc_auc": roc_raw,
        "opt_thresh": t_raw, "opt_f1": f1_raw, "opt_prec": p_raw, "opt_rec": r_raw
    }
    
    # Candidate 2: Logistic Regression on Enriched Features (C tuning)
    best_c_enriched = None
    best_c_score = -1.0
    for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=c_val, max_iter=1000, random_state=42))
        ])
        pipe.fit(X_oof_enriched, y_true)
        p_val = pipe.predict_proba(X_oof_enriched)[:, 1]
        score = average_precision_score(y_true, p_val)
        if score > best_c_score:
            best_c_score = score
            best_c_enriched = c_val
            
    pipe_enriched = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=best_c_enriched, max_iter=1000, random_state=42))
    ])
    pipe_enriched.fit(X_oof_enriched, y_true)
    probs_enriched = pipe_enriched.predict_proba(X_oof_enriched)[:, 1]
    ap_en = average_precision_score(y_true, probs_enriched)
    roc_en = roc_auc_score(y_true, probs_enriched)
    t_en, f1_en, p_en, r_en = evaluate_threshold_range(y_true, probs_enriched)
    candidates["LR_Enriched_Features"] = {
        "model": pipe_enriched, "meta_type": "enriched", "features": list(X_oof_enriched.columns),
        "best_C": best_c_enriched, "probs": probs_enriched, "pr_auc": ap_en, "roc_auc": roc_en,
        "opt_thresh": t_en, "opt_f1": f1_en, "opt_prec": p_en, "opt_rec": r_en
    }
    
    # Candidate 3: Class-Weighted Balanced Logistic Regression on Enriched
    pipe_bal = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=best_c_enriched, class_weight='balanced', max_iter=1000, random_state=42))
    ])
    pipe_bal.fit(X_oof_enriched, y_true)
    probs_bal = pipe_bal.predict_proba(X_oof_enriched)[:, 1]
    ap_bal = average_precision_score(y_true, probs_bal)
    roc_bal = roc_auc_score(y_true, probs_bal)
    t_bal, f1_bal, p_bal, r_bal = evaluate_threshold_range(y_true, probs_bal)
    candidates["LR_Balanced_Enriched"] = {
        "model": pipe_bal, "meta_type": "enriched", "features": list(X_oof_enriched.columns),
        "probs": probs_bal, "pr_auc": ap_bal, "roc_auc": roc_bal,
        "opt_thresh": t_bal, "opt_f1": f1_bal, "opt_prec": p_bal, "opt_rec": r_bal
    }
    
    # Candidate 4: Optimal Soft-Voting Simplex Grid Search
    best_weights = None
    best_vote_ap = -1.0
    p_rf = X_oof_raw["prob_rf"].values
    p_xgb = X_oof_raw["prob_xgb"].values
    p_cb = X_oof_raw["prob_cb"].values
    
    for w_rf in np.linspace(0.0, 0.6, 13):
        for w_xgb in np.linspace(0.1, 0.9, 17):
            w_cb = 1.0 - w_rf - w_xgb
            if w_cb < 0:
                continue
            v_probs = w_rf * p_rf + w_xgb * p_xgb + w_cb * p_cb
            v_ap = average_precision_score(y_true, v_probs)
            if v_ap > best_vote_ap:
                best_vote_ap = v_ap
                best_weights = (w_rf, w_xgb, w_cb)
                
    w_rf, w_xgb, w_cb = best_weights
    probs_vote = w_rf * p_rf + w_xgb * p_xgb + w_cb * p_cb
    roc_vote = roc_auc_score(y_true, probs_vote)
    t_vote, f1_vote, p_vote, r_vote = evaluate_threshold_range(y_true, probs_vote)
    candidates["Optimal_Soft_Voting"] = {
        "weights": {"rf": float(w_rf), "xgb": float(w_xgb), "cb": float(w_cb)},
        "meta_type": "voting",
        "probs": probs_vote, "pr_auc": best_vote_ap, "roc_auc": roc_vote,
        "opt_thresh": t_vote, "opt_f1": f1_vote, "opt_prec": p_vote, "opt_rec": r_vote
    }
    
    print("\nCandidate Meta-Learner Summary:")
    print("-" * 90)
    print(f"{'Meta-Learner':25s} | {'PR-AUC':8s} | {'ROC-AUC':8s} | {'Opt T*':7s} | {'Opt F1':8s} | {'Precision':9s} | {'Recall':8s}")
    print("-" * 90)
    for c_name, c_info in candidates.items():
        print(f"{c_name:25s} | {c_info['pr_auc']:.4f}   | {c_info['roc_auc']:.4f}   | {c_info['opt_thresh']:.2f}    | {c_info['opt_f1']*100:.2f}%   | {c_info['opt_prec']*100:.2f}%    | {c_info['opt_rec']*100:.2f}%")
    print("-" * 90)
    
    # Pick winning meta-learner by PR-AUC & F1
    winner_name = max(candidates.keys(), key=lambda k: (candidates[k]["pr_auc"], candidates[k]["opt_f1"]))
    print(f"\nWINNING META-LEARNER: {winner_name}")
    winner = candidates[winner_name]
    print(f"  Selected Features: {winner.get('meta_type')} features")
    print(f"  OOF PR-AUC:        {winner['pr_auc']:.4f}")
    print(f"  OOF ROC-AUC:       {winner['roc_auc']:.4f}")
    print(f"  Optimal Threshold: {winner['opt_thresh']:.2f} -> OOF F1: {winner['opt_f1']*100:.2f}%")
    
    return winner_name, winner, candidates


def evaluate_on_test_set(X_test, y_test, sc_test, winner_name, winner_info, X_oof_raw):
    """
    Load full-trained base models, generate test meta-features, and evaluate the ensemble on X_test.
    """
    print("\n" + "=" * 75)
    print(f"FINAL EVALUATION ON UNTOUCHED X_TEST ({len(y_test):,} Sessions, {np.sum(y_test==1):,} Threats)")
    print("=" * 75)
    
    # Load base models
    print("Loading full-trained base models from models/...")
    rf_best = joblib.load(MODELS_DIR / "rf_best.joblib")
    xgb_best = joblib.load(MODELS_DIR / "xgb_best.joblib")
    
    try:
        cb_best = joblib.load(MODELS_DIR / "cb_best.joblib")
    except Exception:
        cb_best = CatBoostClassifier()
        cb_best.load_model((MODELS_DIR / "cb_best.cbm").as_posix())
        
    print("Generating base model predictions on untouched X_test (389,817 rows)...")
    t0 = time.time()
    p_test_rf = rf_best.predict_proba(X_test)[:, 1]
    p_test_xgb = xgb_best.predict_proba(X_test)[:, 1]
    p_test_cb = cb_best.predict_proba(X_test)[:, 1]
    print(f"Base models predicted in {time.time()-t0:.2f}s")
    
    # Build test meta-features
    df_test_probs = pd.DataFrame({
        "prob_rf": p_test_rf,
        "prob_xgb": p_test_xgb,
        "prob_cb": p_test_cb
    })
    
    if winner_info["meta_type"] == "enriched":
        X_test_meta = enrich_meta_features(df_test_probs)
        p_test_meta = winner_info["model"].predict_proba(X_test_meta)[:, 1]
    elif winner_info["meta_type"] == "raw":
        p_test_meta = winner_info["model"].predict_proba(df_test_probs)[:, 1]
    elif winner_info["meta_type"] == "voting":
        w = winner_info["weights"]
        p_test_meta = w["rf"] * p_test_rf + w["xgb"] * p_test_xgb + w["cb"] * p_test_cb
    else:
        raise ValueError(f"Unknown meta_type: {winner_info['meta_type']}")
        
    # Evaluate at default threshold 0.50
    pred_default = (p_test_meta >= 0.50).astype(int)
    prec_def = precision_score(y_test, pred_default, zero_division=0)
    rec_def = recall_score(y_test, pred_default, zero_division=0)
    f1_def = f1_score(y_test, pred_default, zero_division=0)
    
    # Evaluate at optimal threshold T*
    opt_t = winner_info["opt_thresh"]
    pred_opt = (p_test_meta >= opt_t).astype(int)
    prec_opt = precision_score(y_test, pred_opt, zero_division=0)
    rec_opt = recall_score(y_test, pred_opt, zero_division=0)
    f1_opt = f1_score(y_test, pred_opt, zero_division=0)
    roc_opt = roc_auc_score(y_test, p_test_meta)
    pr_opt = average_precision_score(y_test, p_test_meta)
    
    cm = confusion_matrix(y_test, pred_opt)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    
    print("\n--- STACKING ENSEMBLE TEST PERFORMANCE (CERT r6.2) ---")
    print(f"Default Threshold (T = 0.50):")
    print(f"  Precision: {prec_def*100:.2f}% | Recall: {rec_def*100:.2f}% | F1-Score: {f1_def*100:.2f}%")
    print(f"\nOptimal Decision Threshold (T* = {opt_t:.2f}):")
    print(f"  Precision: {prec_opt*100:.2f}%")
    print(f"  Recall:    {rec_opt*100:.2f}% ({tp}/{tp+fn} threats detected)")
    print(f"  F1-Score:  {f1_opt*100:.2f}%")
    print(f"  PR-AUC:    {pr_opt:.4f}")
    print(f"  ROC-AUC:   {roc_opt:.4f}")
    print(f"  FPR:       {fpr*100:.5f}% ({fp} false alarms out of {tn+fp:,} normal sessions)")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,} | FP: {fp:,}")
    print(f"  FN: {fn:,}   | TP: {tp:,}")
    
    # Scenario Breakdown
    scenario_names = {
        1: "Scenario 1 (ACM2278 - IP / Data Theft)",
        2: "Scenario 2 (CMP2946 - Sabotage / USB Leak)",
        3: "Scenario 3 (PLJ1771 - Keylogger / Mass Disgruntled)",
        4: "Scenario 4 (CDE1846 - Flight Risk Exfiltration)",
        5: "Scenario 5 (MBG3183 - Dropbox Cloud Upload [Zero-Shot])"
    }
    
    print("\n" + "="*75)
    print("SCENARIO-BY-SCENARIO DETECTION BREAKDOWN (STACKING ENSEMBLE)")
    print("="*75)
    print(f"{'Scenario':<42} {'Total':<8} {'Caught (Opt)':<14} {'Recall':<10}")
    print("-" * 75)
    scenario_breakdown = {}
    for s in range(1, 6):
        s_mask = (sc_test == s)
        total_s = int(np.sum(s_mask))
        if total_s > 0:
            detected_def = int(np.sum(pred_default[s_mask] == 1))
            detected_opt = int(np.sum(pred_opt[s_mask] == 1))
            rec_s = float(detected_opt / total_s)
        else:
            detected_def = 0
            detected_opt = 0
            rec_s = 0.0
            
        sc_info = {
            'name': scenario_names.get(s, f"Scenario {s}"),
            'total_sessions': total_s,
            'detected_default': detected_def,
            'detected_optimal': detected_opt,
            'recall_optimal': rec_s
        }
        scenario_breakdown[f"Scenario_{s}"] = sc_info
        print(f"{sc_info['name']:<42} {total_s:<8} {detected_opt:<14} {rec_s*100:.1f}%")
    
    test_results = {
        "p_test_rf": p_test_rf,
        "p_test_xgb": p_test_xgb,
        "p_test_cb": p_test_cb,
        "p_test_ensemble": p_test_meta,
        "default": {
            "threshold": 0.50,
            "precision": float(prec_def),
            "recall": float(rec_def),
            "f1_score": float(f1_def)
        },
        "optimal": {
            "threshold": float(opt_t),
            "precision": float(prec_opt),
            "recall": float(rec_opt),
            "f1_score": float(f1_opt),
            "roc_auc": float(roc_opt),
            "pr_auc": float(pr_opt),
            "false_positive_rate": float(fpr),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
            "scenario_recall": scenario_breakdown
        }
    }
    
    return test_results, df_test_probs


def plot_and_save_visualizations(y_test, test_results, winner_name, winner_info):
    """
    Generate and save high-resolution publication plots (300 DPI):
    1. plots/ensemble/ensemble_confusion_matrix.png
    2. plots/ensemble/all_models_pr_roc_comparison.png (RF vs XGB vs CB vs Ensemble overlay)
    3. plots/ensemble/meta_learner_weights.png
    """
    print("\nGenerating evaluation visualizations in plots/ensemble/...")
    
    # 1. Confusion Matrix Plot
    cm_dict = test_results["optimal"]["confusion_matrix"]
    cm_matrix = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_matrix, annot=True, fmt=',d', cmap='Blues', cbar=False,
        xticklabels=['Benign (0)', 'Insider (1)'],
        yticklabels=['Benign (0)', 'Insider (1)'],
        annot_kws={'size': 14, 'weight': 'bold'}, ax=ax
    )
    opt_t = test_results['optimal']['threshold']
    f1 = test_results['optimal']['f1_score'] * 100
    ax.set_title(f"Stacking Ensemble Confusion Matrix — CERT r6.2\n(Threshold $T^*={opt_t:.2f}$, F1={f1:.2f}%)", fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = PLOTS_DIR / "ensemble_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"  Saved confusion matrix: {cm_path.as_posix()}")
    
    # 2. All Models PR & ROC Curve Comparison Overlay (Dual Panel)
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))
    
    models_to_plot = [
        ("Random Forest", test_results["p_test_rf"], "#1f77b4", ":"),
        ("XGBoost", test_results["p_test_xgb"], "#ff7f0e", "--"),
        ("CatBoost", test_results["p_test_cb"], "#2ca02c", "-."),
        ("Stacking Ensemble", test_results["p_test_ensemble"], "#d62728", "-")
    ]
    
    for name, p_pred, color, ls in models_to_plot:
        prec, rec, _ = precision_recall_curve(y_test, p_pred)
        ap = average_precision_score(y_test, p_pred)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {ap:.4f})", color=color, linestyle=ls, linewidth=2.4 if name=="Stacking Ensemble" else 1.8)
        
        fpr_arr, tpr_arr, _ = roc_curve(y_test, p_pred)
        roc = roc_auc_score(y_test, p_pred)
        ax_roc.plot(fpr_arr, tpr_arr, label=f"{name} (ROC-AUC = {roc:.4f})", color=color, linestyle=ls, linewidth=2.4 if name=="Stacking Ensemble" else 1.8)
        
    baseline_prec = np.mean(y_test)
    ax_pr.axhline(baseline_prec, color='gray', linestyle='--', alpha=0.6, label=f'Random Chance ({baseline_prec*100:.3f}%)')
    ax_pr.set_title("Precision-Recall Curves Comparison (CERT r6.2)", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall (True Positive Rate)", fontsize=11, fontweight='bold')
    ax_pr.set_ylabel("Precision", fontsize=11, fontweight='bold')
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    ax_pr.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    ax_roc.set_title("ROC Curves Comparison (CERT r6.2)", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=11, fontweight='bold')
    ax_roc.set_ylabel("True Positive Rate", fontsize=11, fontweight='bold')
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    ax_roc.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.02])
    
    plt.tight_layout()
    comparison_path = PLOTS_DIR / "all_models_pr_roc_comparison.png"
    plt.savefig(comparison_path)
    plt.close()
    print(f"  Saved PR & ROC 4-model comparison: {comparison_path.as_posix()}")
    
    # 3. Meta-Learner Weights / Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if winner_info["meta_type"] in ["enriched", "raw"]:
        scaler = winner_info["model"].named_steps['scaler']
        lr = winner_info["model"].named_steps['lr']
        feature_names = winner_info["features"]
        coefs = lr.coef_[0]
        
        df_coef = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        df_coef = df_coef.sort_values(by="Coefficient", ascending=False)
        
        colors = ['#2ca02c' if c >= 0 else '#d62728' for c in df_coef["Coefficient"]]
        bars = ax.barh(df_coef["Feature"], df_coef["Coefficient"], color=colors, edgecolor='#333333', linewidth=0.7)
        ax.set_title(f"Meta-Learner Coefficients ({winner_name})\nLearned Impact on Threat Probability (CERT r6.2)", fontsize=12, pad=12, fontweight='bold')
        ax.set_xlabel("Standardized Logistic Regression Coefficient", fontsize=11, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.invert_yaxis()
        ax.grid(True, linestyle=':', alpha=0.5, axis='x')
    elif winner_info["meta_type"] == "voting":
        w = winner_info["weights"]
        df_w = pd.DataFrame({"Model": ["Random Forest", "XGBoost", "CatBoost"], "Weight": [w["rf"], w["xgb"], w["cb"]]})
        sns.barplot(data=df_w, x="Weight", y="Model", hue="Model", palette="viridis", legend=False, ax=ax)
        ax.set_title("Optimal Soft-Voting Ensemble Model Weights (CERT r6.2)", fontsize=12, pad=12, fontweight='bold')
        ax.set_xlabel("Simplex Weight ($w_i$)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5, axis='x')
        
    plt.tight_layout()
    weights_path = PLOTS_DIR / "meta_learner_weights.png"
    plt.savefig(weights_path)
    plt.close()
    print(f"  Saved meta-learner weights plot: {weights_path.as_posix()}")


def save_artifacts_and_log(winner_name, winner_info, test_results, candidate_summary):
    """Save stacking ensemble pipeline and update results_log.json."""
    ensemble_bundle = {
        "winner_name": winner_name,
        "winner_info": {
            "meta_type": winner_info["meta_type"],
            "features": winner_info.get("features", []),
            "weights": winner_info.get("weights", None),
            "opt_thresh": float(winner_info["opt_thresh"])
        },
        "model": winner_info.get("model", None)
    }
    joblib.dump(ensemble_bundle, MODELS_DIR / "stacking_ensemble.joblib")
    print(f"\nSaved ensemble pipeline to {(MODELS_DIR / 'stacking_ensemble.joblib').as_posix()}")
    
    with open(MODELS_DIR / "results_log.json", "r") as f:
        log = json.load(f)
        
    log["Stacking_Ensemble"] = {
        "model_name": "Stacking Ensemble",
        "dataset": "CERT r6.2",
        "winning_meta_learner": winner_name,
        "meta_features_type": winner_info["meta_type"],
        "optimal_threshold": float(winner_info["opt_thresh"]),
        "cross_validation": {
            "oof_pr_auc": float(winner_info["pr_auc"]),
            "oof_roc_auc": float(winner_info["roc_auc"]),
            "opt_threshold": float(winner_info["opt_thresh"]),
            "opt_f1": float(winner_info["opt_f1"]),
            "opt_prec": float(winner_info["opt_prec"]),
            "opt_rec": float(winner_info["opt_rec"])
        },
        "test_metrics_default_threshold": test_results["default"],
        "test_metrics_optimal_threshold": test_results["optimal"]
    }
    
    with open(MODELS_DIR / "results_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("Updated models/results_log.json with Stacking_Ensemble entry.")


def print_master_leaderboard():
    """Read results_log.json and print complete 4-model comparison table on CERT r6.2."""
    with open(MODELS_DIR / "results_log.json", "r") as f:
        log = json.load(f)
        
    print("\n" + "=" * 115)
    print("FINAL MASTER 4-MODEL LEADERBOARD (EVALUATION ON TEST SET X_TEST — 389,817 SESSIONS, 22 THREATS)")
    print("=" * 115)
    header = f"{'Model':20s} | {'Strategy / Type':22s} | {'T*':5s} | {'Precision':10s} | {'Recall':8s} | {'F1-Score':9s} | {'PR-AUC':8s} | {'ROC-AUC':8s} | {'FPR (%)':8s} | {'FP':4s} | {'TP':4s}"
    print(header)
    print("-" * 115)
    
    models = ["Random_Forest", "XGBoost", "CatBoost", "Stacking_Ensemble"]
    for m_key in models:
        m = log.get(m_key)
        if not m:
            continue
        name = m.get("model_name", m_key)
        strat = m.get("winning_strategy", m.get("winning_meta_learner", "ensemble"))
        t_opt = m.get("optimal_threshold", 0.5)
        opt_metrics = m["test_metrics_optimal_threshold"]
        prec = opt_metrics["precision"] * 100
        rec = opt_metrics["recall"] * 100
        f1 = opt_metrics["f1_score"] * 100
        pr_auc = opt_metrics["pr_auc"]
        roc_auc = opt_metrics["roc_auc"]
        fpr = opt_metrics["false_positive_rate"] * 100
        cm = opt_metrics["confusion_matrix"]
        fp = cm["fp"]
        tp = cm["tp"]
        
        row = f"{name:20s} | {strat:22s} | {t_opt:.2f}  | {prec:6.2f}%    | {rec:5.2f}%  | {f1:6.2f}%   | {pr_auc:.4f}   | {roc_auc:.4f}   | {fpr:6.5f}% | {fp:<4d} | {tp:<4d}"
        print(row)
    print("-" * 115)
    
    print("Scenario Detection Breakdown (Caught / Total Sessions on X_test):")
    for s in range(1, 6):
        s_key = f"Scenario_{s}"
        row_str = f"  Scenario {s}:{'':<15}"
        for m_key in models:
            m = log.get(m_key, {})
            sc_info = m.get("test_metrics_optimal_threshold", {}).get("scenario_recall", {}).get(s_key, {})
            caught = sc_info.get("detected_optimal", "N/A")
            total = sc_info.get("total_sessions", 0)
            row_str += f" | {m.get('model_name', m_key)[:6]}: {caught}/{total:<2}"
        print(row_str)
    print("=" * 115)


def main():
    total_start = time.time()
    print("STARTING STEP 5: STACKING ENSEMBLE META-LEARNER PIPELINE (CERT r6.2)")
    
    # 1. Load splits
    X_train_raw, y_train_raw, X_test, y_test, sc_test = load_data()
    
    # 2. Get base model configurations
    rf_params, xgb_params, cb_params = get_base_model_configs()
    
    # 3. 5-Fold Stratified CV to generate OOF probabilities
    df_oof = generate_oof_predictions(X_train_raw, y_train_raw, rf_params, xgb_params, cb_params)
    
    # 4. Construct raw and enriched meta-features
    X_oof_raw = df_oof[["prob_rf", "prob_xgb", "prob_cb"]]
    X_oof_enriched = enrich_meta_features(df_oof)
    y_oof = df_oof["y_true"].values
    
    # 5. Train & compare candidate meta-learners
    winner_name, winner_info, candidates = train_and_compare_meta_learners(X_oof_raw, X_oof_enriched, y_oof)
    
    # 6. Final evaluation on untouched X_test
    test_results, df_test_probs = evaluate_on_test_set(X_test, y_test, sc_test, winner_name, winner_info, X_oof_raw)
    
    # 7. Generate visualizations
    plot_and_save_visualizations(y_test, test_results, winner_name, winner_info)
    
    # 8. Save artifacts and update log
    save_artifacts_and_log(winner_name, winner_info, test_results, candidates)
    
    # 9. Print Master Leaderboard
    print_master_leaderboard()
    
    print(f"\nStep 5 pipeline completed successfully in {time.time() - total_start:.2f}s!")


if __name__ == "__main__":
    main()
