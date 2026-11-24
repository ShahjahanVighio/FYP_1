"""
Step 4: Train & Evaluate Model 3 — CatBoost (train_cb.py)
CERT r6.2 Insider Threat Detection
- Compares Cost-Sensitive (auto_class_weights across [None, 'SqrtBalanced', 'Balanced']) vs. Resampled (SMOTE + RUS).
- Stratified 5-Fold Cross-Validation with early stopping (50 rounds) on validation PRAUC.
- Hyperparameter tuning via RandomizedSearchCV calibrated for 1.56M train rows.
- Decision threshold optimization (T*) strictly on OOF validation predictions.
- Final evaluation on untouched X_test (389,817 sessions, 22 threats).
- Full Scenario Recall Breakdown (Scenarios 1-5, including zero-shot Scenario 5).
- Saves model to models/cb_best.cbm and models/cb_best.joblib, appends to models/results_log.json,
  generates 300 DPI plots in plots/cb/, and outputs a full 3-Model Leaderboard (RF vs. XGBoost vs. CatBoost).
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# --- Paths & Directories ---
ROOT_DIR = Path(__file__).resolve().parent
PREP_DIR = ROOT_DIR / "prepared_data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots" / "cb"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Loads prepared data splits from prepared_data/ for CERT r6.2."""
    print("Loading prepared data splits for CERT r6.2...")
    X_train_raw = pd.read_parquet(PREP_DIR / "X_train_raw.parquet")
    y_train_raw = pd.read_parquet(PREP_DIR / "y_train_raw.parquet")['insider'].values
    
    X_train_res = pd.read_parquet(PREP_DIR / "X_train_res.parquet")
    y_train_res = pd.read_parquet(PREP_DIR / "y_train_res.parquet")['insider'].values
    
    X_test = pd.read_parquet(PREP_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PREP_DIR / "y_test.parquet")['insider'].values
    sc_test = pd.read_parquet(PREP_DIR / "y_test_scenario.parquet")['scenario'].values
    
    with open(PREP_DIR / "selected_features.json", "r") as f:
        feature_names = json.load(f)
        
    neg_count = int(np.sum(y_train_raw == 0))
    pos_count = int(np.sum(y_train_raw == 1))
    ratio = neg_count / pos_count
    
    print(f"X_train_raw: {X_train_raw.shape}, Benign: {neg_count:,}, Threats: {pos_count:,} (Ratio = {ratio:.2f}:1)")
    print(f"X_train_res: {X_train_res.shape}, Benign: {np.sum(y_train_res == 0):,}, Threats: {np.sum(y_train_res == 1):,}")
    print(f"X_test:      {X_test.shape}, Benign: {np.sum(y_test == 0):,}, Threats: {np.sum(y_test == 1):,} [ISOLATED TEST SET]")
    print(f"Selected Features: {len(feature_names)} features")
    
    return X_train_raw, y_train_raw, X_train_res, y_train_res, X_test, y_test, sc_test, feature_names


def find_optimal_threshold(y_true, y_probs):
    """
    Finds decision threshold in [0.01, 0.99] that maximizes F1 score on validation set.
    """
    best_thresh = 0.50
    best_f1 = 0.0
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return best_thresh, best_f1


def evaluate_variant_cv(X_train_raw, y_train_raw, variant="cost_sensitive", n_iter=5, random_state=42):
    """
    Performs 5-Fold Stratified Cross-Validation on X_train_raw using CatBoost.
    - variant='cost_sensitive': Tests auto_class_weights across ['None', 'SqrtBalanced', 'Balanced'] on raw imbalanced folds.
    - variant='resampled': Applies fold-level SMOTE + Undersampling inside each training fold (auto_class_weights=None).
    Validates strictly on untouched imbalanced validation folds with early stopping (50 rounds).
    Returns: oof_probs, best_params, cv_summary, fold_best_iters.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating CatBoost: Variant {variant.upper()}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y_train_raw))
    
    if variant == "cost_sensitive":
        param_dist = {
            'depth': [4, 6, 8],
            'learning_rate': [0.03, 0.06, 0.1],
            'iterations': [300, 500],
            'l2_leaf_reg': [1, 3, 5, 10],
            'random_strength': [0.1, 1.0],
            'auto_class_weights': ['None', 'SqrtBalanced', 'Balanced']
        }
        
        base_cb = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='PRAUC',
            random_seed=random_state,
            thread_count=-1,
            verbose=0
        )
        
        # Fast representative tuning on a stratified 1-fold slice (~311k rows, ~15 threats)
        tune_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        _, tune_idx = next(tune_skf.split(X_train_raw, y_train_raw))
        X_tune, y_tune = X_train_raw.iloc[tune_idx], y_train_raw[tune_idx]
        
        print(f"Running RandomizedSearchCV ({n_iter} iterations) on representative stratified sample ({len(X_tune):,} rows)...")
        t0 = time.time()
        rs = RandomizedSearchCV(
            base_cb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='average_precision',
            cv=3,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        rs.fit(X_tune, y_tune)
        best_params = rs.best_params_
        print(f"Tuning finished in {time.time()-t0:.1f}s. Best parameters: {best_params}")
        
    else:  # resampled
        param_dist = {
            'depth': [4, 6, 8],
            'learning_rate': [0.03, 0.06, 0.1],
            'iterations': [300, 500],
            'l2_leaf_reg': [1, 3, 5, 10],
            'random_strength': [0.1, 1.0],
            'auto_class_weights': ['None']
        }
        
        base_cb = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='PRAUC',
            random_seed=random_state,
            thread_count=-1,
            verbose=0
        )
        
        # Tune on precomputed balanced training set sample (~155k rows)
        X_res_full = pd.read_parquet(PREP_DIR / "X_train_res.parquet")
        y_res_full = pd.read_parquet(PREP_DIR / "y_train_res.parquet")['insider'].values
        tune_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        _, tune_idx = next(tune_skf.split(X_res_full, y_res_full))
        X_tune, y_tune = X_res_full.iloc[tune_idx], y_res_full[tune_idx]
        
        print(f"Running RandomizedSearchCV ({n_iter} iterations) on resampled training sample ({len(X_tune):,} rows)...")
        t0 = time.time()
        rs = RandomizedSearchCV(
            base_cb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='average_precision',
            cv=3,
            random_state=random_state,
            n_jobs=1,
            verbose=0
        )
        rs.fit(X_tune, y_tune)
        best_params = rs.best_params_
        print(f"Tuning finished in {time.time()-t0:.1f}s. Best parameters: {best_params}")

    # 5-Fold Stratified Cross-Validation with Best Parameters & Early Stopping
    fold_ap = []
    fold_roc = []
    fold_f1 = []
    fold_prec = []
    fold_rec = []
    fold_best_iters = []
    
    print("\nRunning 5-Fold Cross-Validation across 1.56M sessions with early stopping (50 rounds)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train_raw), 1):
        t_fold = time.time()
        X_tr, y_tr = X_train_raw.iloc[train_idx], y_train_raw[train_idx]
        X_va, y_va = X_train_raw.iloc[val_idx], y_train_raw[val_idx]
        
        if variant == "cost_sensitive":
            clf = CatBoostClassifier(
                **best_params,
                loss_function='Logloss',
                eval_metric='PRAUC',
                early_stopping_rounds=50,
                random_seed=random_state + fold,
                thread_count=-1,
                verbose=0
            )
            clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        else:  # fold-level resampled
            smote = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=random_state + fold)
            rus = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state + fold)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            X_tr_res, y_tr_res = rus.fit_resample(X_tr_sm, y_tr_sm)
            clf = CatBoostClassifier(
                **best_params,
                loss_function='Logloss',
                eval_metric='PRAUC',
                early_stopping_rounds=50,
                random_seed=random_state + fold,
                thread_count=-1,
                verbose=0
            )
            clf.fit(X_tr_res, y_tr_res, eval_set=(X_va, y_va), verbose=False)
            
        val_probs = clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs
        
        ap = average_precision_score(y_va, val_probs)
        roc = roc_auc_score(y_va, val_probs)
        preds_05 = (val_probs >= 0.50).astype(int)
        f1 = f1_score(y_va, preds_05, zero_division=0)
        prec = precision_score(y_va, preds_05, zero_division=0)
        rec = recall_score(y_va, preds_05, zero_division=0)
        
        fold_ap.append(ap)
        fold_roc.append(roc)
        fold_f1.append(f1)
        fold_prec.append(prec)
        fold_rec.append(rec)
        
        best_iter = clf.get_best_iteration() if clf.get_best_iteration() is not None else best_params.get('iterations', 300)
        fold_best_iters.append(best_iter)
        print(f"  Fold {fold}/5: Best Iter = {best_iter}, PR-AUC = {ap:.4f}, ROC-AUC = {roc:.4f}, F1 = {f1:.4f}, Prec = {prec:.4f}, Rec = {rec:.4f} ({time.time()-t_fold:.1f}s)")
        
    overall_ap = average_precision_score(y_train_raw, oof_probs)
    overall_roc = roc_auc_score(y_train_raw, oof_probs)
    opt_thresh, opt_f1 = find_optimal_threshold(y_train_raw, oof_probs)
    oof_preds_opt = (oof_probs >= opt_thresh).astype(int)
    opt_prec = precision_score(y_train_raw, oof_preds_opt, zero_division=0)
    opt_rec = recall_score(y_train_raw, oof_preds_opt, zero_division=0)
    
    cv_summary = {
        'variant': variant,
        'best_params': {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best_params.items()},
        'cv_mean_ap': float(np.mean(fold_ap)),
        'cv_std_ap': float(np.std(fold_ap)),
        'cv_mean_roc': float(np.mean(fold_roc)),
        'oof_pr_auc': float(overall_ap),
        'oof_roc_auc': float(overall_roc),
        'default_f1': float(np.mean(fold_f1)),
        'default_prec': float(np.mean(fold_prec)),
        'default_rec': float(np.mean(fold_rec)),
        'opt_threshold': float(opt_thresh),
        'opt_f1': float(opt_f1),
        'opt_prec': float(opt_prec),
        'opt_rec': float(opt_rec),
        'mean_best_iteration': int(np.mean(fold_best_iters))
    }
    
    print(f"\n--- {variant.upper()} OOF Summary ---")
    print(f"  OOF PR-AUC (Average Precision): {overall_ap:.4f}")
    print(f"  OOF ROC-AUC:                    {overall_roc:.4f}")
    print(f"  At default threshold (0.50):    F1 = {np.mean(fold_f1):.4f}, Prec = {np.mean(fold_prec):.4f}, Rec = {np.mean(fold_rec):.4f}")
    print(f"  At optimal threshold ({opt_thresh:.2f}):  F1 = {opt_f1:.4f}, Prec = {opt_prec:.4f}, Rec = {opt_rec:.4f}")
    
    return oof_probs, best_params, cv_summary, fold_best_iters


def plot_results(y_test, y_probs, opt_thresh, feature_importances, feature_names):
    """
    Generates and saves high-resolution publication-quality plots (300 DPI).
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Confusion Matrix
    y_pred_opt = (y_probs >= opt_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt=',d', cmap='Purples', cbar=False, ax=ax,
        xticklabels=['Benign (0)', 'Insider (1)'],
        yticklabels=['Benign (0)', 'Insider (1)']
    )
    ax.set_title(f"CatBoost Confusion Matrix — CERT r6.2\n(Threshold = {opt_thresh:.2f})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class", fontweight='bold')
    ax.set_ylabel("True Class", fontweight='bold')
    plt.tight_layout()
    cm_path = PLOTS_DIR / "cb_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path.as_posix()}")
    
    # 2. PR and ROC Curves
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    ap = average_precision_score(y_test, y_probs)
    roc = roc_auc_score(y_test, y_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # PR Curve
    ax1.plot(recall, precision, color='#6f42c1', lw=2.5, label=f'CatBoost (PR-AUC = {ap:.4f})')
    p_opt = precision_score(y_test, y_pred_opt, zero_division=0)
    r_opt = recall_score(y_test, y_pred_opt, zero_division=0)
    ax1.plot(r_opt, p_opt, marker='o', markersize=8, color='black', label=f'Opt Thresh ({opt_thresh:.2f})')
    ax1.set_xlabel('Recall', fontweight='bold')
    ax1.set_ylabel('Precision', fontweight='bold')
    ax1.set_title('Precision-Recall Curve (r6.2 Test Set)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    
    # ROC Curve
    ax2.plot(fpr, tpr, color='#007bff', lw=2.5, label=f'CatBoost (ROC-AUC = {roc:.4f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance')
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Recall)', fontweight='bold')
    ax2.set_title('ROC Curve (r6.2 Test Set)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim([0.0, 1.05])
    ax2.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    pr_roc_path = PLOTS_DIR / "cb_pr_roc_curves.png"
    plt.savefig(pr_roc_path)
    plt.close()
    print(f"Saved: {pr_roc_path.as_posix()}")
    
    # 3. Top 20 Feature Importances
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', palette='mako', legend=False, ax=ax)
    ax.set_title('CatBoost — Top 20 Feature Importances — CERT r6.2', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Feature Importance (PredictionValuesChange)', fontweight='bold')
    ax.set_ylabel('Feature Name', fontweight='bold')
    plt.tight_layout()
    fi_path = PLOTS_DIR / "cb_feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved: {fi_path.as_posix()}")


def main():
    X_train_raw, y_train_raw, X_train_res, y_train_res, X_test, y_test, sc_test, feature_names = load_data()
    
    # 1. Compare both imbalance strategies on 5-fold CV
    oof_cs, params_cs, cv_cs, iters_cs = evaluate_variant_cv(
        X_train_raw, y_train_raw, variant="cost_sensitive", n_iter=5, random_state=42
    )
    
    oof_res, params_res, cv_res, iters_res = evaluate_variant_cv(
        X_train_raw, y_train_raw, variant="resampled", n_iter=5, random_state=42
    )
    
    # 2. Select winning variant based on validation PR-AUC & F1
    print("\n" + "="*75)
    print("VALIDATION COMPARISON: COST-SENSITIVE VS. RESAMPLED (CATBOOST - CERT r6.2)")
    print("="*75)
    print(f"{'Metric':<25} {'Cost-Sensitive (Raw)':<25} {'Resampled (SMOTE+RUS)':<25}")
    print("-" * 75)
    print(f"{'CV Mean PR-AUC':<25} {cv_cs['cv_mean_ap']:<25.4f} {cv_res['cv_mean_ap']:<25.4f}")
    print(f"{'OOF PR-AUC':<25} {cv_cs['oof_pr_auc']:<25.4f} {cv_res['oof_pr_auc']:<25.4f}")
    print(f"{'OOF ROC-AUC':<25} {cv_cs['oof_roc_auc']:<25.4f} {cv_res['oof_roc_auc']:<25.4f}")
    print(f"{'Default F1 (0.50)':<25} {cv_cs['default_f1']:<25.4f} {cv_res['default_f1']:<25.4f}")
    print(f"{'Optimal Threshold':<25} {cv_cs['opt_threshold']:<25.2f} {cv_res['opt_threshold']:<25.2f}")
    print(f"{'Optimal F1':<25} {cv_cs['opt_f1']:<25.4f} {cv_res['opt_f1']:<25.4f}")
    print(f"{'Optimal Precision':<25} {cv_cs['opt_prec']:<25.4f} {cv_res['opt_prec']:<25.4f}")
    print(f"{'Optimal Recall':<25} {cv_cs['opt_rec']:<25.4f} {cv_res['opt_rec']:<25.4f}")
    print(f"{'Mean Best Iteration':<25} {cv_cs['mean_best_iteration']:<25} {cv_res['mean_best_iteration']:<25}")
    
    if cv_res['cv_mean_ap'] > cv_cs['cv_mean_ap']:
        winner = "resampled"
        best_params = params_res
        opt_thresh = cv_res['opt_threshold']
        final_iters = max(int(np.median(iters_res)), 50)
        print(f"\nWinning Strategy: RESAMPLED (SMOTE + UnderSampling) [Higher PR-AUC: {cv_res['cv_mean_ap']:.4f}]")
    else:
        winner = "cost_sensitive"
        best_params = params_cs
        opt_thresh = cv_cs['opt_threshold']
        final_iters = max(int(np.median(iters_cs)), 50)
        weights_name = best_params.get('auto_class_weights', 'None')
        print(f"\nWinning Strategy: COST-SENSITIVE ({weights_name}) [Higher PR-AUC: {cv_cs['cv_mean_ap']:.4f}]")
        
    # 3. Train final model on full training set using winning strategy & tuned iterations
    final_params = dict(best_params)
    final_params['iterations'] = final_iters
    print(f"\nTraining final CatBoost on full 1.56M train set using {winner.upper()} strategy (iterations={final_iters})...")
    t0 = time.time()
    if winner == "resampled":
        final_cb = CatBoostClassifier(
            **final_params,
            loss_function='Logloss',
            eval_metric='PRAUC',
            random_seed=42,
            thread_count=-1,
            verbose=0
        )
        final_cb.fit(X_train_res, y_train_res)
    else:
        final_cb = CatBoostClassifier(
            **final_params,
            loss_function='Logloss',
            eval_metric='PRAUC',
            random_seed=42,
            thread_count=-1,
            verbose=0
        )
        final_cb.fit(X_train_raw, y_train_raw)
    print(f"Final CatBoost model trained in {time.time()-t0:.2f}s!")
    
    # 4. Final Evaluation on Untouched Test Set (389,817 sessions, 22 threats)
    print("\n" + "="*75)
    print("FINAL EVALUATION ON UNTOUCHED TEST SET (X_test — 389,817 Sessions)")
    print("="*75)
    test_probs = final_cb.predict_proba(X_test)[:, 1]
    
    # Default threshold 0.50
    preds_def = (test_probs >= 0.50).astype(int)
    prec_def = precision_score(y_test, preds_def, zero_division=0)
    rec_def = recall_score(y_test, preds_def, zero_division=0)
    f1_def = f1_score(y_test, preds_def, zero_division=0)
    
    # Optimal threshold (tuned strictly on OOF validation)
    preds_opt = (test_probs >= opt_thresh).astype(int)
    prec_opt = precision_score(y_test, preds_opt, zero_division=0)
    rec_opt = recall_score(y_test, preds_opt, zero_division=0)
    f1_opt = f1_score(y_test, preds_opt, zero_division=0)
    
    roc_test = roc_auc_score(y_test, test_probs)
    pr_auc_test = average_precision_score(y_test, test_probs)
    
    # Confusion matrix & False Positive Rate
    cm = confusion_matrix(y_test, preds_opt)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"Test Set Size:              {len(y_test):,} sessions ({np.sum(y_test == 1):,} threats)")
    print(f"ROC-AUC Score:              {roc_test:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc_test:.4f}")
    print(f"Optimal Threshold (from CV):{opt_thresh:.2f}")
    print("\nPerformance Comparison:")
    print(f"{'Threshold':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 65)
    print(f"{'Default (0.50)':<20} {prec_def:<15.4f} {rec_def:<15.4f} {f1_def:<15.4f}")
    print(f"{f'Optimal ({opt_thresh:.2f})':<20} {prec_opt:<15.4f} {rec_opt:<15.4f} {f1_opt:<15.4f}")
    
    print(f"\nConfusion Matrix (at threshold {opt_thresh:.2f}):")
    print(f"  True Negatives (TN):  {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,}")
    print(f"  True Positives (TP):  {tp:,}")
    print(f"  False Positive Rate (FPR): {fpr*100:.5f}% ({fp} false alarms out of {tn+fp:,} benign sessions)")
    print(f"  Threat Detection Rate:     {tp}/{tp+fn} threats caught ({tp/(tp+fn)*100:.1f}%)")
    
    print("\nDetailed Classification Report (Optimal Threshold):")
    print(classification_report(y_test, preds_opt, target_names=['Benign (0)', 'Insider (1)'], digits=4))
    
    # Scenario-by-Scenario Recall Breakdown
    scenario_names = {
        1: "Scenario 1 (ACM2278 - IP / Data Theft)",
        2: "Scenario 2 (CMP2946 - Sabotage / USB Leak)",
        3: "Scenario 3 (PLJ1771 - Keylogger / Mass Disgruntled)",
        4: "Scenario 4 (CDE1846 - Flight Risk Exfiltration)",
        5: "Scenario 5 (MBG3183 - Dropbox Cloud Upload [Zero-Shot])"
    }
    
    print("\n" + "="*75)
    print("SCENARIO-BY-SCENARIO DETECTION BREAKDOWN (CERT r6.2 TEST SET - CATBOOST)")
    print("="*75)
    print(f"{'Scenario':<42} {'Total':<8} {'Caught (Opt)':<14} {'Recall':<10}")
    print("-" * 75)
    scenario_breakdown = {}
    for s in range(1, 6):
        s_mask = (sc_test == s)
        total_s = int(np.sum(s_mask))
        if total_s > 0:
            detected_def = int(np.sum(preds_def[s_mask] == 1))
            detected_opt = int(np.sum(preds_opt[s_mask] == 1))
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
        
    # 5. Save Model & Results Log
    final_cb.save_model((MODELS_DIR / "cb_best.cbm").as_posix())
    joblib.dump(final_cb, MODELS_DIR / "cb_best.joblib")
    print(f"\nSaved best model to: {(MODELS_DIR / 'cb_best.cbm').as_posix()} and .joblib")
    
    results_entry = {
        'model_name': 'CatBoost',
        'dataset': 'CERT r6.2',
        'winning_strategy': winner,
        'best_hyperparameters': {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best_params.items()},
        'optimal_threshold': float(opt_thresh),
        'cross_validation': {
            'cost_sensitive': cv_cs,
            'resampled': cv_res
        },
        'test_metrics_default_threshold': {
            'threshold': 0.50,
            'precision': float(prec_def),
            'recall': float(rec_def),
            'f1_score': float(f1_def)
        },
        'test_metrics_optimal_threshold': {
            'threshold': float(opt_thresh),
            'precision': float(prec_opt),
            'recall': float(rec_opt),
            'f1_score': float(f1_opt),
            'roc_auc': float(roc_test),
            'pr_auc': float(pr_auc_test),
            'false_positive_rate': float(fpr),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'scenario_recall': scenario_breakdown
        }
    }
    
    log_path = MODELS_DIR / "results_log.json"
    results_log = {}
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                results_log = json.load(f)
        except Exception:
            results_log = {}
    results_log['CatBoost'] = results_entry
    with open(log_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"Saved results to: {log_path.as_posix()}")
    
    # 6. Visualizations
    plot_results(y_test, test_probs, opt_thresh, final_cb.get_feature_importance(), feature_names)
    
    # 7. Comprehensive 3-Model Comparison Leaderboard: RF vs. XGBoost vs. CatBoost
    rf_entry = results_log.get('Random_Forest', {})
    xgb_entry = results_log.get('XGBoost', {})
    cb_entry = results_entry
    
    rf_opt = rf_entry.get('test_metrics_optimal_threshold', {})
    xgb_opt = xgb_entry.get('test_metrics_optimal_threshold', {})
    cb_opt = cb_entry.get('test_metrics_optimal_threshold', {})
    
    rf_sc = rf_opt.get('scenario_recall', {})
    xgb_sc = xgb_opt.get('scenario_recall', {})
    cb_sc = scenario_breakdown
    
    print("\n" + "="*95)
    print("COMPREHENSIVE 3-MODEL COMPARISON LEADERBOARD ON UNTOUCHED TEST SET (CERT r6.2)")
    print("="*95)
    header = f"{'Evaluation Metric':<28} {'Random Forest':<22} {'XGBoost':<22} {'CatBoost':<22}"
    print(header)
    print("-" * 95)
    print(f"{'Winning Imbalance Strategy':<28} {rf_entry.get('winning_strategy', 'N/A').upper():<22} {xgb_entry.get('winning_strategy', 'N/A').upper():<22} {winner.upper():<22}")
    print(f"{'Optimal Decision Threshold (T*)':<28} {rf_opt.get('threshold', 0.5):<22.2f} {xgb_opt.get('threshold', 0.5):<22.2f} {opt_thresh:<22.2f}")
    print(f"{'Precision':<28} {rf_opt.get('precision', 0.0)*100:<21.2f}% {xgb_opt.get('precision', 0.0)*100:<21.2f}% {prec_opt*100:<21.2f}%")
    print(f"{'Recall':<28} {rf_opt.get('recall', 0.0)*100:<21.2f}% {xgb_opt.get('recall', 0.0)*100:<21.2f}% {rec_opt*100:<21.2f}%")
    print(f"{'F1-Score':<28} {rf_opt.get('f1_score', 0.0)*100:<21.2f}% {xgb_opt.get('f1_score', 0.0)*100:<21.2f}% {f1_opt*100:<21.2f}%")
    print(f"{'PR-AUC (Average Precision)':<28} {rf_opt.get('pr_auc', 0.0):<22.4f} {xgb_opt.get('pr_auc', 0.0):<22.4f} {pr_auc_test:<22.4f}")
    print(f"{'ROC-AUC':<28} {rf_opt.get('roc_auc', 0.0):<22.4f} {xgb_opt.get('roc_auc', 0.0):<22.4f} {roc_test:<22.4f}")
    print(f"{'False Positive Rate (FPR)':<28} {rf_opt.get('false_positive_rate', 0.0)*100:<21.5f}% {xgb_opt.get('false_positive_rate', 0.0)*100:<21.5f}% {fpr*100:<21.5f}%")
    print(f"{'False Alarms Count (FP)':<28} {rf_opt.get('confusion_matrix', {}).get('fp', 0):<22} {xgb_opt.get('confusion_matrix', {}).get('fp', 0):<22} {fp:<22}")
    print(f"{'Threats Detected (TP/22)':<28} {rf_opt.get('confusion_matrix', {}).get('tp', 0):<22} {xgb_opt.get('confusion_matrix', {}).get('tp', 0):<22} {tp:<22}")
    print("-" * 95)
    print("Scenario Detection Breakdown (Caught / Total):")
    for s in range(1, 6):
        s_key = f"Scenario_{s}"
        rf_c = rf_sc.get(s_key, {}).get('detected_optimal', 'N/A')
        rf_t = rf_sc.get(s_key, {}).get('total_sessions', 0)
        xgb_c = xgb_sc.get(s_key, {}).get('detected_optimal', 'N/A')
        xgb_t = xgb_sc.get(s_key, {}).get('total_sessions', 0)
        cb_c = cb_sc.get(s_key, {}).get('detected_optimal', 'N/A')
        cb_t = cb_sc.get(s_key, {}).get('total_sessions', 0)
        print(f"  Scenario {s}:{'':<18} {rf_c}/{rf_t:<20} {xgb_c}/{xgb_t:<20} {cb_c}/{cb_t:<20}")
    print("="*95)
    
    print("\n" + "="*75)
    print("STEP 4: CATBOOST TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("="*75)


if __name__ == "__main__":
    main()
