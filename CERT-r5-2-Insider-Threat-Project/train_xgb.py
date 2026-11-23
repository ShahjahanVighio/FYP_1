"""
Step 3: Train & Evaluate Model 2 — XGBoost (train_xgb.py)
Cross-validation comparison of Cost-Sensitive (scale_pos_weight) vs. SMOTE+UnderSampling.
Hyperparameter tuning, early stopping, OOF probability threshold optimization,
and final evaluation on untouched X_test with side-by-side comparison against Random Forest.
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

import xgboost as xgb
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
PLOTS_DIR = ROOT_DIR / "plots" / "xgb"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Loads prepared data splits from prepared_data/."""
    print("Loading prepared data splits...")
    X_train_raw = pd.read_parquet(PREP_DIR / "X_train_raw.parquet")
    y_train_raw = pd.read_parquet(PREP_DIR / "y_train_raw.parquet")['insider'].values
    
    X_train_res = pd.read_parquet(PREP_DIR / "X_train_res.parquet")
    y_train_res = pd.read_parquet(PREP_DIR / "y_train_res.parquet")['insider'].values
    
    X_test = pd.read_parquet(PREP_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PREP_DIR / "y_test.parquet")['insider'].values
    
    with open(PREP_DIR / "selected_features.json", "r") as f:
        feature_names = json.load(f)
        
    neg_count = np.sum(y_train_raw == 0)
    pos_count = np.sum(y_train_raw == 1)
    ratio = neg_count / pos_count
    sqrt_ratio = np.sqrt(ratio)
    
    print(f"X_train_raw: {X_train_raw.shape}, Class 0: {neg_count:,}, Class 1: {pos_count:,} (Ratio = {ratio:.2f}, Sqrt Ratio = {sqrt_ratio:.2f})")
    print(f"X_train_res: {X_train_res.shape}, Class 0: {np.sum(y_train_res == 0):,}, Class 1: {np.sum(y_train_res == 1):,}")
    print(f"X_test:      {X_test.shape}, Class 0: {np.sum(y_test == 0):,}, Class 1: {np.sum(y_test == 1):,} [ISOLATED]")
    
    return X_train_raw, y_train_raw, X_train_res, y_train_res, X_test, y_test, feature_names, ratio, sqrt_ratio


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


def evaluate_variant_cv(X_train_raw, y_train_raw, variant="cost_sensitive", ratio=421.5, sqrt_ratio=20.5, n_iter=8, random_state=42):
    """
    Performs 5-Fold Stratified Cross-Validation on X_train_raw using XGBoost.
    - variant='cost_sensitive': Tests scale_pos_weight (1.0, sqrt_ratio, ratio) on raw imbalanced folds.
    - variant='resampled': Applies SMOTE + Undersampling inside each training fold (scale_pos_weight=1.0).
    Validates strictly on untouched imbalanced validation folds.
    Returns: oof_probs, best_params, cv_summary.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating XGBoost: Variant {variant.upper()}")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y_train_raw))
    
    if variant == "cost_sensitive":
        param_dist = {
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [150, 250, 400],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0.0, 0.1, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
            'scale_pos_weight': [1.0, float(sqrt_ratio), float(ratio)]
        }
        
        base_xgb = xgb.XGBClassifier(
            tree_method='hist',
            eval_metric='aucpr',
            random_state=random_state,
            n_jobs=-1
        )
        
        # Fast representative tuning on a stratified sample
        tune_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        tune_idx, _ = next(tune_skf.split(X_train_raw, y_train_raw))
        X_tune, y_tune = X_train_raw.iloc[tune_idx], y_train_raw[tune_idx]
        
        print(f"Running RandomizedSearchCV ({n_iter} iterations) on {len(X_tune):,} sample...")
        t0 = time.time()
        rs = RandomizedSearchCV(
            base_xgb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='average_precision',
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        rs.fit(X_tune, y_tune)
        best_params = rs.best_params_
        print(f"Tuning completed in {time.time()-t0:.1f}s! Best parameters: {best_params}")
        
    else: # resampled
        param_dist = {
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [150, 250, 400],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0.0, 0.1, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
            'scale_pos_weight': [1.0]
        }
        
        base_xgb = xgb.XGBClassifier(
            tree_method='hist',
            eval_metric='aucpr',
            random_state=random_state,
            n_jobs=-1
        )
        
        # Load precomputed balanced training set for fast tuning
        X_res_tune = pd.read_parquet(PREP_DIR / "X_train_res.parquet")
        y_res_tune = pd.read_parquet(PREP_DIR / "y_train_res.parquet")['insider'].values
        
        print(f"Running RandomizedSearchCV ({n_iter} iterations) on resampled training set...")
        t0 = time.time()
        rs = RandomizedSearchCV(
            base_xgb,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='average_precision',
            cv=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        rs.fit(X_res_tune, y_res_tune)
        best_params = rs.best_params_
        print(f"Tuning completed in {time.time()-t0:.1f}s! Best parameters: {best_params}")

    # 5-Fold Stratified Cross-Validation with Best Parameters & Early Stopping
    fold_ap = []
    fold_roc = []
    fold_f1 = []
    fold_prec = []
    fold_rec = []
    
    print("\nRunning 5-Fold Cross-Validation with early stopping (50 rounds)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train_raw), 1):
        t_fold = time.time()
        X_tr, y_tr = X_train_raw.iloc[train_idx], y_train_raw[train_idx]
        X_va, y_va = X_train_raw.iloc[val_idx], y_train_raw[val_idx]
        
        if variant == "cost_sensitive":
            clf = xgb.XGBClassifier(
                **best_params,
                tree_method='hist',
                eval_metric='aucpr',
                early_stopping_rounds=50,
                random_state=random_state + fold,
                n_jobs=-1
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else: # resampled inside fold
            smote = SMOTE(sampling_strategy=0.1, random_state=random_state + fold)
            rus = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state + fold)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            X_tr_res, y_tr_res = rus.fit_resample(X_tr_sm, y_tr_sm)
            clf = xgb.XGBClassifier(
                **best_params,
                tree_method='hist',
                eval_metric='aucpr',
                early_stopping_rounds=50,
                random_state=random_state + fold,
                n_jobs=-1
            )
            clf.fit(X_tr_res, y_tr_res, eval_set=[(X_va, y_va)], verbose=False)
            
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
        
        best_iter = clf.best_iteration if hasattr(clf, 'best_iteration') and clf.best_iteration is not None else best_params.get('n_estimators')
        print(f"  Fold {fold}: Best Iter = {best_iter}, PR-AUC = {ap:.4f}, ROC-AUC = {roc:.4f}, F1 = {f1:.4f}, Prec = {prec:.4f}, Rec = {rec:.4f} ({time.time()-t_fold:.1f}s)")
        
    overall_ap = average_precision_score(y_train_raw, oof_probs)
    overall_roc = roc_auc_score(y_train_raw, oof_probs)
    opt_thresh, opt_f1 = find_optimal_threshold(y_train_raw, oof_probs)
    oof_preds_opt = (oof_probs >= opt_thresh).astype(int)
    opt_prec = precision_score(y_train_raw, oof_preds_opt, zero_division=0)
    opt_rec = recall_score(y_train_raw, oof_preds_opt, zero_division=0)
    
    cv_summary = {
        'variant': variant,
        'best_params': best_params,
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
        'opt_rec': float(opt_rec)
    }
    
    print(f"\n--- {variant.upper()} OOF Summary ---")
    print(f"  OOF PR-AUC (Average Precision): {overall_ap:.4f}")
    print(f"  OOF ROC-AUC:                    {overall_roc:.4f}")
    print(f"  At default threshold (0.50):    F1 = {np.mean(fold_f1):.4f}, Prec = {np.mean(fold_prec):.4f}, Rec = {np.mean(fold_rec):.4f}")
    print(f"  At optimal threshold ({opt_thresh:.2f}):  F1 = {opt_f1:.4f}, Prec = {opt_prec:.4f}, Rec = {opt_rec:.4f}")
    
    return oof_probs, best_params, cv_summary


def plot_results(y_test, y_probs, opt_thresh, feature_importances, feature_names):
    """
    Generates and saves high-resolution publication-quality plots.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Confusion Matrix
    y_pred_opt = (y_probs >= opt_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt=',d', cmap='Oranges', cbar=False, ax=ax,
        xticklabels=['Benign (0)', 'Insider (1)'],
        yticklabels=['Benign (0)', 'Insider (1)']
    )
    ax.set_title(f"XGBoost Confusion Matrix\n(Threshold = {opt_thresh:.2f})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class", fontweight='bold')
    ax.set_ylabel("True Class", fontweight='bold')
    plt.tight_layout()
    cm_path = PLOTS_DIR / "xgb_confusion_matrix.png"
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
    ax1.plot(recall, precision, color='#d62728', lw=2.5, label=f'XGBoost (PR-AUC = {ap:.4f})')
    p_opt = precision_score(y_test, y_pred_opt, zero_division=0)
    r_opt = recall_score(y_test, y_pred_opt, zero_division=0)
    ax1.plot(r_opt, p_opt, marker='o', markersize=8, color='black', label=f'Opt Thresh ({opt_thresh:.2f})')
    ax1.set_xlabel('Recall', fontweight='bold')
    ax1.set_ylabel('Precision', fontweight='bold')
    ax1.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    
    # ROC Curve
    ax2.plot(fpr, tpr, color='#9467bd', lw=2.5, label=f'XGBoost (ROC-AUC = {roc:.4f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance')
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Recall)', fontweight='bold')
    ax2.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim([0.0, 1.05])
    ax2.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    pr_roc_path = PLOTS_DIR / "xgb_pr_roc_curves.png"
    plt.savefig(pr_roc_path)
    plt.close()
    print(f"Saved: {pr_roc_path.as_posix()}")
    
    # 3. Top 20 Feature Importances (Gain)
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', palette='flare_r', legend=False, ax=ax)
    ax.set_title('XGBoost — Top 20 Feature Importances (Gain)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Gain Importance', fontweight='bold')
    ax.set_ylabel('Feature Name', fontweight='bold')
    plt.tight_layout()
    fi_path = PLOTS_DIR / "xgb_feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved: {fi_path.as_posix()}")


def main():
    X_train_raw, y_train_raw, X_train_res, y_train_res, X_test, y_test, feature_names, ratio, sqrt_ratio = load_data()
    
    # 1. Compare both imbalance strategies on 5-fold CV
    oof_cs, params_cs, cv_cs = evaluate_variant_cv(
        X_train_raw, y_train_raw, variant="cost_sensitive", ratio=ratio, sqrt_ratio=sqrt_ratio, n_iter=8, random_state=42
    )
    
    oof_res, params_res, cv_res = evaluate_variant_cv(
        X_train_raw, y_train_raw, variant="resampled", ratio=ratio, sqrt_ratio=sqrt_ratio, n_iter=8, random_state=42
    )
    
    # 2. Select winning variant based on validation PR-AUC & F1
    print("\n" + "="*60)
    print("VALIDATION COMPARISON: COST-SENSITIVE VS. RESAMPLED (XGBOOST)")
    print("="*60)
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
    
    if cv_res['cv_mean_ap'] > cv_cs['cv_mean_ap']:
        winner = "resampled"
        best_params = params_res
        opt_thresh = cv_res['opt_threshold']
        print(f"\nWinning Strategy: RESAMPLED (SMOTE + UnderSampling) [Higher PR-AUC: {cv_res['cv_mean_ap']:.4f}]")
    else:
        winner = "cost_sensitive"
        best_params = params_cs
        opt_thresh = cv_cs['opt_threshold']
        print(f"\nWinning Strategy: COST-SENSITIVE (scale_pos_weight={best_params.get('scale_pos_weight', 1.0):.2f}) [Higher PR-AUC: {cv_cs['cv_mean_ap']:.4f}]")
        
    # 3. Train final model on full training set using winning strategy
    print(f"\nTraining final XGBoost on full training data using {winner.upper()} strategy...")
    t0 = time.time()
    if winner == "resampled":
        final_xgb = xgb.XGBClassifier(
            **best_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        )
        final_xgb.fit(X_train_res, y_train_res)
    else:
        final_xgb = xgb.XGBClassifier(
            **best_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        )
        final_xgb.fit(X_train_raw, y_train_raw)
    print(f"Final XGBoost model trained in {time.time()-t0:.2f}s!")
    
    # 4. Final Evaluation on Untouched Test Set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON UNTOUCHED TEST SET (X_test)")
    print("="*60)
    test_probs = final_xgb.predict_proba(X_test)[:, 1]
    
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
    
    print(f"Test Set Size:              {len(y_test):,} sessions ({np.sum(y_test == 1):,} malicious)")
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
    print(f"  False Positive Rate (FPR): {fpr*100:.4f}% ({fp} false alarms out of {tn+fp:,} benign sessions)")
    
    print("\nDetailed Classification Report (Optimal Threshold):")
    print(classification_report(y_test, preds_opt, target_names=['Benign (0)', 'Insider (1)'], digits=4))
    
    # 5. Save Artifacts & Visualizations
    joblib.dump(final_xgb, MODELS_DIR / "xgb_best.joblib")
    print(f"Saved best model to: {(MODELS_DIR / 'xgb_best.joblib').as_posix()}")
    
    # Results log entry
    results_entry = {
        'model_name': 'XGBoost',
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
            }
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
    results_log['XGBoost'] = results_entry
    with open(log_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"Saved results to: {log_path.as_posix()}")
    
    # Visualizations
    plot_results(y_test, test_probs, opt_thresh, final_xgb.feature_importances_, feature_names)
    
    # 6. Side-by-Side Comparison: Random Forest vs. XGBoost
    rf_entry = results_log.get('Random_Forest', {})
    if rf_entry:
        rf_opt = rf_entry.get('test_metrics_optimal_threshold', {})
        print("\n" + "="*80)
        print("MODEL COMPARISON ON UNTOUCHED TEST SET (X_test): RANDOM FOREST VS. XGBOOST")
        print("="*80)
        print(f"{'Evaluation Metric':<28} {'Random Forest':<25} {'XGBoost':<25}")
        print("-" * 80)
        print(f"{'Winning Imbalance Strategy':<28} {rf_entry.get('winning_strategy', 'N/A').upper():<25} {winner.upper():<25}")
        print(f"{'Optimal Decision Threshold':<28} {rf_opt.get('threshold', 0.5):<25.2f} {opt_thresh:<25.2f}")
        print(f"{'Precision':<28} {rf_opt.get('precision', 0.0)*100:<24.2f}% {prec_opt*100:<24.2f}%")
        print(f"{'Recall':<28} {rf_opt.get('recall', 0.0)*100:<24.2f}% {rec_opt*100:<24.2f}%")
        print(f"{'F1-Score':<28} {rf_opt.get('f1_score', 0.0)*100:<24.2f}% {f1_opt*100:<24.2f}%")
        print(f"{'PR-AUC (Average Precision)':<28} {rf_opt.get('pr_auc', 0.0):<25.4f} {pr_auc_test:<25.4f}")
        print(f"{'ROC-AUC':<28} {rf_opt.get('roc_auc', 0.0):<25.4f} {roc_test:<25.4f}")
        print(f"{'False Positive Rate (FPR)':<28} {rf_opt.get('false_positive_rate', 0.0)*100:<24.4f}% {fpr*100:<24.4f}%")
        print(f"{'False Alarms Count (FP)':<28} {rf_opt.get('confusion_matrix', {}).get('fp', 0):<25} {fp:<25}")
        print(f"{f'Threats Detected (TP/{np.sum(y_test == 1)})':<28} {rf_opt.get('confusion_matrix', {}).get('tp', 0):<25} {tp:<25}")
        print("="*80)
        
    print("\n" + "="*60)
    print("STEP 3: XGBOOST TRAINING & EVALUATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
