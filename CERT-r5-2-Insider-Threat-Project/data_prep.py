"""
Data Preparation & Leakage-Free Preprocessing for CERT r4.2
Strict train-test isolation (80/20 stratified split).
Statistical feature selection (ANOVA F-score + Mutual Information Top 50) fit strictly on train.
Generates two training variants: Raw Imbalanced and Balanced (SMOTE + UnderSampler).
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

ROOT_DIR = Path(__file__).resolve().parent
PREP_DIR = ROOT_DIR / "prepared_data"
PREP_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dataset():
    """Loads session dataset from parquet (or csv fallback)."""
    parquet_path = ROOT_DIR / "session.parquet"
    csv_path = ROOT_DIR / "session.csv"
    
    if parquet_path.exists():
        print(f"Loading session dataset from {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"Loading session dataset from {csv_path.name}...")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Neither session.parquet nor session.csv found in workspace!")
        
    print(f"Loaded raw session matrix: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean_and_separate(df):
    """
    Separates metadata/identifiers from features, handles NaNs/Infs,
    and maps target to binary 0/1.
    """
    print("\n--- Cleaning & Schema Hygiene ---")
    
    # Metadata columns to drop from model features
    metadata_cols = [
        'Unnamed: 0', 'starttime', 'endtime', 'sessionid', 
        'user', 'pc', 'week', 'day'
    ]
    cols_to_drop = [c for c in metadata_cols if c in df.columns]
    
    # Separate target
    if 'insider' not in df.columns:
        raise ValueError("Target column 'insider' missing from dataset!")
        
    # Binary mapping: 0 = normal, 1 = malicious insider (scenarios 1, 2, 3)
    target_raw = df['insider'].values
    y = np.where(target_raw > 0, 1, 0).astype(np.int32)
    
    # Feature matrix
    X = df.drop(columns=cols_to_drop + ['insider']).copy()
    
    # Handle infinite values safely
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target distribution y: Benign (0) = {np.sum(y == 0):,} ({np.mean(y == 0)*100:.3f}%), "
          f"Insider (1) = {np.sum(y == 1):,} ({np.mean(y == 1)*100:.3f}%)")
    
    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Performs stratified 80/20 train/test split.
    Test set remains completely isolated.
    """
    print(f"\n--- Leakage-Free Train/Test Split ({(1-test_size)*100:.0f}/{test_size*100:.0f} Stratified) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    
    print(f"Train Set: {X_train.shape[0]:,} sessions (Insiders: {np.sum(y_train == 1):,}, {np.mean(y_train == 1)*100:.3f}%)")
    print(f"Test Set:  {X_test.shape[0]:,} sessions (Insiders: {np.sum(y_test == 1):,}, {np.mean(y_test == 1)*100:.3f}%)")
    
    return X_train, X_test, y_train, y_test


def select_top_features(X_train, y_train, X_test, top_k=50, random_state=42):
    """
    Statistical Feature Selection fit STRICTLY on X_train:
    Combines ANOVA F-score (linear separation) and Mutual Information (non-linear dependency).
    Transforms X_train and X_test to top_k features.
    """
    print(f"\n--- Statistical Feature Selection (ANOVA F + Mutual Information, Top {top_k}) ---")
    t0 = time.time()
    
    # Impute missing values with median fit strictly on train
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # 1. ANOVA F-score
    print("Computing ANOVA F-scores on Train...")
    f_scores, _ = f_classif(X_train_imp, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Mutual Information
    print("Computing Mutual Information on Train...")
    # Use stratified subsample for MI if train is large to balance precision and speed
    if len(X_train_imp) > 100000:
        sub_X, _, sub_y, _ = train_test_split(
            X_train_imp, y_train, train_size=50000, random_state=random_state, stratify=y_train
        )
        mi_scores = mutual_info_classif(sub_X, sub_y, random_state=random_state, n_neighbors=3)
    else:
        mi_scores = mutual_info_classif(X_train_imp, y_train, random_state=random_state, n_neighbors=3)
        
    ranked_df = pd.DataFrame({
        'Feature': X_train.columns,
        'ANOVA_F': f_scores,
        'MI': mi_scores
    })
    
    ranked_df['ANOVA_Rank'] = ranked_df['ANOVA_F'].rank(ascending=False, method='min')
    ranked_df['MI_Rank'] = ranked_df['MI'].rank(ascending=False, method='min')
    ranked_df['Combined_Rank'] = (ranked_df['ANOVA_Rank'] + ranked_df['MI_Rank']) / 2.0
    ranked_df = ranked_df.sort_values(by='Combined_Rank').reset_index(drop=True)
    
    top_features = ranked_df.head(top_k)['Feature'].tolist()
    
    print(f"Feature selection completed in {time.time()-t0:.2f}s")
    print(f"\nTop 15 Discriminative Features:")
    print(f"{'Rank':<6} {'Feature Name':<35} {'ANOVA F':<12} {'MI Score':<12} {'Combined Rank'}")
    print("-" * 75)
    for idx, row in ranked_df.head(15).iterrows():
        print(f"{idx+1:<6} {row['Feature']:<35} {row['ANOVA_F']:<12.2f} {row['MI']:<12.4f} {row['Combined_Rank']:.1f}")
        
    # Filter datasets to top_k features
    X_train_selected = X_train_imp[top_features].copy()
    X_test_selected = X_test_imp[top_features].copy()
    
    # Save ranked table and feature list
    ranked_df.to_csv(PREP_DIR / "feature_rankings_all.csv", index=False)
    with open(PREP_DIR / "selected_features.json", "w") as f:
        json.dump(top_features, f, indent=2)
        
    # Robust scaling fit strictly on train
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=top_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_selected), columns=top_features)
    
    # Save fitted preprocessors
    joblib.dump(imputer, PREP_DIR / "imputer.joblib")
    joblib.dump(scaler, PREP_DIR / "scaler.joblib")
    
    return X_train_scaled, X_test_scaled, top_features


def handle_imbalance(X_train, y_train, smote_ratio=0.1, undersampling_ratio=0.5, random_state=42):
    """
    Prepares two training variants:
    1. Raw Imbalanced Train: Preserves genuine distribution.
    2. Resampled Balanced Train: SMOTE followed by Random Undersampling.
    """
    print("\n--- Training Set Imbalance Variants (Fit Strictly on Train) ---")
    
    # Variant A: Raw
    X_train_raw = X_train.copy()
    y_train_raw = y_train.copy()
    
    # Variant B: Resampled (SMOTE + RandomUnderSampler)
    print(f"Applying SMOTE (target ratio={smote_ratio}) + RandomUnderSampler (target ratio={undersampling_ratio})...")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    
    rus = RandomUnderSampler(sampling_strategy=undersampling_ratio, random_state=random_state)
    X_train_res, y_train_res = rus.fit_resample(X_sm, y_sm)
    
    print(f"Raw Imbalanced Train:  Class 0 = {np.sum(y_train_raw == 0):,}, Class 1 = {np.sum(y_train_raw == 1):,}")
    print(f"Resampled Balanced Train: Class 0 = {np.sum(y_train_res == 0):,}, Class 1 = {np.sum(y_train_res == 1):,}")
    
    return X_train_raw, y_train_raw, X_train_res, y_train_res


def save_prepared_splits(X_train_raw, y_train_raw, X_train_res, y_train_res, X_test, y_test):
    """Saves prepared data splits to prepared_data/ in Parquet format."""
    print("\n--- Saving Prepared Data Splits ---")
    
    X_train_raw.to_parquet(PREP_DIR / "X_train_raw.parquet", index=False)
    pd.DataFrame({'insider': y_train_raw}).to_parquet(PREP_DIR / "y_train_raw.parquet", index=False)
    
    X_train_res.to_parquet(PREP_DIR / "X_train_res.parquet", index=False)
    pd.DataFrame({'insider': y_train_res}).to_parquet(PREP_DIR / "y_train_res.parquet", index=False)
    
    X_test.to_parquet(PREP_DIR / "X_test.parquet", index=False)
    pd.DataFrame({'insider': y_test}).to_parquet(PREP_DIR / "y_test.parquet", index=False)
    
    print(f"Saved prepared splits to {PREP_DIR.as_posix()}:")
    for p in PREP_DIR.glob("*.parquet"):
        print(f"  - {p.name:<25} ({p.stat().st_size / (1024*1024):.2f} MB)")


def main():
    df = load_raw_dataset()
    X, y = clean_and_separate(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    X_train_scaled, X_test_scaled, top_features = select_top_features(X_train, y_train, X_test, top_k=50)
    X_train_raw, y_train_raw, X_train_res, y_train_res = handle_imbalance(X_train_scaled, y_train)
    save_prepared_splits(X_train_raw, y_train_raw, X_train_res, y_train_res, X_test_scaled, y_test)
    
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION COMPLETED SUCCESSFULLY (ZERO LEAKAGE)")
    print("="*50)


if __name__ == "__main__":
    main()
