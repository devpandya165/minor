#!/usr/bin/env python3
"""
Complete Model Training Pipeline
- Trains GLOBAL model
- Trains per-state RESIDUAL models (only when beneficial)
- Uses REALISTIC data (with anomalies)
- MAINTAINS EXACT SAME INPUT FORMAT (NO CHANGES TO MODEL INTERFACE)
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from typing import Dict, Tuple, List
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names, but LGBMRegressor was fitted with feature names.*",
)

# ============================================================
# CONFIGURATION
# ============================================================

# USE REALISTIC DATA (with anomalies/missingness)
DATA_FILENAME = r"C:\Users\DEV\Desktop\Minor Project\Project\Data\generate_visualize\synthetic_indian_load_data.csv"
TARGET_COL = "Gross_Load_MW"

# Time-series features
LAG_HOURS = [1, 2, 3, 24, 168]
ROLLING_WINDOWS = [3, 24, 168]

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Personalization threshold
MIN_REL_IMPROVEMENT = 0.05  # 5%

# Model paths
MODEL_DIR = r"C:\Users\DEV\Desktop\Minor Project\models"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_model.joblib")
STATE_MODEL_DIR = os.path.join(MODEL_DIR, "state_models")
TRAINING_REPORT_PATH = os.path.join(MODEL_DIR, "training_report. txt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATE_MODEL_DIR, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

log_file = open(TRAINING_REPORT_PATH, 'w', encoding='utf-8')

def log_print(*args, **kwargs):
    """Print to both console and file."""
    message = ' '.join(str(arg) for arg in args)
    print(message, **kwargs)
    log_file.write(message + '\n')
    log_file.flush()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(filename: str) -> pd.DataFrame:
    """Load CSV data."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    log_print(f"📂 Loading data from: {filename}")
    df = pd.read_csv(filename)
    df["Timestamp_UTC"] = pd.to_datetime(df["Timestamp_UTC"])
    df = df.sort_values(["State_Code", "Timestamp_UTC"]).reset_index(drop=True)
    
    log_print(f"   ✓ Loaded {len(df):,} rows, {df['State_Code'].nunique()} states")
    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling window features."""
    df = df.copy()
    group = df.groupby("State_Code")

    # Lag features
    for lag in LAG_HOURS:
        df[f"Load_lag_{lag}h"] = group[TARGET_COL].shift(lag)

    # Rolling window features
    for window in ROLLING_WINDOWS:
        df[f"Load_roll_mean_{window}h"] = (
            group[TARGET_COL]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    return df


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data: 70% train, 15% val, 15% test."""
    df = df.sort_values("Timestamp_UTC").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]. reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:]. reset_index(drop=True)
    
    return train_df, val_df, test_df


def build_feature_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract features and target - SAME FORMAT AS ORIGINAL."""
    feature_cols = [
        "State_Code",
        "Hour_Of_Day",
        "Day_Of_Week",
        "Is_Weekend",
        "Is_Holiday_State",
        "Avg_Temp_C",
        "Temp_Change_6H",
        "Avg_Humidity_Pct",
    ]
    
    # Add lag and rolling features
    lag_roll_cols = [c for c in df.columns if c.startswith("Load_lag_") or c.startswith("Load_roll_mean_")]
    feature_cols.extend(lag_roll_cols)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    
    return X, y, feature_cols


def build_preprocessor(feature_cols: List[str]):
    """Create OneHot encoder for State_Code - SAME AS ORIGINAL."""
    categorical_features = ["State_Code"]
    numeric_features = [c for c in feature_cols if c != "State_Code"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    return preprocessor


def rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, eps=1e-6):
    """Calculate MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_global_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Train global model on all data - EXACT SAME ARCHITECTURE."""
    
    log_print("\n" + "="*80)
    log_print("STEP 1: TRAINING GLOBAL MODEL")
    log_print("="*80)
    
    preprocessor = build_preprocessor(feature_cols)
    
    log_print("\n🔧 Building global LightGBM model...")
    log_print("   Architecture: OneHotEncoder(State_Code) + LightGBM Regressor")
    
    # EXACT SAME HYPERPARAMETERS AS ORIGINAL
    global_model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    global_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", global_model),
        ]
    )

    log_print("📚 Training on", len(X_train), "samples...")
    global_pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    log_print("\n📊 Evaluating on validation set...")
    y_val_pred = global_pipeline.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = rmse(y_val, y_val_pred)
    val_mape = mape(y_val, y_val_pred)
    
    log_print(f"   MAE:  {val_mae:.2f} MW")
    log_print(f"   RMSE: {val_rmse:.2f} MW")
    log_print(f"   MAPE: {val_mape:.2f}%")

    # Evaluate on test set
    log_print("\n📊 Evaluating on test set...")
    y_test_pred = global_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = rmse(y_test, y_test_pred)
    test_mape = mape(y_test, y_test_pred)
    
    log_print(f"   MAE:  {test_mae:.2f} MW")
    log_print(f"   RMSE: {test_rmse:.2f} MW")
    log_print(f"   MAPE: {test_mape:.2f}%")

    # Save model - EXACT SAME FORMAT
    log_print(f"\n💾 Saving global model to: {GLOBAL_MODEL_PATH}")
    joblib.dump(global_pipeline, GLOBAL_MODEL_PATH)
    log_print("   ✓ Model saved")

    return global_pipeline, test_rmse, test_mape, y_test_pred


def train_residual_models(train_df, val_df, test_df, global_pipeline, feature_cols, global_test_rmse):
    """Train per-state residual correction models - SAME INPUT/OUTPUT FORMAT."""
    
    log_print("\n" + "="*80)
    log_print("STEP 2: TRAINING PER-STATE RESIDUAL MODELS")
    log_print("="*80)
    log_print(f"Threshold: Only keep models that improve global RMSE by > {MIN_REL_IMPROVEMENT*100:.1f}%\n")

    # Calculate global predictions for residuals
    X_train, _, _ = build_feature_matrices(train_df)
    X_val, _, _ = build_feature_matrices(val_df)
    X_test, _, _ = build_feature_matrices(test_df)
    
    train_df["global_pred"] = global_pipeline. predict(X_train)
    train_df["residual"] = train_df[TARGET_COL] - train_df["global_pred"]
    
    val_df["global_pred"] = global_pipeline.predict(X_val)
    val_df["residual"] = val_df[TARGET_COL] - val_df["global_pred"]
    
    test_df["global_pred"] = global_pipeline.predict(X_test)
    test_df["residual"] = test_df[TARGET_COL] - test_df["global_pred"]

    # Train per-state models
    states = sorted(train_df["State_Code"].unique())
    kept_states = []
    state_results = []

    for idx, state in enumerate(states, 1):
        log_print(f"[{idx}/{len(states)}] Training residual model for state: {state}")
        
        # Get state-specific splits
        train_s = train_df[train_df["State_Code"] == state]
        val_s = val_df[val_df["State_Code"] == state]
        test_s = test_df[test_df["State_Code"] == state]

        # Check minimum samples
        if len(train_s) < 300:
            log_print(f"        ⚠️  Skipped: Only {len(train_s)} training samples (need >= 300)")
            continue

        if len(test_s) == 0:
            log_print(f"        ⚠️  Skipped: No test data")
            continue

        # Build residual training data - SAME FEATURE FORMAT
        X_train_s, _, _ = build_feature_matrices(train_s)
        y_train_s = train_s["residual"]
        
        X_val_s, _, _ = build_feature_matrices(val_s)
        
        X_test_s, _, _ = build_feature_matrices(test_s)
        y_test_true = test_s[TARGET_COL].values
        y_test_global = test_s["global_pred"].values

        # Train residual model with SAME ARCHITECTURE
        preprocessor_s = build_preprocessor(feature_cols)
        
        model_s = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.07,
            num_leaves=48,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        pipe_s = Pipeline(
            steps=[
                ("preprocess", preprocessor_s),
                ("model", model_s),
            ]
        )

        log_print(f"        📚 Training on {len(train_s)} samples...")
        pipe_s.fit(X_train_s, y_train_s)

        # Evaluate correction
        y_test_residual = pipe_s.predict(X_test_s)
        y_test_corrected = y_test_global + y_test_residual

        rmse_global = rmse(y_test_true, y_test_global)
        rmse_corrected = rmse(y_test_true, y_test_corrected)
        mape_global = mape(y_test_true, y_test_global)
        mape_corrected = mape(y_test_true, y_test_corrected)
        
        improvement = (rmse_global - rmse_corrected) / (rmse_global + 1e-9)

        log_print(f"        Global RMSE:    {rmse_global:.2f} MW")
        log_print(f"        Corrected RMSE: {rmse_corrected:.2f} MW")
        log_print(f"        Improvement:    {improvement*100:.2f}%")

        # Decision: Keep or discard
        if improvement > MIN_REL_IMPROVEMENT:
            kept_states.append(state)
            model_path = os.path.join(STATE_MODEL_DIR, f"state_{state}_residual_model.joblib")
            joblib.dump(pipe_s, model_path)
            log_print(f"        ✅ SAVED (improvement > {MIN_REL_IMPROVEMENT*100:.1f}%)")
        else:
            log_print(f"        ❌ NOT saved (improvement < {MIN_REL_IMPROVEMENT*100:.1f}%)")

        state_results.append({
            'State': state,
            'Train_Samples': len(train_s),
            'Test_Samples': len(test_s),
            'Global_RMSE': rmse_global,
            'Corrected_RMSE': rmse_corrected,
            'Improvement_%': improvement * 100,
            'Saved': 'Yes' if improvement > MIN_REL_IMPROVEMENT else 'No',
        })

    return kept_states, state_results


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    log_print("\n" + "#"*80)
    log_print("# COMPLETE MODEL TRAINING PIPELINE")
    log_print("# Training Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log_print("# Data Source: REALISTIC (with anomalies)")
    log_print("#"*80)

    try:
        # ---- STEP 0: DATA LOADING & PREPROCESSING ----
        log_print("\n" + "="*80)
        log_print("STEP 0: DATA PREPARATION")
        log_print("="*80)

        df = load_dataset(DATA_FILENAME)
        
        # Drop missing targets
        initial_rows = len(df)
        df = df.dropna(subset=[TARGET_COL]). reset_index(drop=True)
        dropped_missing = initial_rows - len(df)
        log_print(f"   ✓ Dropped {dropped_missing} rows with missing targets")

        # Add time-series features
        log_print("\n📈 Adding time-series features...")
        log_print(f"   Lags: {LAG_HOURS}")
        log_print(f"   Rolling windows: {ROLLING_WINDOWS}")
        df = add_time_series_features(df)
        
        initial_rows = len(df)
        df = df.dropna(). reset_index(drop=True)
        dropped_features = initial_rows - len(df)
        log_print(f"   ✓ Dropped {dropped_features} rows with NaN features")

        # Split data
        log_print("\n📊 Time-based split (70/15/15)...")
        train_df, val_df, test_df = time_based_split(df)
        log_print(f"   Train: {len(train_df):,} samples")
        log_print(f"   Val:   {len(val_df):,} samples")
        log_print(f"   Test:  {len(test_df):,} samples")

        # Build feature matrices
        X_train, y_train, feature_cols = build_feature_matrices(train_df)
        X_val, y_val, _ = build_feature_matrices(val_df)
        X_test, y_test, _ = build_feature_matrices(test_df)
        
        log_print(f"\n   Features ({len(feature_cols)}): ")
        log_print(f"   {', '.join(feature_cols)}")

        # ---- STEP 1: TRAIN GLOBAL MODEL ----
        global_pipeline, global_test_rmse, global_test_mape, y_test_pred_global = train_global_model(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
        )

        # ---- STEP 2: TRAIN RESIDUAL MODELS ----
        kept_states, state_results = train_residual_models(
            train_df, val_df, test_df, global_pipeline, feature_cols, global_test_rmse
        )

        # ---- SUMMARY ----
        log_print("\n" + "="*80)
        log_print("TRAINING SUMMARY")
        log_print("="*80)
        log_print(f"\n✅ GLOBAL MODEL:")
        log_print(f"   Test RMSE: {global_test_rmse:.2f} MW")
        log_print(f"   Test MAPE: {global_test_mape:.2f}%")
        log_print(f"   Location:  {GLOBAL_MODEL_PATH}")
        log_print(f"   Input Format: DataFrame with {len(feature_cols)} features")

        log_print(f"\n✅ PER-STATE RESIDUAL MODELS:")
        log_print(f"   Total states: 36")
        log_print(f"   States with residual models: {len(kept_states)}")
        log_print(f"   States: {', '.join(sorted(kept_states))}")
        log_print(f"   Location: {STATE_MODEL_DIR}")

        log_print(f"\n📊 DETAILED RESULTS:")
        results_df = pd.DataFrame(state_results)
        results_csv = os.path.join(MODEL_DIR, "training_results. csv")
        results_df.to_csv(results_csv, index=False)
        log_print(f"   Saved to: {results_csv}")
        
        log_print("\n" + "="*80)
        log_print("Top 10 states by improvement:")
        log_print("="*80)
        results_sorted = results_df.sort_values('Improvement_%', ascending=False).head(10)
        for idx, (_, row) in enumerate(results_sorted.iterrows(), 1):
            status = "✅ SAVED" if row['Saved'] == 'Yes' else "❌ REJECTED"
            log_print(f"{idx:2d}. {row['State']:>3} | RMSE: {row['Global_RMSE']:>8.2f} → {row['Corrected_RMSE']:>8.2f} | "
                     f"Improvement: {row['Improvement_%']:>6.2f}% | {status}")

        log_print("\n" + "="*80)
        log_print("Bottom 10 states (least improvement):")
        log_print("="*80)
        results_sorted_bottom = results_df.sort_values('Improvement_%', ascending=True).head(10)
        for idx, (_, row) in enumerate(results_sorted_bottom.iterrows(), 1):
            status = "✅ SAVED" if row['Saved'] == 'Yes' else "❌ REJECTED"
            log_print(f"{idx:2d}. {row['State']:>3} | RMSE: {row['Global_RMSE']:>8.2f} → {row['Corrected_RMSE']:>8.2f} | "
                     f"Improvement: {row['Improvement_%']:>6.2f}% | {status}")

        log_print("\n" + "#"*80)
        log_print("# ✅ TRAINING COMPLETE - ALL MODELS READY FOR PRODUCTION")
        log_print("# Consumer pipeline expects SAME input format - NO CHANGES NEEDED")
        log_print("#"*80)

    except Exception as e:
        log_print(f"\n❌ ERROR: {e}")
        import traceback
        log_print(traceback.format_exc())
        sys.exit(1)

    finally:
        log_file.close()
        print(f"\n📄 Full report saved to: {TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()