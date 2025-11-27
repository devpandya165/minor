import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb
import warnings

# Suppress LightGBM / sklearn feature name warnings
warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names, but LGBMRegressor was fitted with feature names.*",
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILENAME = r"C:\Users\DEV\Desktop\Minor Project\Project\Data\Model_Training\synthetic_indian_load_data.csv"  # from your updated dataset.py
TARGET_COL = "Gross_Load_MW"

# Time-series features
LAG_HOURS = [1, 2, 3, 24, 168]
ROLLING_WINDOWS = [3, 24, 168]

# Time-based global split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # test ratio = 1 - TRAIN_RATIO - VAL_RATIO

# Performance threshold for “good” per-state models
# We keep only states where corrected model RMSE improves global RMSE by at least this %.
MIN_REL_IMPROVEMENT = 0.05  # 5%

# Where to save models
MODEL_DIR = "models"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_model.joblib")
STATE_MODEL_DIR = os.path.join(MODEL_DIR, "state_models")  # one file per state

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATE_MODEL_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(filename: str) -> pd.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Data file '{filename}' not found. Run dataset.py first to generate it."
        )
    df = pd.read_csv(filename)
    df["Timestamp_UTC"] = pd.to_datetime(df["Timestamp_UTC"])
    df = df.sort_values(["State_Code", "Timestamp_UTC"]).reset_index(drop=True)
    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby("State_Code")

    for lag in LAG_HOURS:
        df[f"Load_lag_{lag}h"] = group[TARGET_COL].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"Load_roll_mean_{window}h"] = (
            group[TARGET_COL]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
    return df


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Timestamp_UTC").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def build_feature_matrices(df: pd.DataFrame):
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
    lag_roll_cols = [c for c in df.columns if c.startswith("Load_lag_") or c.startswith("Load_roll_mean_")]
    feature_cols.extend(lag_roll_cols)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y, feature_cols


def build_preprocessor(feature_cols):
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
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Loading dataset ===")
    df = load_dataset(DATA_FILENAME)

    # Drop missing target rows (from realistic anomalies)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Total rows after dropping missing target: {len(df)}")

    print("=== Adding time-series features ===")
    df = add_time_series_features(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Total rows after dropping NaNs from lag/rolling: {len(df)}")

    print("=== Global time-based split ===")
    train_df, val_df, test_df = time_based_split(df)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # Global features and target
    X_train, y_train, feature_cols = build_feature_matrices(train_df)
    X_val, y_val, _ = build_feature_matrices(val_df)
    X_test, y_test, _ = build_feature_matrices(test_df)

    preprocessor = build_preprocessor(feature_cols)

    global_model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    global_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", global_model),
        ]
    )

    print("=== Training GLOBAL model ===")
    global_pipeline.fit(X_train, y_train)

    # Evaluate global model
    print("=== Evaluating GLOBAL model ===")
    y_val_pred_global = global_pipeline.predict(X_val)
    y_test_pred_global = global_pipeline.predict(X_test)

    print("\nGlobal model performance (Validation):")
    print(f"  MAE  : {mean_absolute_error(y_val, y_val_pred_global):.3f}")
    print(f"  RMSE : {rmse(y_val, y_val_pred_global):.3f}")
    print(f"  MAPE : {mape(y_val, y_val_pred_global):.2f}%")

    print("\nGlobal model performance (Test):")
    global_mae_test = mean_absolute_error(y_test, y_test_pred_global)
    global_rmse_test = rmse(y_test, y_test_pred_global)
    global_mape_test = mape(y_test, y_test_pred_global)
    print(f"  MAE  : {global_mae_test:.3f}")
    print(f"  RMSE : {global_rmse_test:.3f}")
    print(f"  MAPE : {global_mape_test:.2f}%")

    # Save global model
    print(f"\nSaving GLOBAL model to {GLOBAL_MODEL_PATH}")
    joblib.dump(global_pipeline, GLOBAL_MODEL_PATH)

    # --------------------------------------------------------
    # 1) Compute residuals for train/val/test
    # --------------------------------------------------------
    train_df["global_pred"] = global_pipeline.predict(X_train)
    train_df["residual"] = train_df[TARGET_COL] - train_df["global_pred"]

    val_df["global_pred"] = global_pipeline.predict(X_val)
    val_df["residual"] = val_df[TARGET_COL] - val_df["global_pred"]

    test_df["global_pred"] = global_pipeline.predict(X_test)
    test_df["residual"] = test_df[TARGET_COL] - test_df["global_pred"]

    # --------------------------------------------------------
    # 2) Train per-state residual models (only where useful)
    # --------------------------------------------------------
    print("\n=== Training PER-STATE residual models (corrections) ===")
    states = sorted(df["State_Code"].unique())
    kept_states = []

    per_state_summary = []

    for state in states:
        # Per state splits
        train_s = train_df[train_df["State_Code"] == state]
        val_s = val_df[val_df["State_Code"] == state]
        test_s = test_df[test_df["State_Code"] == state]

        if len(train_s) < 300:  # skip tiny states
            print(f"  Skipping state {state}: only {len(train_s)} training samples")
            continue

        # Build X,y for residual prediction
        X_train_s, _, _ = build_feature_matrices(train_s)
        y_train_s = train_s["residual"]

        X_val_s, _, _ = build_feature_matrices(val_s)
        y_val_s = val_s["residual"]

        X_test_s, _, _ = build_feature_matrices(test_s)
        y_test_s_true = test_s[TARGET_COL].values
        y_test_s_global = test_s["global_pred"].values

        # If no test data for state, skip
        if len(test_s) == 0:
            print(f"  Skipping state {state}: no test data")
            continue

        preproc_s = build_preprocessor(feature_cols)

        model_s = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.07,
            num_leaves=48,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

        pipe_s = Pipeline(
            steps=[
                ("preprocess", preproc_s),
                ("model", model_s),
            ]
        )

        print(f"  Training residual model for state {state} on {len(train_s)} samples")
        pipe_s.fit(X_train_s, y_train_s)

        # Evaluate correction on test
        y_test_residual_pred = pipe_s.predict(X_test_s)
        y_test_corrected = y_test_s_global + y_test_residual_pred  # global + residual

        rmse_global_s = rmse(y_test_s_true, y_test_s_global)
        rmse_corrected_s = rmse(y_test_s_true, y_test_corrected)

        mae_global_s = mean_absolute_error(y_test_s_true, y_test_s_global)
        mae_corrected_s = mean_absolute_error(y_test_s_true, y_test_corrected)

        mape_global_s = mape(y_test_s_true, y_test_s_global)
        mape_corrected_s = mape(y_test_s_true, y_test_corrected)

        rel_improvement = (rmse_global_s - rmse_corrected_s) / (rmse_global_s + 1e-9)

        print(
            f"    {state} GLOBAL   Test RMSE={rmse_global_s:.3f}, MAPE={mape_global_s:.2f}%"
        )
        print(
            f"    {state} CORRECTED Test RMSE={rmse_corrected_s:.3f}, MAPE={mape_corrected_s:.2f}% "
            f"(improvement={rel_improvement*100:.2f}%)"
        )

        per_state_summary.append(
            {
                "state": state,
                "rmse_global": rmse_global_s,
                "rmse_corrected": rmse_corrected_s,
                "mape_global": mape_global_s,
                "mape_corrected": mape_corrected_s,
                "rel_improvement": rel_improvement,
            }
        )

        # Keep this state model only if it improves RMSE by at least MIN_REL_IMPROVEMENT
        if rel_improvement > MIN_REL_IMPROVEMENT:
            kept_states.append(state)
            model_path = os.path.join(STATE_MODEL_DIR, f"state_{state}_residual_model.joblib")
            joblib.dump(pipe_s, model_path)
            print(f"      -> SAVED residual model for state {state} (improved).")
        else:
            print(f"      -> NOT saving model for state {state} (improvement too small).")

    # --------------------------------------------------------
    # 3) Print summary
    # --------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"Global test RMSE : {global_rmse_test:.3f}")
    print(f"Global test MAPE : {global_mape_test:.2f}%")
    print(f"States with saved residual models (improved > {MIN_REL_IMPROVEMENT*100:.1f}%):")
    print(", ".join(kept_states) if kept_states else "None")

    # Optional: show per-state RMSE comparison
    if per_state_summary:
        summary_df = pd.DataFrame(per_state_summary).sort_values("rel_improvement", ascending=False)
        print("\nPer-state RMSE comparison (top 10 by improvement):")
        print(
            summary_df[["state", "rmse_global", "rmse_corrected", "rel_improvement"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()