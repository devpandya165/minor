import os
import sys
import json
import time
from collections import defaultdict, deque
from typing import Dict, Deque, Any, List

import pandas as pd
import joblib
from kafka import KafkaConsumer

# ============ CONFIG (WINDOWS PATHS) ============

KAFKA_BOOTSTRAP = "172.17.14.155:9092"  # TODO: your WSL IP

# Models (Windows paths)
GLOBAL_MODEL_PATH = r"C:\\Users\\DEV\\Desktop\\Minor Project\\models\\global_model.joblib"
STATE_MODEL_DIR = r"C:\\Users\\DEV\\Desktop\\Minor Project\\models\\state_models"

# Data (Windows paths)
HISTORY_SEED_CSV = r"C:\\Users\\DEV\\Desktop\\Minor Project\\Project\\Data\\history_seed.csv"
# Kafka config
TOPIC_NAME = "electricity_topic"
GROUP_ID = "electricity_prediction_group"
BATCH_SIZE = 36  # 1 hour = 36 states
HISTORY_HOURS = 220

# Output (Windows paths)
OUTPUT_CSV_DIR = r"C:\Users\DEV\Desktop\Minor Project\Project\Data\stream_predictions_csv"
GLOBAL_PREDICTIONS_CSV = os.path.join(OUTPUT_CSV_DIR, "global_predictions. csv")

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

print("=" * 70)
print("HOURLY ELECTRICITY LOAD PREDICTION PIPELINE")
print("=" * 70)
print(f"[CFG] Kafka Bootstrap   : {KAFKA_BOOTSTRAP}")
print(f"[CFG] Topic             : {TOPIC_NAME}")
print(f"[CFG] Batch Size        : {BATCH_SIZE} (1 hour = 36 states)")
print(f"[CFG] Global Model      : {GLOBAL_MODEL_PATH}")
print(f"[CFG] State Models Dir  : {STATE_MODEL_DIR}")
print(f"[CFG] History Seed CSV  : {HISTORY_SEED_CSV}")
print(f"[CFG] Output CSV Dir    : {OUTPUT_CSV_DIR}")
print(f"[CFG] Global Predictions CSV : {GLOBAL_PREDICTIONS_CSV}")
print("=" * 70)

# ============ LOAD MODELS ============

print("\n[STEP 1] Loading ML Models from Windows...")

# Load global model
if not os.path.exists(GLOBAL_MODEL_PATH):
    print(f"[MODEL] ❌ Global model not found: {GLOBAL_MODEL_PATH}")
    sys.exit(1)

try:
    print(f"[MODEL] Loading global model from {GLOBAL_MODEL_PATH}")
    global_model = joblib.load(GLOBAL_MODEL_PATH)
    print("[MODEL] ✅ Global model loaded")
except Exception as e:
    print(f"[MODEL] ❌ Failed to load global model: {e}")
    sys. exit(1)

# Load state residual models
state_models = {}
if os.path.isdir(STATE_MODEL_DIR):
    try:
        for fname in os.listdir(STATE_MODEL_DIR):
            if not fname.endswith(". joblib"):
                continue
            state_code = fname. replace("state_", ""). replace("_residual. joblib", "")
            full_path = os.path.join(STATE_MODEL_DIR, fname)
            state_models[state_code] = joblib.load(full_path)
        
        print(f"[MODEL] ✅ Loaded {len(state_models)} per-state residual models")
    except Exception as e:
        print(f"[MODEL] ❌ Failed to load state models: {e}")
        sys.exit(1)
else:
    print(f"[MODEL] ⚠️  State models directory not found: {STATE_MODEL_DIR}")

# ============ HISTORY STATE ============

print("\n[STEP 2] Initializing Per-State History...")
history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
    lambda: deque(maxlen=HISTORY_HOURS)
)
print("[HISTORY] Ready to store 220 hours per state")

# ============ LOAD HISTORY SEED ============

print("\n[STEP 3] Pre-loading History Seed (168 hours per state)...")


def load_history_seed():
    """Load history_seed.csv into per-state history."""
    if not os.path.exists(HISTORY_SEED_CSV):
        print(f"[HISTORY] ⚠️  File not found: {HISTORY_SEED_CSV}")
        return

    try:
        df = pd.read_csv(HISTORY_SEED_CSV)
        print(f"[HISTORY] Loaded {len(df)} rows")

        # Type conversion
        df["Gross_Load_MW"] = pd.to_numeric(df["Gross_Load_MW"], errors="coerce")
        df["event_time"] = pd.to_datetime(df["Timestamp_UTC"], format="%Y-%m-%d %H:%M:%S")

        # Populate history per state
        for state, g in df.groupby("State_Code"):
            g = g.sort_values("event_time"). reset_index(drop=True)
            for _, row in g.iterrows():
                history[state]. append({
                    "Timestamp_UTC": row["Timestamp_UTC"],
                    "event_time": row["event_time"],
                    "Gross_Load_MW_imputed": row["Gross_Load_MW"],
                    "Avg_Temp_C_imputed": pd.to_numeric(row["Avg_Temp_C"], errors="coerce"),
                    "Avg_Humidity_Pct_imputed": pd.to_numeric(row["Avg_Humidity_Pct"], errors="coerce"),
                })

        states = len(history)
        hours = len(df) // states if states > 0 else 0
        print(f"[HISTORY] ✅ Pre-loaded {states} states, {hours} hours each")

    except Exception as e:
        print(f"[HISTORY] ❌ Error loading history seed: {e}")
        import traceback
        traceback.print_exc()


# Load history seed
load_history_seed()

# ============ FEATURE ENGINEERING FUNCTION ============

def engineer_features(state_code: str, current_record: Dict[str, Any]) -> Dict[str, Any]:

    record = current_record.copy()
    state_history = history[state_code]
    
    if len(state_history) == 0:
        # Not enough history, use default values
        record["Load_lag_1h"] = 0.0
        record["Load_lag_2h"] = 0.0
        record["Load_lag_3h"] = 0.0
        record["Load_lag_24h"] = 0.0
        record["Load_lag_168h"] = 0.0
        record["Load_roll_mean_3h"] = 0.0
        record["Load_roll_mean_24h"] = 0.0
        record["Load_roll_mean_168h"] = 0.0
        return record
    
    # Extract historical loads
    hist_loads = [h. get("Gross_Load_MW_imputed", 0.0) for h in state_history]
    
    # Calculate lags
    record["Load_lag_1h"] = float(hist_loads[-1]) if len(hist_loads) >= 1 else 0.0
    record["Load_lag_2h"] = float(hist_loads[-2]) if len(hist_loads) >= 2 else 0.0
    record["Load_lag_3h"] = float(hist_loads[-3]) if len(hist_loads) >= 3 else 0.0
    record["Load_lag_24h"] = float(hist_loads[-24]) if len(hist_loads) >= 24 else 0.0
    record["Load_lag_168h"] = float(hist_loads[-168]) if len(hist_loads) >= 168 else 0.0
    
    # Calculate rolling means
    record["Load_roll_mean_3h"] = float(pd.Series(hist_loads[-3:]).mean()) if len(hist_loads) >= 3 else float(pd.Series(hist_loads).mean())
    record["Load_roll_mean_24h"] = float(pd.Series(hist_loads[-24:]).mean()) if len(hist_loads) >= 24 else float(pd.Series(hist_loads).mean())
    record["Load_roll_mean_168h"] = float(pd. Series(hist_loads[-168:]).mean()) if len(hist_loads) >= 168 else float(pd.Series(hist_loads).mean())
    
    return record

# ============ PREDICTION FUNCTION ============

def predict_batch(records: List[Dict[str, Any]]) -> List[float]:
    """
    Predict using global model + per-state residual models. 
    
    Process:
    1. Engineer features from history
    2. Global model predicts base load
    3. For each state, residual model adds correction
    4. Return corrected predictions
    """
    if not records:
        return []

    # Engineer features for each record
    engineered_records = []
    for record in records:
        state_code = record. get("State_Code")
        engineered_record = engineer_features(state_code, record)
        engineered_records.append(engineered_record)

    df = pd.DataFrame(engineered_records)

    # Global pipeline predicts
    try:
        global_preds = global_model.predict(df)
    except Exception as e:
        print(f"[PREDICT] ❌ Error during global model prediction: {e}")
        print(f"[PREDICT] Available columns: {df.columns.tolist()}")
        return []

    corrected_preds = []
    for i, row in df.iterrows():
        state_code = row.get("State_Code")
        base_pred = global_preds[i]

        if state_code in state_models:
            row_df = row.to_frame().T
            residual_model = state_models[state_code]
            try:
                residual_pred = residual_model.predict(row_df)[0]
                corrected = float(base_pred + residual_pred)
                corrected_preds.append(corrected)
            except Exception as e:
                print(f"[PREDICT] ❌ Error during residual prediction for {state_code}: {e}")
                corrected_preds.append(float(base_pred))
        else:
            corrected_preds.append(float(base_pred))

    return corrected_preds

# ============ GLOBAL PREDICTIONS FUNCTION ============

def append_global_predictions_to_csv(df: pd.DataFrame, global_preds: List[float]) -> None:

    try:
        # Create output DataFrame with global predictions
        df_global = df.copy()
        df_global["Global_Predicted_Load_MW"] = global_preds
        
        # Select columns to save
        cols_to_save = [
            "State_Code", "Timestamp_UTC", "Gross_Load_MW", "Hour_Of_Day", 
            "Day_Of_Week", "Is_Weekend", "Is_Holiday_State", "Avg_Temp_C", 
            "Temp_Change_6H", "Avg_Humidity_Pct",
            "Load_lag_1h", "Load_lag_2h", "Load_lag_3h", "Load_lag_24h", "Load_lag_168h",
            "Load_roll_mean_3h", "Load_roll_mean_24h", "Load_roll_mean_168h",
            "Global_Predicted_Load_MW"
        ]
        
        # Filter to only available columns
        available_cols = [col for col in cols_to_save if col in df_global.columns]
        df_global = df_global[available_cols]
        
        # Check if file exists
        if os.path.exists(GLOBAL_PREDICTIONS_CSV):
            # Append to existing file
            df_existing = pd.read_csv(GLOBAL_PREDICTIONS_CSV)
            df_combined = pd.concat([df_existing, df_global], ignore_index=True)
            df_combined.to_csv(GLOBAL_PREDICTIONS_CSV, index=False)
            print(f"[GLOBAL_CSV] ✅ Appended {len(df_global)} records to global_predictions.csv")
        else:
            # Create new file
            df_global.to_csv(GLOBAL_PREDICTIONS_CSV, index=False)
            print(f"[GLOBAL_CSV] ✅ Created global_predictions.csv with {len(df_global)} records")
    
    except Exception as e:
        print(f"[GLOBAL_CSV] ❌ Error appending to global_predictions.csv: {e}")
        import traceback
        traceback.print_exc()

# ============ KAFKA CONSUMER ============

print("\n[STEP 4] Starting Kafka Consumer...")

try:
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=[KAFKA_BOOTSTRAP],
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_records=BATCH_SIZE,
    )
    print("[KAFKA] ✅ Connected to Kafka broker")
except Exception as e:
    print(f"[KAFKA] ❌ Failed to connect to Kafka: {e}")
    sys.exit(1)

# ============ MAIN LOOP ============

print("\n[STEP 5] Starting Prediction Pipeline.. .\n")

batch_count = 0
all_predictions = []

try:
    for message in consumer:
        try:
            record = message.value
            state_code = record.get("State_Code")
            
            # Parse timestamp if needed
            if "event_time" not in record and "Timestamp_UTC" in record:
                try:
                    record["event_time"] = pd.to_datetime(record["Timestamp_UTC"], format="%Y-%m-%d %H:%M:%S")
                except:
                    record["event_time"] = pd. Timestamp.now()
            
            # Parse load value
            record["Gross_Load_MW_imputed"] = pd.to_numeric(record. get("Gross_Load_MW", 0), errors="coerce")
            record["Avg_Temp_C_imputed"] = pd.to_numeric(record.get("Avg_Temp_C", 0), errors="coerce")
            record["Avg_Humidity_Pct_imputed"] = pd.to_numeric(record.get("Avg_Humidity_Pct", 0), errors="coerce")
            
            # Add record to history
            history[state_code].append(record)
            
            # Check if we have enough records for a batch
            if len(all_predictions) < BATCH_SIZE:
                all_predictions.append(record)
            
            if len(all_predictions) == BATCH_SIZE:
                batch_count += 1
                
                # Make predictions
                predictions = predict_batch(all_predictions)
                
                if predictions:  # Only process if predictions were successful
                    # Create output DataFrame
                    df_output = pd.DataFrame(all_predictions)
                    
                    # Add predictions
                    df_output["Predicted_Load_MW"] = predictions
                    
                    # Append global predictions to CSV
                    append_global_predictions_to_csv(df_output, predictions)
                    
                    avg_pred = sum(predictions) / len(predictions) if predictions else 0
                    print(f"[BATCH {batch_count}] ✅ Processed {BATCH_SIZE} records | Avg Prediction: {avg_pred:.2f} MW")
                
                all_predictions = []
                
        except Exception as e:
            print(f"[ERROR] Error processing record: {e}")
            import traceback
            traceback.print_exc()
            continue

except KeyboardInterrupt:
    print("\n[SHUTDOWN] Received interrupt signal")
except Exception as e:
    print(f"[FATAL] Consumer error: {e}")
    import traceback
    traceback.print_exc()
finally:
    consumer.close()
    print("[SHUTDOWN] ✅ Kafka consumer closed")
    print(f"[STATS] Total batches processed: {batch_count}")
    print(f"[STATS] Global predictions saved to: {GLOBAL_PREDICTIONS_CSV}")