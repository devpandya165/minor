import os
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd


GLOBAL_MODEL_PATH = "/home/devpandya/models/global_model.joblib"
STATE_MODEL_DIR = "/home/devpandya/models/state_models"

class ElectricityLoadPredictor:
    """
    Loads the global LightGBM pipeline and per-state residual pipelines,
    and exposes a predict_batch() method.
    """

    def __init__(self):
        if not os.path.exists(GLOBAL_MODEL_PATH):
            raise FileNotFoundError(f"Global model not found at {GLOBAL_MODEL_PATH}")

        print(f"Loading GLOBAL model from {GLOBAL_MODEL_PATH}")
        self.global_model = joblib.load(GLOBAL_MODEL_PATH)

        self.state_models = {}
        if os.path.isdir(STATE_MODEL_DIR):
                        for fname in os.listdir(STATE_MODEL_DIR):
                if not fname.endswith(".joblib"):
                    continue
                state_code = fname.replace("state_", "").replace("_residual>                full_path = os.path.join(STATE_MODEL_DIR, fname)
                print(f"Loading residual model for state {state_code} from >                self.state_models[state_code] = joblib.load(full_path)
        else:
            print(f"STATE_MODEL_DIR does not exist: {STATE_MODEL_DIR}")

        print(f"Loaded {len(self.state_models)} per-state residual models.")

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[float]:
        """
        records: list of dictionaries, each having the same columns used du>        including lag and rolling features (Load_lag_..., Load_roll_mean_..>

        Returns a list of predictions (floats) - one per record.
        """
        if not records:
            return []

        df = pd.DataFrame(records)

        # Global pipeline includes preprocessing
        global_preds = self.global_model.predict(df)

        corrected_preds = []
        for i, row in df.iterrows():
            state_code = row.get("State_Code")
            base_pred = global_preds[i]

            if state_code in self.state_models:
                row_df = row.to_frame().T
                residual_model = self.state_models[state_code]
                                residual_pred = residual_model.predict(row_df)[0]
                corrected = float(base_pred + residual_pred)
                corrected_preds.append(corrected)
            else:
                corrected_preds.append(float(base_pred))

        return corrected_preds

