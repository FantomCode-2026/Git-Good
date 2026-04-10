from pathlib import Path

import joblib
import numpy as np


class CrowdPredictor:
    def __init__(self, model_path: str | None = None):
        default_model = Path(__file__).with_name("temple_predictor.pkl")
        resolved_model_path = Path(model_path) if model_path else default_model
        try:
            self.model = joblib.load(resolved_model_path)
        except Exception:
            self.model = None

    def predict_future_risk(self, density, speed, flow_status):
        if not self.model:
            return 0.0
        chaotic = 1 if flow_status == "chaotic" else 0
        features = np.array([[density, speed, chaotic]], dtype=np.float32)
        return float(self.model.predict(features)[0])
