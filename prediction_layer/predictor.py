import joblib
import numpy as np

class CrowdPredictor:
    def __init__(self, model_path='crowd_predictor.pkl'):
        try:
            self.model = joblib.load(model_path)
        except:
            self.model = None

    def predict_future_risk(self, density, speed, flow_status):
        if not self.model: return 0
        chaotic = 1 if flow_status == "chaotic" else 0
        features = np.array([[density, speed, chaotic]])
        return self.model.predict(features)[0]
