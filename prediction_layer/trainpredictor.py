# train_predictor.py (Updated for Temple Queueing)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

def generate_temple_data(samples=5000):
    data = []
    for _ in range(samples):
        # Temple specific metrics
        density = np.random.uniform(0, 15)  # Higher density potential in queues
        speed = max(0.05, 2.5 - (density * 0.15) + np.random.normal(0, 0.2))
        
        # Scenario: A "Dense Flag" event is high density AND speed < 0.5
        is_dense_queue = 1 if (density > 8 and speed < 0.6) else 0
        
        # Target: Predicting the risk of a "Stall"
        # If speed is already low and density is rising, risk is near 100
        future_risk = (density * 4) + (40 if speed < 0.4 else 0)
        future_risk = min(100, future_risk + (20 if is_dense_queue else 0))
        
        data.append([density, speed, is_dense_queue, future_risk])
    
    return pd.DataFrame(data, columns=['density', 'speed', 'is_dense_queue', 'target_risk'])

# Train and Save
df = generate_temple_data()
model = RandomForestRegressor(n_estimators=100)
model.fit(df[['density', 'speed', 'is_dense_queue']], df['target_risk'])
joblib.dump(model, 'temple_predictor.pkl')
print("Temple-Specific Model Trained.")