from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def generate_temple_data(samples=5000):
    data = []
    for _ in range(samples):
        density = np.random.uniform(0, 15)
        speed = max(0.05, 2.2 - (density * 0.10) + np.random.normal(0, 0.22))
        is_dense_queue = 1 if (density > 8 and speed < 0.6) else 0

        future_risk = 5.0
        future_risk += max(0.0, density - 5.0) * 6.0
        future_risk += max(0.0, 0.9 - speed) * 35.0
        future_risk += 16.0 if is_dense_queue else 0.0
        future_risk += np.random.normal(0, 2.5)
        future_risk = min(100, max(0, future_risk))
        data.append([density, speed, is_dense_queue, future_risk])

    return pd.DataFrame(data, columns=["density", "speed", "is_dense_queue", "target_risk"])


def main():
    df = generate_temple_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(
        df[["density", "speed", "is_dense_queue"]].to_numpy(dtype=np.float32),
        df["target_risk"].to_numpy(dtype=np.float32),
    )

    output_path = Path(__file__).with_name("temple_predictor.pkl")
    joblib.dump(model, output_path)
    print(f"Temple-specific model trained: {output_path}")


if __name__ == "__main__":
    main()
