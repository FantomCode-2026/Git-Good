"""
Train small venue-specific risk regressors from synthetic data.

Temple: density / queue stall → risk (original).
Stadium: directional flow coherence vs radial imbalance → risk.
Concert: forward surge / pit crush vs lateral spread → risk.

Each generator encodes “normal” vs stressed behaviour; labels are future risk targets.
The digital twin exporter uses the same conceptual split (baseline vs prediction, riskDeviation).
"""

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


def generate_stadium_data(samples=5000):
    """
    Directional circulation: coherent tangential flow vs radial drift / gate imbalance.
    Higher risk when flow is misaligned (low directional_coherence) or exits are overloaded.
    """
    rows = []
    for _ in range(samples):
        radial_gradient = np.random.uniform(0, 1)
        directional_coherence = np.random.uniform(0, 1)
        gate_imbalance = np.random.uniform(0, 1)
        exit_pressure = np.random.uniform(0, 1)

        risk = 8.0
        risk += radial_gradient * 26.0
        risk += (1.0 - directional_coherence) * 34.0
        risk += gate_imbalance * 20.0
        risk += exit_pressure * 18.0
        risk += np.random.normal(0, 3.0)
        risk = float(min(100, max(0, risk)))
        rows.append([radial_gradient, directional_coherence, gate_imbalance, exit_pressure, risk])

    return pd.DataFrame(
        rows,
        columns=[
            "radial_gradient",
            "directional_coherence",
            "gate_imbalance",
            "exit_pressure",
            "target_risk",
        ],
    )


def generate_concert_data(samples=5000):
    """
    Aggressive forward surge toward stage / pit vs lateral spread-out (normal behaviour).
    """
    rows = []
    for _ in range(samples):
        forward_velocity = np.random.uniform(0, 1)
        pit_density = np.random.uniform(0, 1)
        lateral_spread = np.random.uniform(0, 1)
        surge_coherence = np.random.uniform(0, 1)

        risk = 8.0
        risk += forward_velocity * 36.0
        risk += pit_density * 32.0
        risk += (1.0 - lateral_spread) * 18.0
        risk += surge_coherence * 14.0
        risk += np.random.normal(0, 3.0)
        risk = float(min(100, max(0, risk)))
        rows.append([forward_velocity, pit_density, lateral_spread, surge_coherence, risk])

    return pd.DataFrame(
        rows,
        columns=["forward_velocity", "pit_density", "lateral_spread", "surge_coherence", "target_risk"],
    )


def _train_and_save(
    label: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    filename: str,
) -> Path:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(
        df[feature_cols].to_numpy(dtype=np.float32),
        df["target_risk"].to_numpy(dtype=np.float32),
    )
    output_path = Path(__file__).with_name(filename)
    joblib.dump(model, output_path)
    print(f"{label} model trained: {output_path}")
    return output_path


def main():
    _train_and_save(
        "Temple",
        generate_temple_data(),
        ["density", "speed", "is_dense_queue"],
        "temple_predictor.pkl",
    )
    _train_and_save(
        "Stadium",
        generate_stadium_data(),
        [
            "radial_gradient",
            "directional_coherence",
            "gate_imbalance",
            "exit_pressure",
        ],
        "stadium_predictor.pkl",
    )
    _train_and_save(
        "Concert",
        generate_concert_data(),
        ["forward_velocity", "pit_density", "lateral_spread", "surge_coherence"],
        "concert_predictor.pkl",
    )


if __name__ == "__main__":
    main()
