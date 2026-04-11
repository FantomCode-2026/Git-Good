from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

from heatmap.logic import detect_problem, get_risk_score, get_severity, suggest_action
from prediction_layer.predictor import CrowdPredictor


class PredictionLayerBridge:
    """
    Adapts live pipeline metrics into the smaller prediction layer model.
    """

    def __init__(self, model_path: Optional[str] = None):
        default_model_path = Path(__file__).with_name("temple_predictor.pkl")
        resolved_model_path = Path(model_path) if model_path else default_model_path
        self.predictor = CrowdPredictor(str(resolved_model_path))
        self.speed_history: Deque[float] = deque(maxlen=20)
        self.risk_history: Deque[float] = deque(maxlen=20)
        self.density_history: Deque[float] = deque(maxlen=20)
        self.people_history: Deque[float] = deque(maxlen=20)
        self.males_history: Deque[float] = deque(maxlen=20)
        self.females_history: Deque[float] = deque(maxlen=20)

    def evaluate(
        self,
        people_count: int,
        density_score: float,
        density_grid: np.ndarray,
        tracked_objects: List[Dict],
        behavior_results: Dict[int, Dict],
        fall_risk_results: Dict[int, Dict],
        metrics,
        males: int = 0,
        females: int = 0,
    ) -> Dict[str, object]:
        avg_speed = self._compute_average_speed(tracked_objects, metrics)
        flow_snapshot = self._compute_flow_snapshot(tracked_objects, metrics)
        density_index, hotspot_ratio = self._compute_density_index(
            people_count, density_score, density_grid
        )
        density_trend = self._compute_trend(self.density_history, density_score)
        people_trend = self._compute_trend(self.people_history, float(people_count))
        males_trend = self._compute_trend(self.males_history, float(males))
        females_trend = self._compute_trend(self.females_history, float(females))
        speed_for_model = avg_speed if avg_speed is not None else 1.35
        speed_trend = self._compute_trend(self.speed_history, speed_for_model)

        recent_baseline = float(np.mean(self.speed_history)) if self.speed_history else speed_for_model
        speed_drop = bool(
            avg_speed is not None
            and self.speed_history
            and avg_speed < recent_baseline * 0.72
        )
        density_rising = bool(self.density_history and density_trend > 0.015)
        congestion_building = bool(
            avg_speed is not None
            and density_rising
            and avg_speed <= max(recent_baseline * 0.92, 0.45)
            and people_trend >= 0.0
        )
        compression = bool(
            avg_speed is not None
            and (
                (density_score > 0.62 and avg_speed < 0.65)
                or (density_score > 0.52 and speed_drop and density_rising)
                or congestion_building
            )
        )
        unusual_behavior = any(
            result.get("behavior_label") not in {"NORMAL", "FALL_RISK"}
            for result in behavior_results.values()
        )
        fallen = any(
            result.get("fall_probability", 0.0) >= 0.85
            for result in fall_risk_results.values()
        )
        queue_pressure = bool(
            density_score >= 0.48 and hotspot_ratio >= 0.08 and speed_for_model < 1.0
        )
        gate_pressure = bool(
            density_score >= 0.56 and hotspot_ratio >= 0.12 and people_count >= 30
        )
        data = {
            "density": density_index,
            "avg_speed": speed_for_model,
            "flow_status": flow_snapshot["flow_status"],
            "speed_drop": speed_drop or speed_trend < -0.08,
            "opposing_flow": flow_snapshot["opposing_flow"],
            "compression": compression,
            "unusual_behavior": unusual_behavior,
            "fallen": fallen,
            "queue_pressure": queue_pressure,
            "gate_pressure": gate_pressure,
        }

        problem = detect_problem(data)
        if avg_speed is None and problem in {"HIGH_COMPRESSION", "BOTTLENECK", "SPEED_DROP"}:
            problem = "FLOW_CONFLICT" if flow_snapshot["flow_status"] == "chaotic" else "NORMAL"
        base_risk = float(get_risk_score(problem))
        predicted_future_risk = float(
            np.clip(
                self.predictor.predict_future_risk(
                    density=data["density"],
                    speed=data["avg_speed"],
                    flow_status=data["flow_status"],
                ),
                0.0,
                100.0,
            )
        )
        predicted_males, predicted_females = self.predictor.predict_future_people_count(
            current_males=males,
            current_females=females,
            trend_males=males_trend,
            trend_females=females_trend,
        )
        trend_risk = self._estimate_trend_risk(
            density_score=density_score,
            density_trend=density_trend,
            people_count=people_count,
            people_trend=people_trend,
            avg_speed=speed_for_model,
            speed_trend=speed_trend,
            speed_drop=data["speed_drop"],
            opposing_flow=data["opposing_flow"],
            compression=data["compression"],
            unusual_behavior=data["unusual_behavior"],
            fallen=data["fallen"],
        )
        fused_risk = float(
            np.clip(0.35 * base_risk + 0.40 * predicted_future_risk + 0.25 * trend_risk, 0.0, 100.0)
        )

        self.density_history.append(density_score)
        self.people_history.append(float(people_count))
        self.males_history.append(float(males))
        self.females_history.append(float(females))
        self.speed_history.append(speed_for_model)
        self.risk_history.append(fused_risk)

        # Claustrophobia alerts based on predicted gender counts
        claustrophobia_alerts = []
        if predicted_females > 15:  # Threshold for female claustrophobia
            claustrophobia_alerts.append("FEMALE_CLAUSTROPHOBIA")
        if predicted_males > 25:  # Threshold for male claustrophobia
            claustrophobia_alerts.append("MALE_CLAUSTROPHOBIA")

        return {
            "problem": problem,
            "base_risk": base_risk,
            "predicted_future_risk": predicted_future_risk,
            "trend_risk": trend_risk,
            "fused_risk": fused_risk,
            "severity": get_severity(fused_risk),
            "recommended_action": suggest_action(problem, data=data, risk=fused_risk),
            "density_index": density_index,
            "hotspot_ratio": hotspot_ratio,
            "avg_speed": speed_for_model,
            "speed_observed": avg_speed is not None,
            "flow_status": data["flow_status"],
            "density_trend": density_trend,
            "people_trend": people_trend,
            "males_trend": males_trend,
            "females_trend": females_trend,
            "predicted_males": predicted_males,
            "predicted_females": predicted_females,
            "claustrophobia_alerts": claustrophobia_alerts,
            "speed_trend": speed_trend,
            "speed_drop": speed_drop,
            "opposing_flow": data["opposing_flow"],
            "compression": data["compression"],
            "queue_pressure": queue_pressure,
            "gate_pressure": gate_pressure,
            "model_loaded": self.predictor.model is not None,
        }

    def reset(self):
        self.density_history.clear()
        self.people_history.clear()
        self.males_history.clear()
        self.females_history.clear()
        self.speed_history.clear()
        self.risk_history.clear()

    def _compute_average_speed(self, tracked_objects: List[Dict], metrics) -> Optional[float]:
        speeds = []
        for obj in tracked_objects:
            movement_stats = metrics.compute_movement_stats(obj["track_id"])
            if movement_stats is not None:
                speeds.append(float(movement_stats.avg_speed))

        if not speeds:
            return None

        normalized_speed = float(np.mean(speeds) / max(metrics.velocity_threshold_normal, 1.0))
        return float(np.clip(normalized_speed * 2.5, 0.0, 2.5))

    def _compute_flow_snapshot(self, tracked_objects: List[Dict], metrics) -> Dict[str, object]:
        direction_vectors = []
        for obj in tracked_objects:
            history = metrics.movement_history.get(obj["track_id"])
            if history is None or len(history) < 2:
                continue

            (prev_center, _), (curr_center, _) = history[-2], history[-1]
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            norm = float(np.hypot(dx, dy))
            if norm > 1e-6:
                direction_vectors.append((dx / norm, dy / norm))

        if len(direction_vectors) < 4:
            return {"flow_status": "smooth", "opposing_flow": False}

        opposing_pairs = 0
        total_pairs = 0
        angles = []
        for dx, dy in direction_vectors:
            angles.append(float(np.arctan2(dy, dx)))
        for idx in range(len(direction_vectors)):
            for jdx in range(idx + 1, len(direction_vectors)):
                total_pairs += 1
                dot = (
                    direction_vectors[idx][0] * direction_vectors[jdx][0]
                    + direction_vectors[idx][1] * direction_vectors[jdx][1]
                )
                if dot < -0.25:
                    opposing_pairs += 1

        opposing_flow = total_pairs > 0 and (opposing_pairs / total_pairs) > 0.28
        chaotic = len(direction_vectors) >= 8 and (np.std(angles) > 1.35 or opposing_flow)

        return {
            "flow_status": "chaotic" if chaotic else "smooth",
            "opposing_flow": opposing_flow,
        }

    @staticmethod
    def _compute_trend(history: Deque[float], current_value: float) -> float:
        if len(history) < 3:
            return 0.0
        values = np.array(list(history)[-5:] + [current_value], dtype=np.float32)
        x = np.arange(len(values), dtype=np.float32)
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)

    @staticmethod
    def _estimate_trend_risk(
        density_score: float,
        density_trend: float,
        people_count: int,
        people_trend: float,
        avg_speed: float,
        speed_trend: float,
        speed_drop: bool,
        opposing_flow: bool,
        compression: bool,
        unusual_behavior: bool,
        fallen: bool,
    ) -> float:
        if fallen:
            return 95.0

        risk = 15.0
        risk += np.clip(density_score, 0.0, 1.0) * 35.0
        risk += np.clip(density_trend * 800.0, 0.0, 14.0)
        risk += np.clip(people_count / 80.0, 0.0, 1.0) * 6.0
        risk += np.clip(people_trend * 5.0, 0.0, 5.0)
        risk += np.clip((0.85 - avg_speed) / 0.85, 0.0, 1.0) * 12.0
        risk += np.clip((-speed_trend) * 60.0, 0.0, 8.0)
        if speed_drop:
            risk += 5.0
        if opposing_flow:
            risk += 10.0
        if compression:
            risk += 12.0
        if unusual_behavior:
            risk += 8.0
        return float(np.clip(risk, 0.0, 100.0))

    @staticmethod
    def _compute_density_index(
        people_count: int,
        density_score: float,
        density_grid: np.ndarray,
    ) -> tuple[float, float]:
        hotspot_ratio = float(np.count_nonzero(density_grid > 0.65)) / max(density_grid.size, 1)
        density_index = (
            density_score * 8.0
            + hotspot_ratio * 10.0
            + min(people_count / 18.0, 2.5)
        )
        return float(np.clip(density_index, 0.0, 15.0)), hotspot_ratio
