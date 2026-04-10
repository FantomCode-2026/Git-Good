"""Demo Simulator – generates realistic live crowd data without camera."""

from __future__ import annotations
import logging, math, random, threading, time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from heatmap.logic import detect_problem, get_risk_score, get_severity, suggest_action
from digitaltwin.exporter import build_temple_simulation_state

logger = logging.getLogger(__name__)

SCENARIOS = [
    {"name": "Normal Flow",       "base_density": 0.22, "base_speed": 1.45, "duration": 18},
    {"name": "Crowd Building",    "base_density": 0.44, "base_speed": 1.10, "duration": 14},
    {"name": "Queue Pressure",    "base_density": 0.61, "base_speed": 0.82, "duration": 14},
    {"name": "HIGH COMPRESSION",  "base_density": 0.84, "base_speed": 0.42, "duration": 10},
    {"name": "Bottleneck",        "base_density": 0.73, "base_speed": 0.62, "duration": 12},
    {"name": "Clearing",          "base_density": 0.33, "base_speed": 1.25, "duration": 14},
]

ZONE_NAMES = ["A", "B", "C", "D"]

ZONE_OFFSETS = {
    "A": {"density": 0.00, "speed":  0.00, "phase": 0.0},
    "B": {"density": 0.10, "speed": -0.10, "phase": 1.2},
    "C": {"density": 0.05, "speed":  0.05, "phase": 2.4},
    "D": {"density": 0.15, "speed": -0.15, "phase": 0.8},
}


class DemoSimulator:
    def __init__(self, control_engine, interval: float = 2.0):
        self.engine   = control_engine
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scenario_idx  = 0
        self._scenario_time = 0.0
        self._tick          = 0
        self._metrics_history: List[Dict] = []
        self._alerts_history:  List[Dict] = []

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[DemoSimulator] Started – cycling crowd scenarios")

    def stop(self):
        self._running = False

    def current_scenario(self):
        return SCENARIOS[self._scenario_idx]["name"]

    def _loop(self):
        while self._running:
            try:
                self._tick_update()
            except Exception as e:
                logger.error("[DemoSimulator] %s", e)
            time.sleep(self.interval)

    def _tick_update(self):
        scenario = SCENARIOS[self._scenario_idx]
        self._scenario_time += self.interval
        if self._scenario_time >= scenario["duration"]:
            self._scenario_idx  = (self._scenario_idx + 1) % len(SCENARIOS)
            self._scenario_time = 0.0
            scenario = SCENARIOS[self._scenario_idx]
            logger.info("[DemoSimulator] Scenario → %s", scenario["name"])

        t = self._tick * self.interval
        self._tick += 1

        for zone in ZONE_NAMES:
            off     = ZONE_OFFSETS[zone]
            noise_d = 0.04 * math.sin(t * 0.3 + off["phase"]) + random.gauss(0, 0.02)
            noise_s = 0.08 * math.cos(t * 0.2 + off["phase"]) + random.gauss(0, 0.03)

            density = float(max(0.05, min(0.99, scenario["base_density"] + off["density"] + noise_d)))
            speed   = float(max(0.10, min(2.50, scenario["base_speed"]   + off["speed"]   + noise_s)))
            count   = int(density * 120 + random.gauss(0, 3))

            data = {
                "density":          density * 10,
                "avg_speed":        speed,
                "flow_status":      "chaotic" if density > 0.75 and speed < 0.6 else "smooth",
                "speed_drop":       speed < 0.7,
                "opposing_flow":    density > 0.65 and random.random() < 0.2,
                "compression":      density > 0.70 and speed < 0.65,
                "unusual_behavior": density > 0.80 and random.random() < 0.3,
                "fallen":           density > 0.90 and speed < 0.35 and random.random() < 0.05,
                "queue_pressure":   density > 0.55 and speed < 0.9,
                "gate_pressure":    density > 0.65 and count > 50,
            }
            problem    = detect_problem(data)
            base_risk  = get_risk_score(problem)
            fused_risk = float(min(99, base_risk + density * 15 + random.gauss(0, 2)))
            severity   = get_severity(fused_risk)
            action_txt = suggest_action(problem, data=data, risk=fused_risk)

            self.engine.process_evaluation({
                "problem":              problem,
                "base_risk":            base_risk,
                "fused_risk":           fused_risk,
                "predicted_future_risk":fused_risk + random.gauss(0, 3),
                "severity":             severity,
                "recommended_action":   action_txt,
                "density_index":        density * 10,
                "avg_speed":            speed,
                "flow_status":          data["flow_status"],
                "density_trend":        noise_d,
                "people_trend":         random.gauss(0, 0.5),
                "speed_trend":          noise_s,
                "speed_drop":           data["speed_drop"],
                "opposing_flow":        data["opposing_flow"],
                "compression":          data["compression"],
                "queue_pressure":       data["queue_pressure"],
                "gate_pressure":        data["gate_pressure"],
            }, zone=zone)

            self._metrics_history.append({
                "people_count": count, "density": density,
                "fused_risk": fused_risk, "trend_risk": fused_risk * 0.9,
                "timestamp": datetime.utcnow().isoformat(),
            })
            if severity in ("MEDIUM", "HIGH"):
                self._alerts_history.append({
                    "type": problem, "description": action_txt,
                    "severity": severity, "zone": zone,
                    "timestamp": datetime.utcnow().isoformat(),
                    "fallen_count": 1 if data["fallen"] else 0,
                })

        if len(self._metrics_history) > 200:
            self._metrics_history = self._metrics_history[-200:]
        if len(self._alerts_history) > 100:
            self._alerts_history  = self._alerts_history[-100:]

    def build_twin_state(self) -> Dict:
        dash   = self.engine.get_dashboard_state()
        zones  = dash.get("zones", [])
        worst  = max(zones, key=lambda z: z["fused_risk"]) if zones else {}
        max_risk = dash["overall_risk"]
        problem  = worst.get("problem", "NORMAL")

        summary = {
            "prediction_layer": {
                "latest_fused_risk":  max_risk,
                "latest_future_risk": max_risk * 1.05,
                "latest_problem":     problem,
                "latest_flow_status": "chaotic" if max_risk > 70 else "smooth",
            },
            "avg_density":      max_risk / 120,
            "avg_people_count": 60 + max_risk * 0.8,
            "avg_fused_risk":   max_risk,
            "max_fused_risk":   max_risk,
            "demographic_distribution": {"male": 55, "female": 40, "unknown": 5},
        }
        metrics = self._metrics_history[-10:] if self._metrics_history else [{}]
        alerts  = self._alerts_history[-10:]  if self._alerts_history  else []
        return build_temple_simulation_state(summary, metrics, alerts)
