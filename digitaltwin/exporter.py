from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Expected “normal operations” baseline used to score deviation from prediction.
BASELINE_FUSED_RISK = 26.0
BASELINE_FUTURE_RISK = 28.0


def build_temple_simulation_state(
    summary: Dict,
    metrics_history: List[Dict],
    alerts_history: List[Dict],
    demographic_summary: Dict | None = None,
) -> Dict:
    latest_metrics = metrics_history[-1] if metrics_history else {}

    latest_fused_risk = float(summary.get("prediction_layer", {}).get("latest_fused_risk", 0.0))
    latest_future_risk = float(summary.get("prediction_layer", {}).get("latest_future_risk", 0.0))
    latest_problem = summary.get("prediction_layer", {}).get("latest_problem", "NORMAL")
    latest_flow_status = summary.get("prediction_layer", {}).get("latest_flow_status", "smooth")
    avg_density = float(summary.get("avg_density", 0.0))
    avg_people_count = float(summary.get("avg_people_count", 0.0))
    avg_fused_risk = float(summary.get("avg_fused_risk", latest_fused_risk))
    max_fused_risk = float(summary.get("max_fused_risk", latest_fused_risk))
    current_people_count = float(latest_metrics.get("people_count", avg_people_count))
    current_density = float(latest_metrics.get("density", avg_density))

    scenario_level = "normal"
    if latest_fused_risk >= 75.0:
        scenario_level = "critical"
    elif latest_fused_risk >= 55.0:
        scenario_level = "watch"

    agent_target = int(
        np.clip(
            current_people_count * (1.1 + latest_fused_risk / 220.0) + current_density * 22.0,
            30.0,
            220.0,
        )
    )

    zone_distribution = _build_zone_distribution(
        latest_problem=latest_problem,
        latest_flow_status=latest_flow_status,
        latest_fused_risk=latest_fused_risk,
    )
    zone_counts = {
        zone: int(round(agent_target * ratio))
        for zone, ratio in zone_distribution.items()
    }
    _rebalance_counts(zone_counts, agent_target)

    latest_alerts = alerts_history[-5:] if alerts_history else []
    latest_alert_text = latest_alerts[-1].get("description", "No active alerts") if latest_alerts else "No active alerts"

    # Stampede stats from alerts
    stampede_events = sum(1 for a in alerts_history if a.get("type") == "stampede_active")
    latest_fallen = 0
    for a in reversed(alerts_history):
        if a.get("type") == "stampede_active":
            latest_fallen = a.get("fallen_count", 0)
            break

    # Demographics
    demo = demographic_summary or summary.get("demographic_distribution", {})

    baseline_agent = int(
        np.clip(avg_people_count * 1.05 + avg_density * 8.0, 30.0, 220.0)
    )
    baseline_dist = _baseline_zone_distribution()
    baseline_zone_counts: Dict[str, int] = {
        z: int(round(baseline_agent * r)) for z, r in baseline_dist.items()
    }
    _rebalance_counts(baseline_zone_counts, baseline_agent)

    risk_deviation = float(
        np.clip(
            abs(latest_fused_risk - BASELINE_FUSED_RISK) * 1.05
            + abs(latest_future_risk - BASELINE_FUTURE_RISK) * 0.35
            + (18.0 if latest_flow_status == "chaotic" else 0.0),
            0.0,
            100.0,
        )
    )

    flow_angle, flow_narrative = _stadium_flow_angle_narrative(
        latest_problem, latest_flow_status, latest_fused_risk
    )
    venue_stadium = {
        "directionalFlow": {
            "angleRadians": round(flow_angle, 4),
            "strength": round(min(1.0, risk_deviation / 100.0), 4),
            "narrative": flow_narrative,
        },
        "circulationBias": float(
            np.clip(0.38 + latest_fused_risk / 210.0 + (0.12 if latest_flow_status == "chaotic" else 0), 0.2, 0.95)
        ),
    }
    surge = float(
        np.clip(
            risk_deviation / 82.0 + (0.18 if latest_flow_status == "chaotic" else 0.05) + latest_fused_risk / 220.0,
            0.0,
            1.0,
        )
    )
    venue_concert = {
        "forwardSurge": round(surge, 4),
        "lateralSpread": round(float(np.clip(0.48 - risk_deviation / 320.0, 0.12, 0.62)), 4),
        "pitIntensity": round(float(np.clip(latest_fused_risk / 105.0, 0.0, 1.0)), 4),
        "narrative": _concert_surge_narrative(surge, latest_problem),
    }

    behavior_model = {
        "baseline": {
            "label": "normal_operations",
            "fusedRisk": round(BASELINE_FUSED_RISK, 2),
            "futureRisk": round(BASELINE_FUTURE_RISK, 2),
            "flowStatus": "smooth",
            "agentTarget": baseline_agent,
            "zoneCounts": baseline_zone_counts,
        },
        "prediction": {
            "fusedRisk": round(latest_fused_risk, 2),
            "futureRisk": round(latest_future_risk, 2),
            "flowStatus": latest_flow_status,
            "agentTarget": agent_target,
            "zoneCounts": dict(zone_counts),
        },
        "riskDeviation": round(risk_deviation, 2),
        "interpretation": _risk_deviation_text(risk_deviation, latest_fused_risk),
    }

    return {
        "source": "prediction_layer",
        "generatedAt": datetime.now().isoformat(),
        "behaviorModel": behavior_model,
        "venueSimulation": {
            "temple": {"mode": "queue_and_sanctuary", "linkedTo": "zoneCounts"},
            "stadium": venue_stadium,
            "concert": venue_concert,
        },
        "scenario": {
            "level": scenario_level,
            "title": _scenario_title(latest_problem, latest_fused_risk),
            "problem": latest_problem,
            "flowStatus": latest_flow_status,
            "fusedRisk": round(latest_fused_risk, 2),
            "futureRisk": round(latest_future_risk, 2),
            "trendRisk": round(float(latest_metrics.get("trend_risk", 0.0)), 2),
            "avgDensity": round(avg_density, 4),
            "avgPeopleCount": round(avg_people_count, 2),
            "avgFusedRisk": round(avg_fused_risk, 2),
            "peakFusedRisk": round(max_fused_risk, 2),
            "currentPeopleCount": round(current_people_count, 2),
            "currentDensity": round(current_density, 4),
            "agentTarget": agent_target,
            "rushMode": latest_fused_risk >= 88.0 and latest_flow_status == "chaotic",
            "alertText": latest_alert_text,
        },
        "crowdTuning": {
            "globalSpeedScale": float(np.clip(1.02 - latest_fused_risk / 260.0, 0.65, 1.0)),
            "pauseScale": float(np.clip(1.0 + latest_fused_risk / 180.0, 1.0, 1.55)),
            "pressureScale": float(np.clip(0.95 + latest_fused_risk / 200.0, 0.95, 1.45)),
            "queueBias": float(np.clip(1.0 + current_density * 0.8, 1.0, 1.5)),
            "templeBias": float(np.clip(1.0 + latest_fused_risk / 260.0, 1.0, 1.35)),
        },
        "zoneCounts": zone_counts,
        "recentAlerts": latest_alerts,
        "demographics": {
            "male": demo.get("male", 0),
            "female": demo.get("female", 0),
            "unknown": demo.get("unknown", 0),
        },
        "stampede": {
            "totalEvents": stampede_events,
            "latestFallenCount": latest_fallen,
            "active": latest_fallen > 0,
        },
    }


def export_temple_simulation_state(
    output_dir: Path,
    summary: Dict,
    metrics_history: List[Dict],
    alerts_history: List[Dict],
    demographic_summary: Dict | None = None,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_temple_simulation_state(
        summary, metrics_history, alerts_history, demographic_summary
    )

    json_path = output_dir / "temple_prediction_state.json"
    js_path = output_dir / "temple_prediction_state.js"

    json_path.write_text(json.dumps(state, indent=2))
    js_payload = "window.__TEMPLE_SIMULATION_STATE__ = " + json.dumps(state, indent=2) + ";\n"
    js_path.write_text(js_payload)
    return {
        "state": state,
        "json_path": str(json_path),
        "js_path": str(js_path),
    }


def _baseline_zone_distribution() -> Dict[str, float]:
    d = {
        "entry": 0.18,
        "queue": 0.30,
        "courtyard": 0.18,
        "temple": 0.22,
        "exit": 0.12,
    }
    t = sum(d.values())
    return {k: v / t for k, v in d.items()}


def _stadium_flow_angle_narrative(problem: str, flow_status: str, fused_risk: float) -> Tuple[float, str]:
    """Primary circulation direction (radians) + short copy for UI."""
    if problem == "BOTTLENECK":
        angle = math.pi * 0.85
        msg = "Directional bias toward queue relief sectors (bottleneck)."
    elif problem == "HIGH_COMPRESSION":
        angle = math.pi * 1.15
        msg = "Circulation steered toward exits / outer concourse (compression)."
    elif flow_status == "chaotic":
        angle = math.pi * 0.35 + (fused_risk / 220.0)
        msg = "Elevated directional shear while flow is chaotic."
    else:
        angle = math.pi * 0.2
        msg = "Baseline clockwise stadium circulation (normal event)."

    return angle, msg


def _concert_surge_narrative(surge: float, problem: str) -> str:
    if surge >= 0.65:
        return "Aggressive forward surge toward stage / pit — high deviation from normal spread."
    if surge >= 0.35:
        return "Moderate forward surge — pockets compress toward the stage axis."
    if problem == "HIGH_COMPRESSION":
        return "Compression detected — surge potential building at front zones."
    return "Forward surge near baseline — mostly lateral movement."


def _risk_deviation_text(deviation: float, fused_risk: float) -> str:
    if deviation < 12:
        return "Prediction close to normal operating envelope."
    if deviation < 35:
        return "Moderate deviation from baseline crowd behaviour; monitor key zones."
    if deviation < 60:
        return "Substantial deviation — future risk diverges from typical patterns."
    return "Severe deviation — predicted conditions far from normal baseline."


def _scenario_title(problem: str, fused_risk: float) -> str:
    if fused_risk >= 85.0:
        return f"{problem} evacuation pressure"
    if fused_risk >= 65.0:
        return f"{problem} crowd buildup"
    return f"{problem} monitored flow"


def _build_zone_distribution(latest_problem: str, latest_flow_status: str, latest_fused_risk: float) -> Dict[str, float]:
    if latest_problem == "HIGH_COMPRESSION":
        distribution = {
            "entry": 0.10,
            "queue": 0.44,
            "courtyard": 0.16,
            "temple": 0.22,
            "exit": 0.08,
        }
    elif latest_problem == "BOTTLENECK":
        distribution = {
            "entry": 0.20,
            "queue": 0.42,
            "courtyard": 0.14,
            "temple": 0.16,
            "exit": 0.08,
        }
    elif latest_flow_status == "chaotic":
        distribution = {
            "entry": 0.14,
            "queue": 0.34,
            "courtyard": 0.23,
            "temple": 0.18,
            "exit": 0.11,
        }
    else:
        distribution = {
            "entry": 0.18,
            "queue": 0.30,
            "courtyard": 0.18,
            "temple": 0.22,
            "exit": 0.12,
        }

    if latest_fused_risk >= 85.0:
        distribution["temple"] += 0.04
        distribution["queue"] += 0.03
        distribution["entry"] -= 0.03
        distribution["exit"] -= 0.04

    total = sum(distribution.values())
    return {key: value / total for key, value in distribution.items()}


def _rebalance_counts(zone_counts: Dict[str, int], agent_target: int):
    current_total = sum(zone_counts.values())
    if current_total == agent_target:
        return

    preferred_order = ["queue", "temple", "courtyard", "entry", "exit"]
    delta = agent_target - current_total
    idx = 0
    while delta != 0:
        zone = preferred_order[idx % len(preferred_order)]
        if delta > 0:
            zone_counts[zone] += 1
            delta -= 1
        elif zone_counts[zone] > 0:
            zone_counts[zone] -= 1
            delta += 1
        idx += 1
