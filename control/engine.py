"""Layer E – Control Engine"""

from __future__ import annotations
import logging, threading, time
from datetime import datetime
from typing import Dict, List, Optional

from control.scheduler import SchedulingController
from control.announcer import AnnouncementSystem

logger = logging.getLogger(__name__)

ZONE_LABELS = {
    "A": "Main Entrance",
    "B": "North Stand",
    "C": "South Stand",
    "D": "Food Court",
}
RISK_HIGH_THRESHOLD  = 70
RISK_MED_THRESHOLD   = 40
AUTO_ACTION_COOLDOWN = 30   # seconds


class ControlEngine:
    def __init__(self):
        self.scheduler  = SchedulingController()
        self.announcer  = AnnouncementSystem()
        self._lock      = threading.Lock()
        self._last_auto: Dict[str, float] = {}
        self._zone_states: Dict[str, Dict] = {
            z: self._default_zone(z, lbl) for z, lbl in ZONE_LABELS.items()
        }
        self._latest_eval: Optional[Dict] = None
        self._action_log: List[Dict] = []

    # ── Called by pipeline / simulator ────────────────────────────────
    def process_evaluation(self, eval_result: Dict, zone: str = "A") -> None:
        severity   = eval_result.get("severity", "LOW")
        fused_risk = eval_result.get("fused_risk", 0.0)
        problem    = eval_result.get("problem", "NORMAL")
        action_txt = eval_result.get("recommended_action", "Monitor situation")

        with self._lock:
            self._latest_eval = eval_result
            self._zone_states[zone] = {
                "zone":               zone,
                "label":              ZONE_LABELS.get(zone, zone),
                "fused_risk":         round(fused_risk, 1),
                "severity":           severity,
                "problem":            problem,
                "recommended_action": action_txt,
                "density_index":      eval_result.get("density_index", 0.0),
                "avg_speed":          eval_result.get("avg_speed", 0.0),
                "flow_status":        eval_result.get("flow_status", "smooth"),
                "updated_at":         datetime.utcnow().isoformat() + "Z",
            }

        if severity == "HIGH" and self._should_auto_act(zone):
            self._auto_respond(zone, fused_risk, problem, action_txt)
        elif severity == "MEDIUM" and self._should_auto_act(zone, cooldown=60):
            self.announcer.broadcast(
                message  = f"Zone {zone} ({ZONE_LABELS.get(zone,zone)}): "
                           f"Elevated crowd density. Risk {round(fused_risk)}% — {action_txt}",
                severity = "WARNING", source="risk_engine")
            self._record("WARNING_ANNOUNCEMENT", zone, fused_risk)

        # Check for claustrophobia alerts
        claustrophobia_alerts = eval_result.get("claustrophobia_alerts", [])
        for alert in claustrophobia_alerts:
            if alert == "FEMALE_CLAUSTROPHOBIA":
                self.announcer.broadcast(
                    message=f"Zone {zone}: Predicted high female density may cause claustrophobia. "
                            f"Predicted females: {eval_result.get('predicted_females', 0)}",
                    severity="WARNING", source="prediction_layer")
            elif alert == "MALE_CLAUSTROPHOBIA":
                self.announcer.broadcast(
                    message=f"Zone {zone}: Predicted high male density may cause claustrophobia. "
                            f"Predicted males: {eval_result.get('predicted_males', 0)}",
                    severity="WARNING", source="prediction_layer")

    def process_simulation_state(self, sim_state: Dict) -> None:
        scenario   = sim_state.get("scenario", {})
        fused_risk = scenario.get("fusedRisk", 0.0)
        problem    = scenario.get("problem", "NORMAL")
        from heatmap.logic import get_severity, suggest_action
        severity   = get_severity(fused_risk)
        action_txt = suggest_action(problem, risk=fused_risk)
        zone_counts= sim_state.get("zoneCounts", {})
        dominant   = max(zone_counts, key=lambda z: zone_counts.get(z, 0), default="queue")
        zone_map   = {"entry":"A","queue":"B","courtyard":"C","temple":"D","exit":"A"}
        self.process_evaluation({
            "severity": severity, "fused_risk": fused_risk, "problem": problem,
            "recommended_action": action_txt,
            "density_index": scenario.get("avgDensity", 0.0) * 10,
            "avg_speed": 1.0, "flow_status": scenario.get("flowStatus", "smooth"),
        }, zone=zone_map.get(dominant, "A"))

    # ── Manual API ─────────────────────────────────────────────────────
    def manual_schedule(self, zone, action, minutes):
        result = self.scheduler.apply(zone, action, minutes, triggered_by="manual")
        if result["success"]:
            self.announcer.broadcast(
                f"[Manual] Zone {zone}: {action} for {minutes} minutes",
                severity="INFO", source="operator")
            self._record(f"MANUAL_{action}", zone, None)
        return result

    def manual_announcement(self, message, severity="INFO"):
        return self.announcer.broadcast(message, severity=severity, source="operator")

    # ── Dashboard state ────────────────────────────────────────────────
    def get_dashboard_state(self) -> Dict:
        with self._lock:
            zones      = list(self._zone_states.values())
            latest_ev  = dict(self._latest_eval) if self._latest_eval else {}
            log        = list(self._action_log[-20:])
        max_risk = max((z["fused_risk"] for z in zones), default=0.0)
        worst    = max(zones, key=lambda z: z["fused_risk"]) if zones else {}
        return {
            "overall_risk":     round(max_risk, 1),
            "overall_severity": worst.get("severity", "LOW"),
            "zones":            zones,
            "active_schedules": self.scheduler.get_active(),
            "alerts":           self.announcer.get_feed(limit=30),
            "action_log":       log,
            "latest_eval":      latest_ev,
            "timestamp":        datetime.utcnow().isoformat() + "Z",
        }

    # ── Internals ──────────────────────────────────────────────────────
    def _auto_respond(self, zone, fused_risk, problem, action_txt):
        minutes = 20 if fused_risk >= 85 else 10
        self.scheduler.apply(zone, "DELAY", minutes, triggered_by="risk_engine")
        self.announcer.broadcast(
            message  = (f"⚠️ HIGH RISK – Zone {zone} ({ZONE_LABELS.get(zone,zone)})! "
                        f"Risk {round(fused_risk)}% · {problem}. Action: {action_txt}"),
            severity = "CRITICAL", source="risk_engine")
        self._record("AUTO_DELAY+ALERT", zone, fused_risk)
        logger.warning("[ControlEngine] HIGH risk Zone %s (%.1f%% %s) → DELAY %dmin + alert",
                       zone, fused_risk, problem, minutes)

    def _should_auto_act(self, zone, cooldown=AUTO_ACTION_COOLDOWN):
        now  = time.time()
        last = self._last_auto.get(zone, 0.0)
        if now - last >= cooldown:
            self._last_auto[zone] = now
            return True
        return False

    def _record(self, action_type, zone, risk):
        entry = {"action": action_type, "zone": zone,
                 "risk": round(risk, 1) if risk is not None else None,
                 "timestamp": datetime.utcnow().isoformat() + "Z"}
        self._action_log.append(entry)
        if len(self._action_log) > 100:
            self._action_log.pop(0)

    @staticmethod
    def _default_zone(zone, label):
        return {
            "zone": zone, "label": label, "fused_risk": 0.0,
            "severity": "LOW", "problem": "NORMAL",
            "recommended_action": "No action needed",
            "density_index": 0.0, "avg_speed": 1.35,
            "flow_status": "smooth",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
