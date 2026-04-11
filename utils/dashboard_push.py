"""HTTP helpers to push pipeline state and alerts into the CrowdGuard Flask dashboard."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from loguru import logger


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def post_json(base_url: str, path: str, payload: dict, timeout: float = 5.0) -> bool:
    url = base_url.rstrip("/") + path
    body = json.dumps(json_safe(payload)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        logger.warning("Dashboard HTTP {} {} — {} (is `python app.py` running?)", path, e.code, e.reason)
        return False
    except urllib.error.URLError as e:
        logger.warning("Dashboard unreachable {} — {} (start: python app.py --port …)", path, e.reason)
        return False


def push_evaluation(base_url: str, prediction_result: dict, zone: str = "A") -> bool:
    body = dict(prediction_result)
    body["zone"] = zone
    return post_json(base_url, "/api/notify", body)


def push_pipeline_alert(
    base_url: str,
    message: str,
    severity: str = "WARNING",
    source: str = "pipeline",
) -> bool:
    return post_json(
        base_url,
        "/api/pipeline_alert",
        {"message": message, "severity": severity, "source": source},
    )


def dashboard_origin_from_url(open_url: str) -> Optional[str]:
    """http://127.0.0.1:5000/ → http://127.0.0.1:5000"""
    if not (open_url or "").strip():
        return None
    p = urlparse(open_url.strip())
    if not p.netloc:
        return None
    scheme = p.scheme or "http"
    return f"{scheme}://{p.netloc}"


def ingest_alerts_bulk_to_dashboard(base_url: str, alerts: List[Dict[str, Any]], limit: int = 50) -> bool:
    """Map report-style alert dicts and POST in one request (after run, or when live push failed)."""
    rows = []
    for a in alerts[-limit:]:
        pair = pipeline_alert_to_message_severity(a)
        if pair:
            msg, sev = pair
            rows.append({"message": msg, "severity": sev, "source": "pipeline"})
    if not rows:
        return True
    return post_json(base_url, "/api/ingest_pipeline_alerts", {"alerts": rows})


def pipeline_alert_to_message_severity(alert: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Map metrics alert dict to (message, severity) for the dashboard announcer."""
    t = alert.get("type")
    if t == "crowd_anomaly":
        msg = alert.get("description") or "Crowd anomaly detected"
        return msg, "WARNING"
    if t == "prediction_alert":
        msg = alert.get("description") or "Prediction layer alert"
        risk = float(alert.get("fused_risk") or 0)
        sev = "CRITICAL" if risk >= 88 else "WARNING"
        return msg, sev
    if t == "behavior_anomaly":
        tid = alert.get("track_id")
        beh = alert.get("behavior", "?")
        score = float(alert.get("anomaly_score") or 0)
        return (
            f"Behavior anomaly · person {tid}: {beh} (score {score:.2f})",
            "WARNING",
        )
    if t == "fall_risk":
        tid = alert.get("track_id")
        p = float(alert.get("fall_risk") or 0)
        return f"Fall risk · person {tid}: p={p:.2f}", "WARNING"
    if t == "stampede_active":
        msg = alert.get("description") or "Stampede / crowd collapse risk"
        return msg, "CRITICAL"
    return None


class DashboardPushMixin:
    """Optional mixin for CrowdBehaviorSystem: POST eval + alerts to Flask (`python app.py`)."""

    _dashboard_base_url: Optional[str]
    _last_dashboard_eval_ts: float
    _dashboard_eval_interval: float
    _dashboard_alert_dedup: Dict[str, float]
    _dashboard_fail_log_ts: float

    def _init_dashboard_push(self) -> None:
        self._dashboard_base_url = None
        self._last_dashboard_eval_ts = 0.0
        self._dashboard_eval_interval = 1.0
        self._dashboard_alert_dedup = {}
        self._dashboard_fail_log_ts = 0.0
        self._dashboard_push_ok = False

    def configure_dashboard_push(self, base_url: Optional[str]) -> None:
        self._dashboard_base_url = base_url.rstrip("/") if base_url else None
        if self._dashboard_base_url:
            logger.info("Dashboard live push → %s", self._dashboard_base_url)

    def _maybe_push_dashboard_evaluation(self, prediction_result: dict) -> None:
        if not getattr(self, "_dashboard_base_url", None):
            return
        now = time.time()
        if now - self._last_dashboard_eval_ts < self._dashboard_eval_interval:
            return
        self._last_dashboard_eval_ts = now
        ok = push_evaluation(self._dashboard_base_url, prediction_result)
        if ok:
            self._dashboard_push_ok = True
        elif now - self._dashboard_fail_log_ts > 30:
            self._dashboard_fail_log_ts = now
            logger.warning(
                "Dashboard notify failed — start the server: python app.py "
                "(same host/port as --dashboard_url / --dashboard_port)"
            )

    def _maybe_push_dashboard_alert(self, alert: Dict[str, Any]) -> None:
        if not getattr(self, "_dashboard_base_url", None):
            return
        pair = pipeline_alert_to_message_severity(alert)
        if not pair:
            return
        msg, sev = pair
        key = self._dashboard_alert_dedup_key(alert)
        now = time.time()
        if now - self._dashboard_alert_dedup.get(key, 0) < self._dashboard_alert_cooldown(alert):
            return
        self._dashboard_alert_dedup[key] = now
        ok = push_pipeline_alert(self._dashboard_base_url, msg, severity=sev)
        if ok:
            self._dashboard_push_ok = True
        elif now - self._dashboard_fail_log_ts > 30:
            self._dashboard_fail_log_ts = now
            logger.warning(
                "Dashboard alert push failed — is `python app.py` running?"
            )

    @staticmethod
    def _dashboard_alert_dedup_key(alert: Dict[str, Any]) -> str:
        t = alert.get("type")
        if t in ("behavior_anomaly", "fall_risk"):
            return f"{t}:{alert.get('track_id')}"
        if t == "stampede_active":
            return f"stampede:{alert.get('fallen_count')}"
        if t == "crowd_anomaly":
            return "crowd_anomaly"
        if t == "prediction_alert":
            return "prediction_alert"
        return str(t)

    @staticmethod
    def _dashboard_alert_cooldown(alert: Dict[str, Any]) -> float:
        t = alert.get("type")
        if t == "crowd_anomaly":
            return 8.0
        if t == "prediction_alert":
            return 2.0
        if t in ("behavior_anomaly", "fall_risk"):
            return 4.0
        if t == "stampede_active":
            return 3.0
        return 5.0

    def _reset_dashboard_push_state(self) -> None:
        self._last_dashboard_eval_ts = 0.0
        self._dashboard_alert_dedup.clear()
        self._dashboard_push_ok = False
