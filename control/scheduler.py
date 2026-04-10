"""Layer E – Scheduling Control"""

from __future__ import annotations
import logging, threading
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)
_VALID_ACTIONS = {"DELAY", "REDIRECT"}


class ScheduleEntry:
    def __init__(self, zone, action, minutes, triggered_by="manual"):
        self.zone         = zone.upper()
        self.action       = action.upper()
        self.minutes      = minutes
        self.triggered_by = triggered_by
        self.timestamp    = datetime.utcnow().isoformat() + "Z"
        self.id           = f"{self.zone}-{self.action}-{int(datetime.utcnow().timestamp())}"

    def to_dict(self):
        return {"id": self.id, "zone": self.zone, "action": self.action,
                "minutes": self.minutes, "triggered_by": self.triggered_by,
                "timestamp": self.timestamp}


class SchedulingController:
    def __init__(self, max_history=100):
        self._lock    = threading.Lock()
        self._history: List[ScheduleEntry] = []
        self._active:  Dict[str, ScheduleEntry] = {}
        self.max_history = max_history

    def apply(self, zone, action, minutes, triggered_by="manual"):
        zone, action = zone.strip().upper(), action.strip().upper()
        if action not in _VALID_ACTIONS:
            return {"success": False, "error": f"Invalid action '{action}'"}
        try:
            minutes = int(minutes)
        except Exception:
            return {"success": False, "error": "minutes must be integer"}
        if not (1 <= minutes <= 120):
            return {"success": False, "error": "minutes must be 1-120"}
        entry = ScheduleEntry(zone, action, minutes, triggered_by)
        with self._lock:
            self._active[zone] = entry
            self._history.append(entry)
            if len(self._history) > self.max_history:
                self._history.pop(0)
        logger.info("[SchedulingCtrl] %s Zone %s %d min (by=%s)", action, zone, minutes, triggered_by)
        return {"success": True, "entry": entry.to_dict(),
                "message": f"Zone {zone}: {action} applied for {minutes} minutes"}

    def get_active(self):
        with self._lock:
            return [e.to_dict() for e in self._active.values()]

    def get_history(self, limit=20):
        with self._lock:
            return [e.to_dict() for e in self._history[-limit:]]

    def clear_zone(self, zone):
        with self._lock:
            return self._active.pop(zone.upper(), None) is not None
