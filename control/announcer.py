"""Layer E – Announcement System"""

from __future__ import annotations
import logging, os, threading, time, urllib.request, json
from datetime import datetime
from typing import List, Dict, Callable, Optional

logger = logging.getLogger(__name__)


class Announcement:
    _LEVELS = {"INFO", "WARNING", "CRITICAL"}

    def __init__(self, message, severity="INFO", source="system"):
        self.message   = message
        self.severity  = severity.upper() if severity.upper() in self._LEVELS else "INFO"
        self.source    = source
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.id        = f"ann-{int(time.time()*1000)}"

    def to_dict(self):
        return {"id": self.id, "message": self.message, "severity": self.severity,
                "source": self.source, "timestamp": self.timestamp}


class AnnouncementSystem:
    def __init__(self, max_feed=50):
        self._lock       = threading.Lock()
        self._feed: List[Announcement] = []
        self._callbacks: List[Callable] = []
        self.max_feed    = max_feed
        self._tg_token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._tg_enabled = bool(self._tg_token and self._tg_chat_id)
        if self._tg_enabled:
            logger.info("[Announcer] Telegram enabled")

    def broadcast(self, message, severity="INFO", source="system"):
        ann = Announcement(message, severity, source)
        with self._lock:
            self._feed.append(ann)
            if len(self._feed) > self.max_feed:
                self._feed.pop(0)
            callbacks = list(self._callbacks)
        logger.info("[Announcer] [%s] %s", ann.severity, message)
        for cb in callbacks:
            try:
                cb(ann.to_dict())
            except Exception as e:
                logger.warning("[Announcer] Callback error: %s", e)
        if ann.severity == "CRITICAL" and self._tg_enabled:
            threading.Thread(target=self._send_telegram,
                             args=(f"🚨 CROWD ALERT\n{message}",), daemon=True).start()
        return {"success": True, "announcement": ann.to_dict()}

    def register_callback(self, fn):
        with self._lock:
            self._callbacks.append(fn)

    def get_feed(self, limit=20):
        with self._lock:
            return [a.to_dict() for a in self._feed[-limit:]]

    def _send_telegram(self, text):
        try:
            payload = json.dumps({"chat_id": self._tg_chat_id, "text": text,
                                  "parse_mode": "Markdown"}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self._tg_token}/sendMessage",
                data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            logger.warning("[Telegram] Failed: %s", e)
