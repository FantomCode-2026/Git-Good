"""
CrowdGuard – Flask Server (Layer F)
Serves the dashboard UI and all REST + SSE endpoints.

Usage:
    python app.py --demo            # demo simulator, no camera
    python app.py --demo --port 5001
    python app.py                   # real mode: pipeline POSTs to /api/notify
"""

from __future__ import annotations
import argparse, json, logging, queue, sys, time, threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

from flask import (Flask, Response, jsonify, render_template,
                   request, send_from_directory)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crowdguard")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from control.engine    import ControlEngine
from control.simulator import DemoSimulator

app = Flask(
    __name__,
    template_folder=str(ROOT / "dashboard" / "templates"),
    static_folder   =str(ROOT / "dashboard" / "static"),
)

_engine:    ControlEngine  = None
_simulator: DemoSimulator  = None
_demo_mode: bool           = False
_sse_subs:  list           = []          # list[queue.Queue]


# ══════════════════════════════════════════════════════════════════════
# UI ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/twin")
def digital_twin():
    return send_from_directory(ROOT / "digitaltwin", "temple_simulation (3).html")


@app.route("/twin/<path:filename>")
def twin_assets(filename):
    return send_from_directory(ROOT / "digitaltwin", filename)


# ══════════════════════════════════════════════════════════════════════
# LAYER E – CONTROL ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.route("/control/schedule", methods=["POST"])
def control_schedule():
    data    = request.get_json(force=True) or {}
    zone    = str(data.get("zone", "")).strip()
    action  = str(data.get("action", "")).strip()
    minutes = data.get("minutes", 10)
    if not zone or not action:
        return jsonify({"success": False, "error": "zone and action required"}), 400
    result = _engine.manual_schedule(zone, action, minutes)
    _push("schedule_update", result)
    return jsonify(result), 200 if result["success"] else 400


@app.route("/control/announcement", methods=["POST"])
def control_announcement():
    data     = request.get_json(force=True) or {}
    message  = str(data.get("message", "")).strip()
    severity = str(data.get("severity", "INFO")).strip().upper()
    if not message:
        return jsonify({"success": False, "error": "message required"}), 400
    result = _engine.manual_announcement(message, severity)
    _push("announcement", result)
    return jsonify(result), 200


# ══════════════════════════════════════════════════════════════════════
# LAYER F – DATA ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/state")
def api_state():
    state = _engine.get_dashboard_state()
    if _demo_mode and _simulator:
        try:
            state["twin_state"] = _simulator.build_twin_state()
            state["scenario"]   = _simulator.current_scenario()
        except Exception as e:
            logger.warning("Twin state error: %s", e)
    return jsonify(state)


@app.route("/api/twin_state")
def api_twin_state():
    if _demo_mode and _simulator:
        try:
            return jsonify(_simulator.build_twin_state())
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    js_path = ROOT / "digitaltwin" / "temple_prediction_state.json"
    if js_path.exists():
        return Response(js_path.read_text(), mimetype="application/json")
    return jsonify({}), 204


@app.route("/api/alerts")
def api_alerts():
    limit = int(request.args.get("limit", 30))
    return jsonify(_engine.announcer.get_feed(limit=limit))


@app.route("/api/schedules")
def api_schedules():
    return jsonify({"active": _engine.scheduler.get_active(),
                    "history": _engine.scheduler.get_history()})


@app.route("/api/zones")
def api_zones():
    return jsonify(_engine.get_dashboard_state()["zones"])


# ── External pipeline hook ───────────────────────────────────────────
@app.route("/api/notify", methods=["POST"])
def api_notify():
    """Real pipeline POSTs prediction results here."""
    data = request.get_json(force=True) or {}
    zone = str(data.pop("zone", "A"))
    _engine.process_evaluation(data, zone=zone)
    return jsonify({"success": True})


# ══════════════════════════════════════════════════════════════════════
# SERVER-SENT EVENTS
# ══════════════════════════════════════════════════════════════════════

@app.route("/stream")
def sse_stream():
    def generate() -> Iterator[str]:
        q: queue.Queue = queue.Queue(maxsize=50)
        _sse_subs.append(q)
        try:
            # Send initial state immediately
            yield _fmt("state_update", _engine.get_dashboard_state())
            while True:
                try:
                    evt_type, payload = q.get(timeout=2.0)
                    yield _fmt(evt_type, payload)
                except queue.Empty:
                    yield f"event: ping\ndata: {json.dumps({'ts': time.time()})}\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                _sse_subs.remove(q)
            except ValueError:
                pass

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


def _fmt(evt_type: str, data) -> str:
    return f"event: {evt_type}\ndata: {json.dumps(data)}\n\n"


def _push(evt_type: str, data) -> None:
    dead = []
    for q in _sse_subs:
        try:
            q.put_nowait((evt_type, data))
        except queue.Full:
            dead.append(q)
    for q in dead:
        try:
            _sse_subs.remove(q)
        except ValueError:
            pass


def _broadcaster():
    """Push full state to all SSE clients every 2s."""
    while True:
        try:
            _push("state_update", _engine.get_dashboard_state())
        except Exception as e:
            logger.warning("[Broadcaster] %s", e)
        time.sleep(2)


# ══════════════════════════════════════════════════════════════════════
# APP FACTORY
# ══════════════════════════════════════════════════════════════════════

def create_app(demo: bool = False) -> Flask:
    global _engine, _simulator, _demo_mode
    _demo_mode = demo
    _engine    = ControlEngine()

    # Push CRITICAL announcements via SSE immediately
    _engine.announcer.register_callback(
        lambda ann: _push("announcement", ann)
    )

    if demo:
        _simulator = DemoSimulator(_engine, interval=2.0)
        _simulator.start()
        logger.info("[App] Demo simulator running")
    else:
        logger.info("[App] Real mode – pipeline should POST to /api/notify")

    threading.Thread(target=_broadcaster, daemon=True).start()
    return app


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdGuard Dashboard Server")
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    flask_app = create_app(demo=args.demo)

    print("\n" + "═"*55)
    print("  🧠  CrowdGuard – Crowd Monitoring Dashboard")
    print("═"*55)
    print(f"  Dashboard  → http://localhost:{args.port}/")
    print(f"  Twin       → http://localhost:{args.port}/twin")
    print(f"  API State  → http://localhost:{args.port}/api/state")
    print(f"  Demo mode  → {args.demo}")
    print("═"*55 + "\n")

    flask_app.run(host=args.host, port=args.port,
                  debug=args.debug, threaded=True, use_reloader=False)
