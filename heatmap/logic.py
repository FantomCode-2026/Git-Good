# logic.py

def detect_problem(data):

    density = data.get("density", 0)
    avg_speed = data.get("avg_speed", 0)
    flow_status = data.get("flow_status", "smooth")
    speed_drop = data.get("speed_drop", False)
    opposing_flow = data.get("opposing_flow", False)
    compression = data.get("compression", False)
    unusual_behavior = data.get("unusual_behavior", False)
    fallen = data.get("fallen", False)
    gate_pressure = data.get("gate_pressure", False)
    queue_pressure = data.get("queue_pressure", False)

    if fallen:
        return "FALL_DETECTED"

    if compression and density >= 8.5 and avg_speed < 0.7:
        return "HIGH_COMPRESSION"

    if gate_pressure and density >= 7.0 and avg_speed < 1.0:
        return "GATE_PRESSURE"

    if queue_pressure and density >= 6.2 and (avg_speed < 0.95 or speed_drop):
        return "BOTTLENECK"

    if opposing_flow and avg_speed < 1.2:
        return "OPPOSING_FLOW"

    if flow_status == "chaotic" and density >= 4.0:
        return "FLOW_CONFLICT"

    if speed_drop and density >= 3.5:
        return "SPEED_DROP"

    if unusual_behavior:
        return "UNUSUAL_BEHAVIOR"

    return "NORMAL"


def get_risk_score(problem):

    risk_map = {
        "NORMAL": 20,
        "SPEED_DROP": 42,
        "UNUSUAL_BEHAVIOR": 50,
        "FLOW_CONFLICT": 58,
        "OPPOSING_FLOW": 68,
        "GATE_PRESSURE": 72,
        "BOTTLENECK": 76,
        "HIGH_COMPRESSION": 88,
        "FALL_DETECTED": 95
    }

    return risk_map.get(problem, 20)


def get_severity(risk):

    if risk < 40:
        return "LOW"
    elif risk < 70:
        return "MEDIUM"
    else:
        return "HIGH"


def suggest_action(problem, data=None, risk=None):

    density = (data or {}).get("density", 0)
    flow_status = (data or {}).get("flow_status", "smooth")

    if problem == "NORMAL":
        if risk is not None and risk >= 35:
            return "Monitor inflow and keep stewards on standby"
        return "No action needed"

    if problem == "SPEED_DROP":
        return "Monitor queue progression and slow new inflow if needed"

    if problem == "UNUSUAL_BEHAVIOR":
        return "Alert staff to investigate the affected area"

    if problem == "FLOW_CONFLICT":
        return "Guide pedestrians into a single direction and reinforce lane discipline"

    if problem == "OPPOSING_FLOW":
        return "Separate opposing streams and redirect one side immediately"

    if problem == "GATE_PRESSURE":
        if density >= 8.5:
            return "Open additional gates or queue lanes and meter inflow"
        return "Prepare extra gate capacity and deploy queue marshals"

    if problem == "BOTTLENECK":
        if flow_status == "chaotic":
            return "Meter inflow, clear the choke point, and add queue marshals"
        return "Reduce inflow briefly and smooth queue movement at the choke point"

    if problem == "HIGH_COMPRESSION":
        return "Stop inflow, open relief routes, and disperse the compressed zone immediately"

    if problem == "FALL_DETECTED":
        return "Dispatch emergency response and create space around the casualty"

    return "No action available"
