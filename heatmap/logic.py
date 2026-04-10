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

    # 🔴 Highest priority
    if fallen:
        return "FALL_DETECTED"

    # 🔴 Compression
    if compression and avg_speed < 1.5:
        return "HIGH_COMPRESSION"

    # 🔴 Bottleneck
    if density > 5 and avg_speed < 2:
        return "BOTTLENECK"

    # 🔴 Opposing movement
    if opposing_flow:
        return "OPPOSING_FLOW"

    # 🟡 Flow instability
    if flow_status == "chaotic":
        return "FLOW_CONFLICT"

    # 🟡 Speed drop
    if speed_drop:
        return "SPEED_DROP"

    # 🟡 Unusual behavior
    if unusual_behavior:
        return "UNUSUAL_BEHAVIOR"

    return "NORMAL"


def get_risk_score(problem):

    risk_map = {
        "NORMAL": 20,
        "SPEED_DROP": 45,
        "UNUSUAL_BEHAVIOR": 50,
        "FLOW_CONFLICT": 60,
        "OPPOSING_FLOW": 70,
        "BOTTLENECK": 80,
        "HIGH_COMPRESSION": 85,
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
    
def suggest_action(problem):

    action_map = {
        "NORMAL": "No action needed",
        "SPEED_DROP": "Monitor crowd movement",
        "UNUSUAL_BEHAVIOR": "Alert authorities to investigate",
        "FLOW_CONFLICT": "Stop entry temporarily and guide crowd",
        "OPPOSING_FLOW": "Redirect crowd to one direction",
        "BOTTLENECK": "Open additional gates immediately",
        "HIGH_COMPRESSION": "Disperse crowd and reduce density",
        "FALL_DETECTED": "Emergency response required immediately"
    }

    return action_map.get(problem, "No action available")