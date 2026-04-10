"""
Advanced Decision + Control Layer (6 Gate System) - NO SPEECH VERSION

Features:
- Supports 6 gates (A–F)
- Risk trend prediction
- Zone-based notifications
- Smart exit suggestion
- False alarm filtering
- Smart repeat control
- Emergency police alert (simulated)
"""

import time


risk_history = []
last_problem = None
repeat_count = 0
problem_start_time = 0
last_call_time = 0



def can_call():
    global last_call_time
    if time.time() - last_call_time > 30:
        last_call_time = time.time()
        return True
    return False


def call_police(location, risk):
    print(f"\n🚨 [POLICE ALERT]")
    print(f"Location: {location}")
    print(f"Risk Level: {risk}")
    print("Action: Immediate assistance required\n")




def get_risk_trend(current_risk):
    risk_history.append(current_risk)

    if len(risk_history) < 2:
        return "stable"

    if risk_history[-1] > risk_history[-2]:
        return "increasing"
    elif risk_history[-1] < risk_history[-2]:
        return "decreasing"
    else:
        return "stable"



def get_best_exit(zone_density, current_location):
    filtered = {k: v for k, v in zone_density.items() if k != current_location}
    return min(filtered, key=filtered.get)



def is_persistent(problem):
    global problem_start_time, last_problem

    current_time = time.time()

    if problem != last_problem:
        problem_start_time = current_time
        last_problem = problem
        return False

    return (current_time - problem_start_time) > 5



def notify(title, message):
    print(f"\n🔔 [{title}]")
    print(message)



def handle_event(problem, location, risk, zone_density):

    global repeat_count

    trend = get_risk_trend(risk)

    if not is_persistent(problem):
        print("[INFO] Ignoring temporary fluctuation")
        return

    best_exit = get_best_exit(zone_density, location)

    # Smart repeat control
    if problem == last_problem:
        repeat_count += 1
        if repeat_count > 2:
            return
    else:
        repeat_count = 0

    if problem == "normal":
        notify("STATUS", "Everything is okay. Crowd is moving normally.")


    elif problem == "slow":
        notify("WARNING", "Movement is slow. Please keep walking.")


    elif problem == "bottleneck":
        notify(
            "BOTTLENECK DETECTED",
            f"Too many people at {location}. Suggested exit: {best_exit}"
        )


    elif problem == "panic":
        notify(
            "PANIC ALERT",
            "People are panicking. Stay calm. Help is being arranged."
        )


    elif problem == "fallen_person":
        notify(
            "EMERGENCY",
            f"Person fallen at {location}. Clear space immediately."
        )

  
    if risk > 80:
        notify(
            "HIGH RISK ALERT",
            "Emergency situation. Move slowly to nearest exit."
        )

        if can_call():
            call_police(location, risk)

    
    if trend == "increasing" and risk > 50:
        notify(
            "TREND ALERT",
            "Crowd is increasing. Situation may worsen."
        )



def decision_control(problem, location, risk, zone_density):

    print(f"\n[INFO] Problem: {problem} | Location: {location} | Risk: {risk}")

    handle_event(problem, location, risk, zone_density)

    if risk > 80:
        print("[ALERT] 🔴 HIGH RISK DETECTED")

if __name__ == "__main__":

    zones = {
        "Gate A": 0.9,
        "Gate B": 0.7,
        "Gate C": 0.5,
        "Gate D": 0.2,
        "Gate E": 0.3,
        "Gate F": 0.4
    }

    print("---- NORMAL ----")
    decision_control("normal", "Gate A", 20, zones)
    time.sleep(6)

    print("---- BOTTLENECK ----")
    decision_control("bottleneck", "Gate A", 60, zones)
    time.sleep(6)

    print("---- INCREASING RISK ----")
    zones["Gate A"] = 0.95
    decision_control("bottleneck", "Gate A", 75, zones)
    time.sleep(6)

    print("---- EMERGENCY ----")
    decision_control("fallen_person", "Gate A", 90, zones)
