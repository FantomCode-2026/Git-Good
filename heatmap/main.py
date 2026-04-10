import cv2
import numpy as np

from heatmap.heatmap import generate_heatmap
from heatmap.logic import detect_problem, get_risk_score, get_severity, suggest_action


def main():
    # --- Simulated positions (replace later with YOLO output) ---
    positions = [
        (100, 200),
        (120, 210),
        (400, 300),
        (420, 310),
        (410, 320),
    ]

    frame_width = 800
    frame_height = 600
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    frame = generate_heatmap(frame, positions)

    density = len(positions)
    data = {
        "density": density,
        "avg_speed": 1.2,
        "flow_status": "chaotic",
        "speed_drop": True,
        "opposing_flow": True,
        "compression": True,
        "unusual_behavior": False,
        "fallen": False,
    }

    problem = detect_problem(data)
    risk = get_risk_score(problem)
    severity = get_severity(risk)
    action = suggest_action(problem)

    cv2.rectangle(frame, (0, 0), (frame_width, 120), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "Crowd Intelligence System",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Problem: {problem}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Risk: {risk}",
        (300, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if severity == "HIGH":
        sev_color = (0, 0, 255)
    elif severity == "MEDIUM":
        sev_color = (0, 255, 255)
    else:
        sev_color = (0, 255, 0)

    cv2.putText(
        frame,
        f"Severity: {severity}",
        (500, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        sev_color,
        2,
    )
    cv2.putText(
        frame,
        f"Action: {action}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        sev_color,
        2,
    )

    cv2.imshow("Crowd Monitoring System", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
