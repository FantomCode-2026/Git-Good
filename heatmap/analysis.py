# analysis.py

import math
from collections import Counter


# -------------------------------
# 1. EXTRACT POSITIONS FROM YOLO
# -------------------------------
def get_positions(results):
    positions = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            positions.append((cx, cy))

    return positions


# -------------------------------
# 2. COMPUTE MOVEMENTS
# -------------------------------
def compute_movements(prev_positions, curr_positions):

    movements = []

    min_len = min(len(prev_positions), len(curr_positions))

    for i in range(min_len):
        x1, y1 = prev_positions[i]
        x2, y2 = curr_positions[i]

        dx = x2 - x1
        dy = y2 - y1

        speed = math.sqrt(dx**2 + dy**2)

        movements.append({
            "dx": dx,
            "dy": dy,
            "speed": speed
        })

    return movements


# -------------------------------
# 3. AVERAGE SPEED
# -------------------------------
def get_avg_speed(movements):

    if not movements:
        return 0

    return sum(m["speed"] for m in movements) / len(movements)


# -------------------------------
# 4. DIRECTIONS
# -------------------------------
def get_directions(movements):

    directions = []

    for m in movements:
        dx = m["dx"]
        dy = m["dy"]

        if abs(dx) > abs(dy):
            if dx > 2:
                directions.append("RIGHT")
            elif dx < -2:
                directions.append("LEFT")
        else:
            if dy > 2:
                directions.append("DOWN")
            elif dy < -2:
                directions.append("UP")

    return directions


# -------------------------------
# 5. FLOW STATUS
# -------------------------------
def get_flow_status(directions):

    if not directions:
        return "smooth"

    count = Counter(directions)
    dominant = count.most_common(1)[0][1]

    if dominant / len(directions) > 0.7:
        return "smooth"

    return "chaotic"


# -------------------------------
# 6. OPPOSING FLOW
# -------------------------------
def detect_opposing_flow(directions):

    has_left = "LEFT" in directions
    has_right = "RIGHT" in directions

    has_up = "UP" in directions
    has_down = "DOWN" in directions

    return (has_left and has_right) or (has_up and has_down)


# -------------------------------
# 7. SPEED DROP
# -------------------------------
def detect_speed_drop(prev_speed, curr_speed):

    if prev_speed == 0:
        return False

    return curr_speed < prev_speed * 0.7


# -------------------------------
# 8. DENSITY
# -------------------------------
def get_density(positions):
    return len(positions)


# -------------------------------
# 9. COMPRESSION
# -------------------------------
def detect_compression(density, avg_speed):
    return density > 6 and avg_speed < 1.2


# -------------------------------
# 10. FALL DETECTION
# -------------------------------
def detect_fall(results):

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            aspect_ratio = width / height

            if aspect_ratio >= 1.1:
                return True

    return False


# -------------------------------
# 11. OVERLAP DETECTION (CROWD COLLAPSE)
# -------------------------------
def is_overlapping(box1, box2):

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)


def detect_overlap(results):

    boxes = []

    for r in results:
        for box in r.boxes:
            boxes.append(tuple(map(int, box.xyxy[0])))

    overlap_count = 0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if is_overlapping(boxes[i], boxes[j]):
                overlap_count += 1

    return overlap_count > 3


# -------------------------------
# 12. STUCK / DISTRESS DETECTION
# -------------------------------
def detect_stuck(avg_speed, density):
    return avg_speed < 0.3 and density > 6


# -------------------------------
# 13. MASTER FUNCTION (IMPORTANT)
# -------------------------------
def build_data(results, prev_positions, prev_speed):

    # Positions
    positions = get_positions(results)

    # Movement
    movements = compute_movements(prev_positions, positions)

    # Speed
    avg_speed = get_avg_speed(movements)

    # Directions
    directions = get_directions(movements)

    # Flow
    flow_status = get_flow_status(directions)
    opposing_flow = detect_opposing_flow(directions)

    # Speed drop
    speed_drop = detect_speed_drop(prev_speed, avg_speed)

    # Density
    density = get_density(positions)

    # Compression
    compression = detect_compression(density, avg_speed)

    # Fall
    fallen = detect_fall(results)

    # Overlap (crowd collapse)
    overlap = detect_overlap(results)

    # Distress
    stuck = detect_stuck(avg_speed, density)

    # Final data package
    data = {
        "density": density,
        "avg_speed": avg_speed,
        "flow_status": flow_status,
        "speed_drop": speed_drop,
        "opposing_flow": opposing_flow,
        "compression": compression,
        "fallen": fallen,
        "overlap": overlap,
        "stuck": stuck
    }

    return data, positions, avg_speed