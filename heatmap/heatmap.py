import cv2
import numpy as np

def generate_heatmap(frame, positions):

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Add "influence" for each person
    for (x, y) in positions:
        cv2.circle(heatmap, (x, y), 50, 1, -1)  # radius = influence

    # Smooth it (THIS MAKES IT LOOK PRO)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # Normalize to 0–255
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Convert to color (JET = red/yellow/blue)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    return overlay