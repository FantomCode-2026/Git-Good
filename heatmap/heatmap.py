import cv2
import numpy as np


def _clean_positions(positions, width, height):
    cleaned = []
    for point in positions:
        if point is None or len(point) < 2:
            continue
        x, y = point[:2]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < width and 0 <= y < height:
            cleaned.append((x, y))
    return cleaned


def generate_heatmap(frame, positions, influence_radius=None):
    h, w = frame.shape[:2]
    cleaned_positions = _clean_positions(positions, w, h)

    if not cleaned_positions:
        return frame

    radius = influence_radius or max(12, int(round(min(h, w) * 0.045)))
    kernel = max(3, radius * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1

    density = np.zeros((h, w), dtype=np.float32)
    for x, y in cleaned_positions:
        mask = np.zeros_like(density)
        cv2.circle(mask, (x, y), radius, 1.0, -1)
        density += mask

    density = cv2.GaussianBlur(density, (kernel, kernel), radius / 2.0)

    # Scale by an expected local crowd cap instead of clipping every overlap to 1.
    crowd_cap = max(2.0, min(8.0, np.ceil(len(cleaned_positions) / 3.0)))
    density = np.clip(density / crowd_cap, 0.0, 1.0)
    heatmap = (density * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    alpha = (density * 0.55)[..., None]
    overlay = (frame.astype(np.float32) * (1.0 - alpha) + heatmap_color.astype(np.float32) * alpha)

    return overlay.astype(np.uint8)
