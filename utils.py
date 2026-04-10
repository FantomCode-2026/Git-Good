# utils.py
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def preprocess_video_for_dataset(
    video_path: str, output_dir: str, config, save_visualization: bool = False
):
    """
    Extract tracked feature sequences from a video for later labeling or training.
    """
    from main import CrowdBehaviorSystem

    system = CrowdBehaviorSystem(
        config_override={
            "visualization": {"show_display": False, "save_output_video": save_visualization}
        }
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_sequences = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = system.process_frame(frame, frame_id)
        for track_id, sequence in results["behavior_results"].items():
            del sequence  # behavior results are not used for dataset extraction here

        for track_id, person_sequence in system.sequence_manager.sequences.items():
            if len(person_sequence.features) >= system.sequence_manager.min_sequence_length:
                sequence_array = np.stack(list(person_sequence.features)).astype(np.float32)
                seq_id = f"track_{track_id}_frame_{frame_id}"
                np.save(output_dir / f"{seq_id}.npy", sequence_array)
                saved_sequences[seq_id] = {
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "length": int(sequence_array.shape[0]),
                }

        frame_id += 1

    cap.release()

    metadata = {
        "video_path": video_path,
        "total_frames": frame_id,
        "num_sequences": len(saved_sequences),
        "config": config.to_dict() if hasattr(config, "to_dict") else {},
        "sequences": saved_sequences,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {len(saved_sequences)} sequences from {video_path}")


def visualize_attention(video_path: str, model, config, output_path: str):
    """
    Placeholder for future transformer attention visualization support.
    """
    del video_path, model, config, output_path
    raise NotImplementedError(
        "Attention visualization is not implemented for the current transformer pipeline."
    )


def create_alert(frame: np.ndarray, message: str, severity: str = "high"):
    """
    Create alert overlay on frame.
    """
    colors = {
        "high": (0, 0, 255),
        "medium": (0, 165, 255),
        "low": (0, 255, 255),
    }

    color = colors.get(severity, (0, 0, 255))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    cv2.putText(
        frame,
        "ALERT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
    )
    cv2.putText(
        frame,
        message,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return frame


def export_predictions_to_json(predictions: List[Dict], output_path: str):
    """Export predictions to JSON format."""
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


def verify_tensor_shapes():
    """Utility to print current tensor shape expectations in the pipeline."""
    print("=== Tensor Shape Verification ===\n")
    print("1. CNN Feature Extractor:")
    print("   Input: (B, 3, 224, 224)")
    print("   Output: (B, 512)\n")

    print("2. Sequence Buffer:")
    print("   Input per frame: (512,)")
    print("   Output sequence: (T, 512)\n")

    print("3. Transformer:")
    print("   Input: (B, T, 512)")
    print("   After positional encoding: (B, T, 512)")
    print("   After transformer encoder: (B, T, 512)")
    print("   After pooling: (B, 512)\n")

    print("4. Classifier:")
    print("   Fall probability: (B,)")
    print("   Anomaly score: (B,)")
    print("   Behavior probabilities: (B, 3)\n")


if __name__ == "__main__":
    verify_tensor_shapes()
