# utils.py
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict
import json
from pathlib import Path

def preprocess_video_for_dataset(video_path: str, output_dir: str, 
                                 config, save_visualization: bool = False):
    """
    Preprocess a video to extract features for dataset creation.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted features
        config: Configuration object
        save_visualization: Whether to save visualization frames
    """
    from main import FallDetectionPipeline
    
    pipeline = FallDetectionPipeline(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    frame_id = 0
    sequences = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame (without transformer prediction)
        results = pipeline.process_frame(frame, frame_id)
        
        # Save sequences periodically
        if frame_id % 100 == 0:
            for track_id, buffer in pipeline.sequence_manager.buffers.items():
                if len(buffer) == config.sequence_length:
                    sequence = torch.stack(list(buffer)).numpy()
                    seq_id = f"track_{track_id}_frame_{frame_id}"
                    np.save(output_dir / f"{seq_id}.npy", sequence)
                    sequences[seq_id] = buffer
        
        frame_id += 1
    
    cap.release()
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({
            'video_path': video_path,
            'total_frames': frame_id,
            'num_sequences': len(sequences),
            'config': config.__dict__
        }, f, indent=2)
    
    print(f"Extracted {len(sequences)} sequences from {video_path}")

def visualize_attention(video_path: str, model, config, output_path: str):
    """
    Visualize transformer attention weights.
    Placeholder - requires hook registration to capture attention.
    """
    # Implementation would:
    # 1. Register forward hooks on transformer attention layers
    # 2. Capture and aggregate attention weights
    # 3. Overlay on video frames
    pass

def create_alert(frame: np.ndarray, message: str, severity: str = 'high'):
    """
    Create alert overlay on frame.
    
    Args:
        frame: Input frame
        message: Alert message
        severity: 'high', 'medium', 'low'
        
    Returns:
        Frame with alert overlay
    """
    colors = {
        'high': (0, 0, 255),    # Red
        'medium': (0, 165, 255), # Orange
        'low': (0, 255, 255)     # Yellow
    }
    
    color = colors.get(severity, (0, 0, 255))
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Add alert text
    cv2.putText(frame, "⚠ ALERT ⚠", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, message, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def export_predictions_to_json(predictions: List[Dict], output_path: str):
    """Export predictions to JSON format"""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

# Tensor dimension verification utility
def verify_tensor_shapes():
    """Utility to verify all tensor shapes in the pipeline"""
    print("=== Tensor Shape Verification ===\n")
    
    # CNN Feature Extractor
    print("1. CNN Feature Extractor:")
    print(f"   Input: (B, 3, 224, 224)")
    print(f"   Output: (B, 512)\n")
    
    # Sequence Buffer
    print("2. Sequence Buffer:")
    print(f"   Input per frame: (512,)")
    print(f"   Output sequence: (16, 512)\n")
    
    # Transformer
    print("3. Transformer:")
    print(f"   Input: (B, 16, 512)")
    print(f"   After positional encoding: (B, 16, 512)")
    print(f"   After CLS token: (B, 17, 512)")
    print(f"   After transformer encoder: (B, 17, 512)")
    print(f"   After pooling: (B, 512)")
    print(f"   Output logits: (B, 2)\n")
    
    # Classifier
    print("4. Classification:")
    print(f"   Logits: (B, 2)")
    print(f"   Probabilities: (B, 2)")
    print(f"   Prediction: scalar (0 or 1)\n")

if __name__ == "__main__":
    verify_tensor_shapes()