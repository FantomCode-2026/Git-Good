"""
Visualization utilities for drawing bounding boxes, alerts, and crowd statistics
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger

class Visualizer:
    """
    Handles all visualization for the crowd behavior analysis system
    Draws bounding boxes, person IDs, behavior labels, and crowd statistics
    """
    
    def __init__(self, config):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Configuration object with visualization settings
        """
        self.config = config
        
        # Color mapping for different behavior types
        self.behavior_colors = {
            'NORMAL': config.visualization.bbox_color_normal,
            'FALL_DETECTED': config.visualization.bbox_color_fall,
            'ABNORMAL_BEHAVIOR': config.visualization.bbox_color_abnormal,
            'SUSPICIOUS': (0, 165, 255),  # Orange
            'VIOLENT': (0, 0, 255),  # Red
            'UNCERTAIN': (128, 128, 128)  # Gray
        }
        
        logger.info("Initialized Visualizer")
    
    def draw_bbox(self, 
                  frame: np.ndarray, 
                  bbox: List[int], 
                  label: str = "",
                  color: Optional[Tuple[int, int, int]] = None,
                  confidence: Optional[float] = None) -> np.ndarray:
        """
        Draw a single bounding box with label
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Text label to display
            color: BGR color tuple
            confidence: Confidence score to display
            
        Returns:
            Frame with bounding box drawn
        """
        x1, y1, x2, y2 = bbox
        
        # Use default color if not provided
        if color is None:
            color = self.config.visualization.bbox_color_normal
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        if confidence is not None and self.config.visualization.show_confidence:
            label = f"{label} ({confidence:.2f})" if label else f"{confidence:.2f}"
        
        if label:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.config.visualization.font_scale
            thickness = self.config.visualization.font_thickness
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                self.config.visualization.text_color,
                thickness
            )
        
        return frame
    
    def draw_tracked_person(self,
                           frame: np.ndarray,
                           track_id: int,
                           bbox: List[int],
                           behavior_label: str = "NORMAL",
                           confidence: float = 1.0,
                           anomaly_score: Optional[float] = None) -> np.ndarray:
        """
        Draw a tracked person with ID and behavior information
        
        Args:
            frame: Input frame
            track_id: Person's track ID
            bbox: Bounding box coordinates
            behavior_label: Behavior classification label
            confidence: Detection or behavior confidence
            anomaly_score: Anomaly score (0-1)
            
        Returns:
            Frame with person visualization
        """
        # Get color based on behavior
        color = self.behavior_colors.get(behavior_label, self.behavior_colors['NORMAL'])
        
        # Prepare label
        label = f"ID:{track_id} | {behavior_label}"
        if anomaly_score is not None:
            label += f" | A:{anomaly_score:.2f}"
        
        # Draw bounding box
        frame = self.draw_bbox(frame, bbox, label, color, confidence)
        
        return frame
    
    def draw_crowd_stats(self,
                        frame: np.ndarray,
                        people_count: int,
                        density: float,
                        active_tracks: int,
                        fps: float = 0) -> np.ndarray:
        """
        Draw crowd statistics overlay
        
        Args:
            frame: Input frame
            people_count: Number of people detected
            density: Density estimation (0-1)
            active_tracks: Number of active tracked persons
            fps: Current FPS
            
        Returns:
            Frame with statistics overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 130), 
                     self.config.visualization.background_color, -1)
        frame = cv2.addWeighted(overlay, self.config.visualization.opacity, 
                               frame, 1 - self.config.visualization.opacity, 0)
        
        # Draw statistics text
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.visualization.font_scale
        thickness = self.config.visualization.font_thickness
        color = self.config.visualization.text_color
        
        stats = [
            f"People Count: {people_count}",
            f"Active Tracks: {active_tracks}",
            f"Density: {density:.2%}",
            f"FPS: {fps:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, y_offset + i * 25), 
                       font, font_scale, color, thickness)
        
        return frame
    
    def draw_density_heatmap(self, 
                            frame: np.ndarray, 
                            density_grid: np.ndarray,
                            alpha: float = 0.4) -> np.ndarray:
        """
        Draw density heatmap overlay
        
        Args:
            frame: Input frame
            density_grid: 2D array of density values (grid_size x grid_size)
            alpha: Heatmap transparency
            
        Returns:
            Frame with heatmap overlay
        """
        if not self.config.visualization.show_density_map:
            return frame
        
        h, w = frame.shape[:2]
        grid_h, grid_w = density_grid.shape
        
        # Create heatmap
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate cell dimensions
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        # Color map for density (blue -> green -> red)
        for i in range(grid_h):
            for j in range(grid_w):
                density = density_grid[i, j]
                
                # Color based on density
                if density < 0.3:
                    color = (255, 0, 0)  # Blue (low density)
                elif density < 0.6:
                    # Green to yellow
                    ratio = (density - 0.3) / 0.3
                    color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                else:
                    # Yellow to red
                    ratio = (density - 0.6) / 0.4
                    color = (0, int(255 * (1 - ratio)), 255)
                
                # Fill cell
                y1 = i * cell_h
                y2 = (i + 1) * cell_h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w
                
                heatmap[y1:y2, x1:x2] = color
        
        # Blend heatmap with original frame
        frame = cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def draw_alert(self,
                  frame: np.ndarray,
                  alert_message: str,
                  position: Tuple[int, int] = (50, 200),
                  alert_type: str = "warning") -> np.ndarray:
        """
        Draw alert message on frame
        
        Args:
            frame: Input frame
            alert_message: Alert text
            position: (x, y) position for alert
            alert_type: 'warning', 'danger', or 'info'
            
        Returns:
            Frame with alert drawn
        """
        # Color mapping for alert types
        alert_colors = {
            'warning': (0, 165, 255),  # Orange
            'danger': (0, 0, 255),     # Red
            'info': (255, 255, 0)      # Cyan
        }
        
        color = alert_colors.get(alert_type, alert_colors['warning'])
        
        # Draw alert background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            alert_message, font, font_scale, thickness
        )
        
        x, y = position
        cv2.rectangle(frame, (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + 10), color, -1)
        cv2.rectangle(frame, (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + 10), (255, 255, 255), 2)
        
        # Draw alert text
        cv2.putText(frame, alert_message, (x, y), font, font_scale,
                   (255, 255, 255), thickness)
        
        return frame
    
    def draw_movement_trajectory(self,
                                frame: np.ndarray,
                                track_id: int,
                                positions: List[Tuple[int, int]],
                                max_trail: int = 30) -> np.ndarray:
        """
        Draw movement trajectory trail for a person
        
        Args:
            frame: Input frame
            track_id: Person's track ID
            positions: List of (x, y) positions
            max_trail: Maximum number of points to show
            
        Returns:
            Frame with trajectory drawn
        """
        if len(positions) < 2:
            return frame
        
        # Use only recent positions
        recent_positions = positions[-max_trail:]
        
        # Draw trail with fading opacity
        for i in range(1, len(recent_positions)):
            alpha = i / len(recent_positions)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, recent_positions[i-1], recent_positions[i], color, 2)
        
        return frame
    
    def draw_anomaly_indicator(self,
                              frame: np.ndarray,
                              anomaly_score: float,
                              position: Tuple[int, int] = (50, 300)) -> np.ndarray:
        """
        Draw anomaly score indicator bar
        
        Args:
            frame: Input frame
            anomaly_score: Anomaly score (0-1)
            position: (x, y) position for indicator
            
        Returns:
            Frame with anomaly indicator
        """
        x, y = position
        bar_width = 200
        bar_height = 20
        
        # Draw background bar
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     (128, 128, 128), -1)
        
        # Draw filled portion
        filled_width = int(bar_width * anomaly_score)
        if anomaly_score < 0.3:
            color = (0, 255, 0)  # Green
        elif anomaly_score < 0.7:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (x, y), (x + filled_width, y + bar_height), 
                     color, -1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     (255, 255, 255), 2)
        
        # Draw text
        label = f"Anomaly Score: {anomaly_score:.2f}"
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)
        
        return frame

class VideoWriter:
    """
    Handles video writing for output annotated videos
    """
    
    def __init__(self, output_path: str, fps: int, frame_size: Tuple[int, int]):
        """
        Initialize video writer
        
        Args:
            output_path: Path for output video file
            fps: Frames per second
            frame_size: (width, height) of output video
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        logger.info(f"Initialized VideoWriter to {output_path}")
    
    def write_frame(self, frame: np.ndarray):
        """Write a single frame to video"""
        if self.writer is not None:
            self.writer.write(frame)
    
    def release(self):
        """Release video writer resources"""
        if self.writer is not None:
            self.writer.release()
            logger.info(f"Released VideoWriter, saved to {self.output_path}")