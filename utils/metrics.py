"""
Metrics calculation module for crowd statistics and movement analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from loguru import logger

@dataclass
class MovementStats:
    """Container for movement statistics of a person"""
    positions: List[Tuple[int, int]]
    velocities: List[float]
    accelerations: List[float]
    avg_speed: float
    max_speed: float
    movement_pattern: str  # 'stationary', 'walking', 'running', 'erratic'

class CrowdMetrics:
    """
    Calculates crowd statistics including density, movement patterns, and anomalies
    """
    
    def __init__(self, config):
        """
        Initialize crowd metrics calculator
        
        Args:
            config: Configuration object with metrics settings
        """
        self.config = config
        self.frame_area_threshold = config.metrics.frame_area_threshold
        self.density_grid_size = config.metrics.density_grid_size
        self.movement_history_length = config.metrics.movement_history_length
        self.velocity_threshold_normal = config.metrics.velocity_threshold_normal
        self.velocity_threshold_anomaly = config.metrics.velocity_threshold_anomaly
        
        # Store movement history for each person
        self.movement_history: Dict[int, deque] = {}
        
        logger.info("Initialized CrowdMetrics")
    
    def compute_density(self, 
                       frame_shape: Tuple[int, int], 
                       detections: List[Dict]) -> Tuple[float, np.ndarray]:
        """
        Compute crowd density in the frame
        
        Args:
            frame_shape: (height, width) of frame
            detections: List of detection dictionaries with 'bbox'
            
        Returns:
            Tuple of (density_score, density_grid)
            density_score: Overall density (0-1)
            density_grid: 2D array of per-cell density
        """
        h, w = frame_shape
        total_area = h * w
        
        # Calculate total area occupied by people
        total_person_area = 0
        for det in detections:
            bbox = det['bbox']
            person_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_person_area += person_area
        
        # Compute density score
        density_score = min(1.0, total_person_area / (total_area * self.frame_area_threshold))
        
        # Create density grid
        grid_h = self.density_grid_size
        grid_w = self.density_grid_size
        density_grid = np.zeros((grid_h, grid_w))
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        for det in detections:
            bbox = det['bbox']
            # Find center of bounding box
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            
            # Determine grid cell
            grid_x = min(grid_w - 1, cx // cell_w)
            grid_y = min(grid_h - 1, cy // cell_h)
            
            density_grid[grid_y, grid_x] += 1
        
        # Normalize grid
        max_people_per_cell = max(1, np.max(density_grid))
        density_grid = density_grid / max_people_per_cell
        
        return density_score, density_grid
    
    def compute_people_count(self, detections: List[Dict]) -> int:
        """
        Count number of people in frame
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Number of people detected
        """
        return len(detections)
    
    def update_movement(self, track_id: int, center: Tuple[int, int], timestamp: float):
        """
        Update movement history for a person
        
        Args:
            track_id: Person's track ID
            center: (x, y) center coordinates
            timestamp: Frame timestamp or index
        """
        if track_id not in self.movement_history:
            self.movement_history[track_id] = deque(maxlen=self.movement_history_length)
        
        self.movement_history[track_id].append((center, timestamp))
    
    def compute_movement_stats(self, track_id: int) -> Optional[MovementStats]:
        """
        Compute movement statistics for a person
        
        Args:
            track_id: Person's track ID
            
        Returns:
            MovementStats object or None if insufficient data
        """
        if track_id not in self.movement_history:
            return None
        
        history = self.movement_history[track_id]
        if len(history) < 2:
            return None
        
        positions = []
        velocities = []
        accelerations = []
        
        # Calculate velocities between consecutive positions
        for i in range(1, len(history)):
            pos1, t1 = history[i-1]
            pos2, t2 = history[i]
            
            # Calculate Euclidean distance
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Calculate time difference
            dt = t2 - t1
            if dt > 0:
                velocity = distance / dt
                velocities.append(velocity)
                positions.append(pos2)
                
                # Calculate acceleration if we have previous velocity
                if i >= 2:
                    acc = (velocity - velocities[-2]) / dt
                    accelerations.append(acc)
        
        if not velocities:
            return None
        
        # Compute statistics
        avg_speed = np.mean(velocities)
        max_speed = np.max(velocities)
        
        # Determine movement pattern
        if avg_speed < 10:
            pattern = "stationary"
        elif avg_speed < self.velocity_threshold_normal:
            pattern = "walking"
        elif avg_speed < self.velocity_threshold_anomaly:
            pattern = "running"
        else:
            pattern = "erratic"
        
        return MovementStats(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            avg_speed=avg_speed,
            max_speed=max_speed,
            movement_pattern=pattern
        )
    
    def detect_crowd_anomaly(self, 
                            density_score: float, 
                            people_count: int,
                            historical_density: List[float]) -> Tuple[bool, str]:
        """
        Detect anomalies in crowd behavior
        
        Args:
            density_score: Current density score
            people_count: Current number of people
            historical_density: List of historical density scores
            
        Returns:
            Tuple of (is_anomaly, anomaly_description)
        """
        is_anomaly = False
        description = ""
        
        # Check for sudden density change
        if len(historical_density) >= 5:
            avg_density = np.mean(historical_density[-5:])
            density_change = density_score - avg_density
            
            if density_change > 0.3:
                is_anomaly = True
                description = f"Sudden crowd surge (+{density_change:.1%})"
            elif density_change < -0.3:
                is_anomaly = True
                description = f"Sudden crowd dispersal ({density_change:.1%})"
        
        # Check for overcrowding
        if density_score > 0.7:
            is_anomaly = True
            description = f"Overcrowding detected (density: {density_score:.1%})"
        
        return is_anomaly, description
    
    def compute_flow_vectors(self, 
                            current_detections: List[Dict],
                            previous_detections: List[Dict]) -> np.ndarray:
        """
        Compute optical flow-like vectors for crowd movement
        
        Args:
            current_detections: Detections in current frame
            previous_detections: Detections in previous frame
            
        Returns:
            Array of flow vectors (N x 2)
        """
        # Simple matching based on position proximity
        flow_vectors = []
        
        for curr_det in current_detections:
            curr_center = self.get_center(curr_det['bbox'])
            best_match = None
            min_distance = float('inf')
            
            for prev_det in previous_detections:
                prev_center = self.get_center(prev_det['bbox'])
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Max matching distance
                    min_distance = distance
                    best_match = prev_center
            
            if best_match is not None:
                flow_vector = (curr_center[0] - best_match[0], 
                              curr_center[1] - best_match[1])
                flow_vectors.append(flow_vector)
        
        return np.array(flow_vectors)
    
    def compute_crowd_velocity(self, flow_vectors: np.ndarray) -> Tuple[float, float]:
        """
        Compute average crowd velocity magnitude and direction
        
        Args:
            flow_vectors: Array of flow vectors
            
        Returns:
            Tuple of (avg_speed, avg_direction_angle)
        """
        if len(flow_vectors) == 0:
            return 0.0, 0.0
        
        avg_vector = np.mean(flow_vectors, axis=0)
        avg_speed = np.linalg.norm(avg_vector)
        avg_direction = np.arctan2(avg_vector[1], avg_vector[0])
        
        return avg_speed, avg_direction
    
    @staticmethod
    def get_center(bbox: List[int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Remove movement history for tracks that are no longer active
        
        Args:
            active_track_ids: List of currently active track IDs
        """
        inactive_ids = [tid for tid in self.movement_history.keys() 
                       if tid not in active_track_ids]
        
        for tid in inactive_ids:
            del self.movement_history[tid]
        
        if inactive_ids:
            logger.debug(f"Cleaned up movement history for {len(inactive_ids)} tracks")

class MetricsAggregator:
    """
    Aggregates metrics over time and generates reports
    """
    
    def __init__(self):
        """Initialize metrics aggregator"""
        self.metrics_history = []
        self.alerts_history = []
        
        logger.info("Initialized MetricsAggregator")
    
    def add_frame_metrics(self, frame_metrics: Dict):
        """
        Add metrics for a single frame
        
        Args:
            frame_metrics: Dictionary of metrics for the frame
        """
        self.metrics_history.append(frame_metrics)
    
    def add_alert(self, alert: Dict):
        """
        Add alert to history
        
        Args:
            alert: Dictionary with alert information
        """
        self.alerts_history.append(alert)
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics over all frames
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        people_counts = [m['people_count'] for m in self.metrics_history]
        densities = [m['density'] for m in self.metrics_history]
        anomaly_scores = [m.get('avg_anomaly_score', 0) for m in self.metrics_history]
        
        summary = {
            'total_frames': len(self.metrics_history),
            'avg_people_count': np.mean(people_counts),
            'max_people_count': np.max(people_counts),
            'avg_density': np.mean(densities),
            'max_density': np.max(densities),
            'avg_anomaly_score': np.mean(anomaly_scores),
            'total_alerts': len(self.alerts_history),
            'alerts_by_type': self._count_alerts_by_type()
        }
        
        return summary
    
    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by type"""
        alert_counts = {}
        for alert in self.alerts_history:
            alert_type = alert.get('type', 'unknown')
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        return alert_counts
    
    def export_to_csv(self, filepath: str):
        """
        Export metrics history to CSV file
        
        Args:
            filepath: Path to output CSV file
        """
        import pandas as pd
        
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported metrics to {filepath}")
    
    def reset(self):
        """Reset aggregator state"""
        self.metrics_history = []
        self.alerts_history = []
        logger.info("Reset MetricsAggregator")