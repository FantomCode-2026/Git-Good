"""
Configuration module for Crowd Behavior Analysis System
Contains all hyperparameters, paths, and settings for the pipeline
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import torch

def get_best_device():
    """Get the best available device for PyTorch on current hardware"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.zeros(1).to("mps")
            return "mps"
        except RuntimeError:
            return "cpu"
    else:
        return "cpu"
    
@dataclass
class ModelConfig:
    """Model architecture and pretrained weights configuration"""
    yolo_model_name: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.4
    yolo_iou_threshold: float = 0.45
    detection_classes: list = field(default_factory=lambda: [0])  # COCO class 0 = person
    
    # Feature Extraction
    resnet_feature_dim: int = 512
    resnet_pretrained: bool = True
    
    # Transformer
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 2048
    transformer_dropout: float = 0.1
    transformer_weights: Optional[str] = None
    classifier_weights: Optional[str] = None
    
    # Classification
    num_classes: int = 3  # [normal, fall, other_anomaly]
    anomaly_threshold: float = 0.7

@dataclass
class PipelineConfig:
    """Video processing and tracking configuration"""
    # Video processing
    target_fps: int = 30
    frame_skip: int = 2  # Process every 3rd frame by default for speed
    max_frame_queue_size: int = 128
    detector_imgsz: int = 960
    enable_pose_features: bool = True
    
    # Tracking (DeepSORT)
    max_cosine_distance: float = 0.35
    nn_budget: Optional[int] = 200
    max_iou_distance: float = 0.85
    max_age: int = 60
    n_init: int = 1
    tracker_type: str = "iou"
    
    # Sequence Management
    sequence_length: int = 30  # Number of frames to buffer per person
    min_sequence_length: int = 10  # Minimum frames before inference
    overlap_threshold: float = 0.5  # IOU for person re-identification
    
    # Feature extraction batch size
    feature_batch_size: int = 64

@dataclass
class PathConfig:
    """File and directory paths"""
    base_dir: Path = field(default_factory=lambda: Path(os.path.dirname(os.path.abspath(__file__))))
    
    def __post_init__(self):
        # Data directories
        self.data_dir = self.base_dir / "data"
        self.sample_videos_dir = self.data_dir / "sample_videos"
        
        # Output directories
        self.output_dir = self.base_dir / "outputs"
        self.logs_dir = self.output_dir / "logs"
        self.videos_dir = self.output_dir / "videos"
        self.reports_dir = self.output_dir / "reports"
        
        # Model weights (optional custom paths)
        self.yolo_weights: Optional[Path] = None
        self.resnet_weights: Optional[Path] = None
        self.transformer_weights: Optional[Path] = None
        self.pose_landmarker_task: Optional[Path] = self.base_dir / "pose_landmarker.task"
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist"""
        for dir_path in [self.data_dir, self.sample_videos_dir, self.output_dir, 
                        self.logs_dir, self.videos_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class VisualizationConfig:
    """Visualization and display settings"""
    # Display settings
    show_display: bool = True
    save_output_video: bool = True
    save_annotated_frames: bool = False
    
    # Colors (BGR format for OpenCV)
    bbox_color_normal: Tuple[int, int, int] = (0, 255, 0)  # Green
    bbox_color_abnormal: Tuple[int, int, int] = (0, 0, 255)  # Red
    bbox_color_fall: Tuple[int, int, int] = (0, 0, 255)  # Red
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    
    # Font settings
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # UI elements
    show_confidence: bool = True
    show_density_map: bool = True
    opacity: float = 0.6

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_name: str = "crowd_ai.log"
    max_log_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class MetricsConfig:
    """Metrics calculation configuration"""
    # Density estimation
    frame_area_threshold: float = 0.3  # Minimum area percentage for density calculation
    density_grid_size: int = 10  # Grid cells for density heatmap
    
    # Movement analysis
    movement_history_length: int = 10
    velocity_threshold_normal: float = 50.0  # pixels per second
    velocity_threshold_anomaly: float = 200.0
    
    # Alert thresholds
    fall_confidence_threshold: float = 0.8
    anomaly_score_threshold: float = 0.7
    crowd_density_threshold: float = 0.5  # 50% of frame area
    hybrid_fall_heuristic_weight: float = 0.75
    hybrid_fall_model_weight: float = 0.25
    fall_probability_smoothing: float = 0.65
    hybrid_fall_pose_weight: float = 0.35
    hybrid_fall_motion_weight: float = 0.65
    prediction_alert_threshold: float = 75.0
    prediction_watch_threshold: float = 55.0
    prediction_trend_weight: float = 0.35

class Config:
    """Main configuration class aggregating all configs"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.pipeline = PipelineConfig()
        self.paths = PathConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        self.metrics = MetricsConfig()
        
        # Create directories
        self.paths.ensure_directories()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary (useful for CLI overrides)"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                sub_config = getattr(self, key)
                if isinstance(sub_config, (ModelConfig, PipelineConfig, PathConfig, 
                                          VisualizationConfig, LoggingConfig, MetricsConfig)):
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_value(val):
            if isinstance(val, Path):
                return str(val)
            return val
        
        return {
            'model': {k: convert_value(v) for k, v in self.model.__dict__.items()},
            'pipeline': {k: convert_value(v) for k, v in self.pipeline.__dict__.items()},
            'paths': {k: convert_value(v) for k, v in self.paths.__dict__.items() if not k.startswith('_')},
            'visualization': {k: convert_value(v) for k, v in self.visualization.__dict__.items()},
            'logging': {k: convert_value(v) for k, v in self.logging.__dict__.items()},
            'metrics': {k: convert_value(v) for k, v in self.metrics.__dict__.items()}
        }

# Global configuration instance
config = Config()

DEFAULT_DEVICE = get_best_device()
