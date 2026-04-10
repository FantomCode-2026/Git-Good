"""
Main entry point for Crowd Behavior Analysis System
Provides CLI interface and orchestrates the complete pipeline
"""

import argparse
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import json
from loguru import logger

# Import all pipeline components
from config import config
from pipeline.detector import PersonDetector
from pipeline.tracker import PersonTracker
from pipeline.feature_extractor import FeatureExtractor
from pipeline.sequence_manager import SequenceManager, SequenceProcessor
from pipeline.inference import InferencePipeline
from utils.visualization import Visualizer, VideoWriter
from utils.metrics import CrowdMetrics, MetricsAggregator

class CrowdBehaviorSystem:
    """
    Main system orchestrating the entire crowd behavior analysis pipeline
    """
    
    def __init__(
        self,
        config_override: Optional[Dict] = None,
        device_override: Optional[str] = None,
    ):
        """
        Initialize the crowd behavior analysis system
        
        Args:
            config_override: Optional dictionary to override configuration
        """
        # Update configuration if provided
        if config_override:
            config.update_from_dict(config_override)
        
        # Set device
        # With this:
        if device_override is not None:
            self.device = self._resolve_device(device_override)
            logger.info(f"Using user-selected device: {self.device}")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA (NVIDIA GPU)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS (Apple Silicon GPU)")
        else:
            self.device = "cpu"
            logger.info("Using CPU")
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Initialize metrics aggregator
        self.metrics_aggregator = MetricsAggregator()
        
        # State variables
        self.frame_count = 0
        self.total_frames = 0
        self.video_writer = None
        self.visualizer = Visualizer(config)
        
        logger.info("CrowdBehaviorSystem initialized successfully")

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Return a working device, falling back to CPU if MPS is unavailable at runtime."""
        if requested_device != "mps":
            return requested_device

        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            logger.warning("MPS was requested but is not available. Falling back to CPU.")
            return "cpu"

        try:
            torch.zeros(1).to("mps")
            return "mps"
        except RuntimeError as exc:
            logger.warning(f"MPS was requested but failed to initialize ({exc}). Falling back to CPU.")
            return "cpu"
    
    def _initialize_components(self):
        """Initialize all pipeline components with MPS support"""
        logger.info("Initializing pipeline components...")
        
        # Detection (YOLO works on MPS via ultralytics)
        self.detector = PersonDetector(
            model_name=config.model.yolo_model_name,
            confidence_threshold=config.model.yolo_confidence_threshold,
            iou_threshold=config.model.yolo_iou_threshold,
            imgsz=config.pipeline.detector_imgsz,
            device=self.device
        )
        
        # Tracking (DeepSORT runs on CPU, that's fine)
        self.tracker = PersonTracker(
            max_age=config.pipeline.max_age,
            n_init=config.pipeline.n_init,
            max_iou_distance=config.pipeline.max_iou_distance,
            max_cosine_distance=config.pipeline.max_cosine_distance,
            nn_budget=config.pipeline.nn_budget,
            tracker_type=config.pipeline.tracker_type,
        )
        
        # Feature extraction - critical for MPS
        # For MPS, we need to ensure the model is in float32 and moved correctly
        self.feature_extractor = FeatureExtractor(
            feature_dim=config.model.resnet_feature_dim,
            target_size=(224, 224),
            batch_size=config.pipeline.feature_batch_size,
            device=self.device
        )
        
        # Sequence management
        self.sequence_manager = SequenceManager(
            sequence_length=config.pipeline.sequence_length,
            min_sequence_length=config.pipeline.min_sequence_length,
            overlap_threshold=config.pipeline.overlap_threshold
        )
        self.sequence_processor = SequenceProcessor(self.sequence_manager)
        
        # Inference pipeline - also MPS sensitive
        self.inference_pipeline = InferencePipeline(
            feature_dim=config.model.resnet_feature_dim,
            sequence_length=config.pipeline.sequence_length,
            device=self.device,
            transformer_weights=config.model.transformer_weights,
            classifier_weights=config.model.classifier_weights,
        )
        
        # Metrics
        self.metrics = CrowdMetrics(config)
        
        logger.info("All pipeline components initialized")
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Process a single frame through the entire pipeline
        
        Args:
            frame: Input video frame
            frame_number: Current frame number
            
        Returns:
            Dictionary with frame results and metrics
        """
        # 1. Detect persons
        detections = self.detector.detect(frame)
        
        # 2. Track persons (assign persistent IDs)
        tracked_objects = self.tracker.update(detections, frame)
        
        # 3. Extract features for each tracked person
        features = self.feature_extractor.extract_features_per_person(frame, tracked_objects)
        
        # 4. Update sequences and check for inference
        sequences_ready = {}
        for obj in tracked_objects:
            track_id = obj['track_id']
            if track_id in features:
                feature = features[track_id]
                sequence = self.sequence_processor.update_and_check(
                    track_id, feature, frame_number
                )
                if sequence is not None:
                    sequences_ready[track_id] = sequence
        
        # 5. Run inference on ready sequences
        behavior_results = {}
        if sequences_ready:
            behavior_results = self.inference_pipeline.analyze(sequences_ready)
        
        # 6. Update movement metrics
        for obj in tracked_objects:
            track_id = obj['track_id']
            bbox = obj['bbox']
            center = self.metrics.get_center(bbox)
            self.metrics.update_movement(track_id, center, frame_number)
        
        # 7. Compute crowd metrics
        people_count = self.metrics.compute_people_count(tracked_objects)
        density_score, density_grid = self.metrics.compute_density(
            frame.shape[:2], tracked_objects
        )
        
        # 8. Compute average anomaly score
        avg_anomaly = 0
        if behavior_results:
            anomaly_scores = [r['anomaly_score'] for r in behavior_results.values()]
            avg_anomaly = np.mean(anomaly_scores)
        
        # 9. Detect crowd anomalies
        historical_densities = [m.get('density', 0) for m in self.metrics_aggregator.metrics_history[-10:]]
        crowd_anomaly, anomaly_desc = self.metrics.detect_crowd_anomaly(
            density_score, people_count, historical_densities
        )
        
        # 10. Prepare frame results
        frame_results = {
            'frame_number': frame_number,
            'people_count': people_count,
            'density': density_score,
            'avg_anomaly_score': avg_anomaly,
            'tracked_objects': tracked_objects,
            'behavior_results': behavior_results,
            'density_grid': density_grid,
            'crowd_anomaly': crowd_anomaly,
            'anomaly_description': anomaly_desc
        }
        
        # Store metrics
        self.metrics_aggregator.add_frame_metrics({
            'frame': frame_number,
            'people_count': people_count,
            'density': density_score,
            'avg_anomaly_score': avg_anomaly,
            'num_tracks': len(tracked_objects),
            'num_inferences': len(behavior_results)
        })
        
        # Log alerts
        if crowd_anomaly:
            alert = {
                'frame': frame_number,
                'type': 'crowd_anomaly',
                'description': anomaly_desc,
                'density': density_score
            }
            self.metrics_aggregator.add_alert(alert)
            logger.warning(f"Frame {frame_number}: {anomaly_desc}")
        
        for track_id, behavior in behavior_results.items():
            if behavior['anomaly_score'] > config.model.anomaly_threshold:
                alert = {
                    'frame': frame_number,
                    'track_id': track_id,
                    'type': 'behavior_anomaly',
                    'behavior': behavior['behavior_label'],
                    'anomaly_score': behavior['anomaly_score'],
                    'fall_prob': behavior['fall_prob']
                }
                self.metrics_aggregator.add_alert(alert)
                logger.warning(f"Frame {frame_number}: Person {track_id} - {behavior['behavior_label']} "
                             f"(anomaly: {behavior['anomaly_score']:.2f})")
        
        return frame_results
    
    def visualize_results(self, 
                         frame: np.ndarray, 
                         results: Dict) -> np.ndarray:
        """
        Visualize results on frame
        
        Args:
            frame: Input frame
            results: Frame results from process_frame
            
        Returns:
            Annotated frame
        """
        # Draw tracked persons
        for obj in results['tracked_objects']:
            track_id = obj['track_id']
            bbox = obj['bbox']
            
            # Get behavior info if available
            behavior_label = "NORMAL"
            confidence = 1.0
            anomaly_score = None
            
            if track_id in results['behavior_results']:
                behavior = results['behavior_results'][track_id]
                behavior_label = behavior['behavior_label']
                confidence = behavior['confidence']
                anomaly_score = behavior['anomaly_score']
            
            frame = self.visualizer.draw_tracked_person(
                frame, track_id, bbox, behavior_label, confidence, anomaly_score
            )
        
        # Draw crowd statistics
        frame = self.visualizer.draw_crowd_stats(
            frame,
            results['people_count'],
            results['density'],
            len(results['tracked_objects']),
            fps=30  # Will be updated with actual FPS
        )
        
        # Draw density heatmap
        if config.visualization.show_density_map:
            frame = self.visualizer.draw_density_heatmap(
                frame, results['density_grid'], alpha=0.3
            )
        
        # Draw anomaly indicator if high anomaly
        if results['avg_anomaly_score'] > 0.5:
            frame = self.visualizer.draw_anomaly_indicator(
                frame, results['avg_anomaly_score'], (50, 300)
            )
        
        # Draw crowd anomaly alert
        if results['crowd_anomaly']:
            frame = self.visualizer.draw_alert(
                frame, results['anomaly_description'], (50, 200), 'warning'
            )
        
        return frame
    
    def process_video(self, 
                     video_path: str,
                     output_video_path: Optional[str] = None,
                     max_frames: Optional[int] = None) -> Dict:
        """
        Process entire video file
        
        Args:
            video_path: Path to input video file
            output_video_path: Path for output annotated video
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary with summary statistics
        """
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            self.total_frames = min(self.total_frames, max_frames)
        
        frame_stride = max(1, config.pipeline.frame_skip + 1)

        logger.info(f"Processing video: {video_path}")
        logger.info(
            f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {self.total_frames}, "
            f"Frame stride: {frame_stride}"
        )
        
        # Initialize video writer if needed
        if output_video_path and config.visualization.save_output_video:
            self.video_writer = VideoWriter(str(output_video_path), fps, (width, height))
        
        # Process frames
        self.frame_count = 0
        source_frame_count = 0
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret or source_frame_count >= self.total_frames:
                break

            current_frame_number = source_frame_count
            source_frame_count += 1

            if current_frame_number % frame_stride != 0:
                continue
            
            # Process frame
            import time
            start_time = time.time()
            
            results = self.process_frame(frame, current_frame_number)
            
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # Visualize
            if config.visualization.show_display or output_video_path:
                annotated_frame = self.visualize_results(frame, results)
                
                # Display
                if config.visualization.show_display:
                    cv2.imshow('Crowd Behavior Analysis', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save to video
                if self.video_writer:
                    self.video_writer.write_frame(annotated_frame)
            
            # Progress update
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                avg_fps = self.frame_count / sum(processing_times) if processing_times else 0
                logger.info(
                    f"Processed {self.frame_count} frames out of {source_frame_count} read "
                    f"(stride {frame_stride}, ~{avg_fps:.1f} FPS)"
                )
        
        # Cleanup
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Get summary statistics
        summary = self.metrics_aggregator.get_summary_stats()
        summary['processed_frames'] = self.frame_count
        summary['source_frames_read'] = source_frame_count
        summary['frame_stride'] = frame_stride
        summary['total_processing_time'] = sum(processing_times)
        summary['avg_fps'] = self.frame_count / sum(processing_times) if processing_times else 0
        
        logger.info(f"Video processing completed. Summary: {summary}")
        
        return summary

    def _make_json_safe(self, value):
        """Recursively convert numpy types to JSON-safe Python values."""
        if isinstance(value, dict):
            return {str(k): self._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._make_json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    
    def save_report(self, report_path: str, summary: Dict):
        """
        Save analysis report to JSON file
        
        Args:
            report_path: Path to save report
            summary: Summary statistics dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'summary': self._make_json_safe(summary),
            'alerts': self._make_json_safe(self.metrics_aggregator.alerts_history[-100:]),  # Last 100 alerts
            'device': self.device
        }
        
        # Ensure directory exists
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        
        # Also export metrics to CSV
        csv_path = str(report_path).replace('.json', '_metrics.csv')
        self.metrics_aggregator.export_to_csv(csv_path)
    
    def reset(self):
        """Reset system state for new video"""
        self.frame_count = 0
        self.total_frames = 0
        self.tracker.reset()
        self.sequence_manager.reset()
        self.metrics_aggregator.reset()
        self.inference_pipeline.clear_cache()
        logger.info("System reset")

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(description='Crowd Behavior Analysis System')
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path for output annotated video')
    parser.add_argument('--report', type=str, default=None,
                       help='Path for output report (JSON)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Sequence length for temporal analysis')
    parser.add_argument('--frame_skip', type=int, default=None,
                       help='Number of frames to skip between processed frames')
    parser.add_argument('--tracker', type=str, default=None,
                       choices=['iou', 'deepsort'],
                       help='Tracker backend to use')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='YOLO inference image size; smaller is faster')
    parser.add_argument('--transformer_weights', type=str, default=None,
                       help='Path to trained transformer checkpoint')
    parser.add_argument('--classifier_weights', type=str, default=None,
                       help='Path to trained classifier checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Computation device')
    
    args = parser.parse_args()
    
    # Update configuration
    config_override = {}
    if args.no_display:
        config_override['visualization'] = {'show_display': False}
    
    if args.confidence:
        config_override['model'] = {'yolo_confidence_threshold': args.confidence}
    
    if args.sequence_length:
        config_override['pipeline'] = {'sequence_length': args.sequence_length}

    if args.frame_skip is not None:
        config_override.setdefault('pipeline', {})
        config_override['pipeline']['frame_skip'] = max(0, args.frame_skip)

    if args.tracker is not None:
        config_override.setdefault('pipeline', {})
        config_override['pipeline']['tracker_type'] = args.tracker

    if args.imgsz is not None:
        config_override.setdefault('pipeline', {})
        config_override['pipeline']['detector_imgsz'] = max(320, args.imgsz)

    if args.transformer_weights is not None:
        config_override.setdefault('model', {})
        config_override['model']['transformer_weights'] = args.transformer_weights

    if args.classifier_weights is not None:
        config_override.setdefault('model', {})
        config_override['model']['classifier_weights'] = args.classifier_weights
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=config.logging.log_level)
    
    # Fix: Convert Path to string for log file
    log_file_path = config.paths.logs_dir / config.logging.log_file_name
    logger.add(
        str(log_file_path),  # Convert Path to string
        rotation=f"{config.logging.max_log_file_size_mb} MB",
        retention=config.logging.backup_count,
        level=config.logging.log_level
    )
    
    try:
        # Initialize system
        logger.info("Starting Crowd Behavior Analysis System")
        device_override = None if args.device == 'auto' else args.device
        system = CrowdBehaviorSystem(config_override, device_override=device_override)
        
        # Generate output paths if not provided
        input_path = Path(args.video)
        if not input_path.exists():
            logger.error(f"Video file not found: {args.video}")
            return 1
        
        if args.output is None:
            output_path = config.paths.videos_dir / f"{input_path.stem}_annotated.mp4"
        else:
            output_path = Path(args.output)
        
        if args.report is None:
            report_path = config.paths.reports_dir / f"{input_path.stem}_report.json"
        else:
            report_path = Path(args.report)
        
        # Process video
        summary = system.process_video(
            str(input_path),
            str(output_path),
            args.max_frames
        )
        
        # Save report
        system.save_report(str(report_path), summary)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("Processing Complete!")
        logger.info(f"Output video: {output_path}")
        logger.info(f"Report: {report_path}")
        logger.info(f"Average FPS: {summary['avg_fps']:.2f}")
        logger.info(f"Total alerts: {summary['total_alerts']}")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.exception(f"System error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
