"""
Main entry point for Crowd Behavior Analysis System
Provides CLI interface and orchestrates the complete pipeline
"""

import argparse
import sys
import webbrowser
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
from digitaltwin.exporter import export_temple_simulation_state
from pipeline.detector import PersonDetector
from pipeline.tracker import PersonTracker
from pipeline.feature_extractor import FeatureExtractor
from pipeline.pose_estimator import PoseEstimator
from pipeline.sequence_manager import SequenceManager, SequenceProcessor
from pipeline.inference import InferencePipeline
from pipeline.demographic_estimator import DemographicEstimator
from pipeline.behavior_profile import DEFAULT_PROFILE
from prediction_layer import PredictionLayerBridge
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
        self.fall_probability_state = {}
        self.visualizer = Visualizer(config)
        self.prediction_layer = PredictionLayerBridge()
        self.last_prediction_alert_frame = None

        # Demographic estimator for male/female behavior modeling
        self.demographic_estimator = DemographicEstimator(config.demographics)
        # Per-track profile cache for quick access
        self._track_profiles = {}
        
        # ── Layer E+F: Control Engine (optional – activated by app.py) ──
        self._control_engine = None  # set externally by app.py when running dashboard

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
        self.pose_estimator = PoseEstimator(
            enabled=config.pipeline.enable_pose_features,
            task_model_path=str(config.paths.pose_landmarker_task)
            if config.paths.pose_landmarker_task is not None
            else None,
        )
        pose_status = self.pose_estimator.get_status()
        logger.info(
            f"Pose Status: enabled={pose_status['enabled']} "
            f"mode={pose_status['mode']} message={pose_status['message']}"
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

    def _build_demo_behavior_result(self, track_id: int, fall_risk_result: Dict) -> Dict:
        """
        Build believable demo behavior labels from heuristic risk signals.
        """
        movement_stats = self.metrics.compute_movement_stats(track_id)
        fall_probability = float(fall_risk_result.get("fall_probability", 0.0))
        motion_probability = float(fall_risk_result.get("motion_probability", 0.0))
        pose_probability = float(fall_risk_result.get("pose_probability", 0.0))
        acceleration_score = float(fall_risk_result.get("acceleration_score", 0.0))
        direction_change_score = float(fall_risk_result.get("direction_change_score", 0.0))

        if fall_probability >= 0.85:
            return {
                "fall_prob": fall_probability,
                "anomaly_score": max(fall_probability, motion_probability),
                "behavior_probs": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "behavior_label": "FALL_DETECTED",
                "confidence": fall_probability,
                "model_trained": False,
            }

        if fall_probability >= 0.65:
            return {
                "fall_prob": fall_probability,
                "anomaly_score": max(fall_probability, pose_probability),
                "behavior_probs": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "behavior_label": "FALL_RISK",
                "confidence": fall_probability,
                "model_trained": False,
            }

        if acceleration_score > 0.55 and direction_change_score > 0.45:
            confidence = max(acceleration_score, direction_change_score)
            return {
                "fall_prob": fall_probability,
                "anomaly_score": confidence,
                "behavior_probs": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "behavior_label": "ABNORMAL_BEHAVIOR",
                "confidence": confidence,
                "model_trained": False,
            }

        if movement_stats is not None and movement_stats.movement_pattern == "erratic":
            confidence = max(motion_probability, 0.55)
            return {
                "fall_prob": fall_probability,
                "anomaly_score": confidence,
                "behavior_probs": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "behavior_label": "SUSPICIOUS",
                "confidence": confidence,
                "model_trained": False,
            }

        return {
            "fall_prob": fall_probability,
            "anomaly_score": max(0.0, min(motion_probability * 0.5, 0.25)),
            "behavior_probs": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "behavior_label": "NORMAL",
            "confidence": 0.95,
            "model_trained": False,
        }

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
        active_track_ids = [obj['track_id'] for obj in tracked_objects]
        self.metrics.cleanup_old_tracks(active_track_ids)
        
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
            pose_features = self.pose_estimator.extract_pose_features(frame, bbox)
            if pose_features is not None:
                self.metrics.update_pose(track_id, pose_features)
                obj['pose_features'] = pose_features
        
        # 7. Compute crowd metrics
        people_count = self.metrics.compute_people_count(tracked_objects)
        density_score, density_grid = self.metrics.compute_density(
            frame.shape[:2], tracked_objects
        )

        # 7.5 Assign demographic profiles (male/female) to each tracked person
        for obj in tracked_objects:
            track_id = obj['track_id']
            pose_feats = obj.get('pose_features')
            profile = self.demographic_estimator.assign_profile(
                track_id, obj['bbox'], pose_feats
            )
            self._track_profiles[track_id] = profile
            obj['demographic_profile'] = profile.label
            obj['gender'] = profile.gender
        self.demographic_estimator.cleanup(active_track_ids)

        # 8. Compute per-person fall risk with density coupling and demographic profiles
        fall_risk_results = {}
        for obj in tracked_objects:
            track_id = obj['track_id']
            profile = self._track_profiles.get(track_id, DEFAULT_PROFILE)
            local_density = self.metrics.compute_local_density(
                track_id, tracked_objects,
                radius=config.stampede.propagation_radius,
            )
            heuristic_result = self.metrics.estimate_fall_risk(
                track_id, obj['bbox'],
                density_score=density_score,
                local_density=local_density,
                profile=profile,
            )
            model_fall_prob = None
            if track_id in behavior_results:
                model_fall_prob = behavior_results[track_id].get('fall_prob')
            fall_probability_result = self.metrics.compute_hybrid_fall_probability(
                heuristic_result,
                model_fall_prob=model_fall_prob,
                prior_probability=self.fall_probability_state.get(track_id),
                smoothing=config.metrics.fall_probability_smoothing,
                density_factor=density_score,
            )
            combined_result = {**heuristic_result, **fall_probability_result}
            fall_risk_results[track_id] = combined_result
            self.fall_probability_state[track_id] = combined_result['fall_probability']
            obj['fall_probability'] = combined_result['fall_probability']
            obj['fall_risk'] = combined_result['heuristic_probability']

        # 8.5 Stampede / chain-reaction propagation
        self.metrics.propagate_stampede_pressure(
            tracked_objects=tracked_objects,
            fall_risk_results=fall_risk_results,
            fall_probability_state=self.fall_probability_state,
            profiles=self._track_profiles,
        )
        # Update tracked objects with post-stampede fall probabilities
        for obj in tracked_objects:
            track_id = obj['track_id']
            obj['fall_probability'] = self.fall_probability_state.get(
                track_id, obj.get('fall_probability', 0.0)
            )

        if self.inference_pipeline.is_demo_behavior_mode():
            behavior_results = {
                track_id: self._build_demo_behavior_result(track_id, result)
                for track_id, result in fall_risk_results.items()
            }
        
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

        prediction_result = self.prediction_layer.evaluate(
            people_count=people_count,
            density_score=density_score,
            density_grid=density_grid,
            tracked_objects=tracked_objects,
            behavior_results=behavior_results,
            fall_risk_results=fall_risk_results,
            metrics=self.metrics,
        )

        # ── Layer E: feed result into Control Engine (non-blocking) ──
        if hasattr(self, '_control_engine') and self._control_engine is not None:
            try:
                import threading as _threading
                _threading.Thread(
                    target=self._control_engine.process_evaluation,
                    args=(prediction_result, 'A'),
                    daemon=True,
                ).start()
            except Exception:
                pass

        prediction_threshold = config.metrics.prediction_alert_threshold
        if prediction_result['fused_risk'] >= prediction_threshold:
            prediction_desc = (
                f"Future crowd risk {prediction_result['fused_risk']:.1f}/100 "
                f"({prediction_result['problem']}, flow={prediction_result['flow_status']}, "
                f"action={prediction_result['recommended_action']})"
            )
            if anomaly_desc:
                anomaly_desc = f"{anomaly_desc} | {prediction_desc}"
            else:
                anomaly_desc = prediction_desc
            crowd_anomaly = True

        # Count stampede / fallen stats
        stampede_fallen_count = sum(
            1 for p in self.fall_probability_state.values()
            if p >= config.stampede.fallen_threshold
        )

        # 10. Prepare frame results
        frame_results = {
            'frame_number': frame_number,
            'people_count': people_count,
            'density': density_score,
            'avg_anomaly_score': avg_anomaly,
            'tracked_objects': tracked_objects,
            'behavior_results': behavior_results,
            'fall_risk_results': fall_risk_results,
            'density_grid': density_grid,
            'prediction_result': prediction_result,
            'crowd_anomaly': crowd_anomaly,
            'anomaly_description': anomaly_desc,
            'stampede_fallen_count': stampede_fallen_count,
            'demographic_summary': self.demographic_estimator.get_demographic_summary(),
        }
        
        # Store metrics
        self.metrics_aggregator.add_frame_metrics({
            'frame': frame_number,
            'people_count': people_count,
            'density': density_score,
            'avg_anomaly_score': avg_anomaly,
            'avg_fall_probability': np.mean(
                [r['fall_probability'] for r in fall_risk_results.values()]
            ) if fall_risk_results else 0.0,
            'stampede_fallen_count': stampede_fallen_count,
            'base_prediction_risk': prediction_result['base_risk'],
            'predicted_future_risk': prediction_result['predicted_future_risk'],
            'trend_risk': prediction_result['trend_risk'],
            'fused_risk': prediction_result['fused_risk'],
            'flow_status': prediction_result['flow_status'],
            'predicted_problem': prediction_result['problem'],
            'num_tracks': len(tracked_objects),
            'num_inferences': len(behavior_results),
            'demographic_males': self.demographic_estimator.get_demographic_summary().get('male', 0),
            'demographic_females': self.demographic_estimator.get_demographic_summary().get('female', 0),
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

        if prediction_result['fused_risk'] >= prediction_threshold:
            prediction_desc = (
                f"Future crowd risk {prediction_result['fused_risk']:.1f}/100 "
                f"({prediction_result['problem']})"
            )
            if self.last_prediction_alert_frame != frame_number:
                alert = {
                    'frame': frame_number,
                    'type': 'prediction_alert',
                    'description': prediction_desc,
                    'fused_risk': prediction_result['fused_risk'],
                    'predicted_future_risk': prediction_result['predicted_future_risk'],
                    'trend_risk': prediction_result['trend_risk'],
                    'problem': prediction_result['problem']
                }
                self.metrics_aggregator.add_alert(alert)
                self.last_prediction_alert_frame = frame_number
                logger.warning(f"Frame {frame_number}: {prediction_desc}")
        
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

        for track_id, fall_risk in fall_risk_results.items():
            if fall_risk['fall_probability'] > config.metrics.fall_confidence_threshold:
                alert = {
                    'frame': frame_number,
                    'track_id': track_id,
                    'type': 'fall_risk',
                    'fall_risk': fall_risk['fall_probability'],
                    'stampede_boost': fall_risk.get('stampede_boost', 0.0),
                }
                self.metrics_aggregator.add_alert(alert)
                logger.warning(
                    f"Frame {frame_number}: Person {track_id} high fall risk "
                    f"({fall_risk['fall_probability']:.2f}, stampede_boost={fall_risk.get('stampede_boost', 0.0):.2f})"
                )

        # Log stampede event if active
        if stampede_fallen_count > 0:
            alert = {
                'frame': frame_number,
                'type': 'stampede_active',
                'fallen_count': stampede_fallen_count,
                'description': f"STAMPEDE: {stampede_fallen_count} agents fallen",
            }
            self.metrics_aggregator.add_alert(alert)
            logger.critical(
                f"Frame {frame_number}: STAMPEDE ACTIVE — {stampede_fallen_count} agent(s) fallen"
            )
        
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
            fall_probability = obj.get('fall_probability')
            
            if track_id in results['behavior_results']:
                behavior = results['behavior_results'][track_id]
                behavior_label = behavior['behavior_label']
                confidence = behavior['confidence']
                anomaly_score = behavior['anomaly_score']

            if fall_probability is not None and fall_probability > 0.7 and behavior_label == "NORMAL":
                behavior_label = "FALL_RISK"
                confidence = fall_probability
            
            frame = self.visualizer.draw_tracked_person(
                frame,
                track_id,
                bbox,
                behavior_label,
                confidence,
                anomaly_score,
                fall_probability=fall_probability,
                gender=obj.get('gender'),
                demographic_label=obj.get('demographic_profile'),
            )

        # Draw stampede overlay if active
        stampede_fallen = results.get('stampede_fallen_count', 0)
        if stampede_fallen > 0:
            frame = self.visualizer.draw_stampede_overlay(frame, stampede_fallen)
        
        # Draw crowd statistics
        frame = self.visualizer.draw_crowd_stats(
                frame,
                results['people_count'],
                results['density'],
                len(results['tracked_objects']),
                fps=30,  # Will be updated with actual FPS
                prediction_result=results.get('prediction_result'),
                demographic_summary=results.get('demographic_summary'),
                stampede_fallen=stampede_fallen,
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

        prediction_result = results.get('prediction_result')
        if prediction_result and prediction_result['severity'] in {'MEDIUM', 'HIGH'}:
            alert_type = 'danger' if prediction_result['severity'] == 'HIGH' else 'warning'
            frame = self.visualizer.draw_alert(
                frame,
                (
                    f"{prediction_result['problem']} | "
                    f"risk {prediction_result['fused_risk']:.1f} | "
                    f"{prediction_result['recommended_action']}"
                ),
                (50, 250),
                alert_type,
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

        if self.metrics_aggregator.metrics_history:
            final_metrics = self.metrics_aggregator.metrics_history[-1]
            summary['prediction_layer'] = {
                'latest_future_risk': final_metrics.get('predicted_future_risk', 0.0),
                'latest_fused_risk': final_metrics.get('fused_risk', 0.0),
                'latest_problem': final_metrics.get('predicted_problem', 'NORMAL'),
                'latest_flow_status': final_metrics.get('flow_status', 'smooth'),
                'model_loaded': self.prediction_layer.predictor.model is not None,
            }

        # Add demographic summary
        summary['demographic_distribution'] = self.demographic_estimator.get_demographic_summary()
        summary['stampede_events'] = sum(
            1 for a in self.metrics_aggregator.alerts_history if a.get('type') == 'stampede_active'
        )
        
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

        digital_twin_export = export_temple_simulation_state(
            output_dir=config.paths.base_dir / "digitaltwin",
            summary=self._make_json_safe(summary),
            metrics_history=self._make_json_safe(self.metrics_aggregator.metrics_history),
            alerts_history=self._make_json_safe(self.metrics_aggregator.alerts_history),
            demographic_summary=self._make_json_safe(
                self.demographic_estimator.get_demographic_summary()
            ),
        )
        logger.info(
            "Temple digital twin state exported to "
            f"{digital_twin_export['js_path']}"
        )

        digital_twin_pages = [
            config.paths.base_dir / "digitaltwin" / "temple_simulation (3).html",
            config.paths.base_dir / "digitaltwin" / "digital-twin-stadium-enhanced.html",
        ]
        for page in digital_twin_pages:
            try:
                if page.exists():
                    webbrowser.open(page.resolve().as_uri())
                    logger.info(f"Opened digital twin: {page}")
                else:
                    logger.warning(f"Digital twin page not found: {page}")
            except Exception as exc:
                logger.warning(f"Failed to open digital twin {page}: {exc}")
    
    def reset(self):
        """Reset system state for new video"""
        self.frame_count = 0
        self.total_frames = 0
        self.fall_probability_state = {}
        self._track_profiles = {}
        self.last_prediction_alert_frame = None
        self.tracker.reset()
        self.sequence_manager.reset()
        self.metrics.reset()
        self.metrics_aggregator.reset()
        self.inference_pipeline.clear_cache()
        self.prediction_layer.reset()
        self.demographic_estimator.reset()
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
            args.max_frames,
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
