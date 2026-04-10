"""
YOLOv8 detector module for person detection
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
from loguru import logger

class PersonDetector:
    """
    YOLOv8-based person detector for crowd monitoring
    Detects persons in video frames with bounding boxes and confidence scores
    """
    
    def __init__(self, 
                 model_name: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 imgsz: int = 960,
                 device: str = "cpu"):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_name: YOLOv8 model name or path
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        
        # Load YOLO model
        try:
            self.model = YOLO(model_name)
            self.model.to(device)
            logger.info(f"Loaded YOLOv8 model from {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in a single frame
        
        Args:
            frame: Input image in BGR format (H, W, 3)
            
        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2] in absolute coordinates
                - confidence: Detection confidence score
                - class_id: Class ID (should be 0 for person)
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided for detection")
            return []
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=[self.person_class_id],  # Only detect persons
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            
            # Extract bounding boxes, confidences, and class IDs
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detection = {
                        'bbox': box.astype(np.int32).tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id)
                    }
                    detections.append(detection)
        
        logger.debug(f"Detected {len(detections)} persons in frame")
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect persons in multiple frames (batch processing)
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists for each frame
        """
        batch_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            batch_detections.append(detections)
        
        return batch_detections
    
    def filter_detections_by_area(self, 
                                  detections: List[Dict], 
                                  frame_area: Tuple[int, int],
                                  min_area_ratio: float = 0.01,
                                  max_area_ratio: float = 0.8) -> List[Dict]:
        """
        Filter detections based on bounding box area relative to frame
        
        Args:
            detections: List of detection dictionaries
            frame_area: Frame dimensions (width, height)
            min_area_ratio: Minimum area ratio (remove very small boxes)
            max_area_ratio: Maximum area ratio (remove huge boxes)
            
        Returns:
            Filtered list of detections
        """
        frame_width, frame_height = frame_area
        frame_total_area = frame_width * frame_height
        
        filtered = []
        
        for det in detections:
            bbox = det['bbox']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ratio = bbox_area / frame_total_area
            
            if min_area_ratio <= area_ratio <= max_area_ratio:
                filtered.append(det)
            else:
                logger.debug(f"Filtered detection with area ratio {area_ratio:.3f}")
        
        return filtered
    
    def get_detection_centers(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Extract center points of all detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of (cx, cy) center coordinates
        """
        centers = []
        
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            centers.append((cx, cy))
        
        return centers
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold dynamically"""
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")
    
    def set_device(self, device: str):
        """Change inference device"""
        self.device = device
        self.model.to(device)
        logger.info(f"Moved detector to {device}")
