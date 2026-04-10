"""
Feature extraction module for cropping and processing person images.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from models.cnn import ResNetFeatureExtractor


class FeatureExtractor:
    """
    Handles cropping person images and extracting appearance features.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.target_size = target_size
        self.batch_size = batch_size
        self.device = device
        self.model = ResNetFeatureExtractor(feature_dim=feature_dim, pretrained=True)

        if device == "mps":
            self.model = self.model.float()

        self.model.to(device)
        self.model.eval()
        self.use_cpu_for_features = False
        logger.info(f"Initialized FeatureExtractor on {device}")

    def crop_person(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x1 >= x2 or y1 >= y2:
            return None

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        return cv2.resize(cropped, self.target_size)

    @torch.no_grad()
    def extract_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.model.feature_dim), dtype=np.float32)

        all_features = []
        active_device = "cpu" if self.use_cpu_for_features else self.device

        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            try:
                features = self.model.extract_features_batch(batch, active_device)
            except RuntimeError as exc:
                if "mps" in str(exc).lower() and not self.use_cpu_for_features:
                    logger.warning(
                        f"MPS feature extraction failed, retrying on CPU: {exc}"
                    )
                    self.use_cpu_for_features = True
                    active_device = "cpu"
                    features = self.model.extract_features_batch(batch, active_device)
                else:
                    raise
            all_features.append(features)

        return np.vstack(all_features) if all_features else np.empty((0, 0))

    def extract_features_per_person(
        self, frame: np.ndarray, tracked_objects: List[Dict]
    ) -> Dict[int, np.ndarray]:
        if not tracked_objects:
            return {}

        cropped_images = []
        track_ids = []
        for obj in tracked_objects:
            cropped = self.crop_person(frame, obj["bbox"])
            if cropped is not None:
                cropped_images.append(cropped)
                track_ids.append(obj["track_id"])

        if not cropped_images:
            return {}

        features = self.extract_features_batch(cropped_images)
        feature_dict = {
            track_id: feature for track_id, feature in zip(track_ids, features)
        }
        logger.debug(f"Extracted features for {len(feature_dict)} persons")
        return feature_dict

    def extract_feature_single(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        cropped = self.crop_person(frame, bbox)
        if cropped is None:
            return None

        features = self.extract_features_batch([cropped])
        return features[0] if len(features) > 0 else None

    def set_device(self, device: str):
        self.device = device
        self.model.to(device)
        logger.info(f"Moved feature extractor to {device}")
