"""
CNN module for feature extraction using ResNet18.
"""

from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet18-based feature extractor for person appearance features.
    """

    def __init__(self, feature_dim: int = 512, pretrained: bool = True):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            # Older torchvision fallback.
            self.backbone = models.resnet18(pretrained=pretrained)

        self.feature_dim = feature_dim
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.eval()

        logger.info(
            f"Initialized ResNetFeatureExtractor with output dimension {feature_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.projection(features)
        return nn.functional.normalize(features, p=2, dim=1)

    @torch.no_grad()
    def extract_features_batch(self, images: List[np.ndarray], device: str) -> np.ndarray:
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        target_device = torch.device(device)
        if next(self.parameters()).device != target_device:
            self.to(target_device)

        batch_tensors = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_normalized = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_normalized = (img_normalized - mean) / std
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
            batch_tensors.append(img_tensor)

        batch = torch.stack(batch_tensors).to(target_device)
        if target_device.type == "mps":
            batch = batch.contiguous()

        features = self.forward(batch)
        return features.cpu().numpy()

    def to_device(self, device: torch.device):
        self.to(device)
        logger.info(f"Moved feature extractor to {device}")
        return self
