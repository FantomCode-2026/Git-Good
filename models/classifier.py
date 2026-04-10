"""
Classification head for behavior prediction
Predicts fall detection, anomaly scores, and behavior classification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from loguru import logger

class BehaviorClassifier(nn.Module):
    """
    Multi-task classification head for behavior analysis
    Outputs fall probability, anomaly score, and behavior class
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.2):
        """
        Initialize behavior classifier
        
        Args:
            input_dim: Input feature dimension (from transformer)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(BehaviorClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        # Build shared layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        shared_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Task-specific heads
        # 1. Fall detection (binary classification)
        self.fall_head = nn.Sequential(
            nn.Linear(shared_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 2. Anomaly score (regression, 0-1)
        self.anomaly_head = nn.Sequential(
            nn.Linear(shared_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 3. Behavior classification (3 classes: normal, suspicious, violent)
        self.behavior_head = nn.Sequential(
            nn.Linear(shared_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
        logger.info(f"Initialized BehaviorClassifier with input_dim={input_dim}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through classifier
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
                - fall_prob: Fall detection probability
                - anomaly_score: Anomaly score (0-1)
                - behavior_logits: Raw logits for behavior classes
                - behavior_probs: Softmax probabilities for behavior classes
        """
        # Shared features
        shared_features = self.shared_layers(x)
        
        # Task-specific outputs
        fall_prob = self.fall_head(shared_features).squeeze(-1)
        anomaly_score = self.anomaly_head(shared_features).squeeze(-1)
        behavior_logits = self.behavior_head(shared_features)
        behavior_probs = torch.softmax(behavior_logits, dim=-1)
        
        return {
            'fall_prob': fall_prob,
            'anomaly_score': anomaly_score,
            'behavior_logits': behavior_logits,
            'behavior_probs': behavior_probs
        }
    
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict behavior from features (numpy interface)
        
        Args:
            features: Numpy array of shape (batch_size, input_dim)
            
        Returns:
            Dictionary with numpy array predictions
        """
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            outputs = self.forward(x)
            
            # Convert to numpy
            results = {}
            for key, value in outputs.items():
                results[key] = value.numpy()
            
            return results
    
    def get_behavior_label(self, behavior_probs: np.ndarray, 
                          anomaly_score: float,
                          fall_prob: float) -> Tuple[str, float]:
        """
        Get human-readable behavior label with confidence
        
        Args:
            behavior_probs: Probability distribution over behavior classes
            anomaly_score: Anomaly score from anomaly head
            fall_prob: Fall probability from fall head
            
        Returns:
            Tuple of (behavior_label, confidence)
        """
        # Check for fall first (highest priority)
        if fall_prob > 0.8:
            return "FALL_DETECTED", float(fall_prob)
        
        # Check behavior class
        class_idx = np.argmax(behavior_probs)
        confidence = float(behavior_probs[class_idx])
        
        # Adjust based on anomaly score
        if anomaly_score > 0.7:
            if class_idx == 0:  # Normal but high anomaly
                return "ABNORMAL_BEHAVIOR", anomaly_score
            elif class_idx == 1:
                return "SUSPICIOUS_BEHAVIOR", confidence
            elif class_idx == 2:
                return "VIOLENT_BEHAVIOR", confidence
        
        # Normal behavior
        if class_idx == 0 and anomaly_score < 0.3:
            return "NORMAL", confidence
        elif class_idx == 1:
            return "SUSPICIOUS", confidence
        elif class_idx == 2:
            return "VIOLENT", confidence
        else:
            return "UNCERTAIN", max(confidence, anomaly_score)

class BehaviorAnalyzer:
    """
    Wrapper class for behavior analysis in the pipeline
    Handles inference and result aggregation
    """
    
    def __init__(self, model: BehaviorClassifier, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"BehaviorAnalyzer initialized on {device}")
    
    @torch.no_grad()
    def analyze_sequences(self, sequence_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze multiple sequences
        
        Args:
            sequence_embeddings: Array of shape (batch_size, input_dim)
            
        Returns:
            Dictionary with predictions for all sequences
        """
        if len(sequence_embeddings) == 0:
            return {
                'fall_prob': np.array([]),
                'anomaly_score': np.array([]),
                'behavior_probs': np.array([])
            }
        
        # Convert to tensor
        x = torch.from_numpy(sequence_embeddings).float().to(self.device)
        
        # Forward pass
        outputs = self.model(x)
        
        # Convert to numpy
        results = {
            'fall_prob': outputs['fall_prob'].cpu().numpy(),
            'anomaly_score': outputs['anomaly_score'].cpu().numpy(),
            'behavior_probs': outputs['behavior_probs'].cpu().numpy()
        }
        
        return results
    
    def analyze_single_sequence(self, sequence_embedding: np.ndarray) -> Dict[str, float]:
        """
        Analyze a single sequence
        
        Args:
            sequence_embedding: Array of shape (input_dim,)
            
        Returns:
            Dictionary with prediction values
        """
        results = self.analyze_sequences(sequence_embedding.reshape(1, -1))
        
        return {
            'fall_prob': float(results['fall_prob'][0]),
            'anomaly_score': float(results['anomaly_score'][0]),
            'behavior_probs': results['behavior_probs'][0]
        }