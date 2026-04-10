"""
Inference pipeline for transformer-based behavior analysis.
"""

from typing import Dict

import numpy as np
import torch
from loguru import logger

from models.classifier import BehaviorAnalyzer, BehaviorClassifier
from models.transformer import TemporalFeatureProcessor, TemporalTransformerEncoder
from utils.checkpoint import load_state_dict_flexible, read_checkpoint_metadata


class BehaviorInference:
    def __init__(
        self,
        transformer_model,
        classifier_model,
        device: str = "cpu",
        use_trained_classifier: bool = False,
        demo_behavior_mode: bool = False,
    ):
        self.device = device
        self.transformer_model = transformer_model
        self.classifier_model = classifier_model
        self.use_trained_classifier = use_trained_classifier
        self.demo_behavior_mode = demo_behavior_mode

        if device == "mps":
            self.transformer_model = self.transformer_model.float().to(device)
            self.classifier_model = self.classifier_model.float().to(device)
        else:
            self.transformer_model.to(device)
            self.classifier_model.to(device)

        self.transformer_processor = TemporalFeatureProcessor(
            self.transformer_model, torch.device(device)
        )
        self.behavior_analyzer = BehaviorAnalyzer(
            self.classifier_model, torch.device(device)
        )

        self.transformer_model.eval()
        self.classifier_model.eval()
        self.embedding_cache: Dict[int, np.ndarray] = {}

        logger.info(f"Initialized BehaviorInference pipeline on {device}")

    def analyze_sequences(self, sequences: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        if not sequences:
            return {}

        track_ids = list(sequences.keys())
        sequence_list = [sequences[track_id] for track_id in track_ids]
        embeddings = self.transformer_processor.process_sequence_batch(sequence_list)
        predictions = self.behavior_analyzer.analyze_sequences(embeddings.numpy())

        results = {}
        for idx, track_id in enumerate(track_ids):
            if self.use_trained_classifier:
                fall_prob = float(predictions["fall_prob"][idx])
                anomaly_score = float(predictions["anomaly_score"][idx])
                behavior_probs = predictions["behavior_probs"][idx]
                behavior_label, confidence = self.classifier_model.get_behavior_label(
                    behavior_probs, anomaly_score, fall_prob
                )
            else:
                fall_prob = 0.0
                anomaly_score = 0.0
                behavior_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                behavior_label = "NORMAL"
                confidence = 1.0
            results[track_id] = {
                "fall_prob": fall_prob,
                "anomaly_score": anomaly_score,
                "behavior_probs": behavior_probs,
                "behavior_label": behavior_label,
                "confidence": float(confidence),
                "model_trained": self.use_trained_classifier,
            }
            self.embedding_cache[track_id] = embeddings[idx].numpy()

        return results

    def clear_cache(self):
        self.embedding_cache.clear()


class InferencePipeline:
    def __init__(
        self,
        feature_dim: int = 512,
        sequence_length: int = 30,
        device: str = "cpu",
        transformer_weights: str = None,
        classifier_weights: str = None,
    ):
        self.device = device
        self.transformer_weights_loaded = bool(transformer_weights)
        self.classifier_weights_loaded = bool(classifier_weights)
        self.demo_behavior_mode = False
        self.transformer_model = TemporalTransformerEncoder(
            d_model=feature_dim, max_sequence_length=sequence_length
        )
        self.classifier_model = BehaviorClassifier(input_dim=feature_dim)
        if transformer_weights:
            metadata = read_checkpoint_metadata(transformer_weights)
            self.demo_behavior_mode = self.demo_behavior_mode or bool(
                metadata.get("demo_behavior_mode", False)
            )
            load_result = load_state_dict_flexible(
                self.transformer_model,
                transformer_weights,
                preferred_key="transformer_state_dict",
            )
            logger.info(
                f"Loaded transformer weights from {load_result['path']} "
                f"(missing={len(load_result['missing_keys'])}, unexpected={len(load_result['unexpected_keys'])})"
            )
        if classifier_weights:
            metadata = read_checkpoint_metadata(classifier_weights)
            self.demo_behavior_mode = self.demo_behavior_mode or bool(
                metadata.get("demo_behavior_mode", False)
            )
            load_result = load_state_dict_flexible(
                self.classifier_model,
                classifier_weights,
                preferred_key="classifier_state_dict",
            )
            logger.info(
                f"Loaded classifier weights from {load_result['path']} "
                f"(missing={len(load_result['missing_keys'])}, unexpected={len(load_result['unexpected_keys'])})"
            )
        self.behavior_inference = BehaviorInference(
            self.transformer_model,
            self.classifier_model,
            device=device,
            use_trained_classifier=self.classifier_weights_loaded and not self.demo_behavior_mode,
            demo_behavior_mode=self.demo_behavior_mode,
        )
        logger.info(f"Initialized complete InferencePipeline on {device}")

    def analyze(self, sequences: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        return self.behavior_inference.analyze_sequences(sequences)

    def clear_cache(self):
        self.behavior_inference.clear_cache()

    def has_trained_classifier(self) -> bool:
        return self.classifier_weights_loaded

    def is_demo_behavior_mode(self) -> bool:
        return self.demo_behavior_mode
