"""
Sequence buffer management for per-person temporal analysis
Maintains sliding window of features for each tracked person
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class PersonSequence:
    """Container for a person's feature sequence"""
    track_id: int
    features: deque  # Queue of feature vectors
    timestamps: deque  # Queue of frame timestamps
    last_update_frame: int
    sequence_ready: bool = False
    
    def __init__(self, track_id: int, max_length: int):
        self.track_id = track_id
        self.features = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
        self.last_update_frame = -1
        self.sequence_ready = False
        self.max_length = max_length

class SequenceManager:
    """
    Manages feature sequences for all tracked persons
    Maintains sliding windows and triggers inference when sequences are ready
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 min_sequence_length: int = 10,
                 overlap_threshold: float = 0.5):
        """
        Initialize sequence manager
        
        Args:
            sequence_length: Maximum length of sequence buffer
            min_sequence_length: Minimum frames needed for inference
            overlap_threshold: IOU threshold for re-identification
        """
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.overlap_threshold = overlap_threshold
        
        # Store sequences for each tracked person
        self.sequences: Dict[int, PersonSequence] = {}
        
        # Store recent positions for re-identification
        self.recent_positions: Dict[int, List[Tuple[int, int]]] = {}
        
        logger.info(f"Initialized SequenceManager with seq_len={sequence_length}, "
                   f"min_len={min_sequence_length}")
    
    def update_person(self, 
                     track_id: int, 
                     feature: np.ndarray,
                     frame_number: int) -> bool:
        """
        Update sequence for a person with new feature
        
        Args:
            track_id: Person's track ID
            feature: Feature vector for current frame
            frame_number: Current frame number
            
        Returns:
            True if sequence is ready for inference
        """
        # Create new sequence if not exists
        if track_id not in self.sequences:
            self.sequences[track_id] = PersonSequence(track_id, self.sequence_length)
        
        sequence = self.sequences[track_id]
        
        # Add feature to buffer
        sequence.features.append(feature)
        sequence.timestamps.append(frame_number)
        sequence.last_update_frame = frame_number
        
        # Check if sequence is ready
        if len(sequence.features) >= self.min_sequence_length:
            sequence.sequence_ready = True
            return True
        
        return False
    
    def get_sequence(self, track_id: int) -> Optional[np.ndarray]:
        """
        Get the current feature sequence for a person
        
        Args:
            track_id: Person's track ID
            
        Returns:
            Feature sequence as numpy array (seq_len, feature_dim) or None
        """
        if track_id not in self.sequences:
            return None
        
        sequence = self.sequences[track_id]
        if len(sequence.features) == 0:
            return None
        
        # Convert deque to numpy array
        features_array = np.stack(list(sequence.features))
        
        return features_array
    
    def get_all_ready_sequences(self) -> Dict[int, np.ndarray]:
        """
        Get all sequences that are ready for inference
        
        Returns:
            Dictionary mapping track_id to feature sequence
        """
        ready_sequences = {}
        
        for track_id, sequence in self.sequences.items():
            if sequence.sequence_ready and len(sequence.features) >= self.min_sequence_length:
                ready_sequences[track_id] = self.get_sequence(track_id)
        
        return ready_sequences
    
    def cleanup_old_sequences(self, current_frame: int, max_age_frames: int = 60):
        """
        Remove sequences that haven't been updated recently
        
        Args:
            current_frame: Current frame number
            max_age_frames: Maximum frames without update before removal
        """
        to_remove = []
        
        for track_id, sequence in self.sequences.items():
            if current_frame - sequence.last_update_frame > max_age_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.sequences[track_id]
            logger.debug(f"Removed old sequence for track {track_id}")
    
    def get_sequence_statistics(self) -> Dict:
        """
        Get statistics about current sequences
        
        Returns:
            Dictionary with sequence statistics
        """
        stats = {
            'total_sequences': len(self.sequences),
            'ready_sequences': sum(1 for s in self.sequences.values() if s.sequence_ready),
            'avg_sequence_length': 0,
            'track_ids': list(self.sequences.keys())
        }
        
        if self.sequences:
            lengths = [len(s.features) for s in self.sequences.values()]
            stats['avg_sequence_length'] = np.mean(lengths)
        
        return stats
    
    def reset(self):
        """Reset all sequences (useful for new video)"""
        self.sequences.clear()
        self.recent_positions.clear()
        logger.info("Sequence manager reset")
    
    def get_sequence_embeddings_batch(self, sequences: Dict[int, np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """
        Prepare sequences for batch processing by transformer
        
        Args:
            sequences: Dictionary mapping track_id to sequence array
            
        Returns:
            List of (track_id, sequence_array) pairs
        """
        batch = []
        for track_id, sequence in sequences.items():
            batch.append((track_id, sequence))
        
        return batch

class SequenceProcessor:
    """
    Higher-level processor that manages sequences and prepares them for transformer
    """
    
    def __init__(self, sequence_manager: SequenceManager):
        self.sequence_manager = sequence_manager
        self.pending_inference = {}
        
    def update_and_check(self, 
                        track_id: int, 
                        feature: np.ndarray, 
                        frame_number: int) -> Optional[np.ndarray]:
        """
        Update sequence and return if ready for inference
        
        Args:
            track_id: Person's track ID
            feature: Feature vector
            frame_number: Current frame number
            
        Returns:
            Sequence if ready, None otherwise
        """
        is_ready = self.sequence_manager.update_person(track_id, feature, frame_number)
        
        if is_ready:
            sequence = self.sequence_manager.get_sequence(track_id)
            return sequence
        
        return None
    
    def get_sequences_for_inference(self) -> Dict[int, np.ndarray]:
        """
        Get all sequences ready for inference
        
        Returns:
            Dictionary of track_id -> sequence
        """
        return self.sequence_manager.get_all_ready_sequences()
    
    def mark_sequence_processed(self, track_id: int):
        """
        Mark a sequence as processed to avoid redundant inference
        
        Args:
            track_id: Person's track ID
        """
        if track_id in self.sequence_manager.sequences:
            # Reset ready flag, but keep sequence for future updates
            self.sequence_manager.sequences[track_id].sequence_ready = False
            # Clear buffer to start fresh? Or keep for overlapping windows
            # For sliding window, we keep all features
            pass