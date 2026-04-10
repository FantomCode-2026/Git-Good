"""
Transformer module for temporal modeling of person behavior sequences
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple
from loguru import logger

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to incorporate sequence order information
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal behavior analysis
    Processes sequences of feature vectors to capture temporal patterns
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_sequence_length: int = 100):
        """
        Initialize temporal transformer encoder
        
        Args:
            d_model: Input feature dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length for positional encoding
        """
        super(TemporalTransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # Input projection (if needed)
        self.input_projection = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_sequence_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_simulated_weights()
        
        logger.info(f"Initialized TemporalTransformerEncoder with {num_layers} layers, "
                   f"{nhead} heads, {d_model} dimensions")

    def _init_simulated_weights(self):
        """
        Initialize the untrained/demo transformer so it preserves useful visual
        features before applying temporal mixing.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if self.input_projection.weight.shape[0] == self.input_projection.weight.shape[1]:
            with torch.no_grad():
                self.input_projection.weight.copy_(
                    torch.eye(self.d_model, device=self.input_projection.weight.device)
                )
                self.input_projection.weight.add_(torch.randn_like(self.input_projection.weight) * 0.005)
                self.input_projection.bias.zero_()
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process sequence through transformer encoder
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            key_padding_mask: Mask for padding positions
            
        Returns:
            Contextual embeddings of shape (batch_size, seq_len, d_model)
        """
        # Project input if needed
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(
            x, 
            mask=mask,
            src_key_padding_mask=key_padding_mask
        )
        
        return output
    
    def get_sequence_embedding(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get a single embedding representing the entire sequence
        Uses mean pooling over the sequence dimension
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Sequence embedding of shape (batch_size, d_model)
        """
        # Pass through transformer
        encoded = self.forward(x, key_padding_mask=key_padding_mask)

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).type_as(encoded)
            valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
            sequence_embedding = (encoded * valid_mask).sum(dim=1) / valid_counts
        else:
            sequence_embedding = encoded.mean(dim=1)
        
        return sequence_embedding
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights for analysis (for debugging/visualization)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            List of attention weights from each layer
        """
        # This requires modifying the forward pass to return attention weights
        # For now, return empty list
        logger.warning("Attention weights extraction not fully implemented")
        return []

class TemporalFeatureProcessor:
    """
    Wrapper class for processing temporal features in the pipeline
    Handles batching and device management
    """
    
    def __init__(self, model: TemporalTransformerEncoder, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"TemporalFeatureProcessor initialized on {device}")
    
    @torch.no_grad()
    def process_sequence_batch(self, sequences: list) -> torch.Tensor:
        """
        Process multiple sequences at once
        
        Args:
            sequences: List of numpy arrays, each shape (seq_len, d_model)
            
        Returns:
            Sequence embeddings of shape (len(sequences), d_model)
        """
        if len(sequences) == 0:
            return torch.empty((0, self.model.d_model))
        
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in sequences)
        padded_sequences = []
        masks = []
        
        for seq in sequences:
            seq_len = seq.shape[0]
            if seq_len < max_len:
                # Pad with zeros
                padding = np.zeros((max_len - seq_len, seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
                mask = torch.zeros(max_len, dtype=torch.bool)
                mask[seq_len:] = True  # True for padding positions
            else:
                padded_seq = seq
                mask = torch.zeros(max_len, dtype=torch.bool)
            
            padded_sequences.append(padded_seq)
            masks.append(mask)
        
        # Convert to tensors
        batch_tensor = torch.from_numpy(np.stack(padded_sequences)).float().to(self.device)
        padding_mask = torch.stack(masks).to(self.device)
        
        # Get sequence embeddings
        embeddings = self.model.get_sequence_embedding(
            batch_tensor,
            key_padding_mask=padding_mask,
        )
        
        return embeddings.cpu()
    
    def process_single_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Process a single sequence
        
        Args:
            sequence: Numpy array of shape (seq_len, d_model)
            
        Returns:
            Sequence embedding as numpy array of shape (d_model,)
        """
        embeddings = self.process_sequence_batch([sequence])
        return embeddings[0].numpy()
