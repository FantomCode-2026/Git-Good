"""
Checkpoint loading helpers.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def load_state_dict_flexible(
    model: torch.nn.Module, checkpoint_path: str, preferred_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint into a model while tolerating common wrapper formats.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        candidate_keys = []
        if preferred_key:
            candidate_keys.append(preferred_key)
        candidate_keys.extend(
            ["state_dict", "model_state_dict", "model", "transformer_state_dict", "classifier_state_dict"]
        )
        for key in candidate_keys:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    return {
        "path": str(Path(checkpoint_path)),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
