"""
Attribution methods for model interpretability.

Provides feature attribution for both baseline and sequence models.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_logistic_attributions(
    model,
    input_features: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """
    Get feature attributions for logistic regression model.

    Args:
        model: Trained LogisticBaseline model
        input_features: Input feature vector [D]
        feature_names: Names of features
        top_k: Number of top features to return

    Returns:
        List of attribution dictionaries
    """
    # Get model coefficients
    coefficients = model.model.coef_[0]

    # Compute contributions (coefficient * feature_value)
    contributions = coefficients * input_features

    # Get top k by absolute contribution
    top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]

    attributions = []
    for idx in top_indices:
        attributions.append(
            {
                "feature": feature_names[idx],
                "value": float(input_features[idx]),
                "coefficient": float(coefficients[idx]),
                "contribution": float(contributions[idx]),
            }
        )

    return attributions


def get_sequence_attention_weights(
    model: nn.Module,
    sequence: torch.Tensor,
    static: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """
    Extract attention weights from Transformer model.

    Args:
        model: Trained TransformerRiskModel
        sequence: Input sequence [1, T, D]
        static: Static features [1, S]
        mask: Sequence mask [1, T]

    Returns:
        Attention weights [T] (averaged across heads/layers)
    """
    model.eval()

    # For Transformer, we can extract attention weights from the encoder
    # This is a simplified version - in practice, you'd modify the model
    # to return attention weights during forward pass

    with torch.no_grad():
        # Forward pass (simplified - actual implementation would need model modification)
        # For now, use gradient-based importance as proxy
        sequence.requires_grad = True

        logits = model(sequence, static, mask)
        logits.backward()

        # Use gradient magnitude as importance
        importance = torch.abs(sequence.grad).sum(dim=-1).squeeze().cpu().numpy()

        # Normalize by mask
        importance = importance * mask.squeeze().cpu().numpy()
        if importance.sum() > 0:
            importance = importance / importance.sum()

    return importance


def get_sequence_gradient_attribution(
    model: nn.Module,
    sequence: torch.Tensor,
    static: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Get gradient-based attribution for sequence model.

    Args:
        model: Trained sequence model
        sequence: Input sequence [1, T, D]
        static: Static features [1, S]
        mask: Sequence mask [1, T]
        device: Device to run on

    Returns:
        Attribution scores per timestep [T]
    """
    model.eval()
    model.to(device)

    sequence = sequence.to(device)
    static = static.to(device)
    mask = mask.to(device)

    sequence.requires_grad = True

    # Forward pass
    logits = model(sequence, static, mask)

    # Backward pass
    logits.backward()

    # Aggregate gradient magnitude across features
    attribution = torch.abs(sequence.grad).sum(dim=-1).squeeze().cpu().numpy()

    # Apply mask
    attribution = attribution * mask.squeeze().cpu().numpy()

    # Normalize
    if attribution.sum() > 0:
        attribution = attribution / attribution.sum()

    return attribution


def get_top_time_steps(
    attribution: np.ndarray, mask: np.ndarray, k: int = 3
) -> List[int]:
    """
    Get indices of top k most important time steps.

    Args:
        attribution: Attribution scores [T]
        mask: Mask indicating valid timesteps [T]
        k: Number of timesteps to return

    Returns:
        List of timestep indices
    """
    # Mask invalid timesteps
    valid_attribution = attribution * mask

    # Get top k
    top_indices = np.argsort(valid_attribution)[::-1][:k]

    # Filter out masked positions
    top_indices = [int(idx) for idx in top_indices if mask[idx] > 0]

    return top_indices[:k]


def get_top_features_in_timestep(
    sequence: np.ndarray,
    timestep: int,
    feature_names: List[str],
    k: int = 5,
) -> List[Dict]:
    """
    Get top features at a specific timestep.

    Args:
        sequence: Input sequence [T, D]
        timestep: Timestep index
        feature_names: Names of features
        k: Number of features to return

    Returns:
        List of feature dictionaries
    """
    features = sequence[timestep]

    # Get non-zero features
    non_zero_mask = features != 0
    non_zero_indices = np.where(non_zero_mask)[0]

    if len(non_zero_indices) == 0:
        return []

    # Get top k by absolute value
    top_local_indices = np.argsort(np.abs(features[non_zero_indices]))[::-1][:k]
    top_indices = non_zero_indices[top_local_indices]

    top_features = []
    for idx in top_indices:
        top_features.append(
            {
                "feature": feature_names[idx],
                "value": float(features[idx]),
            }
        )

    return top_features


def explain_sequence_prediction(
    model: nn.Module,
    sequence: np.ndarray,
    static: np.ndarray,
    mask: np.ndarray,
    feature_names: List[str],
    top_k_timesteps: int = 3,
    top_k_features: int = 5,
    device: str = "cpu",
) -> Dict:
    """
    Generate comprehensive explanation for sequence prediction.

    Args:
        model: Trained sequence model
        sequence: Input sequence [T, D]
        static: Static features [S]
        mask: Sequence mask [T]
        feature_names: Names of sequence features
        top_k_timesteps: Number of important timesteps
        top_k_features: Number of features per timestep
        device: Device to run on

    Returns:
        Dictionary with explanation components
    """
    # Convert to tensors
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).float()
    static_tensor = torch.from_numpy(static).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(
            sequence_tensor.to(device),
            static_tensor.to(device),
            mask_tensor.to(device),
        )
        risk_score = torch.sigmoid(logits).item()

    # Get temporal attribution
    attribution = get_sequence_gradient_attribution(
        model, sequence_tensor, static_tensor, mask_tensor, device
    )

    # Get top timesteps
    top_timesteps = get_top_time_steps(attribution, mask, top_k_timesteps)

    # Get top features for each important timestep
    important_events = []
    for timestep in top_timesteps:
        top_features = get_top_features_in_timestep(
            sequence, timestep, feature_names, top_k_features
        )

        important_events.append(
            {
                "timestep": int(timestep),
                "importance": float(attribution[timestep]),
                "features": top_features,
            }
        )

    return {
        "risk_score": risk_score,
        "temporal_attribution": attribution.tolist(),
        "top_timesteps": top_timesteps,
        "important_events": important_events,
    }
