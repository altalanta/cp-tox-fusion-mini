"""MC-Dropout utilities for uncertainty quantification in cp-tox-fusion-mini models."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger("cp_tox.uncertainty")


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while keeping other layers in eval mode.
    
    Args:
        model: PyTorch model to modify
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_predict(
    model: nn.Module,
    loader: DataLoader,
    T: int = 30,
    temperature: Optional[float] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, np.ndarray]:
    """Perform MC-Dropout prediction with multiple stochastic forward passes.
    
    Args:
        model: PyTorch model with dropout layers
        loader: DataLoader for evaluation data
        T: Number of stochastic forward passes
        temperature: Temperature scaling factor (applied before sigmoid)
        device: Device to run inference on
        
    Returns:
        Dictionary containing:
        - mean_prob: [N] Mean predicted probabilities
        - std_prob: [N] Standard deviation of probabilities
        - entropy: [N] Predictive entropy (Bernoulli)
        - probs_T: [T, N] All probability predictions across T passes
    """
    model.eval()
    enable_dropout(model)  # Enable dropout for uncertainty
    
    # Collect all predictions across T passes
    all_logits = []
    
    for t in range(T):
        logits_list = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                
                logits = model(inputs)
                
                # Apply temperature scaling if provided
                if temperature is not None:
                    logits = logits / temperature
                
                logits_list.append(logits.cpu())
        
        batch_logits = torch.cat(logits_list, dim=0)
        all_logits.append(batch_logits)
    
    # Stack all predictions: [T, N]
    all_logits = torch.stack(all_logits, dim=0)
    
    # Convert to probabilities
    all_probs = torch.sigmoid(all_logits).numpy()  # [T, N]
    
    # Compute statistics
    mean_prob = np.mean(all_probs, axis=0)  # [N]
    std_prob = np.std(all_probs, axis=0)   # [N]
    
    # Compute Bernoulli entropy: H = -p*log(p) - (1-p)*log(1-p)
    eps = 1e-8  # For numerical stability
    mean_prob_clipped = np.clip(mean_prob, eps, 1 - eps)
    entropy = -(mean_prob_clipped * np.log(mean_prob_clipped) + 
                (1 - mean_prob_clipped) * np.log(1 - mean_prob_clipped))
    
    return {
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "entropy": entropy,
        "probs_T": all_probs
    }


def apply_abstention(
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
    thresh: float = 0.5,
    std_cut: Optional[float] = None,
    ent_cut: Optional[float] = None,
    entropy: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Apply abstention based on uncertainty thresholds.
    
    Args:
        mean_prob: Mean predicted probabilities
        std_prob: Standard deviation of probabilities
        thresh: Classification threshold for predictions
        std_cut: Abstain if std_prob > std_cut
        ent_cut: Abstain if entropy > ent_cut (requires entropy array)
        entropy: Predictive entropy values
        
    Returns:
        Dictionary containing:
        - predictions: Binary predictions (only for non-abstained samples)
        - abstained_mask: Boolean mask indicating abstained samples
        - confident_mask: Boolean mask indicating confident samples
    """
    n_samples = len(mean_prob)
    abstained_mask = np.zeros(n_samples, dtype=bool)
    
    # Apply standard deviation threshold
    if std_cut is not None:
        abstained_mask |= (std_prob > std_cut)
    
    # Apply entropy threshold
    if ent_cut is not None and entropy is not None:
        abstained_mask |= (entropy > ent_cut)
    
    confident_mask = ~abstained_mask
    
    # Make predictions only for confident samples
    predictions = np.full(n_samples, -1, dtype=int)  # -1 indicates abstention
    predictions[confident_mask] = (mean_prob[confident_mask] > thresh).astype(int)
    
    LOGGER.info(f"Abstained on {np.sum(abstained_mask)}/{n_samples} samples "
                f"({100 * np.sum(abstained_mask) / n_samples:.1f}%)")
    
    return {
        "predictions": predictions,
        "abstained_mask": abstained_mask, 
        "confident_mask": confident_mask
    }


def compute_coverage_vs_performance(
    y_true: np.ndarray,
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
    std_thresholds: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Compute coverage vs performance curve for different abstention thresholds.
    
    Args:
        y_true: True binary labels
        mean_prob: Mean predicted probabilities
        std_prob: Standard deviation of probabilities
        std_thresholds: Array of std thresholds to evaluate
        
    Returns:
        Dictionary containing:
        - std_thresholds: Threshold values
        - coverage: Fraction of samples retained
        - accuracy: Accuracy on retained samples
        - auroc: AUROC on retained samples (if possible)
    """
    if std_thresholds is None:
        std_thresholds = np.linspace(0, np.max(std_prob), 20)
    
    coverage = []
    accuracy = []
    auroc = []
    
    for thresh in std_thresholds:
        confident_mask = std_prob <= thresh
        
        if np.sum(confident_mask) == 0:
            coverage.append(0.0)
            accuracy.append(np.nan)
            auroc.append(np.nan)
            continue
        
        coverage.append(np.mean(confident_mask))
        
        # Compute accuracy on confident samples
        y_confident = y_true[confident_mask]
        pred_confident = (mean_prob[confident_mask] > 0.5).astype(int)
        accuracy.append(np.mean(y_confident == pred_confident))
        
        # Compute AUROC if we have both classes
        if len(np.unique(y_confident)) > 1:
            from sklearn.metrics import roc_auc_score
            auroc.append(roc_auc_score(y_confident, mean_prob[confident_mask]))
        else:
            auroc.append(np.nan)
    
    return {
        "std_thresholds": std_thresholds,
        "coverage": np.array(coverage),
        "accuracy": np.array(accuracy),
        "auroc": np.array(auroc)
    }


def uncertainty_histogram_data(
    std_prob: np.ndarray,
    entropy: np.ndarray,
    n_bins: int = 30
) -> Dict[str, tuple]:
    """Prepare histogram data for uncertainty visualizations.
    
    Args:
        std_prob: Standard deviation of probabilities
        entropy: Predictive entropy
        n_bins: Number of histogram bins
        
    Returns:
        Dictionary with histogram data for plotting
    """
    std_hist = np.histogram(std_prob, bins=n_bins)
    entropy_hist = np.histogram(entropy, bins=n_bins)
    
    return {
        "std_hist": std_hist,
        "entropy_hist": entropy_hist
    }


def detect_model_dropout_layers(model: nn.Module) -> int:
    """Count dropout layers in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of dropout layers found
    """
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            dropout_count += 1
    
    if dropout_count == 0:
        LOGGER.warning("No dropout layers found in model. MC-Dropout will not provide uncertainty.")
    else:
        LOGGER.info(f"Found {dropout_count} dropout layers for MC-Dropout.")
    
    return dropout_count