"""
Metrics tracking and calculation utilities.

This module provides classes and functions for tracking training metrics,
calculating losses, and managing metric history.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass, field


class MetricTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        """Initialize metric tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.counts = defaultdict(int)
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names and values
            count: Number of samples in batch
        """
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.metrics[name].append(value * count)
            self.counts[name] += count
    
    def add(self, name: str, value: float, count: int = 1):
        """
        Add a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            count: Number of samples
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        self.metrics[name].append(value * count)
        self.counts[name] += count
    
    def get_average(self, name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Average value
        """
        if name not in self.metrics:
            return 0.0
        
        total = sum(self.metrics[name])
        count = self.counts[name]
        
        return total / count if count > 0 else 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {name: self.get_average(name) for name in self.metrics}
    
    def get_last(self, name: str) -> Optional[float]:
        """Get last value for a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        return self.metrics[name][-1] / max(self.counts[name], 1)
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        averages = self.get_all_averages()
        parts = [f"{name}: {value:.4f}" for name, value in sorted(averages.items())]
        return " | ".join(parts)


@dataclass
class EpochMetrics:
    """Container for metrics from a single epoch."""
    epoch: int
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    learning_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            'epoch': self.epoch,
            'train': self.train_metrics,
            'val': self.val_metrics,
            'lr': self.learning_rate
        }


class MetricHistory:
    """Track metric history across epochs."""
    
    def __init__(self):
        """Initialize metric history."""
        self.history: List[EpochMetrics] = []
    
    def add_epoch(self, epoch_metrics: EpochMetrics):
        """Add metrics for an epoch."""
        self.history.append(epoch_metrics)
    
    def get_metric_series(self, metric_name: str, split: str = 'train') -> List[float]:
        """
        Get series of values for a specific metric.
        
        Args:
            metric_name: Name of metric
            split: 'train' or 'val'
            
        Returns:
            List of metric values
        """
        values = []
        for epoch in self.history:
            metrics = epoch.train_metrics if split == 'train' else epoch.val_metrics
            if metric_name in metrics:
                values.append(metrics[metric_name])
        return values
    
    def get_best_epoch(self, metric_name: str, split: str = 'val', mode: str = 'min') -> Optional[int]:
        """
        Get epoch with best metric value.
        
        Args:
            metric_name: Name of metric
            split: 'train' or 'val'
            mode: 'min' or 'max'
            
        Returns:
            Best epoch number (1-indexed)
        """
        values = self.get_metric_series(metric_name, split)
        if not values:
            return None
        
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return self.history[best_idx].epoch
    
    def save_to_csv(self, filepath: str):
        """Save metrics to CSV file."""
        import pandas as pd
        
        data = []
        for epoch in self.history:
            row = {'epoch': epoch.epoch}
            
            # Add train metrics
            for name, value in epoch.train_metrics.items():
                row[f'train_{name}'] = value
            
            # Add val metrics
            for name, value in epoch.val_metrics.items():
                row[f'val_{name}'] = value
            
            if epoch.learning_rate is not None:
                row['lr'] = epoch.learning_rate
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, current_value: float, epoch: int) -> bool:
        """
        Check if should stop.
        
        Args:
            current_value: Current metric value
            epoch: Current epoch
            
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Improved to {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping at epoch {epoch}")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100
) -> float:
    """
    Calculate accuracy for predictions.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        mask: Optional mask for valid positions
        ignore_index: Index to ignore in targets
        
    Returns:
        Accuracy as float
    """
    # Get predicted classes
    if predictions.dim() > targets.dim():
        pred_classes = predictions.argmax(dim=-1)
    else:
        pred_classes = predictions
    
    # Create mask for valid positions
    if mask is None:
        mask = targets != ignore_index
    else:
        mask = mask & (targets != ignore_index)
    
    # Calculate accuracy
    correct = (pred_classes == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity
    """
    return np.exp(loss)


def calculate_rmsd(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate RMSD between predicted and true coordinates.
    
    Args:
        pred_coords: Predicted coordinates (B, N, 3)
        true_coords: True coordinates (B, N, 3)
        mask: Optional mask for valid positions (B, N)
        
    Returns:
        RMSD value
    """
    # Calculate squared differences
    sq_diff = (pred_coords - true_coords).pow(2).sum(dim=-1)  # (B, N)
    
    # Apply mask if provided
    if mask is not None:
        sq_diff = sq_diff * mask
        n_atoms = mask.sum()
    else:
        n_atoms = sq_diff.numel()
    
    # Calculate RMSD
    msd = sq_diff.sum() / n_atoms
    rmsd = torch.sqrt(msd)
    
    return rmsd.item()


def create_metric_dict(
    loss: torch.Tensor,
    seq_loss: Optional[torch.Tensor] = None,
    struct_loss: Optional[torch.Tensor] = None,
    seq_acc: Optional[float] = None,
    struct_acc: Optional[float] = None,
    rmsd: Optional[float] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Create standardized metric dictionary.
    
    Args:
        loss: Total loss
        seq_loss: Sequence reconstruction loss
        struct_loss: Structure reconstruction loss
        seq_acc: Sequence accuracy
        struct_acc: Structure accuracy
        rmsd: RMSD value
        **kwargs: Additional metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {'loss': loss.item() if isinstance(loss, torch.Tensor) else loss}
    
    if seq_loss is not None:
        metrics['seq_loss'] = seq_loss.item() if isinstance(seq_loss, torch.Tensor) else seq_loss
    
    if struct_loss is not None:
        metrics['struct_loss'] = struct_loss.item() if isinstance(struct_loss, torch.Tensor) else struct_loss
    
    if seq_acc is not None:
        metrics['seq_acc'] = seq_acc
    
    if struct_acc is not None:
        metrics['struct_acc'] = struct_acc
    
    if rmsd is not None:
        metrics['rmsd'] = rmsd
    
    # Add perplexity if we have loss
    metrics['perplexity'] = calculate_perplexity(metrics['loss'])
    
    # Add any additional metrics
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        metrics[key] = value
    
    return metrics