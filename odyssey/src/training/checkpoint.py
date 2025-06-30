"""
Checkpoint management utilities.

This module provides utilities for managing model checkpoints,
including saving, loading, and organizing checkpoints.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn


class CheckpointManager:
    """Manages checkpoints for training runs."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_type: str,
        stage: str,
        iteration: int,
        masking_strategy: str,
        keep_last_n: int = 5,
        keep_best_n: int = 3
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            model_type: Model type (SA, GA, RA, SC)
            stage: Training stage
            iteration: Training iteration
            masking_strategy: Masking strategy
            keep_last_n: Number of recent checkpoints to keep
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.stage = stage
        self.iteration = iteration
        self.masking_strategy = masking_strategy
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        
        # Create directories
        self.model_dir = self.checkpoint_dir / f"{model_type}_stage_{stage}_iter{iteration}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoint_history = []
        self.best_checkpoints = []
        
        # Load existing history
        self._load_history()
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        **kwargs
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
            **kwargs: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'model_type': self.model_type,
            'stage': self.stage,
            'iteration': self.iteration,
            'masking_strategy': self.masking_strategy,
            **kwargs
        }
        
        # Determine filename
        filename = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = self.model_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update history
        self.checkpoint_history.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics
        })
        
        # Save as best if needed
        if is_best:
            best_path = self.model_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
            
            self.best_checkpoints.append({
                'path': best_path,
                'epoch': epoch,
                'metrics': metrics
            })
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Save history
        self._save_history()
        
        return checkpoint_path
    
    def load_latest(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, Any]:
        """
        Load latest checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            device: Device to load to
            
        Returns:
            Checkpoint data
        """
        if not self.checkpoint_history:
            raise ValueError("No checkpoints found")
        
        latest = self.checkpoint_history[-1]
        return self.load(latest['path'], model, optimizer, device)
    
    def load_best(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
        metric: str = 'loss',
        mode: str = 'min'
    ) -> Dict[str, Any]:
        """
        Load best checkpoint based on metric.
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            device: Device to load to
            metric: Metric to use for selection
            mode: 'min' or 'max'
            
        Returns:
            Checkpoint data
        """
        best_path = self.model_dir / "best_model.pt"
        
        if best_path.exists():
            return self.load(best_path, model, optimizer, device)
        
        # Find best from history
        if not self.checkpoint_history:
            raise ValueError("No checkpoints found")
        
        best_checkpoint = self._find_best_checkpoint(metric, mode)
        return self.load(best_checkpoint['path'], model, optimizer, device)
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, Any]:
        """
        Load specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optimizer to load into
            device: Device to load to
            
        Returns:
            Checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        return {
            'checkpoint_dir': str(self.model_dir),
            'num_checkpoints': len(self.checkpoint_history),
            'latest_epoch': self.checkpoint_history[-1]['epoch'] if self.checkpoint_history else None,
            'best_metrics': self.best_checkpoints[-1]['metrics'] if self.best_checkpoints else None
        }
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on retention policy."""
        if self.keep_last_n > 0 and len(self.checkpoint_history) > self.keep_last_n:
            # Sort by epoch
            sorted_history = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
            
            # Determine which to remove
            to_remove = sorted_history[:-self.keep_last_n]
            
            # Remove files
            for checkpoint in to_remove:
                if checkpoint['path'].exists():
                    checkpoint['path'].unlink()
            
            # Update history
            self.checkpoint_history = sorted_history[-self.keep_last_n:]
    
    def _find_best_checkpoint(self, metric: str, mode: str = 'min') -> Dict[str, Any]:
        """Find best checkpoint based on metric."""
        if not self.checkpoint_history:
            raise ValueError("No checkpoints found")
        
        # Extract metric values
        values = []
        for checkpoint in self.checkpoint_history:
            if metric in checkpoint['metrics']:
                values.append((checkpoint, checkpoint['metrics'][metric]))
        
        if not values:
            raise ValueError(f"Metric {metric} not found in checkpoints")
        
        # Sort by metric
        if mode == 'min':
            values.sort(key=lambda x: x[1])
        else:
            values.sort(key=lambda x: x[1], reverse=True)
        
        return values[0][0]
    
    def _save_history(self):
        """Save checkpoint history to file."""
        history_file = self.model_dir / "checkpoint_history.json"
        
        history_data = {
            'checkpoint_history': [
                {
                    'path': str(cp['path']),
                    'epoch': cp['epoch'],
                    'metrics': cp['metrics']
                }
                for cp in self.checkpoint_history
            ],
            'best_checkpoints': [
                {
                    'path': str(cp['path']),
                    'epoch': cp['epoch'],
                    'metrics': cp['metrics']
                }
                for cp in self.best_checkpoints
            ]
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _load_history(self):
        """Load checkpoint history from file."""
        history_file = self.model_dir / "checkpoint_history.json"
        
        if not history_file.exists():
            return
        
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        # Restore checkpoint history
        self.checkpoint_history = [
            {
                'path': Path(cp['path']),
                'epoch': cp['epoch'],
                'metrics': cp['metrics']
            }
            for cp in history_data.get('checkpoint_history', [])
        ]
        
        # Restore best checkpoints
        self.best_checkpoints = [
            {
                'path': Path(cp['path']),
                'epoch': cp['epoch'],
                'metrics': cp['metrics']
            }
            for cp in history_data.get('best_checkpoints', [])
        ]


def create_checkpoint_name(
    model_type: str,
    stage: str,
    iteration: int,
    epoch: int,
    masking_strategy: str,
    suffix: str = ""
) -> str:
    """
    Create standardized checkpoint filename.
    
    Args:
        model_type: Model type
        stage: Training stage
        iteration: Training iteration
        epoch: Epoch number
        masking_strategy: Masking strategy
        suffix: Optional suffix
        
    Returns:
        Checkpoint filename
    """
    parts = [
        model_type,
        f"stage_{stage}",
        f"iter{iteration}",
        f"epoch{epoch}",
        masking_strategy
    ]
    
    if suffix:
        parts.append(suffix)
    
    return "_".join(parts) + ".pt"