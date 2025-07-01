"""
Base training classes and utilities.

This module provides base classes for training loops that can be
extended for specific training strategies.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from odyssey.src.training.metrics import MetricTracker, MetricHistory, EpochMetrics
from odyssey.src.training.model_factory import save_checkpoint, load_checkpoint


class BaseTrainer(ABC):
    """Abstract base class for trainers."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory for checkpoints (optional)
            verbose: Whether to print progress
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.verbose = verbose
        
        # Initialize metrics
        self.metric_history = MetricHistory()
        
        # Move model to device
        self.model.to(self.device)
    
    @abstractmethod
    def compute_loss(self, batch: Any, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Input batch
            outputs: Model outputs
            
        Returns:
            Dictionary with loss values
        """
        pass
    
    @abstractmethod
    def forward_pass(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass.
        
        Args:
            batch: Input batch
            
        Returns:
            Model outputs
        """
        pass
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        metric_tracker = MetricTracker()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.verbose)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward_pass(batch)
            
            # Compute loss
            losses = self.compute_loss(batch, outputs)
            total_loss = losses.get('total', losses.get('loss'))
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping if configured
            if hasattr(self.config, 'gradient_clip') and self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            batch_size = self._get_batch_size(batch)
            metric_tracker.update(losses, count=batch_size)
            
            # Update progress bar
            if self.verbose:
                pbar.set_postfix({'loss': total_loss.item()})
        
        return metric_tracker.get_all_averages()
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of average metrics
        """
        self.model.eval()
        metric_tracker = MetricTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", disable=not self.verbose)
            
            for batch in pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.forward_pass(batch)
                
                # Compute loss
                losses = self.compute_loss(batch, outputs)
                
                # Update metrics
                batch_size = self._get_batch_size(batch)
                metric_tracker.update(losses, count=batch_size)
        
        return metric_tracker.get_all_averages()
    
    def train(
        self,
        num_epochs: int,
        start_epoch: int = 1,
        save_every: int = 10,
        validate_every: int = 1
    ) -> MetricHistory:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch number
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
            
        Returns:
            Metric history
        """
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = {}
            if validate_every > 0 and epoch % validate_every == 0:
                val_metrics = self.validate()
            
            # Get learning rate
            lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=lr
            )
            self.metric_history.add_epoch(epoch_metrics)
            
            # Print metrics
            if self.verbose:
                self._print_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if save_every > 0 and epoch % save_every == 0 and self.checkpoint_dir:
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        return self.metric_history
    
    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if hasattr(batch, 'to'):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if hasattr(v, 'to') else v 
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(item) for item in batch)
        else:
            return batch
    
    def _get_batch_size(self, batch: Any) -> int:
        """Get batch size from batch."""
        if hasattr(batch, 'batch_size'):
            return batch.batch_size
        elif hasattr(batch, 'original_seq'):
            return batch.original_seq.size(0)
        elif isinstance(batch, dict) and 'seq' in batch:
            return batch['seq'].size(0)
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            return self._get_batch_size(batch[0])
        else:
            return 1
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary."""
        msg = f"Epoch {epoch}"
        
        # Add train metrics
        if train_metrics:
            train_parts = [f"{k}: {v:.4f}" for k, v in sorted(train_metrics.items())]
            msg += f" | Train: {' '.join(train_parts)}"
        
        # Add val metrics
        if val_metrics:
            val_parts = [f"{k}: {v:.4f}" for k, v in sorted(val_metrics.items())]
            msg += f" | Val: {' '.join(val_parts)}"
        
        print(msg)
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics={'train': train_metrics, 'val': val_metrics},
            config=self.config
        )
        
        if self.verbose:
            print(f"Saved checkpoint to {checkpoint_path}")


class SimpleTrainer(BaseTrainer):
    """Trainer for simple masked language modeling."""
    
    def forward_pass(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Perform forward pass."""
        return self.model(batch)
    
    def compute_loss(self, batch: Any, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for batch."""
        losses = {}
        
        # Extract losses from outputs
        if 'loss' in outputs:
            losses['loss'] = outputs['loss']
        
        if 'seq_loss' in outputs:
            losses['seq_loss'] = outputs['seq_loss']
        
        if 'struct_loss' in outputs:
            losses['struct_loss'] = outputs['struct_loss']
        
        # Calculate total loss if not provided
        if 'loss' not in losses and ('seq_loss' in losses or 'struct_loss' in losses):
            total = 0
            if 'seq_loss' in losses:
                total = total + losses['seq_loss'] * getattr(self.config, 'seq_loss_weight', 1.0)
            if 'struct_loss' in losses:
                total = total + losses['struct_loss'] * getattr(self.config, 'struct_loss_weight', 1.0)
            losses['loss'] = total
        
        return losses