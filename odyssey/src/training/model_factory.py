"""
Model creation and management utilities.

This module provides functions for creating models, loading checkpoints,
and managing model parameters across different architectures.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

from src.models.transformer import Transformer
from src.models.autoencoder import Autoencoder, FSQEncoder
from src.model_librarian import ensure_identical_parameters_all_architectures
from src.training.config import (
    BaseModelConfig, FSQModelConfig, TransformerModelConfig,
    BaseTrainingConfig, CheckpointConfig
)


def create_fsq_encoder(config: FSQModelConfig) -> FSQEncoder:
    """Create FSQ encoder based on configuration."""
    fsq_encoder = FSQEncoder(
        d_in=3,  # x, y, z coordinates
        d_hidden=config.latent_dim,
        n_resnet_blocks=2,
        fsq_levels=config.fsq_levels
    )
    return fsq_encoder


def create_model(
    config: Union[FSQModelConfig, TransformerModelConfig],
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Created model
    """
    if isinstance(config, FSQModelConfig):
        # Create FSQ autoencoder
        if config.fsq_encoder is None:
            config.fsq_encoder = create_fsq_encoder(config)
        
        model = Autoencoder(
            config=config,
            fsq_encoder=config.fsq_encoder
        )
    else:
        # Create standard transformer
        model = Transformer(config=config)
    
    # Ensure identical parameters across architectures if specified
    if config.model_type:
        model = ensure_identical_parameters_all_architectures(
            model, 
            config.model_type,
            seed=getattr(config, 'reference_model_seed', 1234)
        )
    
    return model.to(device)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state dict into (optional)
        optimizer: Optimizer to load state dict into (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state if model provided
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if optimizer provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    iteration: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
    **kwargs
) -> None:
    """
    Save checkpoint to file.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        epoch: Current epoch (optional)
        iteration: Current iteration (optional)
        metrics: Dictionary of metrics to save (optional)
        config: Configuration object to save (optional)
        **kwargs: Additional data to save in checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if iteration is not None:
        checkpoint['iteration'] = iteration
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        # Convert dataclass to dict if needed
        if hasattr(config, '__dict__'):
            checkpoint['config'] = config.__dict__
        else:
            checkpoint['config'] = config
    
    # Add any additional kwargs
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)


def load_fsq_encoder(
    checkpoint_dir: Union[str, Path],
    model_type: str,
    stage: str,
    iteration: int,
    masking_strategy: str,
    device: torch.device = torch.device("cpu"),
    freeze: bool = True
) -> FSQEncoder:
    """
    Load pre-trained FSQ encoder from checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_type: Model type (SA, GA, RA, SC)
        stage: Training stage
        iteration: Training iteration
        masking_strategy: Masking strategy used
        device: Device to load encoder to
        freeze: Whether to freeze encoder parameters
        
    Returns:
        Loaded FSQ encoder
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Build checkpoint path
    checkpoint_name = f"{model_type}_stage_{stage}_iter{iteration}_{masking_strategy}.pt"
    checkpoint_path = checkpoint_dir / "fsq" / checkpoint_name
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FSQ encoder checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract encoder state dict
    if 'fsq_encoder_state_dict' in checkpoint:
        encoder_state_dict = checkpoint['fsq_encoder_state_dict']
    else:
        # Try to extract from full model state dict
        encoder_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('fsq_encoder.'):
                encoder_state_dict[k.replace('fsq_encoder.', '')] = v
    
    # Create encoder
    # Try to get config from checkpoint, otherwise use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        fsq_levels = config.get('fsq_levels', [7, 5, 5, 5, 5])
        latent_dim = config.get('latent_dim', 32)
    else:
        fsq_levels = [7, 5, 5, 5, 5]
        latent_dim = 32
    
    encoder = FSQEncoder(
        d_in=3,
        d_hidden=latent_dim,
        n_resnet_blocks=2,
        fsq_levels=fsq_levels
    )
    
    # Load state dict
    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.to(device)
    
    # Freeze if requested
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
    
    return encoder


def get_checkpoint_path(
    checkpoint_dir: Union[str, Path],
    checkpoint_config: CheckpointConfig,
    model_type: str,
    stage: str,
    iteration: int,
    masking_strategy: str
) -> Path:
    """
    Get checkpoint path based on configuration and parameters.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        checkpoint_config: Checkpoint configuration
        model_type: Model type
        stage: Training stage
        iteration: Training iteration
        masking_strategy: Masking strategy
        
    Returns:
        Path to checkpoint file
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Select pattern based on masking strategy
    if masking_strategy == "simple":
        pattern = checkpoint_config.simple_checkpoint_pattern
    elif masking_strategy == "complex":
        pattern = checkpoint_config.complex_checkpoint_pattern
    elif masking_strategy == "discrete_diffusion":
        pattern = checkpoint_config.discrete_diffusion_checkpoint_pattern
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")
    
    # Format pattern
    checkpoint_name = pattern.format(
        model_type=model_type,
        stage=stage,
        iter=iteration
    )
    
    return checkpoint_dir / checkpoint_name


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """
    Get parameter count for model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def print_model_summary(model: nn.Module, model_name: Optional[str] = None) -> None:
    """Print model summary including parameter counts."""
    if model_name:
        print(f"\n{model_name} Model Summary:")
    else:
        print("\nModel Summary:")
    
    param_counts = get_parameter_count(model)
    
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Frozen parameters: {param_counts['frozen']:,}")
    
    # Print layer summary if model has named modules
    print("\nLayer Summary:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params:,} parameters")