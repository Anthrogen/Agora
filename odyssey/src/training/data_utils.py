"""
Data loading and processing utilities.

This module provides functions for creating datasets, data loaders,
and handling data preprocessing for training.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.dataset import ProteinDataset
from src.dataloader import (
    SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, 
    NoMaskDataLoader, _get_training_dataloader
)
from src.training.config import (
    BaseTrainingConfig, SimpleMaskingConfig, ComplexMaskingConfig,
    DiffusionConfig
)


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker with unique random seed.
    
    Args:
        worker_id: Worker ID from DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataset(
    data_dir: Union[str, Path],
    csv_file: Union[str, Path]
) -> ProteinDataset:
    """
    Create protein dataset.
    
    Args:
        data_dir: Directory containing protein data files
        csv_file: CSV file with dataset information
        
    Returns:
        ProteinDataset instance
    """
    data_dir = Path(data_dir)
    csv_file = Path(csv_file)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    return ProteinDataset(data_dir=str(data_dir), csv_file=str(csv_file))


def split_dataset(
    dataset: Dataset,
    val_split: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    assert 0 < val_split < 1, "val_split must be between 0 and 1"
    
    # Set seed if provided
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    return train_dataset, val_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    masking_strategy: str,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    masking_config: Optional[Union[SimpleMaskingConfig, ComplexMaskingConfig, DiffusionConfig]] = None,
    **kwargs
) -> DataLoader:
    """
    Create appropriate dataloader based on masking strategy.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        masking_strategy: Type of masking ("simple", "complex", "discrete_diffusion", "none")
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        masking_config: Configuration for masking
        **kwargs: Additional arguments for specific dataloaders
        
    Returns:
        DataLoader instance
    """
    # Common dataloader arguments
    common_args = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'worker_init_fn': worker_init_fn
    }
    
    # Create appropriate dataloader based on masking strategy
    if masking_strategy == "simple":
        if masking_config is None:
            masking_config = SimpleMaskingConfig()
        
        dataloader = SimpleDataLoader(
            **common_args,
            mask_prob_seq=masking_config.mask_prob_seq,
            mask_prob_coords=masking_config.mask_prob_coords
        )
    
    elif masking_strategy == "complex":
        if masking_config is None:
            masking_config = ComplexMaskingConfig()
        
        dataloader = ComplexDataLoader(
            **common_args,
            mask_prob=getattr(masking_config, 'mask_prob', 0.2),
            noise_schedule=getattr(masking_config, 'noise_schedule', 'linear'),
            lambda_val=getattr(masking_config, 'lambda_val', 1.0)
        )
    
    elif masking_strategy == "discrete_diffusion":
        if masking_config is None:
            masking_config = DiffusionConfig()
        
        dataloader = DiffusionDataLoader(
            **common_args,
            diffusion_config=masking_config
        )
    
    elif masking_strategy == "none":
        dataloader = NoMaskDataLoader(**common_args)
    
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")
    
    return dataloader


def create_dataloaders(
    config: BaseTrainingConfig,
    masking_config: Optional[Union[SimpleMaskingConfig, ComplexMaskingConfig, DiffusionConfig]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from configuration.
    
    Args:
        config: Training configuration
        masking_config: Masking configuration
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        seed: Random seed for dataset split
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = create_dataset(config.data_dir, config.csv_file)
    
    # Split dataset
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_split=config.val_split,
        seed=seed or config.reference_model_seed
    )
    
    # Create train dataloader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        masking_strategy=config.masking_strategy,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        masking_config=masking_config
    )
    
    # Create validation dataloader (no shuffle)
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        masking_strategy=config.masking_strategy,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        masking_config=masking_config
    )
    
    return train_loader, val_loader


def create_coordinate_mask(
    batch_size: int,
    seq_len: int,
    mask_prob: float,
    device: torch.device
) -> torch.Tensor:
    """
    Create coordinate mask for structure tokens.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_prob: Probability of masking
        device: Device to create mask on
        
    Returns:
        Boolean mask tensor of shape (batch_size, seq_len)
    """
    mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
    return mask


def get_batch_info(batch: Any) -> Dict[str, Any]:
    """
    Extract information from a batch for logging/debugging.
    
    Args:
        batch: Batch from dataloader
        
    Returns:
        Dictionary with batch information
    """
    info = {}
    
    if hasattr(batch, '__dict__'):
        # MaskedBatch or similar
        info['batch_size'] = batch.original_seq.size(0) if hasattr(batch, 'original_seq') else None
        info['seq_length'] = batch.original_seq.size(1) if hasattr(batch, 'original_seq') else None
        info['has_time'] = hasattr(batch, 'time')
        
        if hasattr(batch, 'seq_mask'):
            info['seq_masked_ratio'] = batch.seq_mask.float().mean().item()
        
        if hasattr(batch, 'struct_mask'):
            info['struct_masked_ratio'] = batch.struct_mask.float().mean().item()
    
    return info


def collate_fn_with_padding(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_seq: int,
    pad_token_struct: int,
    max_len: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function with padding.
    
    Args:
        batch: List of samples from dataset
        pad_token_seq: Padding token for sequences
        pad_token_struct: Padding token for structures
        max_len: Maximum sequence length (if None, uses longest in batch)
        
    Returns:
        Collated and padded batch
    """
    # Find max length in batch
    if max_len is None:
        max_len = max(sample['seq'].size(0) for sample in batch)
    
    # Initialize lists for batched data
    seqs = []
    structs = []
    lengths = []
    
    for sample in batch:
        seq = sample['seq']
        struct = sample['struct']
        length = seq.size(0)
        
        # Pad if necessary
        if length < max_len:
            pad_len = max_len - length
            seq = torch.cat([seq, torch.full((pad_len,), pad_token_seq)])
            struct = torch.cat([struct, torch.full((pad_len,), pad_token_struct)])
        
        seqs.append(seq)
        structs.append(struct)
        lengths.append(length)
    
    # Stack into tensors
    return {
        'seq': torch.stack(seqs),
        'struct': torch.stack(structs),
        'lengths': torch.tensor(lengths)
    }