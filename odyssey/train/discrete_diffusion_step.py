"""
Train the unified transformer model with discrete diffusion.
Based on Algorithm 1 from the SEDD paper (Score Entropy Discrete Diffusion).

This implements:
1. Q_absorb noise process with absorbing states
2. Geometric noise schedule
3. Score entropy loss for learning probability ratios p_t(y)/p_t(x)
4. AdaLN modulation in transformer blocks for time conditioning

The model learns to denoise discrete sequences by predicting the clean data
distribution at each noise level.
"""
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List, Dict
import random
import math
from types import SimpleNamespace


# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.transformer import TransformerTrunk, StandardTransformerBlock
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import DiffusionDataLoader, MaskedBatch
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import score_entropy_loss
from odyssey.src.configurations import TrunkConfig, TrainingConfig, ScoreEntropyLossConfig

def discrete_diffusion_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: TrunkConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, ScoreEntropyLossConfig)
    """Perform a single step with discrete diffusion."""
    seq_x_t, struct_x_t, = batch.masked_data['seq'], batch.masked_data['struct']
    seq_x_0, struct_x_0 = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    seq_valid, struct_valid = ~batch.beospank['seq'], ~batch.beospank['struct']
    coords_x_t, coords_x_0 = batch.masked_data['coords'], batch.unmasked_data['coords']
    B, L = seq_x_t.shape

    nonspecial_elements_coords = (~batch.masks['coords'] & ~batch.beospank['coords']).bool()
    #assert not (~unmasked_coords_elements.any(dim=1).any()) # Dataloader should have gauranteed this.
    assert nonspecial_elements_coords.any(dim=1).all() # Need at least one real residue in each sequence
    
    # Pass raw timestep indices following DiT convention
    timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
    cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
    inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
    timesteps = timesteps.float().unsqueeze(-1)
    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t)
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass with time conditioning
        model_type = model.cfg.first_block_cfg.initials()
        if model_type in ("GA", "RA"): outputs = model(inputs, coords_x_t, nonspecial_elements_coords, timesteps)
        else: outputs = model(inputs, timesteps=timesteps)
        seq_logits, struct_logits = outputs
        
        # Compute losses using score entropy loss
        loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, cumulative_noise, inst_noise, model_cfg.seq_absorb_token, valid_mask=seq_valid)
        loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, cumulative_noise, inst_noise, model_cfg.struct_absorb_token, valid_mask=struct_valid)
        
        # Total loss
        loss = train_cfg.loss_config.seq_loss_weight * loss_seq + train_cfg.loss_config.struct_loss_weight * loss_struct
        
        if train_mode:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics
        return {'loss': (loss.item(), B), 'loss_seq': (loss_seq.item(), B), 'loss_struct': (loss_struct.item(), B)}