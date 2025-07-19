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
from odyssey.src.losses import score_entropy_loss_absorb, score_entropy_loss_uniform
from odyssey.src.configurations import TrunkConfig, TrainingConfig, ScoreEntropyLossConfig

def discrete_diffusion_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, scheduler, batch: MaskedBatch, model_cfg: TrunkConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, ScoreEntropyLossConfig)
    """Perform a single step with discrete diffusion."""
    seq_x_t, struct_x_t, = batch.masked_data['seq'], batch.masked_data['struct']
    seq_x_0, struct_x_0 = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    nonbeospank_seq, nonbeospank_struct = ~batch.beospank['seq'], ~batch.beospank['struct']
    coords_x_t, coords_x_0 = batch.masked_data['coords'], batch.unmasked_data['coords']
    ss8_x_0, sasa_x_0 = batch.masked_data['ss8'], batch.masked_data['sasa']
    global_annotation_x_0, per_residue_annotation_x_0 = batch.unmasked_data['global_annotation'], batch.masked_data['per_residue_annotation']
    plddt_x_0 = batch.masked_data['plddt']
    B, L = seq_x_t.shape

    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank = ~batch.beospank['coords'] & ~batch.beospank['seq']
    nonbeospank_ss8 = ~batch.beospank['ss8']
    nonbeospank_sasa = ~batch.beospank['sasa']
    nonbeospank_global_annotation = ~batch.beospank['global_annotation']
    nonbeospank_per_residue_annotation = ~batch.beospank['per_residue_annotation']
    nonbeospank_plddt = ~batch.beospank['plddt']
    assert content_elements.any(dim=1).all() # Need at least one real residue in each sequence
    
    # Pass raw timestep indices following DiT convention
    timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
    cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
    inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
    timesteps = timesteps.float().unsqueeze(-1)
    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t, ss8_x_0, sasa_x_0, global_annotation_x_0, per_residue_annotation_x_0, plddt_x_0)
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass with time conditioning
        model_type = model.cfg.first_block_cfg.initials()
        if model_type in ("GA", "RA"): outputs = model(inputs, coords_x_t, content_elements, nonbeospank, nonbeospank_ss8, nonbeospank_sasa, nonbeospank_global_annotation, nonbeospank_per_residue_annotation, nonbeospank_plddt, timesteps)
        else: outputs = model(inputs, nonbeospank=nonbeospank, nonbeospank_ss8=nonbeospank_ss8, nonbeospank_sasa=nonbeospank_sasa, nonbeospank_global_annotation=nonbeospank_global_annotation, nonbeospank_per_residue_annotation=nonbeospank_per_residue_annotation, nonbeospank_plddt=nonbeospank_plddt, timesteps=timesteps)
        seq_logits, struct_logits = outputs
        
        score_entropy_loss_fn = score_entropy_loss_absorb if train_cfg.mask_config.corruption_mode == "absorb" else score_entropy_loss_uniform
        
        # Find which batch elements have valid elements and count effective batch sizes
        valid_seq_mask = nonbeospank_seq.any(dim=1)  # [B] - which sequences have at least one valid position
        valid_struct_mask = nonbeospank_struct.any(dim=1)  # [B] - which sequences have at least one valid position
        effective_batch_size_seq = valid_seq_mask.sum().item()
        effective_batch_size_struct = valid_struct_mask.sum().item()
        
        # Compute losses only on valid sequences and structures
        if effective_batch_size_seq > 0:
            seq_logits_valid = seq_logits[valid_seq_mask]
            seq_x_0_valid, seq_x_t_valid = seq_x_0[valid_seq_mask], seq_x_t[valid_seq_mask]
            nonbeospank_seq_valid = nonbeospank_seq[valid_seq_mask]
            cumulative_noise_valid, inst_noise_valid = cumulative_noise[valid_seq_mask], inst_noise[valid_seq_mask]
            loss_seq = score_entropy_loss_fn(seq_logits_valid, seq_x_0_valid, seq_x_t_valid, cumulative_noise_valid, inst_noise_valid, model_cfg.seq_absorb_token, valid_mask=nonbeospank_seq_valid)
        else: loss_seq = torch.tensor(0.0, device=seq_logits.device)
        
        if effective_batch_size_struct > 0:
            struct_logits_valid = struct_logits[valid_struct_mask]
            struct_x_0_valid, struct_x_t_valid = struct_x_0[valid_struct_mask], struct_x_t[valid_struct_mask]
            nonbeospank_struct_valid = nonbeospank_struct[valid_struct_mask]
            cumulative_noise_valid, inst_noise_valid = cumulative_noise[valid_struct_mask], inst_noise[valid_struct_mask]
            loss_struct = score_entropy_loss_fn(struct_logits_valid, struct_x_0_valid, struct_x_t_valid, cumulative_noise_valid, inst_noise_valid, model_cfg.struct_absorb_token, valid_mask=nonbeospank_struct_valid)
        else: loss_struct = torch.tensor(0.0, device=struct_logits.device)

        # Compute combined loss
        seq_loss_weight = train_cfg.loss_config.seq_loss_weight * (effective_batch_size_seq / B)
        struct_loss_weight = train_cfg.loss_config.struct_loss_weight * (effective_batch_size_struct / B)
        loss = seq_loss_weight * loss_seq + struct_loss_weight * loss_struct
        
        if train_mode: # Check if we have any valid gradients to backpropagate
            if effective_batch_size_seq > 0 or effective_batch_size_struct > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        # Return metrics as tuples (value, effective_batch_size)
        return {'loss': (loss.item(), B), 'loss_seq': (loss_seq.item(), effective_batch_size_seq), 'loss_struct': (loss_struct.item(), effective_batch_size_struct)}