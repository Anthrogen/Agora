"""
Train the FSQ in two stages.

Parameter Initialization Strategy:
- Regardless of which architecture is being trained (SA, GA, RA, C), all four
  architectures are initialized with identical parameters where possible.
- The function ensure_identical_parameters_all_architectures creates temporary
  models for all architectures and synchronizes their parameters:
  1. All architectures get identical embeddings, self-attention, feedforward, 
     and output layers from SA architecture
  2. GA and RA get identical geometric/reflexive attention parameters
  3. This ensures fair comparison by removing initialization variance
  
This means when training any architecture, it has the same starting point as
the others would have had, allowing us to isolate the effect of architectural
differences on training dynamics.
"""
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Callable, Dict
import random
from types import SimpleNamespace

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder, StandardTransformerBlock
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import MaskedBatch, SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from odyssey.src.configurations import FSQConfig, TrainingConfig, KabschRMSDLossConfig

def stage_1_step(model: Autoencoder, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: FSQConfig, train_cfg: TrainingConfig, train_mode=True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, KabschRMSDLossConfig)
    # Stage 1: Masked coordinate reconstruction
    B, L, H, _ = batch.masked_data['coords'].shape

    # Create masks for GA/RA/SA/SC models (valid positions that are not masked)
    unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank_elements = ~batch.beospank['coords']
    assert unmasked_elements.any(dim=1).all()
    
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass - use only first 3 atoms for standard coordinates
        three_atom_masked_coords = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]
        if model.cfg.first_block_cfg.initials() in ("GA", "RA"): x_rec, _ = model(three_atom_masked_coords, batch.masked_data['coords'], unmasked_elements)
        else: x_rec, _ = model(three_atom_masked_coords, mask=nonbeospank_elements)

        # In order to run KABSCH, we need to isolate only unmasked residues into a [U, 3, 3] tensor for each protein in the batch, where U is number of unmasked residues in a given protein.
        pts_pred = []; pts_true = []
        for batch_idx in range(B):
            real_residues = torch.arange(L, device=batch.masked_data['coords'].device)[unmasked_elements[batch_idx]]
            pred_coords = x_rec[batch_idx][real_residues]  # [U, 3, 3]
            true_coords = batch.masked_data['coords'][batch_idx, real_residues, :3, :]  # [U, 3, 3] - only first 3 atoms!

            # Flatten to [1, U*3, 3]
            pts_pred.append(pred_coords.reshape(1, -1, 3))
            pts_true.append(true_coords.reshape(1, -1, 3))
        
        # Compute squared Kabsch RMSD loss and regular RMSD
        if pts_pred:
            loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
            # Also compute regular Kabsch RMSD for reporting
            with torch.no_grad(): rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
        else:
            loss = torch.tensor(0.0, device=batch.masked_data['coords'].device)
            rmsd = torch.tensor(0.0, device=batch.masked_data['coords'].device)
        
        if train_mode:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics
        return {'loss': (loss.item(), B), 'rmsd': (rmsd.item(), B)}


def stage_2_step(model: Autoencoder, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: FSQConfig, train_cfg: TrainingConfig, train_mode=True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, KabschRMSDLossConfig)
    # Stage 2: Full structure reconstruction from frozen encoder
    B, L, H, _ = batch.masked_data['coords'].shape
    
    # Create masks for GA/RA/SA/SC models
    unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank_elements = ~batch.beospank['coords']
    assert unmasked_elements.any(dim=1).all(), f"Offending protein: Seq={batch.unmasked_data['seq'][:,:25].tolist()}"
    
    model.train(train_mode)
    model.encoder.eval()  # Encoder is frozen
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass through frozen encoder to get z_q
        with torch.no_grad():
            four_atom = batch.masked_data['coords'][:, :, :4, :]  # [B, L, 4, 3] for encoder
            z_q = model.quantizer.indices_to_codes(batch.masked_data['struct'])

        # Zero out BOS/EOS/PAD/UNK positions in z_q
        z_q[batch.beospank['coords']] = 0.0
        
        # Concatenate z_q with seq_tokens along last dimension
        # z_q: [B, L, fsq_dim], seq_tokens: [B, L] -> [B, L, 1]
        seq_tokens_float = batch.masked_data['seq'].unsqueeze(-1).float()  # [B, L, 1]
        decoder_input = torch.cat([z_q, seq_tokens_float], dim=-1)  # [B, L, fsq_dim + 1]
        
        # Decoder forward pass
        if model.cfg.first_block_cfg.initials() in ("GA", "RA"): x_rec = model.decoder(decoder_input, four_atom, unmasked_elements)
        else: x_rec = model.decoder(decoder_input, mask=nonbeospank_elements)
        
        # x_rec is [B, L, 14, 3] for stage 2
        # Compute loss on all valid positions (no masking in stage 2)
        pts_pred = []; pts_true = []
        for batch_idx in range(B):
            real_residues = torch.arange(L, device=batch.masked_data['coords'].device)[unmasked_elements[batch_idx]]
            pred_coords = x_rec[batch_idx][real_residues]  # [M, 14, 3] 
            true_coords = batch.masked_data['coords'][batch_idx][real_residues]  # [M, 14, 3]

            # Flatten to [1, M*14, 3]
            pts_pred.append(pred_coords.reshape(1, -1, 3))
            pts_true.append(true_coords.reshape(1, -1, 3))
        
        # Compute squared Kabsch RMSD loss
        if pts_pred:
            loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
            # Also compute regular Kabsch RMSD for reporting
            with torch.no_grad(): rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
        else:
            loss = torch.tensor(0.0, device=batch.masked_data['coords'].device)
            rmsd = torch.tensor(0.0, device=batch.masked_data['coords'].device)
        
        if train_mode:
            # Backward pass (only decoder parameters will update)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics
        return {'loss': (loss.item(), B), 'rmsd': (rmsd.item(), B)}
