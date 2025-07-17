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
from odyssey.src.dataset import ProteinDataset, ATOMS
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from odyssey.src.configurations import FSQConfig, TrainingConfig, KabschRMSDLossConfig

def _token_to_amino_acid(token: int) -> str:
    """Convert sequence token to amino acid letter"""
    # Create mapping from token value to amino acid letter
    token_to_aa = {v.value: k for k, v in SEQUENCE_TOKENS.__members__.items()}
    return token_to_aa.get(token, 'X')  # Default to 'X' for unknown tokens

def stage_1_step(model: Autoencoder, optimizer: torch.optim.Optimizer, scheduler, batch: MaskedBatch, model_cfg: FSQConfig, train_cfg: TrainingConfig, train_mode=True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, KabschRMSDLossConfig)
    # Stage 1: Masked coordinate reconstruction
    B, L, H, _ = batch.masked_data['coords'].shape

    # Create masks for GA/RA/SA/SC models (valid positions that are not masked)
    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank = ~batch.beospank['coords']
    assert content_elements.any(dim=1).all()
    
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass - use only first 3 atoms for standard coordinates
        three_atom_masked_coords = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]
        if model.cfg.first_block_cfg.initials() in ("GA", "RA"): x_rec, _ = model(three_atom_masked_coords, batch.masked_data['coords'], content_elements, nonbeospank)
        else: x_rec, _ = model(three_atom_masked_coords, nonbeospank=nonbeospank)

        # In order to run KABSCH, we need to isolate only unmasked residues into a [U, 3, 3] tensor for each protein in the batch, where U is number of unmasked residues in a given protein.
        pts_pred = []; pts_true = []
        for batch_idx in range(B):
            real_residues = torch.arange(L, device=batch.masked_data['coords'].device)[content_elements[batch_idx]] # U
            pred_coords = x_rec[batch_idx][real_residues]  # [U, 3, 3]
            true_coords = batch.masked_data['coords'][batch_idx, real_residues, :3, :]  # [U, 3, 3] - only first 3 atoms!

            # Mean center based on backbone atoms (first 3 atoms: N, CA, C)
            # Calculate centroids: mean over both residues and atoms
            pred_centroid = pred_coords.reshape(-1, 3).mean(dim=0)  # [3]
            true_centroid = true_coords.reshape(-1, 3).mean(dim=0)  # [3]
            
            # Center the coordinates
            pred_coords_centered = pred_coords - pred_centroid  # [U, 3, 3]
            true_coords_centered = true_coords - true_centroid  # [U, 3, 3]

            # Flatten to [1, U*3, 3]
            pts_pred.append(pred_coords_centered.reshape(1, -1, 3))
            pts_true.append(true_coords_centered.reshape(1, -1, 3))
        
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
            scheduler.step()
        
        # Return metrics
        return {'loss': (loss.item(), B), 'rmsd': (rmsd.item(), B)}


def stage_2_step(model: Autoencoder, optimizer: torch.optim.Optimizer, scheduler, batch: MaskedBatch, model_cfg: FSQConfig, train_cfg: TrainingConfig, train_mode=True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, KabschRMSDLossConfig)
    # Stage 2: Full structure reconstruction from frozen encoder
    B, L, H, _ = batch.masked_data['coords'].shape
    
    # Create masks for GA/RA/SA/SC models
    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank = ~batch.beospank['coords']
    assert content_elements.any(dim=1).all(), f"Offending protein: Seq={batch.unmasked_data['seq'][:,:25].tolist()}"
    
    model.train(train_mode)
    model.encoder.eval()  # Encoder is frozen
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass through frozen encoder to get z_q
        with torch.no_grad():
            four_atom = batch.masked_data['coords'][:, :, :4, :]  # [B, L, 4, 3] for encoder
            z_q = model.quantizer.indices_to_codes(batch.masked_data['struct'])

        # Zero out BOS/EOS/PAD/UNK positions in z_q
        z_q[batch.beospank['coords']] = 0.0
        assert not batch.masks['coords'].any(), "There is no masking in stage 2"
        
        # Decoder forward pass with context injection
        if model.cfg.first_block_cfg.initials() in ("GA", "RA"): x_rec = model.decoder(z_q, four_atom, content_elements, nonbeospank, batch.masked_data['seq'])
        else: x_rec = model.decoder(z_q, nonbeospank=nonbeospank, seq_tokens=batch.masked_data['seq'])
        
        # x_rec is [B, L, 14, 3] for stage 2
        # Compute loss on all valid positions (no masking in stage 2)
        pts_pred = []; pts_true = []
        for batch_idx in range(B):
            real_residues = torch.arange(L, device=batch.masked_data['coords'].device)[content_elements[batch_idx]]
            pred_coords = x_rec[batch_idx][real_residues]  # [M, 14, 3] 
            true_coords = batch.masked_data['coords'][batch_idx][real_residues]  # [M, 14, 3]
            seq_tokens = batch.masked_data['seq'][batch_idx][real_residues]  # [M]

            # Mean center using backbone atoms (first 3 atoms: N, CA, C)
            pred_backbone = pred_coords[:, :3, :]  # [M, 3, 3] - backbone atoms for all residues
            true_backbone = true_coords[:, :3, :]  # [M, 3, 3] - backbone atoms for all residues
            
            # Calculate centroids from backbone atoms
            pred_centroid = pred_backbone.reshape(-1, 3).mean(dim=0)  # [3]
            true_centroid = true_backbone.reshape(-1, 3).mean(dim=0)  # [3]
            
            # Center all coordinates
            pred_coords_centered = pred_coords - pred_centroid  # [M, 14, 3]
            true_coords_centered = true_coords - true_centroid  # [M, 14, 3]
            
            # Now slice coordinates based on sequence-specific atom counts
            residue_pred_coords = []; residue_true_coords = []
            
            for residue_idx, token in enumerate(seq_tokens):
                # Convert token to amino acid and get atom count
                aa = _token_to_amino_acid(token.item())
                k = len(ATOMS.get(aa, ATOMS['X']))  # Number of atoms for this amino acid
                
                # Slice to sequence-specific atom count
                pred_res = pred_coords_centered[residue_idx, :k, :]  # [k, 3]
                true_res = true_coords_centered[residue_idx, :k, :]  # [k, 3]
                
                residue_pred_coords.append(pred_res)
                residue_true_coords.append(true_res)
            
            if residue_pred_coords:
                # Concatenate all residues for this protein
                pred_coords_full = torch.cat(residue_pred_coords, dim=0)  # [total_atoms, 3]
                true_coords_full = torch.cat(residue_true_coords, dim=0)  # [total_atoms, 3]
                
                # Add as [1, total_atoms, 3]
                pts_pred.append(pred_coords_full.unsqueeze(0))
                pts_true.append(true_coords_full.unsqueeze(0))
        
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
            scheduler.step()
        
        # Return metrics
        return {'loss': (loss.item(), B), 'rmsd': (rmsd.item(), B)}
