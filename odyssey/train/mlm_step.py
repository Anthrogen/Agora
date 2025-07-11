"""
Train the unified transformer model with masked language modeling.
Refactored version with masking logic moved to DataLoader.

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

Masking Strategy:
- All architectures use IDENTICAL masking patterns for each iteration by using
  a fixed seed for the DataLoader workers
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
from types import SimpleNamespace

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.transformer import TransformerTrunk, StandardTransformerBlock
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import _get_training_dataloader, MaskedBatch
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import cross_entropy_loss, calculate_accuracy
from odyssey.src.configurations import TrunkConfig, TrainingConfig, CrossEntropyLossConfig

def mlm_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: TrunkConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, CrossEntropyLossConfig)
    assert train_cfg.loss_config.loss_elements == "masked"
    """Perform a single MLM step with train/validation mode."""
    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    mask_seq, mask_struct, mask_coords= batch.masks['seq'], batch.masks['struct'], batch.masks['coords']
    seq_tokens, struct_tokens = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    ss8_tokens, sasa_tokens = batch.unmasked_data['ss8'], batch.unmasked_data['sasa']
    global_annotation_tokens, per_residue_annotation_tokens = batch.unmasked_data['global_annotation'], batch.unmasked_data['per_residue_annotation']
    plddt_tokens = batch.unmasked_data['plddt']
    B, L = masked_seq.shape

    assert torch.all(mask_coords == mask_struct), f"mask_coords and mask_struct differ in some positions:\nmask_coords:\n{mask_coords}\nmask_struct:\n{mask_struct}"

    # Create mask for GA/RA/SA/SC models and for SS8/SASA
    nonspecial_elements_coords = (~batch.beospank['coords']) & (~mask_struct)
    nonbeospank_elements = ~batch.beospank['coords']
    nonbeospank_ss8 = ~batch.beospank['ss8']
    nonbeospank_sasa = ~batch.beospank['sasa']
    nonbeospank_global_annotation = ~batch.beospank['global_annotation']
    nonbeospank_per_residue_annotation = ~batch.beospank['per_residue_annotation']
    nonbeospank_plddt = ~batch.beospank['plddt']
    assert nonspecial_elements_coords.any(dim=1).all()
    
    inputs = (masked_seq, masked_struct, ss8_tokens, sasa_tokens, global_annotation_tokens, per_residue_annotation_tokens, plddt_tokens) # Prepare model input
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass
        model_type = model.cfg.first_block_cfg.initials()
        if model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, nonspecial_elements_coords, nonbeospank_ss8, nonbeospank_sasa, nonbeospank_global_annotation, nonbeospank_per_residue_annotation, nonbeospank_plddt)
        else: outputs = model(inputs, mask=nonbeospank_elements, mask_ss8=nonbeospank_ss8, mask_sasa=nonbeospank_sasa, mask_global_annotation=nonbeospank_global_annotation, mask_per_residue_annotation=nonbeospank_per_residue_annotation, mask_plddt=nonbeospank_plddt)
        seq_logits, struct_logits = outputs

        if train_cfg.loss_config.loss_elements == "masked":
            loss_elements_seq = batch.masks['seq']
            loss_elements_struct = batch.masks['struct']
        elif train_cfg.loss_config.loss_elements == "non_beospank":
            # Compute loss over all non-BOS/EOS/PAD positions, including masks.
            loss_elements_seq = ~batch.beospank['seq']
            loss_elements_struct = ~batch.beospank['struct']
        elif train_cfg.loss_config.loss_elements == "non_special":
            loss_elements_seq = ~batch.beospank['seq'] & ~batch.masks['seq']
            loss_elements_struct = ~batch.beospank['struct'] & ~batch.masks['struct']
        else:
            raise ValueError(f"What is {train_cfg.loss_config.loss_elements}?")

        # Find which batch elements have valid loss elements and count effective batch sizes
        valid_seq_mask = loss_elements_seq.any(dim=1)  # [B]
        valid_struct_mask = loss_elements_struct.any(dim=1)  # [B]
        effective_batch_size_seq = valid_seq_mask.sum().item()
        effective_batch_size_struct = valid_struct_mask.sum().item()
        
        # Compute losses only on valid sequences and structures
        if effective_batch_size_seq > 0:
            seq_logits_valid = seq_logits[valid_seq_mask]
            seq_labels_valid = batch.unmasked_data['seq'][valid_seq_mask]
            loss_elements_seq_valid = loss_elements_seq[valid_seq_mask]
                
            loss_seq = cross_entropy_loss(seq_logits_valid, seq_labels_valid, loss_elements_seq_valid)
            seq_acc = calculate_accuracy(seq_logits_valid, seq_labels_valid, loss_elements_seq_valid)
        else:
            loss_seq = torch.tensor(0.0, device=seq_logits.device)
            seq_acc = torch.tensor(0.0, device=seq_logits.device)
        
        if effective_batch_size_struct > 0:
            struct_logits_valid = struct_logits[valid_struct_mask]
            struct_labels_valid = batch.unmasked_data['struct'][valid_struct_mask]
            loss_elements_struct_valid = loss_elements_struct[valid_struct_mask]
                
            loss_struct = cross_entropy_loss(struct_logits_valid, struct_labels_valid, loss_elements_struct_valid)
            struct_acc = calculate_accuracy(struct_logits_valid, struct_labels_valid, loss_elements_struct_valid)
        else:
            loss_struct = torch.tensor(0.0, device=struct_logits.device)
            struct_acc = torch.tensor(0.0, device=struct_logits.device)

        # Compute combined loss
        loss = train_cfg.loss_config.seq_loss_weight * loss_seq + train_cfg.loss_config.struct_loss_weight * loss_struct
        
        if train_mode:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics as tuples (value, effective_batch_size)
        return {'loss': (loss.item(), B), 'loss_seq': (loss_seq.item(), effective_batch_size_seq), 'loss_struct': (loss_struct.item(), effective_batch_size_struct), 'seq_acc': (seq_acc.item(), effective_batch_size_seq), 'struct_acc': (struct_acc.item(), effective_batch_size_struct)}