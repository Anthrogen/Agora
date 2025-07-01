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
from src.models.transformer import TransformerTrunk, StandardTransformerBlock
from src.models.autoencoder import FSQEncoder
from src.dataloader import _get_training_dataloader, MaskedBatch
from src.dataset import ProteinDataset
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from src.losses import cross_entropy_loss, calculate_accuracy
from src.configurations import TrunkConfig, TrainingConfig

def mlm_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: TrunkConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    """Perform a single MLM step with train/validation mode."""
    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    mask_seq, mask_struct, mask_coords= batch.masks['seq'], batch.masks['struct'], batch.masks['coords']
    seq_tokens, struct_tokens = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    B, L = masked_seq.shape

    assert torch.all(mask_coords == mask_struct), f"mask_coords and mask_struct differ in some positions:\nmask_coords:\n{mask_coords}\nmask_struct:\n{mask_struct}"

    # Create coord_mask for GA/RA models
    nonspecial_elements_coords = (~batch.beospank['coords']) & (~mask_struct)
    # We need one non-special element in coords for GA/RA models.
    assert nonspecial_elements_coords.any(dim=1).all()
    
    inputs = (masked_seq, masked_struct) # Prepare model input
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass
        model_type = model.cfg.first_block_cfg.initials()
        if model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, nonspecial_elements_coords)
        else: outputs = model(inputs)
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

        loss_seq = cross_entropy_loss(seq_logits, batch.unmasked_data['seq'], loss_elements_seq)
        loss_struct = cross_entropy_loss(struct_logits, batch.unmasked_data['struct'], loss_elements_struct)

        loss = train_cfg.loss_config.seq_loss_weight * loss_seq + train_cfg.loss_config.struct_loss_weight * loss_struct

        seq_acc = calculate_accuracy(seq_logits, batch.unmasked_data['seq'], loss_elements_seq)
        struct_acc = calculate_accuracy(struct_logits, batch.unmasked_data['struct'], loss_elements_struct)
        
        if train_mode:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics
        return {'loss': loss.item()*B, 'loss_seq': loss_seq.item()*B, 'loss_struct': loss_struct.item()*B, 'seq_acc': seq_acc.item()*B, 'struct_acc': struct_acc.item()*B}