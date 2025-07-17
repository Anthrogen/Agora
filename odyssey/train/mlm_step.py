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

def mlm_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, scheduler, batch: MaskedBatch, model_cfg: TrunkConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, CrossEntropyLossConfig)
    assert train_cfg.loss_config.loss_elements == "masked"
    """Perform a single MLM step with train/validation mode."""
    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    ss8_tokens, sasa_tokens = batch.unmasked_data['ss8'], batch.unmasked_data['sasa']
    global_annotation_tokens, per_residue_annotation_tokens = batch.unmasked_data['global_annotation'], batch.unmasked_data['per_residue_annotation']
    plddt_tokens = batch.unmasked_data['plddt']
    B, L = masked_seq.shape

    # Create mask for GA/RA/SA/SC models and for SS8/SASA
    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    nonbeospank = ~batch.beospank['coords'] & ~batch.beospank['seq']
    nonbeospank_ss8 = ~batch.beospank['ss8']
    nonbeospank_sasa = ~batch.beospank['sasa']
    nonbeospank_global_annotation = ~batch.beospank['global_annotation']
    nonbeospank_per_residue_annotation = ~batch.beospank['per_residue_annotation']
    nonbeospank_plddt = ~batch.beospank['plddt']
    assert content_elements.any(dim=1).all()
    
    inputs = (masked_seq, masked_struct, ss8_tokens, sasa_tokens, global_annotation_tokens, per_residue_annotation_tokens, plddt_tokens) # Prepare model input
    nonbeospanks_all = {'nonbeospank': nonbeospank,
                        'nonbeospank_ss8': nonbeospank_ss8,
                        'nonbeospank_sasa': nonbeospank_sasa,
                        'nonbeospank_global_annotation': nonbeospank_global_annotation,
                        'nonbeospank_per_residue_annotation': nonbeospank_per_residue_annotation,
                        'nonbeospank_plddt': nonbeospank_plddt}

    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass
        model_type = model.cfg.first_block_cfg.initials()
        if model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, content_elements, **nonbeospanks_all)
        else: outputs = model(inputs, **nonbeospanks_all)
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
            scheduler.step()
        
        # Return metrics as tuples (value, effective_batch_size)
        return {'loss': (loss.item(), B), 'loss_seq': (loss_seq.item(), effective_batch_size_seq), 'loss_struct': (loss_struct.item(), effective_batch_size_struct), 'seq_acc': (seq_acc.item(), effective_batch_size_seq), 'struct_acc': (struct_acc.item(), effective_batch_size_struct)}

def generate_mlm(model, model_cfg, train_cfg, batch):
    """
    Hard-coded parameters:
    unmask_per_pass = 3
    unmask_strategy = "max_confidence" # or "uniform" or "prob_confidence"
    # max_confidence = unmask the three tokens of which you are most confident.
    # prob_confidence = unmask the three tokens probabilistically, with probability proportional to confidence.
    # uniform = unmask three tokens uniformly at random.
    remask_per_pass # Not yet implemented.
    """

    # We want only one batch element at a time
    B = batch.masked_data['seq'].shape[0]
    assert B == 1, "Right now we only generate for batches of one."

    content_elements_coords = ~batch.beospank['coords'] & ~batch.masks['coords']

    masked_data_all = [batch.masked_data[t] for t in ('seq', 'struct', 'ss8', 'sasa', 'global_annotation', 'per_residue_annotation', 'plddt')]
    nonbeospanks_all = {'nonbeospank': ~batch.beospank['coords'] & ~batch.beospank['seq'],
                        'nonbeospank_ss8': ~batch.beospank['ss8'],
                        'nonbeospank_sasa': ~batch.beospank['sasa'],
                        'nonbeospank_global_annotation': ~batch.beospank['global_annotation'],
                        'nonbeospank_per_residue_annotation': ~batch.beospank['per_residue_annotation'],
                        'nonbeospank_plddt': ~batch.beospank['plddt']}

    if model_cfg.first_block_cfg.initials() in ("GA", "RA"):
        outputs = model(masked_data, batch.masked_data['coords'], content_elements_coords, **nonbeospanks_all)
    else: 
        outputs = model(masked_data, **nonbeospanks_all)

    seq_logits, struct_logits = outputs

    seq_probs = torch.nn.functional.softmax(seq_logits, dim=-1)

    NUM_TO_UNMASK = 3
    UNMASK_STRATEGY = "max_confidence"
    # Pick the NUM_TO_UNMASK favorite masked positions to unmask:
    # Three ways to pick:
    # 1. Randomly, uniformly over all masked positions.
    # 2. Randomly, with probability proportional to likelihoods.
    # 3. Deterministically, picking the three tokens of which you are most confident.

    masked_positions = torch.arange(len(batch.masks['seq']))[batch.masks['seq']]
    probs = seq_probs[masked_positions]

    N = min(len(probs), NUM_TO_UNMASK)
    if UNMASK_STRATEGY == "uniform":
        idxs = torch.randperm(len(masked_positions))[:N]
        positions_to_unmask = masked_positions[idxs]
    elif UNMASK_STRATEGY == "prob_confidence":
        probs_masked = probs[masked_positions]
        idxs = torch.multinomial(probs_masked, N, replacement=False)
        positions_to_unmask = masked_positions[idxs]
    elif UNMASK_STRATEGY == "max_confidence":
        probs_masked = probs[masked_positions]
        idxs = torch.argsort(probs_masked, dim=-1, descending=True)[:N]
        positions_to_unmask = masked_positions[idxs]
    else:
        raise ValueError(f"Unknown unmask strategy: {UNMASK_STRATEGY}")

    # Now, actually go through and unmask at each position:
    # Sample according to a probability law that DOES NOT INCLUDE special tokens.
    
    # probs is [B, L, V] long
    probs = probs.squeeze(0) # Now [L, V] long
    probs = probs[positions_to_unmask, :] # Now [N, V] long

    # Next, we need to "chop off" the special tokens as they are not on the menu.
    V_basic = model_cfg.seq_vocab - len(SPECIAL_TOKENS)



