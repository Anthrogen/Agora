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
import pdb

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.transformer import TransformerTrunk, StandardTransformerBlock
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import _get_training_dataloader
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import cross_entropy_loss, calculate_accuracy
from odyssey.src.configurations import TrunkConfig, TrainingConfig, CrossEntropyLossConfig

def sample_positions_to_unmask(masked_positions, probs_masked, UNMASK_STRATEGY, NUM_TO_UNMASK):
    N = min(len(masked_positions), NUM_TO_UNMASK)
    if UNMASK_STRATEGY == "uniform":
        idxs = torch.randperm(len(masked_positions))[:N]
        positions_to_unmask = masked_positions[idxs]
    elif UNMASK_STRATEGY == "prob_confidence":
        modes = torch.max(probs_masked, dim=-1).values
        idxs = torch.multinomial(modes, N, replacement=False)
        positions_to_unmask = masked_positions[idxs]
    elif UNMASK_STRATEGY == "min_entropy":
        entropies = -torch.sum(probs_masked * torch.log(probs_masked), dim=-1)
        # Want lowest entropy posiitons:
        idxs = torch.argsort(entropies, descending=True)[:N]
        positions_to_unmask = masked_positions[idxs]
    else:
        raise ValueError(f"Unknown unmask strategy: {UNMASK_STRATEGY}")

    return positions_to_unmask

def generate_mlm(model, model_cfg, train_cfg, batch, UNMASK_STRATEGY="min_entropy", NUM_TO_UNMASK=3):
    """
    Hard-coded parameters:
    Pick the NUM_TO_UNMASK favorite masked positions to unmask:
    Three ways to pick:
    1. Randomly, uniformly over all masked positions.
    2. Randomly, with probability proportional to probabilities of logits at only masked positions.
    3. Deterministically, picking the three tokens of which you are most confident (lowest discrete entropy)
    remask_per_pass # Not yet implemented.
    """
    assert UNMASK_STRATEGY in ("uniform", "prob_confidence", "min_entropy")
    assert NUM_TO_UNMASK > 0

    # We want only one batch element at a time
    # pdb.set_trace()
    assert batch is not None
    B = batch.unmasked_data['seq'].shape[0]
    assert B == 1, "Right now we only generate for batches of one."
    device = batch.unmasked_data['seq'].device
    # Recall that orthologous_groups and semantic_description are never masked.

    while batch.masks['seq'].any() or batch.masks['struct'].any():
        #########################################################
        # Forward pass of batch through the model to get logits.
        #########################################################
        content_elements = ~batch.beospank['coords'] & ~batch.masks['coords']

        masked_inputs = ('coords', 'seq', 'struct', 'ss8', 'sasa', 'domains', 'plddt')
        unmasked_inputs = ('orthologous_groups', 'semantic_description')
        
        tok1 = {k: batch.masked_data[k] for k in masked_inputs}
        tok2 = {k: batch.unmasked_data[k] for k in unmasked_inputs}
        input_tokens = {**tok1, **tok2}

        # Forward pass
        if model_cfg.first_block_cfg.initials() in ("GA", "RA"): outputs = model(input_tokens, batch.beospank, coords=batch.masked_data['coords'], content_elements=content_elements)
        else: outputs = model(input_tokens, batch.beospank)
        seq_logits, struct_logits = outputs
        logits =  {'seq': seq_logits, 'struct': struct_logits}

        for track in ('seq', 'struct'):
            probs = torch.nn.functional.softmax(logits[track], dim=-1).squeeze(0)

            masked_positions = torch.arange(batch.masks[track].shape[1], device=device)[batch.masks[track].squeeze(0)]
            probs_masked = probs[masked_positions, :]
            positions_to_unmask = sample_positions_to_unmask(masked_positions, probs_masked, UNMASK_STRATEGY, NUM_TO_UNMASK)

            # probs is [L, V] long
            probs = probs[positions_to_unmask, :] # Now [N, V] long

            # Next, we need to "chop off" the special tokens as they are not on the menu.
            nonspecial_elements = torch.ones_like(probs)  # Remove .float() for fp16 compatibility
            nonspecial_elements[:, -len(SPECIAL_TOKENS):] = 0.0
            probs = probs * nonspecial_elements

            # Renormalize, now that we've zero-ed out many elements.
            probs = probs / probs.sum(dim=-1, keepdim=True) # quotient is [N, V_content]
            # Now, sample from the renormalized probabilities:
            sampled_tokens = torch.multinomial(probs, 1, replacement=False).squeeze(-1) # [N]

            # Now, we need to "write" the sampled tokens to the batch:
            batch.masked_data[track][0,positions_to_unmask] = sampled_tokens
            batch.masks[track][0, positions_to_unmask] = False
        
    return batch
