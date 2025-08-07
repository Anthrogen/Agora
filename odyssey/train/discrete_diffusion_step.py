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
import pdb

import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List, Dict
import random
import math
from types import SimpleNamespace
from odyssey.src.dataloader import _get_noise_levels

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.transformer import TransformerTrunk, StandardTransformerBlock
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import DiffusionDataLoader, MaskedBatch
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import score_entropy_loss_absorb, score_entropy_loss_uniform
from odyssey.src.configurations import TrunkConfig, TrainingConfig, ScoreEntropyLossConfig
from odyssey.src.tokenizer import CorruptionMode

def discrete_diffusion_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, scheduler, batch: MaskedBatch, model_cfg: TrunkConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    assert isinstance(train_cfg.loss_config, ScoreEntropyLossConfig)
    """Perform a single step with discrete diffusion."""

    B, L = batch.masked_data['seq'].shape

    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    assert content_elements.any(dim=1).all() # Need at least one real residue in each sequence
    
    # Pass raw timestep indices following DiT convention
    timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
    cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
    inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
    timesteps = timesteps.to(torch.float32).unsqueeze(-1)  # Explicit float32 for fp16 compatibility
    
    # Prepare inputs
    model.train(train_mode)

    masked_inputs = ('coords', 'seq', 'struct', 'ss8', 'sasa', 'domains', 'plddt')
    unmasked_inputs = ('orthologous_groups', 'semantic_description')

    tok1 = {k: batch.masked_data[k] for k in masked_inputs}
    tok2 = {k: batch.unmasked_data[k] for k in unmasked_inputs}
    input_tokens = {**tok1, **tok2}

    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass with time conditioning
        model_type = model.cfg.first_block_cfg.initials()

        if model_type in ("GA", "RA"): outputs = model(input_tokens, batch.beospank, coords=batch.masked_data['coords'], content_elements=content_elements, timesteps=timesteps)
        else: outputs = model(input_tokens, batch.beospank, timesteps=timesteps)
        seq_logits, struct_logits = outputs
        
        score_entropy_loss_fn = score_entropy_loss_absorb if train_cfg.mask_config.corruption_mode == "absorb" else score_entropy_loss_uniform
        
        # Find which batch elements have valid elements and count effective batch sizes
        valid_seq_mask = (~batch.beospank['seq']).any(dim=1)  # [B] - which sequences have at least one valid position
        valid_struct_mask = (~batch.beospank['struct']).any(dim=1)  # [B] - which sequences have at least one valid position
        effective_batch_size_seq = valid_seq_mask.sum().item()
        effective_batch_size_struct = valid_struct_mask.sum().item()
        
        # Compute losses only on valid sequences and structures
        if effective_batch_size_seq > 0:
            seq_logits_valid = seq_logits[valid_seq_mask]
            seq_x_0_valid, seq_x_t_valid = batch.unmasked_data['seq'][valid_seq_mask], batch.masked_data['seq'][valid_seq_mask]
            nonbeospank_seq_valid = (~batch.beospank['seq'])[valid_seq_mask]
            cumulative_noise_valid, inst_noise_valid = cumulative_noise[valid_seq_mask], inst_noise[valid_seq_mask]
            loss_seq = score_entropy_loss_fn(seq_logits_valid, seq_x_0_valid, seq_x_t_valid, cumulative_noise_valid, inst_noise_valid, model_cfg.seq_absorb_token, valid_mask=nonbeospank_seq_valid)
        else: loss_seq = 0.0 * seq_logits.sum()
        
        if effective_batch_size_struct > 0:
            struct_logits_valid = struct_logits[valid_struct_mask]
            struct_x_0_valid, struct_x_t_valid = batch.unmasked_data['struct'][valid_struct_mask], batch.masked_data['struct'][valid_struct_mask]
            nonbeospank_struct_valid = (~batch.beospank['struct'])[valid_struct_mask]
            cumulative_noise_valid, inst_noise_valid = cumulative_noise[valid_struct_mask], inst_noise[valid_struct_mask]
            loss_struct = score_entropy_loss_fn(struct_logits_valid, struct_x_0_valid, struct_x_t_valid, cumulative_noise_valid, inst_noise_valid, model_cfg.struct_absorb_token, valid_mask=nonbeospank_struct_valid)
        else: loss_struct = 0.0 * struct_logits.sum()

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


# def generate_discrete_diffusion(model, model_cfg, train_cfg, batch):
#     if batch is None:
#         return None
#     seq_x_t, struct_x_t, = batch.masked_data['seq'], batch.masked_data['struct']
#     seq_x_0, struct_x_0 = batch.unmasked_data['seq'], batch.unmasked_data['struct']
#     nonbeospank_seq, nonbeospank_struct = ~batch.beospank['seq'], ~batch.beospank['struct']
#     coords_x_t, coords_x_0 = batch.masked_data['coords'], batch.unmasked_data['coords']
#     ss8_x_0, sasa_x_0 = batch.masked_data['ss8'], batch.masked_data['sasa']
#     orthologous_groups_x_0, semantic_description_x_0, domains_x_0 = batch.unmasked_data['orthologous_groups'], batch.unmasked_data['semantic_description'], batch.masked_data['domains']
#     plddt_x_0 = batch.masked_data['plddt']
#     B, L = seq_x_t.shape
#     V = {'seq': model_cfg.seq_vocab, 'struct': model_cfg.struct_vocab}
#     x_t = {'seq': seq_x_t, 'struct': struct_x_t}

#     assert B == 1, "Batch must be of size 1 for generation."

#     content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
#     nonbeospank = ~batch.beospank['coords'] & ~batch.beospank['seq']
#     nonbeospank_ss8 = ~batch.beospank['ss8']
#     nonbeospank_sasa = ~batch.beospank['sasa']
#     nonbeospank_orthologous_groups = ~batch.beospank['orthologous_groups']
#     nonbeospank_semantic_description = ~batch.beospank['semantic_description']
#     nonbeospank_domains = ~batch.beospank['domains']
#     nonbeospank_plddt = ~batch.beospank['plddt']
#     assert content_elements.any(dim=1).all() # Need at least one real residue in each sequence
    
#     # Pass raw timestep indices following DiT convention
#     timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
#     cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
#     inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
#     timesteps = timesteps.float().unsqueeze(-1)
    
#     # Prepare inputs
#     inputs = (seq_x_t, struct_x_t, ss8_x_0, sasa_x_0, orthologous_groups_x_0, semantic_description_x_0, domains_x_0, plddt_x_0)
#     model.eval()
    
#     # Forward pass with time conditioning
#     model_type = model.cfg.first_block_cfg.initials()
#     if model_type in ("GA", "RA"): outputs = model(inputs, coords_x_t, content_elements, nonbeospank, nonbeospank_ss8, nonbeospank_sasa, nonbeospank_orthologous_groups, nonbeospank_semantic_description, nonbeospank_domains, nonbeospank_plddt, timesteps)
#     else: outputs = model(inputs, nonbeospank=nonbeospank, nonbeospank_ss8=nonbeospank_ss8, nonbeospank_sasa=nonbeospank_sasa, nonbeospank_orthologous_groups=nonbeospank_orthologous_groups, nonbeospank_semantic_description=nonbeospank_semantic_description, nonbeospank_domains=nonbeospank_domains, nonbeospank_plddt=nonbeospank_plddt, timesteps=timesteps)
#     seq_logits, struct_logits = outputs

#     s = {'seq': torch.exp(seq_logits), 'struct': torch.exp(struct_logits)}

#     # Generate a [L,V] matrix where the i,y-th entry is Q_t(x_t^i, y)
#     inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
#         train_cfg.diffusion_cfg.sigma_min, 
#         train_cfg.diffusion_cfg.sigma_max, 
#         train_cfg.diffusion_cfg.num_timesteps,
#         train_cfg.diffusion_cfg.noise_schedule
#     )

#     # Generate a [L,V] matrix where the i,y-th entry is \delta_{x_t^i}(y)
#     one_hot = torch.nn.functional.one_hot(x_t['seq'], num_classes=V['seq'])

#     pdb.set_trace()

def generate_discrete_diffusion(corruption_mode, model, model_cfg, train_cfg, batch):
    """Perform a single step with discrete diffusion."""
    if batch is None:
        assert False

    B, L = batch.masked_data['seq'].shape
    assert corruption_mode == CorruptionMode.UNIFORM
    assert B == 1, "Batch must be of size 1 for generation."
    device = batch.masked_data['seq'].device

    content_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
    assert content_elements.any(dim=1).all() # Need at least one real residue in each sequence
    
    # Pass raw timestep indices following DiT convention
    timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
    cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
    inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
    timesteps = timesteps.to(torch.float32).unsqueeze(-1)  # Explicit float32 for fp16 compatibility
    
    # Prepare inputs
    model.train(False)

    masked_inputs = ('coords', 'seq', 'struct', 'ss8', 'sasa', 'domains', 'plddt')
    unmasked_inputs = ('orthologous_groups', 'semantic_description')


    
    model.train(False)

    V = {'seq': model_cfg.seq_vocab, 'struct': model_cfg.struct_vocab}
    V_nonspecial = {key: value - len(SPECIAL_TOKENS) for key, value in V.items()}

    def prob_nonspecial(s):
        s['seq'] = s['seq'][:,:,:V_nonspecial['seq']] # [B, L, \hat{V}]
        sums = s['seq'].sum(dim=-1).unsqueeze(-1) # [B, L, 1]

        s['seq'] = s['seq'] / sums

    def expontentiate_uniform(noise, dim):
        term1 = torch.exp(-noise) * torch.eye(dim, device=device)
        term2 = ((1-torch.exp(-noise))/dim) * torch.ones(dim, dim, device=device)

        return term1 + term2

    
    with torch.no_grad():
        # Forward pass with time conditioning
        model_type = model.cfg.first_block_cfg.initials()

        # Generate a [L,V] matrix where the i,y-th entry is Q_t(x_t^i, y)
        inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
            train_cfg.mask_config.sigma_min, 
            train_cfg.mask_config.sigma_max, 
            train_cfg.mask_config.num_timesteps,
            train_cfg.mask_config.noise_schedule
        )

        for t in reversed(range(1, len(inst_noise_levels))):
            print(f"Timestep {t}")
            print(f"Inst noise level: {inst_noise_levels[t]}")

            tok1 = {k: batch.masked_data[k] for k in masked_inputs}
            tok2 = {k: batch.unmasked_data[k] for k in unmasked_inputs}
            input_tokens = {**tok1, **tok2}

            if model_type in ("GA", "RA"): outputs = model(input_tokens, batch.beospank, coords=batch.masked_data['coords'], content_elements=content_elements, timesteps=timesteps)
            else: outputs = model(input_tokens, batch.beospank, timesteps=timesteps)
            seq_logits, struct_logits = outputs

            s = {'seq': torch.exp(seq_logits), 'struct': torch.exp(struct_logits)}

            s = prob_nonspecial(s)

            generator_matrix_neg = expontentiate_uniform(cumulative_noise_levels[t-1] - cumulative_noise_levels[t], dim=V_nonspecial['seq'])
            generator_matrix_pos = expontentiate_uniform(cumulative_noise_levels[t] - cumulative_noise_levels[t-1], dim=V_nonspecial['seq'])

            # factor1 = generator_matrix_neg @ s['seq'] # [V,V] @ [B, L, V] -> [B, L, V]
            factor1 = torch.einsum('ij,blj->bli', generator_matrix_neg, s['seq']) # [V,V] @ [B, L, V] -> [B, L, V]
            factor2 = generator_matrix_pos[batch.masked_data['seq']] # [B, L, V]

            # Element-wise multiplication?
            product = factor1 * factor2

            print(f"Product shape: {product.shape}")

        pdb.set_trace()

    return batch
