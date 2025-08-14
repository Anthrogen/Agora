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
from agora.src.dataloader import _get_noise_levels

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agora.src.models.transformer import TransformerTrunk, StandardTransformerBlock
from agora.src.models.autoencoder import FSQEncoder
from agora.src.dataset import ProteinDataset
from agora.src.dataloader import DiffusionDataLoader
from agora.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from agora.src.losses import score_entropy_loss_absorb, score_entropy_loss_uniform
from agora.src.configurations import TrunkConfig, TrainingConfig, ScoreEntropyLossConfig
from agora.src.tokenizer import CorruptionMode

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
    tracks = ('seq', 'struct')


    V = {'seq': model_cfg.seq_vocab, 'struct': model_cfg.struct_vocab}
    V_nonspecial = {key: value - len(SPECIAL_TOKENS) for key, value in V.items()}
    absorb_tokens = {'seq': model_cfg.seq_absorb_token, 'struct': model_cfg.struct_absorb_token}
    num_content_tokens = {key: value - len(SPECIAL_TOKENS) for key, value in V.items()}

    absorb_vocab = {t: list(range(num_content_tokens[t])) + [absorb_tokens[t]] for t in tracks}
    uniform_vocab = {t: list(range(num_content_tokens[t])) for t in tracks}

    # Generate a [L,V] matrix where the i,y-th entry is Q_t(x_t^i, y)
    inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
        train_cfg.mask_config.sigma_min, 
        train_cfg.mask_config.sigma_max, 
        train_cfg.mask_config.num_timesteps,
        train_cfg.mask_config.noise_schedule
    )

    def prob_law(s):
        # Vocab already trimmed down to exclude nonpermitted tokens.
        for track in tracks:
            sums = s[track].sum(dim=-1).unsqueeze(-1) # [B, L, 1]
            s[track] = s[track] / sums
        return s

    if corruption_mode == CorruptionMode.UNIFORM:

        def expontentiate_uniform(noise, dim):
            """
            Calclate exp(noise * Q_uniform)
            """
            term1 = torch.exp(-noise) * torch.eye(dim, device=device)
            term2 = ((1-torch.exp(-noise))/dim) * torch.ones(dim, dim, device=device)

            return term1 + term2

        with torch.no_grad():
            # Forward pass with time conditioning
            model_type = model.cfg.first_block_cfg.initials()

            for t in reversed(range(1, len(inst_noise_levels))):
                print(f"Timestep {t}")
                print(f"Inst noise level: {inst_noise_levels[t]}")

                tok1 = {k: batch.masked_data[k] for k in masked_inputs}
                tok2 = {k: batch.unmasked_data[k] for k in unmasked_inputs}
                input_tokens = {**tok1, **tok2}

                current_timestep = timesteps - t

                assert model_type not in ("GA", "RA"), "see comment below to fix implementation"
                if model_type in ("GA", "RA"): outputs = model(input_tokens, batch.beospank, coords=batch.masked_data['coords'], content_elements=content_elements, timesteps=current_timestep)
                else: outputs = model(input_tokens, batch.beospank, timesteps=current_timestep)
                seq_logits, struct_logits = outputs

                s = {'seq': torch.exp(seq_logits)[:,:,uniform_vocab['seq']], 'struct': torch.exp(struct_logits)[:,:,uniform_vocab['struct']]}
                s = prob_law(s)

                for track in tracks:

                    generator_matrix_neg = expontentiate_uniform(cumulative_noise_levels[t-1] - cumulative_noise_levels[t], dim=V_nonspecial[track])
                    generator_matrix_pos = expontentiate_uniform(cumulative_noise_levels[t] - cumulative_noise_levels[t-1], dim=V_nonspecial[track])

                    # factor1 = generator_matrix_neg @ s['seq'] # [V,V] @ [B, L, V] -> [B, L, V]
                    factor1 = torch.einsum('ij,blj->bli', generator_matrix_neg, s[track]) # [V,V] @ [B, L, V] -> [B, L, V]

                    # Indices corresponding to BEOSPANK do not matter.  
                    indices = batch.masked_data[track] % V_nonspecial[track]
                    factor2 = generator_matrix_pos[indices] # [B, L, V]

                    # Element-wise multiplication?
                    product = factor1 * factor2
                    product = product.squeeze(0)# [L, V]

                    samples = torch.multinomial(product, 1, replacement=False).squeeze(-1) # [B, L]

                    # Only unmasked positions should be affected.
                    batch.masked_data[track] = torch.where(batch.masks[track], samples, batch.masked_data[track])

            # Only do after all timesteps have passed.
            for track in tracks:
                batch.masks[track] = torch.zeros_like(batch.masks[track]).bool()

        return batch

    elif corruption_mode == CorruptionMode.MASK:
        assert False

        # num_content = V[track] - len(SPECIAL_TOKENS)
        # keep_pos = list(range(num_content)) + [mask_token]

        def expontentiate_absorb(noise, dim):
            assert Fasle
            # TODO: columsn or rows?  Should it be transpose of this.

            # term1 = torch.exp(-noise) * torch.eye(dim, device=device)
            # term2 = ((1-torch.exp(-noise))/dim) * torch.ones(dim, dim, device=device)

            # return term1 + term2
            matrix_I = torch.eye(dim, device=device)
            vec_last_hot = torch.nn.functional.one_hot(torch.tensor(dim-1, device=device).long(), num_classes=dim).float().unsqueeze(-1)
            matrix_A = vec_last_hot.tile((1, dim))

            term1 = torch.exp(-noise) * matrix_I
            term2 = ((1-torch.exp(-noise))) * matrix_A

            return term1 + term2

        with torch.no_grad():
            # Forward pass with time conditioning
            model_type = model.cfg.first_block_cfg.initials()
 

            MIN = 1
            for t in reversed(range(MIN, len(inst_noise_levels))):
                print(f"Timestep {t}")
                print(f"Inst noise level: {inst_noise_levels[t]}")

                tok1 = {k: batch.masked_data[k] for k in masked_inputs}
                tok2 = {k: batch.unmasked_data[k] for k in unmasked_inputs}
                input_tokens = {**tok1, **tok2}

                current_timestep = timesteps - t

                assert model_type not in ("GA", "RA"), "see comment below to fix implementation"
                # TODO: content_elements needs to be updated (mask will be filled in after each round of infilling)
                if model_type in ("GA", "RA"): outputs = model(input_tokens, batch.beospank, coords=batch.masked_data['coords'], content_elements=content_elements, timesteps=current_timestep)
                else: outputs = model(input_tokens, batch.beospank, timesteps=current_timestep)
                seq_logits, struct_logits = outputs

                # Trim down logits to exclude special tokens besides the absorbing mask.
                s = {'seq': torch.exp(seq_logits)[:,:,absorb_vocab['seq']], 'struct': torch.exp(struct_logits)[:,:,absorb_vocab['struct']]}
                s = prob_law(s)

                for track in tracks:

                    generator_matrix_neg = expontentiate_absorb(cumulative_noise_levels[t-1] - cumulative_noise_levels[t], dim=len(absorb_vocab[track]))
                    generator_matrix_pos = expontentiate_absorb(cumulative_noise_levels[t] - cumulative_noise_levels[t-1], dim=len(absorb_vocab[track]))

                    # factor1 = generator_matrix_neg @ s['seq'] # [V,V] @ [B, L, V] -> [B, L, V]
                    factor1 = torch.einsum('ij,blj->bli', generator_matrix_neg, s[track]) # [V,V] @ [B, L, V] -> [B, L, V]

                    # Indices corresponding to BEOSPANK do not matter.  
                    indices = torch.where(batch.beospank[track], 0, batch.masked_data[track])
                    # Replace masks with -1 to account for new ordering of vocabulary.
                    indices = torch.where(indices == absorb_tokens[track], -1, indices)

                    factor2 = generator_matrix_pos[indices] # [B, L, V]

                    # Element-wise multiplication?
                    product = factor1 * factor2
                    product = product.squeeze(0) # [L, V]

                    if t > MIN:
                        samples = torch.multinomial(product, 1, replacement=False).squeeze(-1) # [L]

                        # Only unmasked positions should be affected.
                        # batch.masked_data[track] = torch.where(batch.masks[track], samples, batch.masked_data[track])
                        flipping = batch.masks[track] & (samples != absorb_tokens[track])

                        batch.masked_data[track] = torch.where(flipping, samples, batch.masked_data[track])
                        batch.masks[track] = batch.masks[track] & ~flipping
                    
                    elif t == MIN:
                        # Final denoising timestep.  We need to unmask everything NOW.
                        product = product[:,:-1] # [L, V-1], omitting mask token as a possibility
                        # multinomial can accept improperly normalized input -- no need to sum to one.
                        samples = torch.multinomial(product, 1, replacement=False).squeeze(-1) # [L] 

                        flipping = batch.masks[track]

                        batch.masked_data[track] = torch.where(flipping, samples, batch.masked_data[track])
                        batch.masks[track] = torch.zeros_like(batch.masks[track]).bool()
                    else:
                        assert False

        #TODO: should we be unmasking unmasked_data instead of masked_data?
        return batch

    else:
        raise ValueError(f"Unknown corruption mode: {corruption_mode}")
