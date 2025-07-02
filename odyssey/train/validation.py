import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random
from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.transformer import TransformerTrunk
from src.models.autoencoder import FSQEncoder
from src.dataset import ProteinDataset
from src.dataloader import _get_training_dataloader, MaskedBatch, worker_init_fn
from src.losses import score_entropy_loss
from src.dataloader import _get_noise_levels

from mlm_step import mlm_step
from discrete_diffusion_step import discrete_diffusion_step
from fsq_step import stage_1_step, stage_2_step

def compute_mask_prob_for_time(t: int, diffusion_cfg: DiffusionConfig) -> float:
    """Compute mask probability for a given time index using the diffusion schedule."""
    # Get noise levels
    inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
        diffusion_cfg.sigma_min,
        diffusion_cfg.sigma_max,
        diffusion_cfg.num_timesteps,
        schedule_type=diffusion_cfg.noise_schedule
    )
    
    # Get cumulative noise at time t
    cumulative_noise = cumulative_noise_levels[t]
    
    # Compute mask probability
    mask_prob = 1 - torch.exp(-cumulative_noise)
    
    return mask_prob.item()

def evaluate_mlm_model(model: TransformerTrunk, batch: MaskedBatch, model_type: str, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> Dict[str, float]:
    """Evaluate an MLM model using cross-entropy loss on a single batch."""
    with torch.no_grad():
        retval = mlm_step({model_type : model}, optimizer=None, batch=batch, model_cfg=model_cfg, train_cfg=train_cfg, train_mode=False)
        return retval[model_type]

def evaluate_diffusion_model(model: TransformerTrunk, batch: MaskedBatch, model_type: str, timestep: int, model_cfg: ModelConfig, diffusion_cfg: DiffusionConfig,
                           train_cfg: TrainingConfig, device: torch.device) -> Dict[str, float]:
    """Evaluate a diffusion model using score entropy loss on a single batch."""

    B = batch.masked_data['seq'].shape[0]
    psuedo_timestep_indices = torch.full((B,), timestep, dtype=torch.long)

    inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
        diffusion_cfg.sigma_min,
        diffusion_cfg.sigma_max,
        diffusion_cfg.num_timesteps,
        schedule_type=diffusion_cfg.noise_schedule
    )
    pseudo_cumulative_noise = cumulative_noise_levels[psuedo_timestep_indices].unsqueeze(1)
    pseudo_inst_noise = inst_noise_levels[psuedo_timestep_indices].unsqueeze(1)

    batch.metadata['pseudo_timestep_indices'] = psuedo_timestep_indices.to(device)
    batch.metadata['pseudo_cumulative_noise'] = pseudo_cumulative_noise.to(device)
    batch.metadata['pseudo_inst_noise'] = pseudo_inst_noise.to(device)

    with torch.no_grad():
        retval = discrete_diffusion_step({model_type: model}, optimizer=None, batch=batch, model_cfg=model_cfg, train_cfg=train_cfg, train_mode=False)
        return retval[model_type]



def validate(model_checkpoint_path: str, fsq_encoder_checkpoint_path: str = None):

    tracks = {'seq': False, 'struct': False, 'coords': True}
    min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
    dataset_mode = "backbone"
    step_fn = mlm_step
    data_dir = "../../../sample_data/1k.csv"


    ###########################################################################
    #  Model and Config loading from checkpoint
    ###########################################################################
    model, model_cfg, train_cfg = load_model_from_checkpoint(model_checkpoint_path, device)
    # If FSQ encoder needed, load it
    fsq_encoder = None
    if fsq_encoder_checkpoint_path:
        fsq_encoder, _ = load_model_from_checkpoint(fsq_encoder_checkpoint_path, device)

    ###########################################################################
    #  Data Loading
    ###########################################################################
    data_seed = model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create DataLoaders with fixed seed for consistent masking
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000)

    dataset = ProteinDataset(data_dir, mode=dataset_mode, max_length=model_cfg.max_len - 2)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size], generator=g_val)
    
    # Pass appropriate FSQ encoder
    val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder)


    ###########################################################################
    #  Validation Loop
    ###########################################################################

    # Prepare containers to store epoch-level history for CSV saving
    epoch_metrics = []  # List of dicts; one per epoch

    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            # Skip empty/None batches
            if batch_data is None: continue
                
            # Validate single model on batch
            val_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=False)
            
            # Accumulate validation metrics
            batch_size = len(batch_data.masked_data['coords'])
            for k, v in val_metrics.items():
                if k not in val_metrics_sum:
                    val_metrics_sum[k] = 0.0
                val_metrics_sum[k] += v  # v is already loss * batch_size
            val_sample_count += batch_size
        
        val_metrics = {k: v / val_sample_count for k, v in val_metrics_sum.items()}

    # Validation metrics
    print(f"  Val:")
    print(*(f"      {k}: \t{v:.3f}" for k, v in val_metrics.items()), sep="\n")
       
if __name__ == "__main__":
    path_to_model_checkpoint = '/workspace/demo/Odyssey/checkpoints/transformer_trunk/SC_discrete_diffusion_discrete_diffusion_model.pt'
    path_to_fsq_encoder_checkpoint = '/workspace/demo/Odyssey/checkpoints/fsq/SC_stage_1_discrete_diffusion_model.pt'
    validate(path_to_model_checkpoint, path_to_fsq_encoder_checkpoint) 