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
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple, Callable, Dict
import random
from types import SimpleNamespace
import argparse

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder, StandardTransformerBlock
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import MaskedBatch, SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader, worker_init_fn
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from fsq_step import stage_1_step, stage_2_step
from mlm_step import mlm_step
from discrete_diffusion_step import discrete_diffusion_step
from odyssey.src.configurations import *
from odyssey.src.config_loader import load_config
from odyssey.src.model_librarian import ensure_identical_parameters_transformers, ensure_identical_parameters_autoencoders, load_model_from_empty, load_model_from_checkpoint, save_model_checkpoint



def train(model_cfg, train_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # Use different masking strategies for stage 1 vs stage 2
    if model_cfg.style == "stage_1":
        tracks = {'seq': False, 'struct': False, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
    elif model_cfg.style == "stage_2":
        tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 0}
    elif model_cfg.style == "mlm":
        tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
    elif model_cfg.style == "discrete_diffusion":
        tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}

    load_fsq_encoder = model_cfg.style in {"mlm", "discrete_diffusion", "stage_2"}

    step_fns = {'stage_1': stage_1_step, 'stage_2': stage_2_step, 'mlm': mlm_step, 'discrete_diffusion': discrete_diffusion_step}
    step_fn = step_fns[model_cfg.style]

    # Set dataset mode based on style
    dataset_mode = "side_chain" if model_cfg.style == "stage_2" else "backbone"

    ###########################################################################
    #  Model Loading
    ###########################################################################
    # TODO: better printing
    print(f"Starting {model_cfg.style} training")
    print(f"Training model: {model_cfg.first_block_cfg}")
    print(f"Masking: {train_cfg.mask_config}")
    
    # Create model with fixed seed
    print(f"Creating {model_cfg.first_block_cfg.initials()} target model...")
    model = load_model_from_empty(model_cfg, device)
    fsq_encoder = None
    if load_fsq_encoder: fsq_encoder, _, _ = load_model_from_checkpoint(model_cfg.fsq_encoder_path, device)

    if isinstance(model, Autoencoder):
        if fsq_encoder is not None:
            model.encoder = fsq_encoder
    elif isinstance(model, TransformerTrunk):
        pass # FSQ encoder is not part of the Transformer Trunk model.

    # optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.learning_rate) 

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_cfg.first_block_cfg.initials()} total parameters: {total_params:,}")
    
    ###########################################################################
    #  Data Loading
    ###########################################################################
    # Set seed for dataset split
    data_seed = model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create DataLoaders with fixed seed for consistent masking
    g_train = torch.Generator()
    g_train.manual_seed(data_seed)
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000)

    dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_mode, max_length=model_cfg.max_len - 2)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    # We use g_val as the generator of the split, as this is more consistent with validation scripts in which there is no training genreator.
    #  Importantly, the split needs to be reproducible between this train script and validation.py -- this way we can legitimately validate on a common data directory without data leakage.
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g_val)
    
    # Pass appropriate FSQ encoder
    train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder)
    val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder)

    ###########################################################################
    #  Training Loop
    ###########################################################################

    # Prepare containers to store epoch-level history for CSV saving
    epoch_metrics = []  # List of dicts; one per epoch
    # -------------------- Training loop -------------------- #
    for epoch in range(train_cfg.max_epochs):
        # Reset epoch accumulators - now track sum and count separately
        train_metrics_sum = {}
        train_metrics_count = {}
        val_metrics_sum = {}
        val_metrics_count = {}
        
        # Training
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.max_epochs} [{model_cfg.first_block_cfg.initials()} Train]",
                 ascii=True, leave=True, ncols=150) as pbar:
            for batch_data in pbar:
                # Skip empty/None batches
                if batch_data is None: continue
                
                # Train single model on batch
                train_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=True)
                
                # Accumulate metrics (step functions now return (value, count) tuples)
                for k, (value, count) in train_metrics.items():
                    if k not in train_metrics_sum:
                        train_metrics_sum[k] = 0.0
                        train_metrics_count[k] = 0
                    train_metrics_sum[k] += value * count  # value is per-element, so multiply by count
                    train_metrics_count[k] += count
                
                # Update progress bar with running average
                running_avg = {k: train_metrics_sum[k] / train_metrics_count[k] for k in train_metrics_sum.keys()}
                prefix = model_cfg.first_block_cfg.initials()
                pbar.set_postfix({f"{prefix}_{k}": f"{v:.3f}" for k, v in running_avg.items()})
        
        # Calculate final training epoch averages
        epoch_train_metrics = {k: train_metrics_sum[k] / train_metrics_count[k] for k in train_metrics_sum.keys()}
        
        # -------------------- Validation -------------------- #
        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                # Skip empty/None batches
                if batch_data is None: continue
                    
                # Validate single model on batch
                val_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=False)
                
                # Accumulate validation metrics (step functions now return (value, count) tuples)
                for k, (value, count) in val_metrics.items():
                    if k not in val_metrics_sum:
                        val_metrics_sum[k] = 0.0
                        val_metrics_count[k] = 0
                    val_metrics_sum[k] += value * count  # value is per-element, so multiply by count
                    val_metrics_count[k] += count
        
        # Calculate final validation epoch averages
        epoch_val_metrics = {k: val_metrics_sum[k] / val_metrics_count[k] for k in val_metrics_sum.keys()}
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{train_cfg.max_epochs} - {model_cfg.first_block_cfg.initials()}:")
        # Training metrics
        print(f"  Train:")
        print(*(f"      {k}: \t{v:.3f}" for k, v in epoch_train_metrics.items()), sep="\n")
        
        # Validation metrics
        print(f"  Val:")
        print(*(f"      {k}: \t{v:.3f}" for k, v in epoch_val_metrics.items()), sep="\n")

        # -------------------- Store metrics for CSV -------------------- #
        # On first epoch, create ordered list of keys
        if epoch == 0:
            metric_names = sorted(list(epoch_train_metrics.keys()))  # deterministic
            csv_header = [f"train_{k}" for k in metric_names] + [f"val_{k}" for k in metric_names]
        # Build row with metric values in the same order as header
        row = [epoch_train_metrics[k] for k in metric_names] + [epoch_val_metrics[k] for k in metric_names]
        epoch_metrics.append(row)

    final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_cfg.first_block_cfg.initials()}_{model_cfg.style}_{train_cfg.mask_config}_model.pt"

    save_model_checkpoint(final_checkpoint_path, model, model_cfg, train_cfg, optimizer)
    print(f"\nSaved final checkpoint for {model_cfg.first_block_cfg.initials()}")
    
    # -------------------- Save epoch history to a single CSV -------------------- #
    metrics_array = np.array(epoch_metrics)  # shape [E, num_cols]
    history_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_cfg.first_block_cfg.initials()}_{model_cfg.style}_{train_cfg.mask_config}_epoch_metrics.csv"
    np.savetxt(history_csv_path, metrics_array, delimiter=',', header=','.join(csv_header), comments='')
    print(f"Saved epoch metrics to {history_csv_path}")

    return model, epoch_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Odyssey models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    model_cfg, train_cfg, model_config_dict, train_config_dict = load_config(args.config)
    train(model_cfg, train_cfg)
