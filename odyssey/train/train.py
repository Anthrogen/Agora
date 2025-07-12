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
from odyssey.src.configurations import TransformerConfig, TrainingConfig
from odyssey.train.fsq_step import stage_1_step, stage_2_step
from odyssey.train.mlm_step import mlm_step
from odyssey.train.discrete_diffusion_step import discrete_diffusion_step
from odyssey.src.configurations import *
from odyssey.src.config_loader import load_config, load_multi_configs
from odyssey.src.model_librarian import ensure_identical_parameters_transformers, ensure_identical_parameters_autoencoders, load_model_from_empty, load_model_from_checkpoint, save_model_checkpoint, save_summary_history

from odyssey.train.yaml_expander import expand_yaml_to_directory
from odyssey.train.generate_experiment_map import generate_experiment_map


def train(model_cfg_list: List[TransformerConfig], train_cfg_list: List[TrainingConfig], callback=None):
    # If a user just passes in a single set of configs, listify them.
    if isinstance(model_cfg_list, TransformerConfig) and isinstance(train_cfg_list, TrainingConfig):
        model_cfg_list = [model_cfg_list]
        train_cfg_list = [train_cfg_list]

    assert len(model_cfg_list) == len(train_cfg_list), "model_cfg_list and train_cfg_list must have the same length"
    num_models = len(model_cfg_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create checkpoint directories for all models
    for train_cfg in train_cfg_list:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Lists to store model-specific objects
    models = []
    optimizers = []
    step_fns = []
    tracks_list = []
    min_unmasked_list = []
    dataset_modes = []
    
    # Set up each model
    for i, (model_cfg, train_cfg) in enumerate(zip(model_cfg_list, train_cfg_list)):
        # Use different masking strategies for stage 1 vs stage 2
        if model_cfg.style == "stage_1":
            tracks = {'seq': False, 'struct': False, 'coords': True, 'ss8': False, 'sasa': False, 'global_annotation': False, 'per_residue_annotation': False, 'plddt': False}
            min_unmasked = {'seq': 0, 'coords': 1}
        elif model_cfg.style == "stage_2":
            tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': False, 'sasa': False, 'global_annotation': False, 'per_residue_annotation': False, 'plddt': False}
            min_unmasked = {'seq': 0, 'coords': 0}
        elif model_cfg.style == "mlm":
            tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'global_annotation': True, 'per_residue_annotation': True, 'plddt': True}
            min_unmasked = {'seq': 0, 'coords': 1}
        elif model_cfg.style == "discrete_diffusion":
            tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'global_annotation': True, 'per_residue_annotation': True, 'plddt': True}
            min_unmasked = {'seq': 0, 'coords': 1}

        tracks_list.append(tracks)
        min_unmasked_list.append(min_unmasked)
        
        load_fsq_encoder = model_cfg.style in {"mlm", "discrete_diffusion", "stage_2"}
        step_fns_dict = {'stage_1': stage_1_step, 'stage_2': stage_2_step, 'mlm': mlm_step, 'discrete_diffusion': discrete_diffusion_step}
        step_fns.append(step_fns_dict[model_cfg.style])
        
        # Set dataset mode based on style
        dataset_mode = "side_chain" if model_cfg.style == "stage_2" else "backbone"
        dataset_modes.append(dataset_mode)
        
        ###########################################################################
        #  Model Loading
        ###########################################################################
        print(f"Model {i+1}/{num_models}: Starting {model_cfg.style} training")
        print(f"  Training model: {model_cfg.first_block_cfg}")
        print(f"  Masking: {train_cfg.mask_config}")
        
        # Create model with fixed seed
        print(f"  Creating {model_cfg.first_block_cfg.initials()} target model...")
        model = load_model_from_empty(model_cfg, device)
        fsq_encoder = None
        if load_fsq_encoder: 
            fsq_encoder, _, _ = load_model_from_checkpoint(model_cfg.fsq_encoder_path, device)
            if isinstance(model, Autoencoder):
                model.encoder = fsq_encoder
                
        models.append((model, fsq_encoder))
        
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.learning_rate)
        optimizers.append(optimizer)
        
        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {model_cfg.first_block_cfg.initials()} total parameters: {total_params:,}")
    
    ###########################################################################
    #  Data Loading
    ###########################################################################
    train_loaders = []
    val_loaders = []
    
    for i, (model_cfg, train_cfg) in enumerate(zip(model_cfg_list, train_cfg_list)):
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

        dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_modes[i], max_length=model_cfg.max_len - 2, max_length_global=model_cfg.max_len_global - 2)
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size

        # We use g_val as the generator of the split
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g_val)
        
        # Pass appropriate FSQ encoder
        model, fsq_encoder = models[i]
        train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                               batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, 
                                               worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                               fsq_encoder=fsq_encoder)
        val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                            batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, 
                                            worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                            fsq_encoder=fsq_encoder)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    ###########################################################################
    #  Training Loop
    ###########################################################################
    # Prepare containers to store epoch-level history for CSV saving
    epoch_metrics_all = [[] for _ in range(num_models)]  # List of lists
    csv_headers = [None] * num_models
    
    # Use first config for epoch count (they should all be the same)
    max_epochs = train_cfg_list[0].max_epochs
    
    # -------------------- Training loop -------------------- #
    for epoch in range(max_epochs):
        # Reset epoch accumulators for all models
        train_metrics_sum_all = [{} for _ in range(num_models)]
        train_metrics_count_all = [{} for _ in range(num_models)]
        val_metrics_sum_all = [{} for _ in range(num_models)]
        val_metrics_count_all = [{} for _ in range(num_models)]
        
        # Training phase for all models
        for model_idx in range(num_models):
            model, fsq_encoder = models[model_idx]
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            optimizer = optimizers[model_idx]
            train_loader = train_loaders[model_idx]
            step_fn = step_fns[model_idx]
            
            model.train()
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Model {model_idx+1}: {model_cfg.first_block_cfg.initials()} Train]",
                     ascii=True, leave=True, ncols=150) as pbar:
                for batch_data in pbar:
                    # Skip empty/None batches
                    if batch_data is None: continue
                    
                    # Train single model on batch
                    train_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=True)
                    
                    # Accumulate metrics (step functions now return (value, count) tuples)
                    for k, (value, count) in train_metrics.items():
                        if k not in train_metrics_sum_all[model_idx]:
                            train_metrics_sum_all[model_idx][k] = 0.0
                            train_metrics_count_all[model_idx][k] = 0
                        train_metrics_sum_all[model_idx][k] += value * count
                        train_metrics_count_all[model_idx][k] += count
                    
                    # Update progress bar with running average
                    running_avg = {k: train_metrics_sum_all[model_idx][k] / train_metrics_count_all[model_idx][k] 
                                  for k in train_metrics_sum_all[model_idx].keys()}
                    prefix = model_cfg.first_block_cfg.initials()
                    pbar.set_postfix({f"{prefix}_{k}": f"{v:.3f}" for k, v in running_avg.items()})

        # Validation phase for all models
        for model_idx in range(num_models):
            model, fsq_encoder = models[model_idx]
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            optimizer = optimizers[model_idx]
            val_loader = val_loaders[model_idx]
            step_fn = step_fns[model_idx]
            
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    # Skip empty/None batches
                    if batch_data is None: continue
                        
                    # Validate single model on batch
                    val_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=False)
                    
                    # Accumulate validation metrics (step functions now return (value, count) tuples)
                    for k, (value, count) in val_metrics.items():
                        if k not in val_metrics_sum_all[model_idx]:
                            val_metrics_sum_all[model_idx][k] = 0.0
                            val_metrics_count_all[model_idx][k] = 0
                        val_metrics_sum_all[model_idx][k] += value * count
                        val_metrics_count_all[model_idx][k] += count
        
        # Calculate and print epoch averages for all models
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{max_epochs} Summary:")
        print(f"{'='*80}")
        
        for model_idx in range(num_models):
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            
            # Calculate final epoch averages
            epoch_train_metrics = {k: train_metrics_sum_all[model_idx][k] / train_metrics_count_all[model_idx][k] 
                                 for k in train_metrics_sum_all[model_idx].keys()}
            epoch_val_metrics = {k: val_metrics_sum_all[model_idx][k] / val_metrics_count_all[model_idx][k] 
                               for k in val_metrics_sum_all[model_idx].keys()}
            
            # Print model summary
            print(f"\nModel {model_idx+1}: {model_cfg.first_block_cfg.initials()} ({model_cfg.style}):")
            print(f"  Train: " + " | ".join([f"{k}: {v:.3f}" for k, v in epoch_train_metrics.items()]))
            print(f"  Val:   " + " | ".join([f"{k}: {v:.3f}" for k, v in epoch_val_metrics.items()]))
            
            # Store metrics for CSV
            
            if epoch == 0:
                metric_names = sorted(list(epoch_train_metrics.keys()))
                csv_header = [f"train_{k}" for k in metric_names] + [f"val_{k}" for k in metric_names]
                csv_headers[model_idx] = csv_header
            
            row = [epoch_train_metrics[k] for k in metric_names] + [epoch_val_metrics[k] for k in metric_names]
            epoch_metrics_all[model_idx].append(row)

        if callback is not None:
            with torch.no_grad():
                ret = {'train_metrics': train_metrics_sum_all, 'val_metrics': val_metrics_sum_all, 'model': model}
                callback(ret)
            

    # Save final checkpoints and metrics for all models
    for model_idx in range(num_models):
        model, fsq_encoder = models[model_idx]
        model_cfg = model_cfg_list[model_idx]
        train_cfg = train_cfg_list[model_idx]
        optimizer = optimizers[model_idx]
        
        # final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_cfg.first_block_cfg.initials()}_{model_cfg.style}_{train_cfg.mask_config}_model.pt"
        final_checkpoint_path = Path(train_cfg.checkpoint_dir) / "model.pt"
        save_model_checkpoint(final_checkpoint_path, model, model_cfg, train_cfg, optimizer)
        print(f"\nSaved final checkpoint for Model {model_idx+1}")
        
        # Save epoch history to CSV
        metrics_array = np.array(epoch_metrics_all[model_idx])
        # history_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_cfg.first_block_cfg.initials()}_{model_cfg.style}_{train_cfg.mask_config}_epoch_metrics.csv"
        history_csv_path = Path(train_cfg.checkpoint_dir) / "history.csv"
        np.savetxt(history_csv_path, metrics_array, delimiter=',', header=','.join(csv_header), comments='')
        print(f"Saved epoch metrics to {history_csv_path}")

    return models, epoch_metrics_all, csv_headers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Odyssey models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Create expanded directory name based on config file - save to configs folder
    config_path = Path(args.config)
    yaml_name = config_path.stem
    # Go up to the project root and into configs/expanded (base directory)
    expanded_base_dir = Path(__file__).parent.parent.parent / "checkpoints"
    expanded_yaml_dir = expanded_base_dir / yaml_name
    
    # Expand the YAML file
    print(f"Expanding configuration file: {args.config}")
    num_generated = expand_yaml_to_directory(args.config, str(expanded_base_dir))
    
    if num_generated <= 0:
        print("Error: No configurations were generated")
        exit(1)
    
    print(f"Successfully generated {num_generated} configurations")
    
    # Generate experiment map
    print(f"Generating experiment map...")
    try:
        map_file = generate_experiment_map(args.config, str(expanded_yaml_dir))
        print(f"Generated experiment map: {map_file}")
    except Exception as e:
        print(f"Warning: Could not generate experiment map: {e}")
    
    # Verify the expanded directory exists and contains files
    if not expanded_yaml_dir.exists():
        print(f"ERROR: Expanded directory {expanded_yaml_dir} does not exist!")
        exit(1)
    
    # List contents of expanded directory for debugging
    print(f"Contents of {expanded_yaml_dir}:")
    for item in expanded_yaml_dir.iterdir():
        print(f"  {item}")
        if item.is_dir():
            for subitem in item.iterdir():
                print(f"    {subitem}")
    
    # Load all expanded configs and train
    model_cfg_list, train_cfg_list = load_multi_configs(str(expanded_yaml_dir))
    _, epoch_metrics_all, csv_headers = train(model_cfg_list, train_cfg_list)

    base_dir = Path(train_cfg_list[0].checkpoint_dir).parent
    summary_history_csv_path = base_dir / "summary_history.csv"
    save_summary_history(epoch_metrics_all, summary_history_csv_path, header_list=csv_headers, name_prefix=base_dir.stem)
    print(f"Saved summary history to {summary_history_csv_path}")
