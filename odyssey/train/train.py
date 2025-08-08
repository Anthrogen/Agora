import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple, Callable, Dict
import random
from types import SimpleNamespace
import argparse
import yaml

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder, StandardTransformerBlock
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import _get_training_dataloader, worker_init_fn
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

class FakeScheduler():
    def __init__(self): pass
    def step(self): pass

def create_linear_decay(base_learning_rate: float, min_learning_rate: float, num_epochs_decay: int, num_epochs_warmup: int):
    def lr_lambda(current_step): # One step = one forward pass through the model
        if current_step < num_epochs_warmup: # Linear warmup from 0 to base_lr
            warmup_lr = (current_step / num_epochs_warmup) * base_learning_rate
            return warmup_lr / base_learning_rate
        elif current_step < num_epochs_decay: # Linear decay from base_lr to min_lr
            # Calculate decay progress after warmup period
            decay_start = num_epochs_warmup
            decay_progress = (current_step - decay_start) / (num_epochs_decay - decay_start)
            decay_lr = base_learning_rate - decay_progress * (base_learning_rate - min_learning_rate)
            return decay_lr / base_learning_rate
        else: # Stay at min_lr after decay period
            return min_learning_rate / base_learning_rate
    return lr_lambda

def create_decay(base_learning_rate: float, min_learning_rate: float, num_epochs_decay: int):
    def lr_lambda(current_step): # One step = one forward pass through the model
        if current_step < num_epochs_decay: # Linear decay from base_lr to min_lr
            decay_progress = current_step / num_epochs_decay
            decay_lr = base_learning_rate - decay_progress * (base_learning_rate - min_learning_rate)
            return decay_lr / base_learning_rate
        else: # Stay at min_lr after decay period
            return min_learning_rate / base_learning_rate

    return lr_lambda

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
    schedulers = []
    step_fns = []
    tracks_list = []
    min_unmasked_list = []
    dataset_modes = []
    
    # Set up each model
    for i, (model_cfg, train_cfg) in enumerate(zip(model_cfg_list, train_cfg_list)):
        # Use different masking strategies for stage 1 vs stage 2
        if model_cfg.style == "stage_1":
            tracks = {'seq': False, 'struct': False, 'coords': True, 'ss8': False, 'sasa': False, 'orthologous_groups': False, 'semantic_description': False, 'domains': False, 'plddt': False}
            min_unmasked = {'seq': 1, 'coords': 1}
        elif model_cfg.style == "stage_2":
            tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': False, 'sasa': False, 'orthologous_groups': False, 'semantic_description': False, 'domains': False, 'plddt': False}
            min_unmasked = {'seq': 0, 'coords': 0}
        elif model_cfg.style == "mlm":
            tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
            min_unmasked = {'seq': 1, 'coords': 1}
        elif model_cfg.style == "discrete_diffusion":
            tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
            min_unmasked = {'seq': 1, 'coords': 1}

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
        print(f"  Training model: {model_cfg.initials()}")
        print(f"  Masking: {train_cfg.mask_config}")
        
        if train_cfg.jump_start is None: # No jump start provided.  We will cold-start the model. Create model with fixed seed
            print(f"  Creating {model_cfg.initials()} target model...")
            model = load_model_from_empty(model_cfg, device)
        else: # Jump start provided.  Load model from checkpoint.
            print(f"  Loading {model_cfg.initials()} target model from {train_cfg.jump_start}...")
            model, model_cfg_loaded, train_cfg_loaded= load_model_from_checkpoint(train_cfg.jump_start, device)
            # TODO: Assert model_cfg from load and model_cfg "similar enough", but the train_cfg can be different.
            model_cfg = model_cfg_loaded  # Replace the original model_cfg with the loaded one
            model_cfg_list[i] = model_cfg  # Update the list to persist changes

        autoencoder = None
        if load_fsq_encoder: 
            autoencoder, autoencoder_model_cfg, _ = load_model_from_checkpoint(model_cfg.autoencoder_path, device)
            autoencoder.encoder.eval(); autoencoder.encoder.requires_grad_(False)
            autoencoder.quantizer.eval(); autoencoder.quantizer.requires_grad_(False)  # Also freeze quantizer
            if isinstance(model, Autoencoder): 
                model.encoder = autoencoder.encoder  # Update the encoder to match the loaded encoder
                model.quantizer = autoencoder.quantizer  # Update the quantizer to match the loaded quantizer
                model_cfg.encoder_cfg = autoencoder_model_cfg.encoder_cfg  # Update the encoder_cfg to match the loaded encoder_cfg
                model_cfg_list[i] = model_cfg  # Ensure the updated config is saved to the list
                
        models.append((model, autoencoder))

        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {model_cfg.initials()} total parameters: {total_params:,}")

        ###########################################################################
        # Initializing optimizer and scheduler
        ###########################################################################
        if isinstance(train_cfg.optim_schedule_config, FlatSchedulerConfig):
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.optim_schedule_config.learning_rate)
            scheduler = FakeScheduler()
        elif isinstance(train_cfg.optim_schedule_config, LinearDecaySchedulerConfig):
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.optim_schedule_config.base_learning_rate)
            lr_lambda = create_linear_decay(train_cfg.optim_schedule_config.base_learning_rate, train_cfg.optim_schedule_config.min_learning_rate, train_cfg.optim_schedule_config.num_epochs_decay, train_cfg.optim_schedule_config.num_epochs_warmup)
            scheduler = LambdaLR(optimizer, lr_lambda)
        elif isinstance(train_cfg.optim_schedule_config, DecaySchedulerConfig):
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.optim_schedule_config.base_learning_rate)
            lr_lambda = create_decay(train_cfg.optim_schedule_config.base_learning_rate, train_cfg.optim_schedule_config.min_learning_rate, train_cfg.optim_schedule_config.num_epochs_decay)
            scheduler = LambdaLR(optimizer, lr_lambda)
        else: raise ValueError(f"Invalid optimizer schedule: {train_cfg.optim_schedule_config}")

        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
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

        dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_modes[i], max_length=model_cfg.max_len - 2, max_length_orthologous_groups=model_cfg.max_len_orthologous_groups - 2, max_length_semantic_description=model_cfg.max_len_semantic_description - 2) # , critical_tracks=tracks_list[i])
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size

        # We use g_val as the generator of the split
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g_val)
        
        # Pass appropriate FSQ encoder
        model, autoencoder = models[i]
        train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                               batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, 
                                               worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                               autoencoder=autoencoder)
        val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                            batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, 
                                            worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                            autoencoder=autoencoder)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    ###########################################################################
    #  Set default values for checkpoint_freq and max_steps_val if None
    ###########################################################################
    for i, train_cfg in enumerate(train_cfg_list):
        # Set checkpoint_freq to number of training batches per epoch if None
        if train_cfg.checkpoint_freq is None:
            train_cfg.checkpoint_freq = len(train_loaders[i])
            print(f"  Model {i+1}: checkpoint_freq set to {train_cfg.checkpoint_freq} (once per epoch)")
        
        # Set max_steps_val to number of validation batches if None  
        if train_cfg.max_steps_val is None:
            train_cfg.max_steps_val = len(val_loaders[i])
            print(f"  Model {i+1}: max_steps_val set to {train_cfg.max_steps_val} (all validation batches)")

    ###########################################################################
    #  Training Loop
    ###########################################################################
    # Prepare containers to store epoch-level history for CSV saving
    epoch_metrics_all = [[] for _ in range(num_models)]  # List of lists
    csv_headers = [None] * num_models
    
    # Use first config for epoch count (they should all be the same)
    max_epochs = train_cfg_list[0].max_epochs
    
    # -------------------- Training loop -------------------- #
    steps = [0] * num_models  # Track steps per model
    
    for epoch in range(max_epochs):
        # Reset epoch accumulators for all models (for callback)
        train_metrics_sum_all = [{} for _ in range(num_models)]
        train_metrics_count_all = [{} for _ in range(num_models)]
        val_metrics_sum_all = [{} for _ in range(num_models)]
        val_metrics_count_all = [{} for _ in range(num_models)]
        
        # Training phase for all models
        for model_idx in range(num_models):
            model, autoencoder = models[model_idx]
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            optimizer = optimizers[model_idx]
            scheduler = schedulers[model_idx]
            train_loader = train_loaders[model_idx]
            val_loader = val_loaders[model_idx]
            step_fn = step_fns[model_idx]
            
            model.train()
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Model {model_idx+1}: {model_cfg.initials()} Train]", ascii=True, leave=True, ncols=200) as pbar:
                for batch_data in pbar:
                    steps[model_idx] += 1
                    # Skip empty/None batches
                    if batch_data is None: continue
                    
                    # Train single model on batch
                    train_metrics = step_fn(model, optimizer, scheduler, batch_data, model_cfg, train_cfg, train_mode=True)
                    
                    # Accumulate metrics for epoch-level callback
                    for k, (value, count) in train_metrics.items():
                        if k not in train_metrics_sum_all[model_idx]:
                            train_metrics_sum_all[model_idx][k] = 0.0
                            train_metrics_count_all[model_idx][k] = 0
                        train_metrics_sum_all[model_idx][k] += value * count
                        train_metrics_count_all[model_idx][k] += count
                    
                    # Extract single-batch training metrics for display and checkpointing
                    epoch_train_metrics = {k: value for k, (value, count) in train_metrics.items()}
                    
                    # Update progress bar with current batch metrics
                    prefix = model_cfg.initials()
                    pbar.set_postfix({f"{prefix}_{k}": f"{v:.3f}" for k, v in epoch_train_metrics.items()})

                    # Save periodic checkpoints and calculate validation metrics
                    if train_cfg.checkpoint_freq > 0 and steps[model_idx] % train_cfg.checkpoint_freq == 0:
                        print(f"\nSaving checkpoint at step {steps[model_idx]}...")
                        
                        # Save checkpoint for current model
                        checkpoint_path = Path(train_cfg.checkpoint_dir) / f"checkpoint_step_{steps[model_idx]}.pt"
                        save_model_checkpoint(checkpoint_path, model, model_cfg, train_cfg, optimizer)
                        print(f"Saved checkpoint for Model {model_idx+1} at {checkpoint_path}")

                        # Run validation for current model
                        model.eval()
                        val_metrics_sum = {}
                        val_metrics_count = {}
                        
                        with torch.no_grad():
                            for idx_val, val_batch_data in enumerate(val_loader):
                                if idx_val >= train_cfg.max_steps_val: break

                                # Skip empty/None batches
                                if val_batch_data is None: continue
                                    
                                # Validate single model on batch
                                val_metrics = step_fn(model, optimizer, scheduler, val_batch_data, model_cfg, train_cfg, train_mode=False)
                                
                                # Accumulate validation metrics across validation batches
                                for k, (value, count) in val_metrics.items():
                                    if k not in val_metrics_sum:
                                        val_metrics_sum[k] = 0.0
                                        val_metrics_count[k] = 0
                                    val_metrics_sum[k] += value * count
                                    val_metrics_count[k] += count
                                    
                                    # Also accumulate for epoch-level callback
                                    if k not in val_metrics_sum_all[model_idx]:
                                        val_metrics_sum_all[model_idx][k] = 0.0
                                        val_metrics_count_all[model_idx][k] = 0
                                    val_metrics_sum_all[model_idx][k] += value * count
                                    val_metrics_count_all[model_idx][k] += count
                        
                        # Calculate validation averages
                        epoch_val_metrics = {k: val_metrics_sum[k] / val_metrics_count[k] 
                                           for k in val_metrics_sum.keys()}
                        
                        # Create row with current step's training metrics and validation metrics
                        metric_names = sorted(list(epoch_train_metrics.keys()))
                        
                        # Store metrics for CSV headers (only if not set yet)
                        if csv_headers[model_idx] is None:
                            csv_header = [f"train_{k}" for k in metric_names] + [f"val_{k}" for k in metric_names]
                            csv_headers[model_idx] = csv_header
                        
                        row = [epoch_train_metrics[k] for k in metric_names] + [epoch_val_metrics[k] for k in metric_names]
                        epoch_metrics_all[model_idx].append(row)
                        
                        # Set model back to training mode
                        model.train()
        
        # Calculate and print epoch averages for all models (for callback)
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{max_epochs} Summary:")
        print(f"{'='*80}")
        
        for model_idx in range(num_models):
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            
            # Calculate final epoch averages
            epoch_train_metrics_avg = {k: train_metrics_sum_all[model_idx][k] / train_metrics_count_all[model_idx][k] 
                                     for k in train_metrics_sum_all[model_idx].keys()}
            epoch_val_metrics_avg = {k: val_metrics_sum_all[model_idx][k] / val_metrics_count_all[model_idx][k] 
                                   for k in val_metrics_sum_all[model_idx].keys()}
            
            # Print model summary
            print(f"\nModel {model_idx+1}: {model_cfg.initials()} ({model_cfg.style}):")
            print(f"  Train: " + " | ".join([f"{k}: {v:.3f}" for k, v in epoch_train_metrics_avg.items()]))
            if epoch_val_metrics_avg:  # Only check val metrics since they might be empty if no checkpoints occurred
                print(f"  Val:   " + " | ".join([f"{k}: {v:.3f}" for k, v in epoch_val_metrics_avg.items()]))
            
        if callback is not None:
            with torch.no_grad():
                ret = {'train_metrics': train_metrics_sum_all, 'val_metrics': val_metrics_sum_all, 'model': model}
                callback(ret)

    # Save final checkpoints and metrics for all models
    for model_idx in range(num_models):
        model, autoencoder = models[model_idx]
        model_cfg = model_cfg_list[model_idx]
        train_cfg = train_cfg_list[model_idx]
        optimizer = optimizers[model_idx]
        
        final_checkpoint_path = Path(train_cfg.checkpoint_dir) / "model.pt"
        save_model_checkpoint(final_checkpoint_path, model, model_cfg, train_cfg, optimizer)
        print(f"\nSaved final checkpoint for Model {model_idx+1} to {final_checkpoint_path}")
        
        # Save step history to CSV
        metrics_array = np.array(epoch_metrics_all[model_idx])
        history_csv_path = Path(train_cfg.checkpoint_dir) / "history.csv"
        np.savetxt(history_csv_path, metrics_array, delimiter=',', header=','.join(csv_headers[model_idx]), comments='')
        print(f"Saved step metrics to {history_csv_path}")

    return models, epoch_metrics_all, csv_headers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Odyssey models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Create expanded directory name based on config file and checkpoint directory
    config_path = Path(args.config)
    yaml_name = config_path.stem
    
    # Load config temporarily to get the checkpoint directory structure
    with open(args.config, 'r') as f: temp_config = yaml.safe_load(f)
    
    # Extract the checkpoint directory from the config
    checkpoint_dir = temp_config['train_cfg']['training_cfg']['checkpoint_dir']
    checkpoint_path = Path(checkpoint_dir)
    
    # Use the same directory structure as the checkpoints for YAML files
    expanded_yaml_dir = checkpoint_path / yaml_name
    
    # Expand the YAML file
    print(f"Expanding configuration file: {args.config}")
    num_generated = expand_yaml_to_directory(args.config, str(checkpoint_path))
    
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
