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
from typing import Optional, Tuple, Callable, Dict, List
import random
from types import SimpleNamespace
import argparse

# FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
import functools
import torch.distributed as dist

# Wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")

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

# Import transformer block classes for FSDP wrapping policy
from odyssey.src.models.blocks import (
    StandardTransformerBlock,
    GeometricTransformerBlock,
    ReflexiveTransformerBlock,
    ConsensusTransformerBlock
)

class FakeScheduler():
    def __init__(self): pass
    def step(self): pass

def create_transformer_block_wrap_policy():
    """Create FSDP wrapping policy that only wraps transformer blocks.
    
    This policy will wrap only the transformer block classes to avoid
    double wrapping issues with other model components.
    """
    def lambda_policy_fn(module):
        # Only wrap the transformer block modules
        return isinstance(module, (
            StandardTransformerBlock,
            GeometricTransformerBlock,
            ReflexiveTransformerBlock,
            ConsensusTransformerBlock
        ))
    
    return functools.partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda_policy_fn
    )

def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(rank)
        return True, rank, world_size
    return False, 0, 1

def reduce_metrics_across_gpus(metrics_dict, world_size, rank=0, debug=False):
    """Reduce metrics across all GPUs by averaging.
    
    Args:
        metrics_dict: Dictionary of metrics to reduce
        world_size: Number of GPUs
        rank: Current GPU rank
        debug: Whether to print debug information
    
    Returns:
        Dictionary of reduced (averaged) metrics
    """
    if world_size == 1:
        return metrics_dict
    
    reduced_metrics = {}
    
    # Debug: collect all metrics from all GPUs for printing
    if debug and 'loss' in metrics_dict:
        # First, gather all loss values to rank 0 for debugging
        loss_tensor = torch.tensor(metrics_dict['loss'], dtype=torch.float32).cuda()
        if rank == 0:
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.gather(loss_tensor, gathered_losses, dst=0)
            print(f"\n[DEBUG] Loss values from all GPUs:")
            for i, loss in enumerate(gathered_losses):
                print(f"  GPU {i}: {loss.item():.6f}")
        else:
            dist.gather(loss_tensor, dst=0)
    
    for key, value in metrics_dict.items():
        # Convert to tensor if not already
        tensor_value = torch.tensor(value, dtype=torch.float32).cuda()
        # All-reduce across GPUs (sum)
        dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
        # Average by dividing by world size
        reduced_metrics[key] = (tensor_value / world_size).item()
    
    # Debug: print averaged loss
    if debug and rank == 0 and 'loss' in reduced_metrics:
        print(f"  Averaged loss: {reduced_metrics['loss']:.6f}")
        print(f"  (Sum: {reduced_metrics['loss'] * world_size:.6f}, divided by {world_size} GPUs)\n")
    
    return reduced_metrics

def create_linear_decay(base_learning_rate: float, min_learning_rate: float, num_epochs_decay: int, num_epochs_warmup: int):
    def lr_lambda(current_step): # One step = one forward pass through the model
        if current_step < num_epochs_warmup: # Linear warmup from min_lr to base_lr
            warmup_progress = current_step / num_epochs_warmup
            warmup_lr = min_learning_rate + warmup_progress * (base_learning_rate - min_learning_rate)
            return warmup_lr / base_learning_rate
        elif current_step < (num_epochs_warmup + num_epochs_decay): # Linear decay from base_lr to min_lr
            decay_step = current_step - num_epochs_warmup
            decay_progress = decay_step / num_epochs_decay
            decay_lr = base_learning_rate - decay_progress * (base_learning_rate - min_learning_rate)
            return decay_lr / base_learning_rate
        else: # Stay at min_lr after decay period
            return min_learning_rate / base_learning_rate

    return lr_lambda

def train(model_cfg_list: List[TransformerConfig], train_cfg_list: List[TrainingConfig], callback=None, use_fsdp: bool = True, use_wandb: bool = True):
    # If a user just passes in a single set of configs, listify them.
    if isinstance(model_cfg_list, TransformerConfig) and isinstance(train_cfg_list, TrainingConfig):
        model_cfg_list = [model_cfg_list]
        train_cfg_list = [train_cfg_list]

    assert len(model_cfg_list) == len(train_cfg_list), "model_cfg_list and train_cfg_list must have the same length"
    num_models = len(model_cfg_list)
    
    # Setup distributed training if available
    is_distributed, rank, world_size = setup_distributed()
    
    # Set device based on distributed setup
    if is_distributed:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fsdp = False  # Disable FSDP if not distributed
    
    # Initialize wandb (only on rank 0)
    wandb_run = None
    if WANDB_AVAILABLE and use_wandb and (not is_distributed or rank == 0):
        # Create a run name based on the first model config
        run_name = f"{model_cfg_list[0].initials()}_{model_cfg_list[0].style}"
        if len(model_cfg_list) > 1:
            run_name += f"_and_{len(model_cfg_list)-1}_more"
        
        # Initialize wandb
        wandb_run = wandb.init(
            project="odyssey-training",
            name=run_name,
            config={
                "num_models": num_models,
                "model_configs": [asdict(cfg) for cfg in model_cfg_list],
                "training_configs": [asdict(cfg) for cfg in train_cfg_list],
                "distributed": is_distributed,
                "world_size": world_size,
                "use_fsdp": use_fsdp and is_distributed,
            },
            reinit=True
        )
    
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
        if not is_distributed or rank == 0:
            print(f"Model {i+1}/{num_models}: Starting {model_cfg.style} training")
            print(f"  Training model: {model_cfg.initials()}")
            print(f"  Masking: {train_cfg.mask_config}")
        
        if train_cfg.jump_start is None: # No jump start provided.  We will cold-start the model. Create model with fixed seed
            if not is_distributed or rank == 0:
                print(f"  Creating {model_cfg.initials()} target model...")
            model = load_model_from_empty(model_cfg, device)
        else: # Jump start provided.  Load model from checkpoint.
            if not is_distributed or rank == 0:
                print(f"  Loading {model_cfg.initials()} target model from {train_cfg.jump_start}...")
            model, model_cfg_loaded, train_cfg_loaded= load_model_from_checkpoint(train_cfg.jump_start, device)
            # TODO: Assert model_cfg from load and model_cfg "similar enough", but the train_cfg can be different.
            model_cfg = model_cfg_loaded  # Replace the original model_cfg with the loaded one
            model_cfg_list[i] = model_cfg  # Update the list to persist changes

        autoencoder = None
        if load_fsq_encoder: 
            autoencoder, autoencoder_model_cfg, _ = load_model_from_checkpoint(model_cfg.autoencoder_path, device)
            autoencoder.encoder.eval(); autoencoder.encoder.requires_grad_(False)
            if isinstance(model, Autoencoder): 
                model.encoder = autoencoder.encoder  # Update the encoder to match the loaded encoder
                model_cfg.encoder_cfg = autoencoder_model_cfg.encoder_cfg  # Update the encoder_cfg to match the loaded encoder_cfg
                model_cfg_list[i] = model_cfg  # Ensure the updated config is saved to the list
        
        # Wrap model with FSDP if enabled
        if use_fsdp and is_distributed:
            if rank == 0:
                print(f"  Wrapping model with FSDP (only transformer blocks)...")
            
            # Debug: Show which modules will be wrapped
            if rank == 0:
                print("  Modules to be wrapped by FSDP:")
                for name, module in model.named_modules():
                    if isinstance(module, (StandardTransformerBlock, GeometricTransformerBlock, 
                                         ReflexiveTransformerBlock, ConsensusTransformerBlock)):
                        print(f"    - {name}: {module.__class__.__name__}")
            
            # Create the wrapping policy that only wraps transformer blocks
            auto_wrap_policy = create_transformer_block_wrap_policy()
            
            # FSDP configuration
            fsdp_config = dict(
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=None,  # Can be enabled if needed
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=False),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=torch.cuda.current_device(),
                sync_module_states=True,  # Ensure all ranks have same initial weights
            )
            
            model = FSDP(model, **fsdp_config)
            if rank == 0:
                print(f"  FSDP wrapping complete")
                
        models.append((model, autoencoder))

        # Print parameter count (only on rank 0)
        total_params = sum(p.numel() for p in model.parameters())
        if not is_distributed or rank == 0:
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

        dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_modes[i], max_length=model_cfg.max_len - 2, max_length_global=model_cfg.max_len_global - 2) # , critical_tracks=tracks_list[i])
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
            model, autoencoder = models[model_idx]
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            optimizer = optimizers[model_idx]
            scheduler = schedulers[model_idx]
            train_loader = train_loaders[model_idx]
            step_fn = step_fns[model_idx]
            
            model.train()
            # Only show progress bar on rank 0
            disable_tqdm = is_distributed and rank != 0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Model {model_idx+1}: {model_cfg.initials()} Train]",
                     ascii=True, leave=True, ncols=150, disable=disable_tqdm) as pbar:
                for batch_data in pbar:
                    # Skip empty/None batches
                    if batch_data is None: continue
                    
                    # Train single model on batch
                    train_metrics = step_fn(model, optimizer, scheduler, batch_data, model_cfg, train_cfg, train_mode=True)
                    
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
                    prefix = model_cfg.initials()
                    pbar.set_postfix({f"{prefix}_{k}": f"{v:.3f}" for k, v in running_avg.items()})

        # Validation phase for all models
        for model_idx in range(num_models):
            model, autoencoder = models[model_idx]
            model_cfg = model_cfg_list[model_idx]
            train_cfg = train_cfg_list[model_idx]
            optimizer = optimizers[model_idx]
            scheduler = schedulers[model_idx]
            val_loader = val_loaders[model_idx]
            step_fn = step_fns[model_idx]
            
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    # Skip empty/None batches
                    if batch_data is None: continue
                        
                    # Validate single model on batch
                    val_metrics = step_fn(model, optimizer, scheduler, batch_data, model_cfg, train_cfg, train_mode=False)
                    
                    # Accumulate validation metrics (step functions now return (value, count) tuples)
                    for k, (value, count) in val_metrics.items():
                        if k not in val_metrics_sum_all[model_idx]:
                            val_metrics_sum_all[model_idx][k] = 0.0
                            val_metrics_count_all[model_idx][k] = 0
                        val_metrics_sum_all[model_idx][k] += value * count
                        val_metrics_count_all[model_idx][k] += count
        
        # Calculate and print epoch averages for all models (only on rank 0)
        if not is_distributed or rank == 0:
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
            
            # Average metrics across all GPUs for display/wandb logging
            if is_distributed:
                # Enable debug for first epoch and every 10 epochs to see GPU losses
                debug_mode = (epoch == 0) or (epoch % 10 == 0)
                if debug_mode and rank == 0:
                    print(f"\n[DEBUG] Epoch {epoch+1} - Model {model_idx+1} ({model_cfg.initials()})")
                display_train_metrics = reduce_metrics_across_gpus(epoch_train_metrics, world_size, rank, debug=debug_mode)
                display_val_metrics = reduce_metrics_across_gpus(epoch_val_metrics, world_size, rank, debug=debug_mode)
            else:
                display_train_metrics = epoch_train_metrics
                display_val_metrics = epoch_val_metrics
            
            # Print model summary (only on rank 0)
            if not is_distributed or rank == 0:
                print(f"\nModel {model_idx+1}: {model_cfg.initials()} ({model_cfg.style}):")
                print(f"  Train: " + " | ".join([f"{k}: {v:.3f}" for k, v in display_train_metrics.items()]))
                print(f"  Val:   " + " | ".join([f"{k}: {v:.3f}" for k, v in display_val_metrics.items()]))
                
                # Log to wandb
                if wandb_run is not None:
                    # Create a prefix for multi-model scenarios
                    prefix = f"model_{model_idx}/" if num_models > 1 else ""
                    model_name = model_cfg.initials()
                    
                    # Log metrics with model-specific prefix
                    wandb_log_dict = {
                        f"{prefix}{model_name}/train/{k}": v for k, v in display_train_metrics.items()
                    }
                    wandb_log_dict.update({
                        f"{prefix}{model_name}/val/{k}": v for k, v in display_val_metrics.items()
                    })
                    wandb_log_dict["epoch"] = epoch + 1
                    
                    # Log learning rate if available
                    current_lr = optimizers[model_idx].param_groups[0]['lr']
                    wandb_log_dict[f"{prefix}{model_name}/learning_rate"] = current_lr
                    
                    wandb.log(wandb_log_dict)
            
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
        model, autoencoder = models[model_idx]
        model_cfg = model_cfg_list[model_idx]
        train_cfg = train_cfg_list[model_idx]
        optimizer = optimizers[model_idx]
        
        final_checkpoint_path = Path(train_cfg.checkpoint_dir) / "model.pt"
        save_model_checkpoint(final_checkpoint_path, model, model_cfg, train_cfg, optimizer)
        if not is_distributed or rank == 0:
            print(f"\nSaved final checkpoint for Model {model_idx+1} to {final_checkpoint_path}")
        
        # Save epoch history to CSV
        metrics_array = np.array(epoch_metrics_all[model_idx])
        history_csv_path = Path(train_cfg.checkpoint_dir) / "history.csv"
        np.savetxt(history_csv_path, metrics_array, delimiter=',', header=','.join(csv_header), comments='')
        if not is_distributed or rank == 0:
            print(f"Saved epoch metrics to {history_csv_path}")

    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()
    
    # Cleanup distributed training if needed
    if is_distributed:
        dist.destroy_process_group()
    
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
    
    # Check if wandb should be disabled via environment variable
    use_wandb = os.environ.get('DISABLE_WANDB', '').lower() not in ['1', 'true', 'yes']
    
    _, epoch_metrics_all, csv_headers = train(model_cfg_list, train_cfg_list, use_fsdp=True, use_wandb=use_wandb)

    base_dir = Path(train_cfg_list[0].checkpoint_dir).parent
    summary_history_csv_path = base_dir / "summary_history.csv"
    save_summary_history(epoch_metrics_all, summary_history_csv_path, header_list=csv_headers, name_prefix=base_dir.stem)
    print(f"Saved summary history to {summary_history_csv_path}")
