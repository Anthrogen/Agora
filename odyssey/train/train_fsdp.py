import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple, Callable, Dict, List
import random
from types import SimpleNamespace
import argparse
import gc
import atexit
import signal
import functools
import yaml
from datetime import timedelta

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

# Try to import wandb (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking.")

# Import the model and data loader from the src directory
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

def create_transformer_block_wrap_policy():
    """Create FSDP wrapping policy that only wraps transformer blocks.
    
    This policy will wrap only the transformer block classes that have
    trainable parameters to avoid issues with frozen modules.
    """
    def lambda_policy_fn(module):
        # Only wrap transformer block modules that have trainable parameters
        if isinstance(module, (
            StandardTransformerBlock,
            GeometricTransformerBlock,
            ReflexiveTransformerBlock,
            ConsensusTransformerBlock
        )):
            # Check if the module has any trainable parameters
            has_trainable_params = any(p.requires_grad for p in module.parameters())
            return has_trainable_params
        return False
    
    return functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Print debug info from ALL ranks to diagnose connection issues
        print(f"\n[Rank {rank}] Distributed Training Environment:")
        print(f"[Rank {rank}]   RANK: {rank}")
        print(f"[Rank {rank}]   LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
        print(f"[Rank {rank}]   WORLD_SIZE: {world_size}")
        print(f"[Rank {rank}]   MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"[Rank {rank}]   MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        print(f"[Rank {rank}]   GROUP_RANK: {os.environ.get('GROUP_RANK', 'not set')}")
        print(f"[Rank {rank}]   ROLE_RANK: {os.environ.get('ROLE_RANK', 'not set')}")
        print(f"[Rank {rank}]   RDZV vars: BACKEND={os.environ.get('RDZV_BACKEND', 'not set')}, ENDPOINT={os.environ.get('RDZV_ENDPOINT', 'not set')}")
        print(f"[Rank {rank}] About to call dist.init_process_group()...")
        
        # Initialize with timeout for better error messages
        try:
            dist.init_process_group(
                backend='nccl',
                timeout=timedelta(minutes=30)  # Increase timeout for multi-node setup
            )
            print(f"[Rank {rank}] Successfully initialized process group!")
        except Exception as e:
            print(f"[Rank {rank}] ERROR in dist.init_process_group: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Use LOCAL_RANK for CUDA device, not global rank
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Set CUDA device to {local_rank} (local_rank)")
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

class FakeScheduler():
    def __init__(self): pass
    def step(self): pass

def create_warmup_decay(base_learning_rate: float, min_learning_rate: float, num_steps_decay: int, num_steps_warmup: int):
    def lr_lambda(current_step): # One step = one forward pass through the model
        if current_step < num_steps_warmup: # Linear warmup from min_lr to base_lr
            warmup_progress = current_step / num_steps_warmup
            warmup_lr = min_learning_rate + warmup_progress * (base_learning_rate - min_learning_rate)
            return warmup_lr / base_learning_rate
        elif current_step < (num_steps_warmup + num_steps_decay): # Linear decay from base_lr to min_lr
            decay_step = current_step - num_steps_warmup
            decay_progress = decay_step / num_steps_decay
            decay_lr = base_learning_rate - decay_progress * (base_learning_rate - min_learning_rate)
            return decay_lr / base_learning_rate
        else: # Stay at min_lr after decay period
            return min_learning_rate / base_learning_rate

    return lr_lambda

def create_decay(base_learning_rate: float, min_learning_rate: float, num_steps_decay: int):
    def lr_lambda(current_step): # One step = one forward pass through the model
        if current_step < num_steps_decay: # Linear decay from base_lr to min_lr
            decay_progress = current_step / num_steps_decay
            decay_lr = base_learning_rate - decay_progress * (base_learning_rate - min_learning_rate)
            return decay_lr / base_learning_rate
        else: # Stay at min_lr after decay period
            return min_learning_rate / base_learning_rate

    return lr_lambda

def train(model_cfg_list: List[TransformerConfig], train_cfg_list: List[TrainingConfig], verbose=False, use_wandb=True, wandb_project="odyssey-test-fsq-training", wandb_run_name=None, use_fsdp=True):
    
    assert len(model_cfg_list) == 1, "Only one model configuration is supported for training"
    
    # Setup distributed training if available
    is_distributed, rank, world_size = setup_distributed()
    
    # Set device based on distributed setup
    if is_distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device = torch.device(f"cuda:{local_rank}")
        is_fsdp = use_fsdp
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        is_fsdp = False  # Disable FSDP if not distributed
    
    # If a user just passes in a single set of configs, listify them.
    if isinstance(model_cfg_list, TransformerConfig) and isinstance(train_cfg_list, TrainingConfig):
        model_cfg_list = [model_cfg_list]
        train_cfg_list = [train_cfg_list]

    assert len(model_cfg_list) == len(train_cfg_list), "model_cfg_list and train_cfg_list must have the same length"
    num_models = len(model_cfg_list)
    
    # Create checkpoint directories for all models
    if rank == 0:
        for train_cfg in train_cfg_list:
            os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb (only on rank 0)
    if rank == 0 and use_wandb and WANDB_AVAILABLE:
        # Create config dict for wandb
        wandb_config = {
            "num_models": num_models,
            "world_size": world_size,
            "checkpoint_save_freq": train_cfg_list[0].checkpoint_freq,
        }
        
        # Add all model and training configs
        for i, (model_cfg, train_cfg) in enumerate(zip(model_cfg_list, train_cfg_list)):
            model_prefix = f"model_{i}"
            wandb_config[f"{model_prefix}_style"] = model_cfg.style
            # Handle different config types - for AutoencoderConfig, get properties from encoder_cfg
            if hasattr(model_cfg, 'encoder_cfg'):
                wandb_config[f"{model_prefix}_d_model"] = model_cfg.encoder_cfg.d_model
                wandb_config[f"{model_prefix}_n_heads"] = model_cfg.encoder_cfg.n_heads
                wandb_config[f"{model_prefix}_n_layers"] = model_cfg.encoder_cfg.n_layers
            else: # For TrunkConfig, access directly through transformer_cfg
                wandb_config[f"{model_prefix}_d_model"] = model_cfg.transformer_cfg.d_model
                wandb_config[f"{model_prefix}_n_heads"] = model_cfg.transformer_cfg.n_heads
                wandb_config[f"{model_prefix}_n_layers"] = model_cfg.transformer_cfg.n_layers
            wandb_config[f"{model_prefix}_max_len"] = model_cfg.max_len
            wandb_config[f"{model_prefix}_first_block"] = str(model_cfg.first_block_cfg)
            wandb_config[f"{model_prefix}_batch_size"] = train_cfg.batch_size
            wandb_config[f"{model_prefix}_max_epochs"] = train_cfg.max_epochs
            wandb_config[f"{model_prefix}_mask_config"] = str(train_cfg.mask_config)
        
        # Initialize wandb run
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            tags=[model_cfg.style for model_cfg in model_cfg_list],
        )
        
        # Log config files as artifacts
        artifact = wandb.Artifact("training_configs", type="config")
        
        # Find and log YAML files
        for train_cfg in train_cfg_list:
            config_dir = Path(train_cfg.checkpoint_dir)
            # Look for yaml files in the checkpoint directory
            for yaml_file in config_dir.glob("*.yaml"):
                artifact.add_file(str(yaml_file))
            # Look for map.md in parent directory
            map_file = config_dir.parent / "map.md"
            if map_file.exists():
                artifact.add_file(str(map_file))
        
        wandb_run.log_artifact(artifact)
        print(f"  Wandb run initialized: {wandb_run.name}")
        
        if num_models > 1:
            print(f"  Note: Training {num_models} models, but wandb will only log metrics for the first model")
        
        # Define metrics to use step as x-axis for both checkpoint and batch metrics
        wandb.define_metric("step")
        
        # Define model-level metrics for validation (logged at checkpoint intervals)
        wandb.define_metric("model/val_loss", step_metric="step")
        wandb.define_metric("model/val_rmsd", step_metric="step")
        wandb.define_metric("model/val_loss_seq", step_metric="step")
        wandb.define_metric("model/val_loss_struct", step_metric="step")
        wandb.define_metric("model/val_seq_acc", step_metric="step")
        wandb.define_metric("model/val_struct_acc", step_metric="step")
        
        # Define batch-level metrics for training (logged every step)
        wandb.define_metric("batch/train_loss", step_metric="step")
        wandb.define_metric("batch/train_rmsd", step_metric="step")
        wandb.define_metric("batch/train_loss_seq", step_metric="step")
        wandb.define_metric("batch/train_loss_struct", step_metric="step")
        wandb.define_metric("batch/train_seq_acc", step_metric="step")
        wandb.define_metric("batch/train_struct_acc", step_metric="step")
        wandb.define_metric("batch/learning_rate", step_metric="step")
        wandb.define_metric("batch/epoch", step_metric="step")
    else:
        wandb_run = None
    
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
        if rank == 0:
            print(f"Model {i+1}/{num_models}: Starting {model_cfg.style} training")
            print(f"  Training model: {model_cfg.first_block_cfg}")
            print(f"  Masking: {train_cfg.mask_config}")
        
        # Create model with fixed seed
        if train_cfg.jump_start is None: # No jump start provided.  We will cold-start the model. Create model with fixed seed
            if rank == 0: print(f"  Creating {model_cfg.first_block_cfg.initials()} target model...")
            model = load_model_from_empty(model_cfg, device)
        else: # Jump start provided. Load model from checkpoint
            if rank == 0: print(f"  Loading {model_cfg.first_block_cfg.initials()} target model from {train_cfg.jump_start}...")
            model, model_cfg_loaded, train_cfg_loaded= load_model_from_checkpoint(train_cfg.jump_start, device)
            model_cfg = model_cfg_loaded  # Replace the original model_cfg with the loaded one
            model_cfg_list[i] = model_cfg  # Update the list to persist changes

        autoencoder = None
        if load_fsq_encoder: 
            autoencoder, autoencoder_model_cfg, _ = load_model_from_checkpoint(model_cfg.autoencoder_path, device)
            autoencoder.encoder.eval(); autoencoder.encoder.requires_grad_(False)
            autoencoder.quantizer.eval(); autoencoder.quantizer.requires_grad_(False)  # Also freeze quantizer
            if isinstance(model, Autoencoder): 
                model.encoder = autoencoder.encoder # Update the encoder to match the loaded encoder
                model.quantizer = autoencoder.quantizer # Update the quantizer to match the loaded quantizer
                model_cfg.encoder_cfg = autoencoder_model_cfg.encoder_cfg # Update the encoder_cfg to match the loaded encoder_cfg
                model_cfg_list[i] = model_cfg  # Ensure the updated config is saved to the list
        
        # Wrap model with FSDP if enabled
        if is_fsdp and is_distributed:
            if rank == 0: print(f"  Wrapping model with FSDP (only transformer blocks)...")
            
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
                sync_module_states=False,  # Disabled for multi-node training
            )
            
            # For stage_2: only wrap decoder (frozen encoder incompatible with FSDP)
            if model_cfg.style == "stage_2": model.decoder = FSDP(model.decoder, **fsdp_config)
            else: model = FSDP(model, **fsdp_config)
            
            # Debug: Show which modules were actually wrapped and FSDP's view of world
            if rank == 0:
                print("  Modules wrapped by FSDP:")
                for name, module in model.named_modules():
                    if isinstance(module, FSDP): print(f"    - {name}: {module.__class__.__name__}")
            
            # CRITICAL: Verify FSDP sees all ranks
            fsdp_world_size = dist.get_world_size()
            fsdp_rank = dist.get_rank()
            print(f"  [Rank {rank}] FSDP world_size: {fsdp_world_size}, FSDP rank: {fsdp_rank}")
            
            if rank == 0: 
                print(f"  FSDP wrapping complete")
                print(f"  FSDP is sharding across {fsdp_world_size} ranks (expecting {world_size})")
            
        models.append((model, autoencoder))

        # Print parameter count and sharding info
        total_params = sum(p.numel() for p in model.parameters())
        if is_fsdp:
            # For FSDP, each rank only holds a shard of parameters
            local_params = sum(p.numel() for p in model.parameters() if p.is_leaf)
            print(f"  [Rank {rank}] Local parameters (shard): {local_params:,}")
            
        if rank == 0:
            print(f"  {model_cfg.first_block_cfg.initials()} total parameters: {total_params:,}")
            print(f"  Using float32 training")
            if is_fsdp:
                expected_shard_size = total_params // fsdp_world_size
                print(f"  Expected parameters per rank: ~{expected_shard_size:,} (total/{fsdp_world_size})")
        
        ###########################################################################
        # Initializing optimizer and scheduler
        ###########################################################################
        if isinstance(train_cfg.optim_schedule_config, FlatSchedulerConfig):
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.optim_schedule_config.learning_rate)
            scheduler = FakeScheduler()
        elif isinstance(train_cfg.optim_schedule_config, WarmupDecaySchedulerConfig):
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg.optim_schedule_config.base_learning_rate)
            lr_lambda = create_warmup_decay(train_cfg.optim_schedule_config.base_learning_rate, train_cfg.optim_schedule_config.min_learning_rate, train_cfg.optim_schedule_config.num_epochs_decay, train_cfg.optim_schedule_config.num_epochs_warmup)
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

        dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_modes[i], max_length=model_cfg.max_len - 2)
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size

        # We use g_val as the generator of the split
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g_val)
        
        # Pass appropriate FSQ encoder
        model, autoencoder = models[i]
        
        # For FSDP, we need to use DistributedSampler
        if is_fsdp:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = None # DON'T use DistributedSampler for validation - all ranks should see same data to avoid synchronization issues
            
            # Calculate optimal number of workers - For H100s with high throughput, we want more workers
            # Rule of thumb: (num_cpus / num_gpus) * 0.75 to leave some headroom
            # Setting to 0 to avoid generator pickling issues
            num_workers = 0  # int((os.cpu_count() / world_size) * 0.75)  # About 84 workers per GPU
            train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                                   batch_size=train_cfg.batch_size, sampler=train_sampler, 
                                                   shuffle=False, generator=g_train,  # shuffle=False when using sampler
                                                   worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                                   autoencoder=autoencoder, num_workers=num_workers, 
                                                   pin_memory=True, persistent_workers=False, prefetch_factor=None)
            val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                                batch_size=train_cfg.batch_size, sampler=val_sampler, 
                                                shuffle=False, generator=g_val,  # No sampler, all ranks see same data
                                                worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                                autoencoder=autoencoder, num_workers=num_workers, 
                                                pin_memory=True, persistent_workers=False, prefetch_factor=None)
        else:
            # For single GPU, use fewer workers
            # Setting to 0 to avoid generator pickling issues
            num_workers = 0  # int((os.cpu_count() * 0.5))  # Use half the CPUs for single GPU
            train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                                   batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, 
                                                   worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                                   autoencoder=autoencoder, num_workers=num_workers, 
                                                   pin_memory=True, persistent_workers=False, prefetch_factor=None)
            val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks_list[i], device, 
                                                batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, 
                                                worker_init_fn=worker_init_fn, min_unmasked=min_unmasked_list[i], 
                                                autoencoder=autoencoder, num_workers=num_workers, 
                                                pin_memory=True, persistent_workers=False, prefetch_factor=None)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    ###########################################################################
    #  Set default values for checkpoint_freq and max_steps_val if None
    ###########################################################################
    for i, train_cfg in enumerate(train_cfg_list):
        # Set checkpoint_freq to number of training batches per epoch if None
        if train_cfg.checkpoint_freq is None:
            train_cfg.checkpoint_freq = len(train_loaders[i])
            if rank == 0: print(f"  Model {i+1}: checkpoint_freq set to {train_cfg.checkpoint_freq} (once per epoch)")
        
        # Set max_steps_val to number of validation batches if None  
        if train_cfg.max_steps_val is None:
            train_cfg.max_steps_val = len(val_loaders[i])
            if rank == 0: print(f"  Model {i+1}: max_steps_val set to {train_cfg.max_steps_val} (all validation batches)")

    ###########################################################################
    #  Training Loop
    ###########################################################################
    # Prepare containers to store epoch-level history for CSV saving
    epoch_metrics_all = [[] for _ in range(num_models)]  # List of lists
    csv_headers = [None] * num_models
    
    # Use first config for step count (they should all be the same)
    max_epochs = train_cfg_list[0].max_epochs
    
    # -------------------- Training loop -------------------- #
    steps = [0] * num_models  # Track steps per model
    
    # Moving average tracking for smoother model metric plots
    moving_avg_window = 100  # Number of batches to average over
    train_metric_buffers = [{} for _ in range(num_models)]  # Store recent values for each model
    
    for epoch in range(max_epochs):
        # Set epoch for distributed samplers to ensure proper shuffling
        if is_fsdp:
            for i in range(num_models):
                train_loaders[i].sampler.set_epoch(epoch)
        
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
            disable_pbar = rank != 0
            with tqdm(train_loader, desc=f"Epoch {epoch+1} [Model {model_idx+1}: {model_cfg.first_block_cfg.initials()} Train]", ascii=True, leave=True, ncols=150, disable=disable_pbar) as pbar:
                for batch_data in pbar:
                    steps[model_idx] += 1
                    # Skip empty/None batches
                    if batch_data is None: continue
                    
                    # Train single model on batch with bfloat16
                    train_metrics = step_fn(model, optimizer, scheduler, batch_data, model_cfg, train_cfg, train_mode=True)
                    
                    # Extract single-batch training metrics (no accumulation)
                    epoch_train_metrics = {k: value for k, (value, count) in train_metrics.items()}
                    
                    # Update moving average buffers for this model
                    for metric_name, metric_value in epoch_train_metrics.items():
                        if metric_name not in train_metric_buffers[model_idx]:
                            train_metric_buffers[model_idx][metric_name] = []
                        
                        # Add new value and maintain window size
                        train_metric_buffers[model_idx][metric_name].append(metric_value)
                        if len(train_metric_buffers[model_idx][metric_name]) > moving_avg_window:
                            train_metric_buffers[model_idx][metric_name].pop(0)
                    
                    # Calculate moving averages for display
                    moving_avg_metrics = {}
                    for metric_name in epoch_train_metrics:
                        if metric_name in train_metric_buffers[model_idx] and len(train_metric_buffers[model_idx][metric_name]) > 0:
                            moving_avg_metrics[metric_name] = sum(train_metric_buffers[model_idx][metric_name]) / len(train_metric_buffers[model_idx][metric_name])
                        else: moving_avg_metrics[metric_name] = epoch_train_metrics[metric_name]
                    
                    # Update progress bar with moving average metrics
                    prefix = model_cfg.first_block_cfg.initials()
                    pbar.set_postfix({f"{prefix}_{k}": f"{v:.3f}" for k, v in moving_avg_metrics.items()})

                    # Log batch-level training metrics to wandb (high frequency)
                    if rank == 0 and wandb_run is not None and model_idx == 0:  # Only log first model's batch metrics
                        batch_log = {"step": steps[model_idx]}  # Using 'step' as step counter (it's actually batch number)
                        batch_log["batch/epoch"] = epoch + 1  # Current epoch (actual epoch)
                        
                        # Log all training metrics at batch level
                        for metric_name, metric_value in epoch_train_metrics.items():
                            batch_log[f"batch/train_{metric_name}"] = metric_value
                        
                        # Log learning rate
                        batch_log["batch/learning_rate"] = optimizer.param_groups[0]['lr']
                        
                        wandb.log(batch_log)

                    # Save periodic checkpoints and calculate validation metrics
                    # ALL RANKS must participate in checkpoint saving for FSDP
                    checkpoint_needed = train_cfg.checkpoint_freq > 0 and steps[model_idx] % train_cfg.checkpoint_freq == 0
                    
                    # Synchronize checkpoint decision across all ranks
                    if is_distributed:
                        checkpoint_tensor = torch.tensor(checkpoint_needed, dtype=torch.bool, device=torch.cuda.current_device())
                        dist.broadcast(checkpoint_tensor, src=0)
                        checkpoint_needed = checkpoint_tensor.item()
                    
                    if checkpoint_needed:
                        if is_distributed: dist.barrier()
                        if rank == 0: print(f"\nSaving checkpoint at step {steps[model_idx]}...")
                        
                        # ALL RANKS must call this function for FSDP state dict extraction
                        checkpoint_path = Path(train_cfg.checkpoint_dir) / f"checkpoint_step_{steps[model_idx]}.pt"
                        save_model_checkpoint(checkpoint_path, model, model_cfg, train_cfg, optimizer)
                        
                        if is_distributed: dist.barrier()
                        if rank == 0: print(f"  Saved checkpoint for Model {model_idx+1} at {checkpoint_path}")
                        
                        # Log checkpoint to wandb (only on rank 0)
                        if rank == 0 and wandb_run is not None:
                            checkpoint_artifact = wandb.Artifact(
                                f"checkpoint_model_{model_idx}_step_{steps[model_idx]}", 
                                type="model",
                                metadata={"step": steps[model_idx], "epoch": epoch, "model_idx": model_idx, "model_style": model_cfg.style, "model_type": model_cfg.first_block_cfg.initials()}
                            )
                            checkpoint_artifact.add_file(str(checkpoint_path))
                            wandb_run.log_artifact(checkpoint_artifact)

                        # Validation phase - Only rank 0 does validation, others wait
                        if is_distributed: dist.barrier()
                        model.eval()
                        val_metrics_sum = {}; val_metrics_count = {}
                        val_loader = val_loaders[model_idx] # Get the val_loader for this model
                        
                        # All ranks must iterate together for FSDP synchronization
                        with torch.no_grad():
                            for idx_val, val_batch_data in enumerate(val_loader):
                                if idx_val >= train_cfg.max_steps_val: break

                                # Skip empty/None batches
                                if val_batch_data is None: continue
                                    
                                # All ranks participate in forward pass (FSDP requirement)
                                val_metrics = step_fn(model, optimizer, scheduler, val_batch_data, model_cfg, train_cfg, train_mode=False)
                                
                                # Only rank 0 accumulates metrics to avoid duplicate counting
                                if rank == 0:
                                    # Accumulate validation metrics across validation batches
                                    for k, (value, count) in val_metrics.items():
                                        if k not in val_metrics_sum:
                                            val_metrics_sum[k] = 0.0
                                            val_metrics_count[k] = 0
                                        val_metrics_sum[k] += value * count
                                        val_metrics_count[k] += count
                            
                        # Only rank 0 calculates averages and reports metrics
                        if rank == 0:
                            # Calculate validation averages
                            epoch_val_metrics = {k: val_metrics_sum[k] / val_metrics_count[k] for k in val_metrics_sum.keys()}
                            
                            # During validation, we don't need to reduce training metrics
                            # Just use the last training metrics from this rank
                            display_train_metrics = epoch_train_metrics
                            display_val_metrics = epoch_val_metrics
                            
                            # Print metrics summary
                            print(f"\n{'='*80}")
                            print(f"Epoch {epoch+1}, Step {steps[model_idx]} Summary:")
                            print(f"{'='*80}")
                            print(f"\nModel {model_idx+1}: {model_cfg.first_block_cfg.initials()} ({model_cfg.style}):")
                            print(f"  Train: " + " | ".join([f"{k}: {v:.3f}" for k, v in display_train_metrics.items()]))
                            print(f"  Val:   " + " | ".join([f"{k}: {v:.3f}" for k, v in display_val_metrics.items()]))
                            
                            # Create row with current step's training metrics and validation metrics
                            metric_names = sorted(list(display_train_metrics.keys()))
                            
                            # Store metrics for CSV headers (only if not set yet)
                            if csv_headers[model_idx] is None:
                                csv_header = [f"train_{k}" for k in metric_names] + [f"val_{k}" for k in metric_names]
                                csv_headers[model_idx] = csv_header
                            
                            row = [display_train_metrics[k] for k in metric_names] + [display_val_metrics[k] for k in metric_names]
                            epoch_metrics_all[model_idx].append(row)
                            
                            # Log validation metrics to wandb
                            if wandb_run is not None:
                                wandb_log = {"step": steps[model_idx]}
                                
                                # Log only validation metrics for current model
                                for metric_name in metric_names:
                                    wandb_log[f"model/val_{metric_name}"] = epoch_val_metrics[metric_name]
                                
                                wandb.log(wandb_log)
                        
                        # Set model back to training mode (all ranks)
                        model.train()
                        
                        # CRITICAL: Synchronize all ranks after validation phase
                        # This ensures all ranks stay at the same step count
                        if is_distributed: dist.barrier()
    
    # Clean up dataloaders to terminate persistent workers
    if rank == 0: print("\nCleaning up dataloader workers...")
    
    # Explicitly delete dataloader objects to trigger worker shutdown
    for i in range(len(train_loaders)):
        if hasattr(train_loaders[i], '_iterator'):
            del train_loaders[i]._iterator
        if hasattr(train_loaders[i], '_DataLoader__initialized'):
            train_loaders[i]._DataLoader__initialized = False
            
    for i in range(len(val_loaders)):
        if hasattr(val_loaders[i], '_iterator'):
            del val_loaders[i]._iterator
        if hasattr(val_loaders[i], '_DataLoader__initialized'):
            val_loaders[i]._DataLoader__initialized = False
    
    # Delete the dataloader lists
    del train_loaders
    del val_loaders
    
    # Force garbage collection
    gc.collect()
    
    # Clean up FSDP
    if is_distributed:
        dist.destroy_process_group()
    
    return models, epoch_metrics_all, csv_headers, wandb_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Odyssey models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='odyssey-training',
                        help='Wandb project name (default: odyssey-training)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    args = parser.parse_args()
    
    # Initialize FSDP for main process if needed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        is_distributed = False
    
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
    
    # Only rank 0 should expand the YAML file
    if rank == 0:
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
    
    # Synchronize all processes before loading configs
    if is_distributed:
        # Process group will be initialized in train function
        # Just ensure all processes wait for rank 0 to finish generating configs
        import time
        if rank != 0:
            # Give rank 0 time to generate configs
            time.sleep(5)
    
    # Load all expanded configs and train
    print(f"[Rank {rank}] About to load configs from: {expanded_yaml_dir}")
    try:
        model_cfg_list, train_cfg_list = load_multi_configs(str(expanded_yaml_dir))
        print(f"[Rank {rank}] Successfully loaded configs")
    except Exception as e:
        print(f"[Rank {rank}] ERROR loading configs: {e}")
        raise
    
    print(f"[Rank {rank}] About to call train()")
    _, epoch_metrics_all, csv_headers, wandb_run = train(
        model_cfg_list, 
        train_cfg_list, 
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_fsdp=True
    )

    if rank == 0:
        base_dir = Path(train_cfg_list[0].checkpoint_dir).parent
        summary_history_csv_path = base_dir / "summary_history.csv"
        save_summary_history(epoch_metrics_all, summary_history_csv_path, header_list=csv_headers, name_prefix=base_dir.stem)
        print(f"Saved summary history to {summary_history_csv_path}")
        
        # Log summary history to existing wandb run if available
        if wandb_run is not None:
            summary_artifact = wandb.Artifact("training_summary", type="summary")
            summary_artifact.add_file(str(summary_history_csv_path))
            
            # Also add all generated configs and map
            for yaml_file in expanded_yaml_dir.rglob("*.yaml"):
                summary_artifact.add_file(str(yaml_file))
            if (expanded_yaml_dir / "map.md").exists():
                summary_artifact.add_file(str(expanded_yaml_dir / "map.md"))
                
            wandb_run.log_artifact(summary_artifact)
            
            # Log final summary metrics
            final_metrics = {}
            # Just log the first model's final metrics with simple "model" prefix
            if epoch_metrics_all and len(epoch_metrics_all[0]) > 0:
                final_step = epoch_metrics_all[0][-1]
                metric_names = csv_headers[0]
                
                for i, metric_name in enumerate(metric_names):
                    final_metrics[f"final/model/{metric_name}"] = final_step[i]
            
            if final_metrics:
                wandb.log(final_metrics)
            
            # Finish the wandb run
            wandb.finish()
        
        print("\nTraining completed successfully!")
    
    # Ensure clean exit
    sys.exit(0)
