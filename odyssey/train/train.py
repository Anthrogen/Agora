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
import json
from datetime import datetime
import argparse

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder, StandardTransformerBlock
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import MaskedBatch, SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from synchronize import ensure_identical_parameters_transformers, ensure_identical_parameters_autoencoders
from fsq_step import stage_1_step, stage_2_step
from mlm_step import mlm_step
from discrete_diffusion_step import discrete_diffusion_step

from odyssey.src.configurations import *
from odyssey.src.config_loader import load_config_from_args

def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model_cfg, train_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Store config dictionaries as backup
    model_config_dict = model_cfg.get_config_dict()
    train_config_dict = train_cfg.get_config_dict()
    
    # TODO: better printing
    print(f"Starting {model_cfg.style} training")
    print(f"Training model: {model_cfg.first_block_cfg}")
    print(f"Masking: {train_cfg.mask_config}")
    
    # Create model with fixed seed
    print(f"Creating {model_cfg.first_block_cfg.initials()} target model...")
    torch.manual_seed(model_cfg.reference_model_seed)
    
    desired_constructor = Autoencoder if isinstance(model_cfg, FSQConfig) else TransformerTrunk
    sync_function = ensure_identical_parameters_autoencoders if isinstance(model_cfg, FSQConfig) else ensure_identical_parameters_transformers

    model_cfg_sa = replace(model_cfg, first_block_cfg=SelfAttentionConfig())
    model_sa = desired_constructor(model_cfg_sa).to(device)

    if isinstance(model_cfg.first_block_cfg, SelfAttentionConfig):
        model = model_sa
    else:
        # Synchronize model with baseline SA model
        model_target = desired_constructor(model_cfg).to(device)
        
        # Synchronize target model with SA reference
        print(f"Synchronizing {model_cfg.first_block_cfg.initials()} shared parameters with SA reference...")
        temp_models = {"SA": model_sa, model_cfg.first_block_cfg.initials(): model_target}
        sync_function(temp_models, model_cfg.reference_model_seed)
        
        # Keep target model, delete SA reference
        model = model_target
        del model_sa; del temp_models

    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate)

    # Load checkpoint with dynamic path based on model type
    if model_cfg.style in {"mlm", "discrete_diffusion", "stage_2"}:
        #TODO also, we should be using os.path.join rather than / wherever possible.
        checkpoint = torch.load(model_cfg.fsq_encoder_path, map_location=device, weights_only=False)
        encoder_state = {k.removeprefix('encoder.'): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        fsq_config = checkpoint['model_config'] #TODO: load FSQConfig from checkpoint
        fsq_encoder = FSQEncoder(fsq_config)
        fsq_encoder.load_state_dict(encoder_state)
        print(f"Loaded {model_cfg.first_block_cfg.initials()} encoder weights from: {model_cfg.fsq_encoder_path}")

        fsq_encoder.eval()
        fsq_encoder.requires_grad_(False)
        fsq_encoder = fsq_encoder.to(device)

    if model_cfg.style == "stage_2":
        model.encoder = fsq_encoder
        model.quantizer = model.encoder.quantizer
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_cfg.first_block_cfg.initials()} total parameters: {total_params:,}")
    
    # -------------------- Data loading -------------------- #
    # Set seed for dataset split
    data_seed = model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)
    
    # Set dataset mode based on style
    dataset_mode = "side_chain" if model_cfg.style == "stage_2" else "backbone"
    dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_mode, max_length=model_cfg.max_len - 2)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders with fixed seed for consistent masking
    g_train = torch.Generator()
    g_train.manual_seed(data_seed)
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000)
    
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
        
    # Pass appropriate FSQ encoder based on style
    if model_cfg.style == "stage_1":
        train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
        val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
    elif model_cfg.style == "stage_2":
        train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=model.encoder)
        val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=model.encoder)
    elif model_cfg.style in {"mlm", "discrete_diffusion"}:
        train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder)
        val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder)

    step_fns = {'stage_1': stage_1_step, 'stage_2': stage_2_step, 'mlm': mlm_step, 'discrete_diffusion': discrete_diffusion_step}
    step_fn = step_fns[model_cfg.style]

    history_train = None
    history_val = None
    # Prepare containers to store epoch-level history for CSV saving
    epoch_metrics = []  # List of dicts; one per epoch
    # -------------------- Training loop -------------------- #
    for epoch in range(train_cfg.max_epochs):
        # Training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.max_epochs} [{model_cfg.first_block_cfg.initials()} Train]",
                 ascii=True, leave=True, ncols=150) as pbar:
            for batch_data in pbar:
                # Skip empty/None batches
                if batch_data is None: continue
                
                # Train single model on batch
                train_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=True)

                if history_train is None:
                    history_train_sum = {k: 0 for k in train_metrics.keys()}
                    history_train_cts = {k: 0 for k in train_metrics.keys()}
                    running_average_metric = {k: 0 for k in train_metrics.keys()}

                for k in train_metrics.keys():
                    history_train_sum[k] += train_metrics[k]
                    history_train_cts[k] += len(batch_data.masked_data['coords'])
                    running_average_metric[k] = history_train_sum[k] / history_train_cts[k]

                # Update progress bar
                prefix = model_cfg.first_block_cfg.initials()
                pbar.set_postfix({f"{prefix}_{k}": f"{v:.3f}" for k, v in running_average_metric.items()})
        
        # Calculate epoch averages
        for key in train_metrics: history_train_sum[key] /= history_train_cts[key]
        
        # -------------------- Validation -------------------- #
        with torch.no_grad():
            for batch_data in val_loader:
                # Skip empty/None batches
                if batch_data is None: continue
                    
                # Validate single model on batch
                val_metrics = step_fn(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=False)

                if history_val is None:
                    history_val_sum = {k: 0 for k in val_metrics.keys()}
                    history_val_cts = {k: 0 for k in val_metrics.keys()}

                for k in val_metrics.keys():
                    history_val_sum[k] += val_metrics[k]
                    history_val_cts[k] += len(batch_data.masked_data['coords'])
        
        # Calculate epoch averages
        for key in val_metrics: history_val_sum[key] /= history_val_cts[key]
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{train_cfg.max_epochs} - {model_cfg.first_block_cfg.initials()}:")
        # Training metrics
        print(f"  Train:")
        print(*(f"      {k}: \t{v:.3f}" for k, v in history_train_sum.items()), sep="\n")
        
        # Validation metrics
        print(f"  Val:")
        print(*(f"      {k}: \t{v:.3f}" for k, v in history_val_sum.items()), sep="\n")

        # -------------------- Store metrics for CSV -------------------- #
        # On first epoch, create ordered list of keys
        if epoch == 0:
            metric_names = sorted(list(history_train_sum.keys()))  # deterministic
            csv_header = [f"train_{k}" for k in metric_names] + [f"val_{k}" for k in metric_names]
        # Build row with metric values in the same order as header
        row = [history_train_sum[k] for k in metric_names] + [history_val_sum[k] for k in metric_names]
        epoch_metrics.append(row)

    # Save final checkpoint
    final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_cfg.first_block_cfg.initials()}_{model_cfg.style}_{train_cfg.mask_config}_model.pt"
    torch.save({
        'epoch': train_cfg.max_epochs,
        'model_type': model_cfg.first_block_cfg.initials(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_cfg,
        'model_config_dict': model_config_dict,  # Backup dictionary
        'train_config_dict': train_config_dict   # Backup dictionary
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint for {model_cfg.first_block_cfg.initials()}")
    
    # -------------------- Save epoch history to a single CSV -------------------- #
    metrics_array = np.array(epoch_metrics)  # shape [E, num_cols]
    history_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_cfg.first_block_cfg.initials()}_{model_cfg.style}_{train_cfg.mask_config}_epoch_metrics.csv"
    np.savetxt(history_csv_path, metrics_array, delimiter=',', header=','.join(csv_header), comments='')
    print(f"Saved epoch metrics to {history_csv_path}")

    return model, epoch_metrics

# --------------------------------------------------------------------------- #
#  MLM Training Configurations                                                #
# --------------------------------------------------------------------------- #
# NOTE: These hardcoded configurations are now replaced by YAML config files
# Use: python train.py --config configs/mlm_config.yaml
# _mask_cfg = SimpleMaskConfig(mask_prob_seq=0.15, mask_prob_struct=0.15)
# # _mask_cfg = ComplexMaskConfig()

# _loss_cfg = CrossEntropyLossConfig(seq_loss_weight=1.0, struct_loss_weight=1.0, loss_elements="masked")

# _first_block_cfg = SelfConsensusConfig(consensus_num_iterations=1, consensus_connectivity_type="local_window", consensus_w=2, consensus_r=8, consensus_edge_hidden_dim=24)
# # _first_block_cfg = SelfAttentionConfig()
# # _first_block_cfg = GeometricAttentionConfig()
# # _first_block_cfg = ReflexiveAttentionConfig()

# _train_cfg = TrainingConfig(
#     batch_size=4,
#     max_epochs=50,
#     learning_rate=1e-5,
#     mask_config=_mask_cfg,
#     loss_config=_loss_cfg,
#     data_dir="/workspace/demo/Odyssey/sample_data/1k.csv",
#     checkpoint_dir="/workspace/demo/Odyssey/checkpoints/transformer_trunk"
# )

# _model_cfg = TrunkConfig(
#     style='mlm',
#     d_model=128,
#     n_heads=1,
#     n_layers=3,
#     max_len=2048,
#     dropout=0.1,
#     ff_mult=4,
#     first_block_cfg=_first_block_cfg,
#     reference_model_seed=42,
#     fsq_encoder_path="/workspace/demo/Odyssey/checkpoints/fsq/SC_stage_1_simple_model.pt", # complex_model.pt
#     seq_vocab=len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS),
#     struct_vocab=4375 + len(SPECIAL_TOKENS),
#     seq_absorb_token=SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS),
#     struct_absorb_token=SPECIAL_TOKENS.MASK.value + 4375
# )

# --------------------------------------------------------------------------- #
#  Discrete Diffusion Training Configurations                                 #
# --------------------------------------------------------------------------- #
# _mask_cfg = DiffusionConfig(noise_schedule = "uniform", sigma_min = 0.31, sigma_max = 5.68, num_timesteps = 100)

# _loss_cfg = ScoreEntropyLossConfig(seq_loss_weight=1.0, struct_loss_weight=1.0)

# _first_block_cfg = SelfConsensusConfig(consensus_num_iterations=1, consensus_connectivity_type="local_window", consensus_w=2, consensus_r=8, consensus_edge_hidden_dim=24)
# # _first_block_cfg = SelfAttentionConfig()
# # _first_block_cfg = GeometricAttentionConfig()
# # _first_block_cfg = ReflexiveAttentionConfig()

# _train_cfg = TrainingConfig(
#     batch_size=4,
#     max_epochs=150,
#     learning_rate=1e-5,
#     mask_config=_mask_cfg,
#     loss_config=_loss_cfg,
#     data_dir="/workspace/demo/Odyssey/sample_data/1k.csv",
#     checkpoint_dir="/workspace/demo/Odyssey/checkpoints/transformer_trunk"
# )

# _model_cfg = TrunkConfig(
#     style='discrete_diffusion',
#     d_model=128,
#     n_heads=1,
#     n_layers=3,
#     max_len=2048,
#     dropout=0.1,
#     ff_mult=4,
#     first_block_cfg=_first_block_cfg,
#     reference_model_seed=42,
#     fsq_encoder_path="/workspace/demo/Odyssey/checkpoints/fsq/SC_stage_1_discrete_diffusion_model.pt",
#     seq_vocab=len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS),
#     struct_vocab=4375 + len(SPECIAL_TOKENS),
#     seq_absorb_token=SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS),
#     struct_absorb_token=SPECIAL_TOKENS.MASK.value + 4375
# )

# --------------------------------------------------------------------------- #
#  FSQ Stage 1 Training Configurations                                        #
# --------------------------------------------------------------------------- #
# _mask_cfg = SimpleMaskConfig(mask_prob_seq=0.15, mask_prob_struct=0.15)
# # _mask_cfg = ComplexMaskConfig()
# # _mask_cfg = DiffusionConfig(noise_schedule = "uniform", sigma_min = 0.31, sigma_max = 5.68, num_timesteps = 100)

# _loss_cfg = KabschRMSDLossConfig()

# # _first_block_cfg = SelfConsensusConfig(consensus_num_iterations=1, consensus_connectivity_type="local_window", consensus_w=2, consensus_r=8, consensus_edge_hidden_dim=24)
# _first_block_cfg = SelfAttentionConfig()
# # _first_block_cfg = GeometricAttentionConfig()
# # _first_block_cfg = ReflexiveAttentionConfig()

# _train_cfg = TrainingConfig(
#     batch_size=4,
#     max_epochs=50,
#     learning_rate=1e-5,
#     mask_config=_mask_cfg,
#     loss_config=_loss_cfg,
#     data_dir="/workspace/demo/Odyssey/sample_data/1k.csv",
#     checkpoint_dir="/workspace/demo/Odyssey/checkpoints/fsq"
# )

# _model_cfg = FSQConfig(
#     style='stage_1',
#     d_model=128,
#     n_heads=1,
#     n_layers=3,
#     max_len=2048,
#     dropout=0.1,
#     ff_mult=4,
#     first_block_cfg=_first_block_cfg,
#     reference_model_seed=42,
#     latent_dim=32,
#     fsq_levels=[7, 5, 5, 5, 5],
#     fsq_encoder_path=None,
#     seq_vocab=len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS),
#     struct_vocab=4375 + len(SPECIAL_TOKENS)
# )

# --------------------------------------------------------------------------- #
#  FSQ Stage 2 Training Configurations                                        #
# --------------------------------------------------------------------------- #
# _mask_cfg = NoMaskConfig()

# _loss_cfg = KabschRMSDLossConfig()

# _first_block_cfg = SelfConsensusConfig(consensus_num_iterations=1, consensus_connectivity_type="local_window", consensus_w=2, consensus_r=8, consensus_edge_hidden_dim=24)
# # _first_block_cfg = SelfAttentionConfig()
# # _first_block_cfg = GeometricAttentionConfig()
# # _first_block_cfg = ReflexiveAttentionConfig()

# _train_cfg = TrainingConfig(
#     batch_size=4,
#     max_epochs=1,
#     learning_rate=1e-5,
#     mask_config=_mask_cfg,
#     loss_config=_loss_cfg,
#     data_dir="/workspace/demo/Odyssey/sample_data/1k.csv",
#     checkpoint_dir="/workspace/demo/Odyssey/checkpoints/fsq"
# )

# _model_cfg = FSQConfig(
#     style='stage_2',
#     d_model=128,
#     n_heads=1,
#     n_layers=3,
#     max_len=2048,
#     dropout=0.1,
#     ff_mult=4,
#     first_block_cfg=_first_block_cfg,
#     reference_model_seed=42,
#     latent_dim=32,
#     fsq_levels=[7, 5, 5, 5, 5],
#     fsq_encoder_path="/workspace/demo/Odyssey/checkpoints/fsq/SC_stage_1_discrete_diffusion_model.pt",
#     seq_vocab=len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS),
#     struct_vocab=4375 + len(SPECIAL_TOKENS)
# )

if __name__ == "__main__":
    # Load configuration from YAML file instead of hardcoded values
    config_loader, model_cfg, train_cfg = load_config_from_args()
    
    # Run training with loaded configurations
    train(model_cfg, train_cfg)