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
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List, Dict
import random
import math
from types import SimpleNamespace


# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.transformer import TransformerTrunk, StandardTransformerBlock
from src.models.autoencoder import FSQEncoder
from src.dataset import ProteinDataset
from src.dataloader import DiffusionDataLoader, MaskedBatch
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from src.losses import score_entropy_loss

# --------------------------------------------------------------------------- #
#  Configurations                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128 # 768  # Model dimensions
    n_heads: int = 1 # 12
    n_layers: int = 3 # 12
    seq_vocab: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab: int = 7*5*5*5*5 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
    max_len: int = 2048
    dropout: float = 0.1  # Other architecture params
    ff_mult: int = 4
    ff_hidden_dim: int = d_model * ff_mult

    
    # Consensus-specific parameters
    consensus_num_iterations: int = 1  # Number of Consensus gradient iterations
    consensus_connectivity_type: str = "local_window"  # "local_window" or "top_w"
    consensus_w: int = 2  # Window size for local_window, or w value for top_w
    consensus_r: int = 8  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim: int = 24  # Hidden dim for edge networks

    # Absorbing state tokens (using MASK token index)
    seq_absorb_token: int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
    struct_absorb_token: int = SPECIAL_TOKENS.MASK.value + 4375

@dataclass
class DiffusionConfig:
    """Discrete diffusion configuration."""
    # Noise schedule parameters
    noise_schedule: str = "uniform"  # Type of noise schedule ("linear", "inverted_u", or "uniform")
    sigma_min: float = 0.31  # Minimum noise level
    sigma_max: float = 5.68  # Maximum noise level
    num_timesteps: int = 100  # Number of discrete timesteps for training

@dataclass
class TrainingConfig:
    """Training process configuration."""
    model_type: str = "SC"  # Model to train - can be "SA", "GA", "RA", or "SC"
    batch_size: int = 4  # Training hyperparameters
    max_epochs: int = 150
    learning_rate: float = 1e-5
    num_iter: int = 3  # Number of iterations to repeat training

    # Loss weights
    seq_loss_weight: float = 1.0
    struct_loss_weight: float = 1.0

    data_dir: str = "../sample_data/1k.csv"  # Data paths
    checkpoint_dir: str = "../checkpoints/transformer_trunk"  # Checkpointing
    reference_model_seed: int = 22 # Reference model seed for consistent parameter initialization

def create_model_with_config(model_type: str, base_config: ModelConfig, device: torch.device) -> TransformerTrunk:
    """Create a model with specific first layer type."""
    config = ModelConfig(
        d_model=base_config.d_model,
        n_heads=base_config.n_heads,
        n_layers=base_config.n_layers,
        seq_vocab=base_config.seq_vocab,
        struct_vocab=base_config.struct_vocab,
        max_len=base_config.max_len,
        dropout=base_config.dropout,
        ff_mult=base_config.ff_mult,
        ff_hidden_dim=base_config.ff_hidden_dim,
        # Consensus-specific parameters
        consensus_num_iterations=base_config.consensus_num_iterations,
        consensus_connectivity_type=base_config.consensus_connectivity_type,
        consensus_w=base_config.consensus_w,
        consensus_r=base_config.consensus_r,
        consensus_edge_hidden_dim=base_config.consensus_edge_hidden_dim
    )
    # Set model_type attribute
    config.model_type = model_type
    return TransformerTrunk(config, use_adaln=True).to(device)

def ensure_identical_parameters_all_models(models: Dict[str, TransformerTrunk], seed: int):
    """Ensure all models have identical parameters where possible.
    
    Strategy:
    1. Set random seed for reproducible initialization
    2. Copy shared embeddings, output layers, and time embeddings from first model
    3. For transformer layers beyond the first (which is architecture-specific),
       copy entire StandardTransformerBlock state dicts when both models use them
    
    Args:
        models: Dictionary mapping model type to model instance
        seed: Random seed for consistent initialization
    """
    if len(models) == 0: return
    
    torch.manual_seed(seed)
    ref_model = next(iter(models.values()))
    
    with torch.no_grad():
        for model_type, model in models.items():
            if model is not ref_model:
                # Copy embeddings
                model.seq_embed.load_state_dict(ref_model.seq_embed.state_dict())
                model.struct_embed.load_state_dict(ref_model.struct_embed.state_dict())
                
                # Copy output layers
                model.final_norm.load_state_dict(ref_model.final_norm.state_dict())
                model.seq_logits.load_state_dict(ref_model.seq_logits.state_dict())
                model.struct_logits.load_state_dict(ref_model.struct_logits.state_dict())
                
                # Copy time embedding network if present (for diffusion models)
                if hasattr(model, 'time_embed') and hasattr(ref_model, 'time_embed'):
                    model.time_embed.load_state_dict(ref_model.time_embed.state_dict())
                
                # For layers beyond the first (which is architecture-specific),
                # copy entire StandardTransformerBlocks when both models use them
                for i in range(1, min(len(model.layers), len(ref_model.layers))):
                    if (isinstance(model.layers[i], StandardTransformerBlock) and 
                        isinstance(ref_model.layers[i], StandardTransformerBlock)):
                        model.layers[i].load_state_dict(ref_model.layers[i].state_dict())

# --------------------------------------------------------------------------- #
#  Training utilities                                                         #
# --------------------------------------------------------------------------- #
def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def discrete_diffusion_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: ModelConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    """Perform a single step with discrete diffusion."""
    seq_x_t, struct_x_t, = batch.masked_data['seq'], batch.masked_data['struct']
    seq_x_0, struct_x_0 = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    seq_valid, struct_valid = ~batch.beospank['seq'], ~batch.beospank['struct']
    coords_x_t, coords_x_0 = batch.masked_data['coords'], batch.unmasked_data['coords']
    B, L = seq_x_t.shape

    nonspecial_elements_coords = (~batch.masks['coords'] & ~batch.beospank['coords']).bool()
    #assert not (~unmasked_coords_elements.any(dim=1).any()) # Dataloader should have gauranteed this.
    assert nonspecial_elements_coords.any(dim=1).all() # Need at least one real residue in each sequence
    
    # Pass raw timestep indices following DiT convention
    timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
    cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
    inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
    timesteps = timesteps.float().unsqueeze(-1)
    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t)
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass with time conditioning
        if train_cfg.model_type in ("GA", "RA"): outputs = model(inputs, coords_x_t, nonspecial_elements_coords, timesteps)
        else: outputs = model(inputs, timesteps=timesteps)
        seq_logits, struct_logits = outputs
        
        # Compute losses using score entropy loss
        loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, cumulative_noise, inst_noise, model_cfg.seq_absorb_token, valid_mask=seq_valid)
        loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, cumulative_noise, inst_noise, model_cfg.struct_absorb_token, valid_mask=struct_valid)
        
        # Total loss
        loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
        
        if train_mode:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics
        return {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item()}

def main():
    # Initialize configurations
    model_cfg = ModelConfig()
    diffusion_cfg = DiffusionConfig()
    train_cfg = TrainingConfig()
    train_cfg.masking_strategy = "discrete_diffusion"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Validate model type
    valid_types = {"SA", "GA", "RA", "SC"}
    if train_cfg.model_type not in valid_types: raise ValueError(f"Invalid model type: {train_cfg.model_type}. Must be one of {valid_types}")
    
    # Arrays to store validation metrics
    all_metrics = {
        'val_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
        'val_seq_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
        'val_struct_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs))
    }
    
    print(f"Starting discrete diffusion training with {train_cfg.num_iter} iterations")
    print(f"Training model: {train_cfg.model_type}")
    print(f"Noise schedule: {diffusion_cfg.noise_schedule} with sigma_min={diffusion_cfg.sigma_min}, sigma_max={diffusion_cfg.sigma_max}")
    print(f"Number of timesteps: {diffusion_cfg.num_timesteps}")
    
    # -------------------- Iteration loop -------------------- #
    for iteration in range(train_cfg.num_iter):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{train_cfg.num_iter}")
        print(f"{'='*60}\n")
        
        # Create model with fixed seed for this iteration
        print(f"Creating model for iteration {iteration + 1}...")
        torch.manual_seed(train_cfg.reference_model_seed + iteration)
        
        # Ensure identical parameter initialization across architectures
        if train_cfg.model_type == "SA":
            # For SA training, just create SA model directly (no synchronization needed)
            print(f"Creating {train_cfg.model_type} model...")
            model = create_model_with_config("SA", model_cfg, device)
        else:
            # For non-SA training, create SA reference and target model, then synchronize
            print(f"Creating SA reference model and {train_cfg.model_type} target model...")
            sa_model = create_model_with_config("SA", model_cfg, device)
            target_model = create_model_with_config(train_cfg.model_type, model_cfg, device)
            
            # Synchronize target model with SA reference
            print(f"Synchronizing {train_cfg.model_type} shared parameters with SA reference...")
            temp_models = {"SA": sa_model, train_cfg.model_type: target_model}
            ensure_identical_parameters_all_models(temp_models, train_cfg.reference_model_seed + iteration)
            
            # Keep target model, delete SA reference
            model = target_model
            del sa_model; del temp_models

        optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate)

        # Load checkpoint with dynamic path based on model type
        #TODO this directory really, really should be configurable.
        #TODO also, we should be using os.path.join rather than / wherever possible.
        encoder_checkpoint_path = f"../checkpoints/fsq/{train_cfg.model_type}_stage_1_iter1_{train_cfg.masking_strategy}.pt"
        checkpoint = torch.load(encoder_checkpoint_path, map_location=device, weights_only=False)
        encoder_state = {k.removeprefix('encoder.'): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        fsq_config = SimpleNamespace(**checkpoint['model_cfg_dict'])
        fsq_encoder = FSQEncoder(fsq_config)
        
        fsq_encoder.load_state_dict(encoder_state)
        print(f"Loaded {train_cfg.model_type} encoder weights from: {encoder_checkpoint_path}")
        
        fsq_encoder.eval()
        fsq_encoder.requires_grad_(False)
        fsq_encoder = fsq_encoder.to(device)
        
        # Print parameter count (only on first iteration)
        if iteration == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"{train_cfg.model_type} total parameters: {total_params:,}")
        
        # -------------------- Data loading -------------------- #
        # Set seed for dataset split
        data_seed = train_cfg.reference_model_seed + iteration * 1000
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        random.seed(data_seed)
        
        dataset = ProteinDataset(train_cfg.data_dir, max_length=model_cfg.max_len - 2)
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders with fixed seed for consistent masking
        g_train = torch.Generator()
        g_train.manual_seed(data_seed)
        g_val = torch.Generator()
        g_val.manual_seed(data_seed + 5000)
        
        diffusion_tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
        train_loader = DiffusionDataLoader(train_ds, model_cfg, train_cfg, diffusion_cfg, diffusion_tracks, device, fsq_encoder=fsq_encoder, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
        val_loader = DiffusionDataLoader(val_ds, model_cfg, train_cfg, diffusion_cfg, diffusion_tracks, device, fsq_encoder=fsq_encoder, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
        
        # Initialize tracking
        history = {'train_loss': [], 'train_seq_loss': [], 'train_struct_loss': [], 'val_loss': [], 'val_seq_loss': [], 'val_struct_loss': []}
        
        # -------------------- Training loop -------------------- #
        for epoch in range(train_cfg.max_epochs):
            # Training metrics accumulators
            train_metrics_sum = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0}
            num_batches = 0

            # Training
            with tqdm(train_loader, desc=f"Iter {iteration+1}/{train_cfg.num_iter}, Epoch {epoch+1}/{train_cfg.max_epochs} [{train_cfg.model_type} Train]",
                     ascii=True, leave=True, ncols=150) as pbar:
                for batch_data in pbar:
                    # Skip empty/None batches
                    if batch_data is None: continue
                    
                    # Train single model on batch
                    batch_metrics = discrete_diffusion_step(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=True)
                    
                    # Accumulate metrics
                    for key in train_metrics_sum: train_metrics_sum[key] += batch_metrics[key]
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({f'{train_cfg.model_type}_loss': f"{batch_metrics['loss']:.3f}"})
            
            # Calculate epoch averages
            for key in train_metrics_sum: train_metrics_sum[key] /= num_batches
            history['train_loss'].append(train_metrics_sum['loss'])
            history['train_seq_loss'].append(train_metrics_sum['loss_seq'])
            history['train_struct_loss'].append(train_metrics_sum['loss_struct'])
            
            # -------------------- Validation -------------------- #
            val_metrics_sum = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0}
            val_num_batches = 0
            
            for batch_data in val_loader:
                # Skip empty/None batches
                if batch_data is None: continue
                
                # Validate single model on batch
                batch_metrics = discrete_diffusion_step(model, optimizer, batch_data, model_cfg, train_cfg, train_mode=False)
                
                # Accumulate metrics
                for key in val_metrics_sum: val_metrics_sum[key] += batch_metrics[key]
                val_num_batches += 1
            
            # Calculate epoch averages
            for key in val_metrics_sum: val_metrics_sum[key] /= val_num_batches
            
            # Store in history
            history['val_loss'].append(val_metrics_sum['loss'])
            history['val_seq_loss'].append(val_metrics_sum['loss_seq'])
            history['val_struct_loss'].append(val_metrics_sum['loss_struct'])
            
            # Store in global metrics arrays
            all_metrics['val_loss'][iteration, epoch] = val_metrics_sum['loss']
            all_metrics['val_seq_loss'][iteration, epoch] = val_metrics_sum['loss_seq']
            all_metrics['val_struct_loss'][iteration, epoch] = val_metrics_sum['loss_struct']
            
            # Print epoch summary
            print(f"\nIteration {iteration+1}, Epoch {epoch+1}/{train_cfg.max_epochs} - {train_cfg.model_type}:")
            # Training metrics
            print(f"  Train:")
            print(f"    Score Entropy Loss: {train_metrics_sum['loss']:.4f} (Seq: {train_metrics_sum['loss_seq']:.4f}, Struct: {train_metrics_sum['loss_struct']:.4f})")
            
            # Validation metrics
            print(f"  Val:")
            print(f"    Score Entropy Loss: {val_metrics_sum['loss']:.4f} (Seq: {val_metrics_sum['loss_seq']:.4f}, Struct: {val_metrics_sum['loss_struct']:.4f})")
                    
        # Save final checkpoints only for the first iteration
        if iteration == 0:
            final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_discrete_diffusion_iter{iteration+1}_final.pt"
            torch.save({
                'iteration': iteration + 1,
                'epoch': train_cfg.max_epochs,
                'model_type': train_cfg.model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'diffusion_config': diffusion_cfg,
                'model_config': model_cfg
            }, final_checkpoint_path)
            print(f"\nSaved final checkpoint for {train_cfg.model_type} iteration {iteration+1}")
        else:
            print(f"\nSkipping checkpoint save for {train_cfg.model_type} iteration {iteration+1} (only saving iteration 1)")
    
    # -------------------- Save metrics to CSV -------------------- #
    # Save validation losses
    loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_discrete_diffusion_val_loss.csv"
    np.savetxt(loss_csv_path, all_metrics['val_loss'], delimiter=',',
               header=f"Validation score entropy losses for {train_cfg.model_type} (discrete diffusion)\n"
                     f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
    
    # Save sequence losses
    seq_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_discrete_diffusion_seq_val_loss.csv"
    np.savetxt(seq_loss_csv_path, all_metrics['val_seq_loss'], delimiter=',', 
               header=f"Sequence validation score entropy losses for {train_cfg.model_type} (discrete diffusion)\n"
                     f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
    
    # Save structure losses
    struct_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_discrete_diffusion_struct_val_loss.csv"
    np.savetxt(struct_loss_csv_path, all_metrics['val_struct_loss'], delimiter=',',
               header=f"Structure validation score entropy losses for {train_cfg.model_type} (discrete diffusion)\n"
                     f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
    
    print(f"\nSaved metrics for {train_cfg.model_type}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("DISCRETE DIFFUSION TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"Model trained: {train_cfg.model_type}")
    print(f"Number of iterations: {train_cfg.num_iter}")
    print(f"Number of epochs per iteration: {train_cfg.max_epochs}")
    print(f"Noise schedule: sigma_min={diffusion_cfg.sigma_min}, sigma_max={diffusion_cfg.sigma_max}")
    
    print(f"\n{train_cfg.model_type}:")
    print(f"  Final validation loss: {all_metrics['val_loss'][:, -1].mean():.4f} "
          f"± {all_metrics['val_loss'][:, -1].std():.4f}")
    print(f"  Best validation loss: {all_metrics['val_loss'].min():.4f}")

if __name__ == "__main__":
    main() 