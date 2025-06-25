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
    n_heads: int = 8 # 12
    n_layers: int = 3 # 12
    seq_vocab: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab: int = 7*5*5*5*5 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
    max_len: int = 2048
    dropout: float = 0.1  # Other architecture params
    ff_mult: int = 4
    ff_hidden_dim: int = d_model * ff_mult
    
    # Consensus-specific parameters
    consensus_num_iterations: int = 5  # Number of Consensus gradient iterations
    consensus_connectivity_type: str = "local_window"  # "local_window" or "top_w"
    consensus_w: int = 4  # Window size for local_window, or w value for top_w
    consensus_r: int = 24  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim: int = 12  # Hidden dim for edge networks


@dataclass
class DiffusionConfig:
    """Discrete diffusion configuration."""
    # Noise schedule parameters
    noise_schedule: str = "skewed_rectangular"  # Type of noise schedule ("linear", "inverted_u", or "skewed_rectangular")
    sigma_min: float = 0.31  # Minimum noise level
    sigma_max: float = 5.68  # Maximum noise level
    num_timesteps: int = 100  # Number of discrete timesteps for training
    
    # Absorbing state tokens (using MASK token index)
    seq_absorb_token: int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
    struct_absorb_token: int = SPECIAL_TOKENS.MASK.value + 4375

@dataclass
class TrainingConfig:
    """Training process configuration."""
    model_types: List[str] = field(default_factory=lambda: ["C"]) # Models to train - can be any subset of ["SA", "GA", "RA", "C"]
    batch_size: int = 4  # Training hyperparameters
    max_epochs: int = 200
    learning_rate: float = 1e-5
    num_iter: int = 3  # Number of iterations to repeat training

    # Loss weights
    seq_loss_weight: float = 1.0
    struct_loss_weight: float = 1.0

    data_dir: str = "../sample_data/1k"  # Data paths
    checkpoint_dir: str = "../checkpoints/transformer_trunk_tmp"  # Checkpointing
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

def train_step(models: Dict[str, TransformerTrunk], optimizers: Dict[str, torch.optim.Optimizer], 
               batch: MaskedBatch, diffusion_cfg: DiffusionConfig,
               train_cfg: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Perform a single training step for all models with discrete diffusion."""
    seq_x_t, struct_x_t, = batch.masked_data['seq'], batch.masked_data['struct']
    seq_x_0, struct_x_0 = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    seq_valid, struct_valid = ~batch.beospad['seq'], ~batch.beospad['struct']
    coords_x_t, coords_x_0 = batch.masked_data['coords'], batch.unmasked_data['coords']
    B, L = seq_x_t.shape

    nonspecial_elements_coords = (~batch.masks['coords'] & ~batch.beospad['coords']).bool()
    #assert not (~unmasked_coords_elements.any(dim=1).any()) # Dataloader should have gauranteed this.
    assert nonspecial_elements_coords.any(dim=1).all() # Need at least one real residue in each sequence
    
    # Pass raw timestep indices following DiT convention
    timesteps = batch.metadata['timestep_indices'].float().unsqueeze(-1) # Timesteps shoudl be the same across all tracks of a protein, though masks are not.
    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t)
    metrics = {}
    
    # Train each model on the same batch
    for model_type, model in models.items():
        model.train()
        optimizer = optimizers[model_type]
        
        # Forward pass with time conditioning
        if model_type in ("GA", "RA"): outputs = model(inputs, coords_x_t, nonspecial_elements_coords, timesteps)
        else: outputs = model(inputs, timesteps=timesteps)
        seq_logits, struct_logits = outputs
        
        # Compute losses using score entropy loss (for training)
        loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, batch.metadata['cumulative_noise'], batch.metadata['inst_noise'], diffusion_cfg.seq_absorb_token, valid_mask=seq_valid)
        loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, batch.metadata['cumulative_noise'], batch.metadata['inst_noise'], diffusion_cfg.struct_absorb_token, valid_mask=struct_valid)
        
        # Total loss (what we train on)
        loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store metrics
        metrics[model_type] = {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item()}
    
    return metrics

def validate_step(models: Dict[str, TransformerTrunk], batch: MaskedBatch, 
                 diffusion_cfg: DiffusionConfig, 
                 train_cfg: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Perform a single validation step for all models."""
    seq_x_t, struct_x_t, = batch.masked_data['seq'], batch.masked_data['struct']
    seq_x_0, struct_x_0 = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    seq_valid, struct_valid = ~batch.beospad['seq'], ~batch.beospad['struct']
    coords_x_t, coords_x_0 = batch.masked_data['coords'], batch.unmasked_data['coords']
    B, L = seq_x_t.shape

    nonspecial_elements_coords = (~batch.masks['coords'] & ~batch.beospad['coords']).bool()
    #assert not (~unmasked_coords_elements.any(dim=1).any()) # Dataloader should gaurantee this.
    assert nonspecial_elements_coords.any(dim=1).all()

    timesteps = batch.metadata['timestep_indices'] if 'timestep_indices' in batch.metadata else batch.metadata['pseudo_timestep_indices']
    cumulative_noise = batch.metadata['cumulative_noise'] if 'cumulative_noise' in batch.metadata else batch.metadata['pseudo_cumulative_noise']
    inst_noise = batch.metadata['inst_noise'] if 'inst_noise' in batch.metadata else batch.metadata['pseudo_inst_noise']
    timesteps = timesteps.float().unsqueeze(-1)

    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t)
    metrics = {}
    
    # Evaluate each model
    for model_type, model in models.items():
        model.eval()
        
        with torch.no_grad():
            # Forward pass with time conditioning
            if model_type in ("GA", "RA"): outputs = model(inputs, coords_x_t, coord_mask=nonspecial_elements_coords, timesteps=timesteps)
            else: outputs = model(inputs, timesteps=timesteps)
            seq_logits, struct_logits = outputs
            
            # Compute losses using score entropy loss (main loss)
            loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, cumulative_noise, inst_noise, diffusion_cfg.seq_absorb_token, valid_mask=seq_valid)
            loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, cumulative_noise, inst_noise, diffusion_cfg.struct_absorb_token, valid_mask=struct_valid)
            
            # Total loss
            loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
            
            # Store metrics
            metrics[model_type] = {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item()}
    
    return metrics

def main():
    # Initialize configurations
    model_cfg = ModelConfig()
    diffusion_cfg = DiffusionConfig()
    train_cfg = TrainingConfig()
    train_cfg.masking_strategy = "discrete_diffusion"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Validate model types
    valid_types = {"SA", "GA", "RA", "C"}
    for mt in train_cfg.model_types:
        if mt not in valid_types:
            raise ValueError(f"Invalid model type: {mt}. Must be one of {valid_types}")
    
    # Arrays to store validation metrics
    all_metrics = {
        model_type: {
            'val_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
            'val_seq_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
            'val_struct_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs))
        }
        for model_type in train_cfg.model_types
    }
    
    print(f"Starting discrete diffusion training with {train_cfg.num_iter} iterations")
    print(f"Training models: {train_cfg.model_types}")
    print(f"Noise schedule: {diffusion_cfg.noise_schedule} with sigma_min={diffusion_cfg.sigma_min}, sigma_max={diffusion_cfg.sigma_max}")
    print(f"Number of timesteps: {diffusion_cfg.num_timesteps}")
    
    # -------------------- Iteration loop -------------------- #
    for iteration in range(train_cfg.num_iter):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{train_cfg.num_iter}")
        print(f"{'='*60}\n")
        
        # Create models with fixed seed for this iteration
        torch.manual_seed(train_cfg.reference_model_seed + iteration)
        
        # Create models and load corresponding FSQ encoders
        # Each model type loads its own pre-trained FSQ encoder from a separate checkpoint
        # This allows each architecture to have been trained with architecture-specific autoencoders
        models = {}; optimizers = {}; fsq_encoders = {}
        
        for model_type in train_cfg.model_types:
            # Create model
            models[model_type] = create_model_with_config(model_type, model_cfg, device)
            optimizers[model_type] = AdamW(models[model_type].parameters(), lr=train_cfg.learning_rate)

            # Load checkpoint with dynamic path based on model type
            #TODO this directory really, really should be configurable.
            #TODO also, we should be using os.path.join rather than / wherever possible.
            encoder_checkpoint_path = f"../checkpoints/fsq/{model_type}_stage_1_iter1_{train_cfg.masking_strategy}.pt"
            checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
            encoder_state = {k.removeprefix('encoder.'): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
            fsq_config = SimpleNamespace(**checkpoint['model_cfg_dict'])
            fsq_encoder = FSQEncoder(fsq_config)
            
            fsq_encoder.load_state_dict(encoder_state)
            print(f"Loaded {model_type} encoder weights from: {encoder_checkpoint_path}")
            
            fsq_encoder.eval()
            fsq_encoder.requires_grad_(False)
            fsq_encoder = fsq_encoder.to(device)
            fsq_encoders[model_type] = fsq_encoder
        
        # Ensure consistent parameter initialization
        print(f"Ensuring all models have identical parameters...")
        ensure_identical_parameters_all_models(models, train_cfg.reference_model_seed + iteration)
        
        # Print parameter count (only on first iteration)
        if iteration == 0:
            for model_type, model in models.items():
                total_params = sum(p.numel() for p in model.parameters())
                print(f"{model_type} total parameters: {total_params:,}")
        
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
        
        # Create DataLoaders with fixed seed for consistent masking: replace w/ model_types[model_idx]
        g_train = torch.Generator()
        g_train.manual_seed(data_seed)
        g_val = torch.Generator()
        g_val.manual_seed(data_seed + 5000)
        

        diffusion_tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
        train_loader = DiffusionDataLoader(train_ds, model_cfg, train_cfg, diffusion_cfg, diffusion_tracks, device, fsq_encoder=fsq_encoders[train_cfg.model_types[0]], batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
        val_loader = DiffusionDataLoader(val_ds, model_cfg, train_cfg, diffusion_cfg, diffusion_tracks, device, fsq_encoder=fsq_encoders[train_cfg.model_types[0]], batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
        
        # Initialize tracking
        history = {
            model_type: {'train_loss': [], 'train_seq_loss': [], 'train_struct_loss': [],
                'val_loss': [], 'val_seq_loss': [], 'val_struct_loss': []
            }
            for model_type in train_cfg.model_types
        }
        
        # -------------------- Training loop -------------------- #
        for epoch in range(train_cfg.max_epochs):
            # Training metrics accumulators
            train_metrics_sum = {model_type: {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0} for model_type in train_cfg.model_types}
            num_batches = 0

            # Training
            with tqdm(train_loader, desc=f"Iter {iteration+1}/{train_cfg.num_iter}, Epoch {epoch+1}/{train_cfg.max_epochs} [Train]",
                     ascii=True, leave=True, ncols=150) as pbar:
                for batch_data in pbar:
                    # Train all models on the same batch
                    batch_metrics = train_step(models, optimizers, batch_data, diffusion_cfg, train_cfg)
                    
                    # Accumulate metrics
                    for model_type in train_cfg.model_types:
                        for key in train_metrics_sum[model_type]:
                            train_metrics_sum[model_type][key] += batch_metrics[model_type][key]
                    num_batches += 1
                    
                    # Update progress bar
                    postfix = {}
                    for model_type in train_cfg.model_types:
                        postfix[f'{model_type}_loss'] = f"{batch_metrics[model_type]['loss']:.3f}"
                    pbar.set_postfix(postfix)
            
            # Calculate epoch averages
            for model_type in train_cfg.model_types:
                for key in train_metrics_sum[model_type]:
                    train_metrics_sum[model_type][key] /= num_batches
                
                history[model_type]['train_loss'].append(train_metrics_sum[model_type]['loss'])
                history[model_type]['train_seq_loss'].append(train_metrics_sum[model_type]['loss_seq'])
                history[model_type]['train_struct_loss'].append(train_metrics_sum[model_type]['loss_struct'])
            
            # -------------------- Validation -------------------- #
            val_metrics_sum = {
                model_type: {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0}
                for model_type in train_cfg.model_types
            }
            val_num_batches = 0
            
            for batch_data in val_loader:
                # Validate all models on the same batch
                batch_metrics = validate_step(models, batch_data, diffusion_cfg, train_cfg)
                
                # Accumulate metrics
                for model_type in train_cfg.model_types:
                    for key in val_metrics_sum[model_type]:
                        val_metrics_sum[model_type][key] += batch_metrics[model_type][key]
                val_num_batches += 1
            
            # Calculate epoch averages
            for model_type in train_cfg.model_types:
                for key in val_metrics_sum[model_type]:
                    val_metrics_sum[model_type][key] /= val_num_batches
                
                # Store in history
                history[model_type]['val_loss'].append(val_metrics_sum[model_type]['loss'])
                history[model_type]['val_seq_loss'].append(val_metrics_sum[model_type]['loss_seq'])
                history[model_type]['val_struct_loss'].append(val_metrics_sum[model_type]['loss_struct'])
                
                # Store in global metrics arrays
                all_metrics[model_type]['val_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss']
                all_metrics[model_type]['val_seq_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss_seq']
                all_metrics[model_type]['val_struct_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss_struct']
            
            # Print epoch summary
            print(f"\nIteration {iteration+1}, Epoch {epoch+1}/{train_cfg.max_epochs}")
            for model_type in train_cfg.model_types:
                print(f"\n{model_type}:")
                # Training metrics
                print(f"  Train:")
                print(f"    Score Entropy Loss: {train_metrics_sum[model_type]['loss']:.4f} "
                      f"(Seq: {train_metrics_sum[model_type]['loss_seq']:.4f}, "
                      f"Struct: {train_metrics_sum[model_type]['loss_struct']:.4f})")
                
                # Validation metrics
                print(f"  Val:")
                print(f"    Score Entropy Loss: {val_metrics_sum[model_type]['loss']:.4f} "
                      f"(Seq: {val_metrics_sum[model_type]['loss_seq']:.4f}, "
                      f"Struct: {val_metrics_sum[model_type]['loss_struct']:.4f})")
                    
        # Save final checkpoints only for the first iteration
        if iteration == 0:
            for model_type in train_cfg.model_types:
                final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_discrete_diffusion_iter{iteration+1}_final.pt"
                torch.save({
                    'iteration': iteration + 1,
                    'epoch': train_cfg.max_epochs,
                    'model_type': model_type,
                    'model_state_dict': models[model_type].state_dict(),
                    'optimizer_state_dict': optimizers[model_type].state_dict(),
                    'history': history[model_type],
                    'diffusion_config': diffusion_cfg,
                    'model_config': model_cfg
                }, final_checkpoint_path)
            print(f"\nSaved final checkpoints for iteration {iteration+1}")
        else:
            print(f"\nSkipping checkpoint save for iteration {iteration+1} (only saving iteration 1)")
    
    # -------------------- Save metrics to CSV -------------------- #
    for model_type in train_cfg.model_types:
        # Save validation losses
        loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_diffusion_val_loss.csv"
        np.savetxt(loss_csv_path, all_metrics[model_type]['val_loss'], delimiter=',',
                   header=f"Validation score entropy losses for {model_type} (discrete diffusion)\n"
                         f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        # Save sequence losses
        seq_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_diffusion_seq_val_loss.csv"
        np.savetxt(seq_loss_csv_path, all_metrics[model_type]['val_seq_loss'], delimiter=',', 
                   header=f"Sequence validation score entropy losses for {model_type} (discrete diffusion)\n"
                         f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        # Save structure losses
        struct_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_diffusion_struct_val_loss.csv"
        np.savetxt(struct_loss_csv_path, all_metrics[model_type]['val_struct_loss'], delimiter=',',
                   header=f"Structure validation score entropy losses for {model_type} (discrete diffusion)\n"
                         f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        print(f"\nSaved metrics for {model_type}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("DISCRETE DIFFUSION TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"Models trained: {train_cfg.model_types}")
    print(f"Number of iterations: {train_cfg.num_iter}")
    print(f"Number of epochs per iteration: {train_cfg.max_epochs}")
    print(f"Noise schedule: sigma_min={diffusion_cfg.sigma_min}, sigma_max={diffusion_cfg.sigma_max}")
    
    for model_type in train_cfg.model_types:
        print(f"\n{model_type}:")
        print(f"  Final validation loss: {all_metrics[model_type]['val_loss'][:, -1].mean():.4f} "
              f"± {all_metrics[model_type]['val_loss'][:, -1].std():.4f}")
        print(f"  Best validation loss: {all_metrics[model_type]['val_loss'].min():.4f}")

if __name__ == "__main__":
    main() 
