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
from src.data_util.dataset import ProteinDataset
from src.dataloader_trunk import DiffusionDataLoader 
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS

# --------------------------------------------------------------------------- #
#  Configurations                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128 # 768  # Model dimensions
    n_heads: int = 8 # 12
    n_layers: int = 3 # 12
    seq_vocab: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab: int = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
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
    noise_schedule: str = "geometric"  # Type of noise schedule
    sigma_min: float = 0.1  # Minimum noise level
    sigma_max: float = 10.0  # Maximum noise level
    num_timesteps: int = 100  # Number of discrete timesteps for training
    
    # Absorbing state tokens (using MASK token index)
    seq_absorb_token: int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
    struct_absorb_token: int = SPECIAL_TOKENS.MASK.value + 4375

@dataclass
class TrainingConfig:
    """Training process configuration."""
    model_types: List[str] = field(default_factory=lambda: ["C"]) # Models to train - can be any subset of ["SA", "GA", "RA", "C"]
    batch_size: int = 4  # Training hyperparameters
    max_epochs: int = 25
    learning_rate: float = 1e-5
    num_iter: int = 3  # Number of iterations to repeat training

    # Loss weights
    seq_loss_weight: float = 1.0
    struct_loss_weight: float = 1.0

    data_dir: str = "../sample_data/1k"  # Data paths
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
#  Discrete Diffusion Utilities                                                #
# --------------------------------------------------------------------------- #
def score_entropy_loss(output, x_0, x_t, cumulative_noise_levels, inst_noise_levels, mask_token, validation):
    """
    Score entropy loss from SEDD paper.
    
    Args:
        output: Model predictions of shape (B, L, V)
        x_0: Original tokens of shape (B, L)
        x_t: Noisy tokens of shape (B, L)
        cumulative_noise_levels: Cumulative noise at time t, shape (B, 1)
        inst_noise_levels: Instantaneous noise at time t, shape (B, 1)
        mask_token: Index of the absorbing/mask token
        validation: Whether to compute training or validation loss
    
    Returns:
        Loss value
    """

    B, L, V = output.shape
    
    # Output represents log score ratios, apply exp to get ratios p_t(y)/p_t(x^i)
    # This ensures the ratios are positive
    output = torch.exp(output)

    # Create one-hot encoding of x_t
    # x_t shape: [B, L] with values in [0, V-1]
    x_t_onecold = 1.0 - F.one_hot(x_t, num_classes=V).float()  # [B, L, V]
    
    # Calculate delta = output @ (1 - one_hot_{x_t})
    # This computes the dot product along the vocabulary dimension
    # (1 - x_t_onehot) zeros out the position corresponding to x_t
    #x_t_onecold = 1.0 - x_t_onehot  # [B, L, V]
    
    # Compute dot product: sum over vocabulary dimension
    # delta[b, l] = sum_v output[b, l, v] * x_t_onecold[b, l, v]
    delta = (output * x_t_onecold).sum(dim=-1)  # [B, L]
    
    # The delta tensor now has shape [B, L] where:
    # delta[b, l] = sum of all output[b, l, v] except where v == x_t[b, l]

    # More numerically stable computation of base
    # base = (1 - exp(-σ)) / exp(-σ) = exp(σ) - 1
    base = torch.exp(cumulative_noise_levels) - 1.0
    base = base.unsqueeze(1)
    masked_positions = x_t == mask_token
    alpha = 1.0 - 2.0 * masked_positions.float()

    # Define "opposite": if x_t is masked, opposite is x_0; if x_t is not masked, opposite is mask_token
    opposite = torch.where(masked_positions, x_0, torch.full_like(x_t, mask_token))
    
    # epsilon_1[b, l] = output[b, l, opposite[b, l]]
    # Gather values from output at opposite positions
    # batch_indices = torch.arange(B, device=output.device).unsqueeze(1).expand(B, L)  # [B, L]
    # position_indices = torch.arange(L, device=output.device).unsqueeze(0).expand(B, L)  # [B, L]
    
    #epsilon_1 = output[batch_indices, position_indices, opposite]  # [B, L]
    epsilon_1 = torch.gather(output, dim=2, index=opposite.unsqueeze(-1)).squeeze(-1)
    
    # epsilon_2: base^alpha
    # base has shape [B, 1, 1] after unsqueeze, alpha has shape [B, L]
    # We need to squeeze base back to [B, 1] for proper broadcasting
    base = base.squeeze(-1)  # [B, 1, 1] -> [B, 1]
    epsilon_2 = torch.pow(base, alpha)  # [B, L]
    epsilon_2 = torch.clamp(epsilon_2, min=1e-10)

    # Clip epsilon_1 to prevent log of very small numbers
    epsilon_1 = torch.clamp(epsilon_1, min=1e-10)
    epsilon = epsilon_2 * torch.log(epsilon_1)

    gamma = (delta - epsilon) # Gamma is (B, L)
    K = epsilon_2 * (torch.log(epsilon_2) - 1) # (B, L)
    gamma += K # now >= 0 element-wise
    
    # inst_noise_levels has shape [B, 1], we want to broadcast with gamma [B, L]
    # We should NOT unsqueeze, as [B, 1] will broadcast correctly to [B, L]
    per_residue = gamma * inst_noise_levels  # [B, L] * [B, 1] -> [B, L]

    if validation:
        # num_masked = masked_positions.sum(dim=1) # (B,)
        # valid = num_masked > 0  # keep samples that have masks
        # if valid.any(): # at least one valid sequence
        #     per_protein = (per_residue.sum(dim=1)[valid] / num_masked[valid]) # (B_valid,)
        # else: # rare corner-case: skip batch
        #     return torch.tensor(0., dtype=gamma.dtype, device=gamma.device, requires_grad=False)
        per_protein = per_residue.sum(dim=1) / L
    else:
        per_protein = per_residue.sum(dim=1) # (B,)

    return per_protein.mean(0) # scalar

# --------------------------------------------------------------------------- #
#  Training utilities                                                         #
# --------------------------------------------------------------------------- #
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask_token: int) -> float:
    """Calculate accuracy for non-masked positions."""
    non_masked = labels != mask_token
    if not non_masked.any():
        return 0.0
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & non_masked
    return (correct.sum().float() / non_masked.sum().float()).item()

def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_step(models: Dict[str, TransformerTrunk], optimizers: Dict[str, torch.optim.Optimizer], 
               batch_data: Tuple, diffusion_cfg: DiffusionConfig, model_cfg: ModelConfig, 
               train_cfg: TrainingConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Perform a single training step for all models with discrete diffusion."""
    # Unpack batch data (now includes masks)
    x_t, x_0, masks, timestep_indices, cumulative_noise_levels, inst_noise, lengths, coords = batch_data
    seq_x_t, struct_x_t = x_t
    seq_x_0, struct_x_0 = x_0
    seq_mask, struct_mask = masks
    B, L = seq_x_t.shape
    
    # Create coord_mask for GA/RA models
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    valid_start = 1  # After BOS
    valid_end = lengths.unsqueeze(1) + 1  # Before EOS
    coord_mask = (positions >= valid_start) & (positions < valid_end)
    # Exclude positions where structure is masked
    coord_mask = coord_mask & (~struct_mask)
    
    # Ensure at least one position is valid in coord_mask for each sequence
    all_masked = ~coord_mask.any(dim=1)  # Check which sequences have all positions masked
    if all_masked.any():
        for idx in torch.where(all_masked)[0]:
            # When all positions are masked, add dummy coordinates to prevent NaNs
            # Create minimal valid coordinates at position 1 (after BOS)
            # Set non-collinear coordinates that form a proper triangle:
            # N at origin, CA displaced in x, C displaced in x and y
            coords[idx, 1, 0, :] = torch.tensor([0., 0., 0.], device=device)  # N
            coords[idx, 1, 1, :] = torch.tensor([1.5, 0., 0.], device=device)  # CA  
            coords[idx, 1, 2, :] = torch.tensor([2.4, 1.3, 0.], device=device)  # C
            # Update coord_mask for this position
            coord_mask[idx, 1] = True
    
    # Pass raw timestep indices following DiT convention
    timesteps = timestep_indices.float()
    timesteps = timesteps.unsqueeze(-1)  # [B] -> [B, 1]
    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t)
    metrics = {}
    
    # Train each model on the same batch
    for model_type, model in models.items():
        model.train()
        optimizer = optimizers[model_type]
        
        # Forward pass with time conditioning
        if model_type in ("GA", "RA"): outputs = model(inputs, coords, coord_mask, timesteps)
        else: outputs = model(inputs, timesteps=timesteps)
        seq_logits, struct_logits = outputs
        
        # Compute losses using score entropy loss (for training)
        loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, cumulative_noise_levels, inst_noise, diffusion_cfg.seq_absorb_token, validation=False)
        loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, cumulative_noise_levels, inst_noise, diffusion_cfg.struct_absorb_token, validation=False)
        
        # Total loss (what we train on)
        loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
        
        # Calculate accuracies (for monitoring)
        seq_acc = calculate_accuracy(seq_logits.view(-1, model_cfg.seq_vocab), seq_x_0.view(-1), diffusion_cfg.seq_absorb_token)
        struct_acc = calculate_accuracy(struct_logits.view(-1, model_cfg.struct_vocab), struct_x_0.view(-1), diffusion_cfg.struct_absorb_token)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Store metrics
        metrics[model_type] = {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item(), 'seq_acc': seq_acc, 'struct_acc': struct_acc}
    
    return metrics

def validate_step(models: Dict[str, TransformerTrunk], batch_data: Tuple, 
                 diffusion_cfg: DiffusionConfig, model_cfg: ModelConfig, 
                 train_cfg: TrainingConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Perform a single validation step for all models."""
    # Unpack batch data (now includes masks)
    x_t, x_0, masks, timestep_indices, cumulative_noise_levels, inst_noise, lengths, coords = batch_data
    seq_x_t, struct_x_t = x_t
    seq_x_0, struct_x_0 = x_0
    seq_mask, struct_mask = masks
    B, L = seq_x_t.shape
    
    # Create coord_mask for GA/RA models
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    valid_start = 1  # After BOS
    valid_end = lengths.unsqueeze(1) + 1  # Before EOS
    coord_mask = (positions >= valid_start) & (positions < valid_end)
    # Exclude positions where structure is masked
    coord_mask = coord_mask & (~struct_mask)
    
    # Ensure at least one position is valid in coord_mask for each sequence
    all_masked = ~coord_mask.any(dim=1)  # Check which sequences have all positions masked
    if all_masked.any():
        for idx in torch.where(all_masked)[0]:
            # When all positions are masked, add dummy coordinates to prevent NaNs
            # Create minimal valid coordinates at position 1 (after BOS)
            # Set non-collinear coordinates that form a proper triangle:
            # N at origin, CA displaced in x, C displaced in x and y
            coords[idx, 1, 0, :] = torch.tensor([0., 0., 0.], device=device)  # N
            coords[idx, 1, 1, :] = torch.tensor([1.5, 0., 0.], device=device)  # CA  
            coords[idx, 1, 2, :] = torch.tensor([2.4, 1.3, 0.], device=device)  # C
            # Update coord_mask for this position
            coord_mask[idx, 1] = True
    
    # Pass raw timestep indices following DiT convention
    timesteps = timestep_indices.float()
    timesteps = timesteps.unsqueeze(-1)  # [B] -> [B, 1]
    
    # Prepare inputs
    inputs = (seq_x_t, struct_x_t)
    metrics = {}
    
    # Evaluate each model
    for model_type, model in models.items():
        model.eval()
        
        with torch.no_grad():
            # Forward pass with time conditioning
            if model_type in ("GA", "RA"): outputs = model(inputs, coords, coord_mask=coord_mask, timesteps=timesteps)
            else: outputs = model(inputs, timesteps=timesteps)
            seq_logits, struct_logits = outputs
            
            # Compute losses using score entropy loss (main loss)
            loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, cumulative_noise_levels, inst_noise, diffusion_cfg.seq_absorb_token, validation=True)
            loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, cumulative_noise_levels, inst_noise,diffusion_cfg.struct_absorb_token, validation=True)
            
            # Total loss
            loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
            
            # Calculate accuracies
            seq_acc = calculate_accuracy(seq_logits.view(-1, model_cfg.seq_vocab), seq_x_0.view(-1), diffusion_cfg.seq_absorb_token)
            struct_acc = calculate_accuracy(struct_logits.view(-1, model_cfg.struct_vocab), struct_x_0.view(-1), diffusion_cfg.struct_absorb_token)
            
            # Store metrics
            metrics[model_type] = {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item(), 'seq_acc': seq_acc, 'struct_acc': struct_acc}
    
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
            'val_struct_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
            'val_seq_acc': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
            'val_struct_acc': np.zeros((train_cfg.num_iter, train_cfg.max_epochs))
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
            encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
            fsq_config = SimpleNamespace(**checkpoint['model_cfg_dict'])
            fsq_encoder = FSQEncoder(fsq_config)
            
            # Handle key mapping for Consensus model (token_weight -> token_encoder.weight)
            if model_type == "C":
                new_encoder_state = {}
                for k, v in encoder_state.items():
                    if 'consensus.token_weight' in k:
                        new_k = k.replace('consensus.token_weight', 'consensus.token_encoder.weight')
                        new_encoder_state[new_k] = v
                    elif 'consensus.token_bias' in k:
                        new_k = k.replace('consensus.token_bias', 'consensus.token_encoder.bias')
                        new_encoder_state[new_k] = v
                    else:
                        new_encoder_state[k] = v
                encoder_state = new_encoder_state
            
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
        
        train_loader = DiffusionDataLoader(
            train_ds, fsq_encoder=fsq_encoders[train_cfg.model_types[0]], model_cfg=model_cfg, diffusion_cfg=diffusion_cfg, 
            device=device, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn
        )
        
        val_loader = DiffusionDataLoader(
            val_ds, fsq_encoder=fsq_encoders[train_cfg.model_types[0]], model_cfg=model_cfg, diffusion_cfg=diffusion_cfg, 
            device=device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn
        )
        
        # Initialize tracking
        history = {
            model_type: {'train_loss': [], 'train_seq_loss': [], 'train_struct_loss': [], 'train_seq_acc': [], 'train_struct_acc': [],
                'val_loss': [], 'val_seq_loss': [], 'val_struct_loss': [], 'val_seq_acc': [], 'val_struct_acc': []
            }
            for model_type in train_cfg.model_types
        }
        
        # -------------------- Training loop -------------------- #
        for epoch in range(train_cfg.max_epochs):
            # Training metrics accumulators
            train_metrics_sum = {model_type: {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0} for model_type in train_cfg.model_types}
            num_batches = 0

            # Training
            with tqdm(train_loader, desc=f"Iter {iteration+1}/{train_cfg.num_iter}, Epoch {epoch+1}/{train_cfg.max_epochs} [Train]",
                     ascii=True, leave=True, ncols=150) as pbar:
                for batch_data in pbar:
                    # Train all models on the same batch
                    batch_metrics = train_step(models, optimizers, batch_data, diffusion_cfg, model_cfg, train_cfg, device)
                    
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
                history[model_type]['train_seq_acc'].append(train_metrics_sum[model_type]['seq_acc'])
                history[model_type]['train_struct_acc'].append(train_metrics_sum[model_type]['struct_acc'])
            
            # -------------------- Validation -------------------- #
            val_metrics_sum = {
                model_type: {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0}
                for model_type in train_cfg.model_types
            }
            val_num_batches = 0
            
            for batch_data in val_loader:
                # Validate all models on the same batch
                batch_metrics = validate_step(models, batch_data, diffusion_cfg, model_cfg, train_cfg, device)
                
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
                history[model_type]['val_seq_acc'].append(val_metrics_sum[model_type]['seq_acc'])
                history[model_type]['val_struct_acc'].append(val_metrics_sum[model_type]['struct_acc'])
                
                # Store in global metrics arrays
                all_metrics[model_type]['val_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss']
                all_metrics[model_type]['val_seq_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss_seq']
                all_metrics[model_type]['val_struct_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss_struct']
                all_metrics[model_type]['val_seq_acc'][iteration, epoch] = val_metrics_sum[model_type]['seq_acc']
                all_metrics[model_type]['val_struct_acc'][iteration, epoch] = val_metrics_sum[model_type]['struct_acc']
            
            # Print epoch summary
            print(f"\nIteration {iteration+1}, Epoch {epoch+1}/{train_cfg.max_epochs}")
            for model_type in train_cfg.model_types:
                print(f"\n{model_type}:")
                # Training metrics
                print(f"  Train:")
                print(f"    Score Entropy Loss: {train_metrics_sum[model_type]['loss']:.4f} "
                      f"(Seq: {train_metrics_sum[model_type]['loss_seq']:.4f}, "
                      f"Struct: {train_metrics_sum[model_type]['loss_struct']:.4f})")
                print(f"    Accuracy: Seq: {train_metrics_sum[model_type]['seq_acc']:.4f}, "
                      f"Struct: {train_metrics_sum[model_type]['struct_acc']:.4f}")
                
                # Validation metrics
                print(f"  Val:")
                print(f"    Score Entropy Loss: {val_metrics_sum[model_type]['loss']:.4f} "
                      f"(Seq: {val_metrics_sum[model_type]['loss_seq']:.4f}, "
                      f"Struct: {val_metrics_sum[model_type]['loss_struct']:.4f})")
                print(f"    Accuracy: Seq: {val_metrics_sum[model_type]['seq_acc']:.4f}, "
                      f"Struct: {val_metrics_sum[model_type]['struct_acc']:.4f}")
                    
        # Save final checkpoints only for the first iteration
        if iteration == 0:
            for model_type in train_cfg.model_types:
                final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"transformer_trunk/{model_type}_discrete_diffusion_iter{iteration+1}_final.pt"
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
