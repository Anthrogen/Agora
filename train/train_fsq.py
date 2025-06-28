"""
Train the FSQ in two stages.

Parameter Initialization Strategy:
- Regardless of which architecture is being trained (SA, GA, RA, C), all four
  architectures are initialized with identical parameters where possible.
- The function ensure_identical_parameters_all_architectures creates temporary
  models for all architectures and synchronizes their parameters:
  1. All architectures get identical embeddings, self-attention, feedforward, 
     and output layers from SA architecture
  2. GA and RA get identical geometric/reflexive attention parameters
  3. This ensures fair comparison by removing initialization variance
  
This means when training any architecture, it has the same starting point as
the others would have had, allowing us to isolate the effect of architectural
differences on training dynamics.
"""
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
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Callable, List, Dict
import random

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.autoencoder import Autoencoder, StandardTransformerBlock
from src.dataloader import MaskedBatch, SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader
from src.dataset import ProteinDataset
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss

# --------------------------------------------------------------------------- #
#  Configurations                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # FSQ parameters
    fsq_dim: int = 5
    fsq_levels: list[int] = field(default_factory=lambda: [7, 5, 5, 5, 5])
    stage: str = "stage_1" # "stage_1" or "stage_2"
    fsq_encoder = None
    
    # Transformer parameters
    model_type: Optional[str] = None # Placeholder for model type
    d_model: int = 128 # 768  # Model dimensions
    latent_dim: int = 32
    n_heads: int = 4 # 12
    n_layers: int = 3 # 12
    seq_vocab: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab: int = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
    max_len: int = 2048
    dropout: float = 0.1  # Other architecture params
    ff_mult: int = 4
    ff_hidden_dim: int = d_model * ff_mult
    
    # Consensus-specific parameters
    consensus_num_iterations: int = 1  # Number of consensus gradient iterations
    consensus_connectivity_type: str = "local_window"  # "local_window" or "top_w"
    consensus_w: int = 2  # Window size for local_window, or w value for top_w
    consensus_r: int = 8  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim: int = 24  # Hidden dim for edge networks

@dataclass
class DiffusionConfig:
    """Discrete diffusion configuration."""
    # Noise schedule parameters
    noise_schedule: str = "linear"  # Type of noise schedule
    sigma_min: float = 0.31  # Minimum noise level
    sigma_max: float = 5.68  # Maximum noise level
    num_timesteps: int = 100  # Number of discrete timesteps for training
    
    # Absorbing state tokens (using MASK token index)
    seq_absorb_token: int = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
    struct_absorb_token: int = SPECIAL_TOKENS.MASK.value + 4375

@dataclass
class TrainingConfig:
    """Training process configuration."""
    model_types: List[str] = field(default_factory=lambda: ["SA","GA","RA","SC"]) # Models to train - can be any subset of ["SA", "GA", "RA", "SC"]
    batch_size: int = 4  # Training hyperparameters
    max_epochs: int = 70
    learning_rate: float = 1e-5
    num_iter: int = 3  # Number of iterations to repeat training
    masking_strategy: str = "simple" # Masking strategy: 'simple' or 'complex' or 'discrete_diffusion'
    
    if masking_strategy == "simple":
        mask_prob_seq: float = 0.2 # Masking probability for sequence tokens
        mask_prob_coords: float = 0.2 # Masking probability for structure tokens

    data_dir: str = "../sample_data/1k.csv"  # Data paths
    checkpoint_dir: str = "../checkpoints/fsq"  # Checkpointing
    reference_model_seed: int = 22 # Reference model seed for consistent parameter initialization across architectures

def create_model_with_config(model_type: str, base_config: ModelConfig, device: torch.device) -> Autoencoder:
    """Create a model with specific first layer type."""
    
    # For stage 1, use the model_type and n_layers from the base config
    if base_config.stage == "stage_1":
        actual_model_type = model_type
        n_layers = base_config.n_layers

    # For stage 2, override model_type to SA and n_layers to 10
    elif base_config.stage == "stage_2":
        actual_model_type = "SA"  # Always use SA for stage 2
        n_layers = 10  # Always use 10 layers for stage 2
    
    config = ModelConfig(
        model_type=actual_model_type,
        fsq_dim=base_config.fsq_dim,
        fsq_levels=base_config.fsq_levels,
        stage=base_config.stage,
        d_model=base_config.d_model,
        latent_dim=base_config.latent_dim,
        n_heads=base_config.n_heads,
        n_layers=n_layers,
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
    return Autoencoder(config).to(device)

def ensure_identical_parameters_all_models(models: Dict[str, Autoencoder], seed: int):
    """Ensure all models have identical parameters where possible.
    
    This is only relevant for stage 1 where different architectures (SA, GA, RA, C) 
    need to start from the same initialization for fair comparison.
    
    In stage 2, all models use SA architecture so this isn't needed.
    
    Strategy:
    1. Set random seed for reproducible initialization
    2. Copy shared encoder components from first model to all others
    3. Copy shared decoder components from first model to all others
    
    Args:
        models: Dictionary mapping model type to model instance
        seed: Random seed for consistent initialization
    """
    if len(models) == 0: return
    torch.manual_seed(seed) # Set random seed
    ref_model = next(iter(models.values())) # Use first model as reference
    
    with torch.no_grad():
        # For stage 1, synchronize parameters across different architectures
        for model_type, model in models.items():
            if model is not ref_model:
                # Copy encoder components
                # Input projection and conv blocks are shared
                model.encoder.input_proj.load_state_dict(ref_model.encoder.input_proj.state_dict())
                model.encoder.encoder_conv1.load_state_dict(ref_model.encoder.encoder_conv1.state_dict())
                model.encoder.encoder_conv2.load_state_dict(ref_model.encoder.encoder_conv2.state_dict())
                model.encoder.encoder_proj.load_state_dict(ref_model.encoder.encoder_proj.state_dict())
                
                # Copy decoder components  
                # Input projection and conv blocks are shared
                model.decoder.decoder_input.load_state_dict(ref_model.decoder.decoder_input.state_dict())
                model.decoder.decoder_conv1.load_state_dict(ref_model.decoder.decoder_conv1.state_dict())
                model.decoder.decoder_conv2.load_state_dict(ref_model.decoder.decoder_conv2.state_dict())
                model.decoder.output_proj.load_state_dict(ref_model.decoder.output_proj.state_dict())
                
                # Handle encoder transformer layers
                # Skip first layer as it's architecture-specific
                # Copy remaining StandardTransformerBlocks where possible
                for i in range(1, min(len(model.encoder.layers), len(ref_model.encoder.layers))):
                    if (type(model.encoder.layers[i]) == type(ref_model.encoder.layers[i]) and
                        isinstance(model.encoder.layers[i], StandardTransformerBlock)):
                        model.encoder.layers[i].load_state_dict(ref_model.encoder.layers[i].state_dict())
                
                # Handle decoder transformer layers similarly
                for i in range(1, min(len(model.decoder.layers), len(ref_model.decoder.layers))):
                    if (type(model.decoder.layers[i]) == type(ref_model.decoder.layers[i]) and
                        isinstance(model.decoder.layers[i], StandardTransformerBlock)):
                        model.decoder.layers[i].load_state_dict(ref_model.decoder.layers[i].state_dict())

# --------------------------------------------------------------------------- #
#  Training utilities                                                          #
# --------------------------------------------------------------------------- #
def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_step(models: Dict[str, Autoencoder], optimizers: Dict[str, torch.optim.Optimizer], batch: MaskedBatch, model_cfg: ModelConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Perform a single training step for all models with the same batch."""
    metrics = {} # Store metrics for each model
    
    if model_cfg.stage == "stage_1":

        # Stage 1: Masked coordinate reconstruction
        B, L, H, _ = batch.masked_data['coords'].shape
        
        # Train each model on the same batch
        for model_type, model in models.items():
            model.train()
            optimizer = optimizers[model_type]

            unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
            assert unmasked_elements.any(dim=1).all()
            
            # Forward pass - use only first 3 atoms for standard coordinates
            three_atom_masked_coords = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]
            if model_type in ("GA", "RA"): x_rec, _ = model(three_atom_masked_coords, batch.masked_data['coords'], unmasked_elements)
            else: x_rec, _ = model(three_atom_masked_coords)

            # In order to run KABSCH, we need to isolate only unmasked residues into a [U, 3, 3] tensor for each protein in the batch, where U is number of unmasked residues in a given protein.
            pts_pred = []; pts_true = []
            for batch_idx in range(B):
                real_residues = torch.arange(L, device=device)[unmasked_elements[batch_idx]]
                pred_coords = x_rec[batch_idx][real_residues]  # [U, 3, 3]
                true_coords = batch.masked_data['coords'][batch_idx, real_residues, :3, :]  # [U, 3, 3] - only first 3 atoms!

                # Flatten to [1, U*3, 3]
                pts_pred.append(pred_coords.reshape(1, -1, 3))
                pts_true.append(true_coords.reshape(1, -1, 3))
            
            # Compute squared Kabsch RMSD loss and regular RMSD
            if pts_pred:
                loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
                # Also compute regular Kabsch RMSD for reporting
                with torch.no_grad():
                    rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
            else:
                loss = torch.tensor(0.0, device=device)
                rmsd = torch.tensor(0.0, device=device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store metrics
            metrics[model_type] = {'loss': loss.item(), 'rmsd': rmsd.item()}
    
    elif model_cfg.stage == "stage_2":
        # Stage 2: Full structure reconstruction from frozen encoder
        B, L, H, _ = batch.masked_data['coords'].shape
        
        # Create coord_mask for GA/RA models
        unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
        assert unmasked_elements.any(dim=1).all()
        
        # Train each model on the same batch
        for model_type, model in models.items():
            model.train()
            model.encoder.eval()  # Encoder is frozen
            optimizer = optimizers[model_type]
            
            # Forward pass through frozen encoder to get z_q
            with torch.no_grad():
                four_atom = batch.masked_data['coords'][:, :, :4, :]  # [B, L, 4, 3] for encoder
                three_atom = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]
                if model_type in ("GA", "RA"): z_q, _ = model.encoder(three_atom, four_atom, unmasked_elements)
                else: z_q, _ = model.encoder(three_atom)
            
            # Zero out BOS/EOS/PAD positions in z_q
            z_q[batch.beospank['coords']] = 0.0
            
            # Concatenate z_q with seq_tokens along last dimension
            # z_q: [B, L, fsq_dim], seq_tokens: [B, L] -> [B, L, 1]
            seq_tokens_float = batch.masked_data['seq'].unsqueeze(-1).float()  # [B, L, 1]
            decoder_input = torch.cat([z_q, seq_tokens_float], dim=-1)  # [B, L, fsq_dim + 1]
            
            # Decoder forward pass
            if model_type in ("GA", "RA"): x_rec = model.decoder(decoder_input, four_atom, unmasked_elements)
            else: x_rec = model.decoder(decoder_input)
            
            # x_rec is [B, L, 14, 3] for stage 2
            # Compute loss on all valid positions (no masking in stage 2)
            pts_pred = []; pts_true = []
            for batch_idx in range(B):
                # real_residues = unmasked_indices[batch_idx]
                real_residues = torch.arange(L, device=device)[unmasked_elements[batch_idx]]
                pred_coords = x_rec[batch_idx][real_residues]  # [M, 14, 3] 
                true_coords = batch.masked_data['coords'][batch_idx][real_residues]  # [M, 14, 3]

                # Flatten to [1, M*14, 3]
                pts_pred.append(pred_coords.reshape(1, -1, 3))
                pts_true.append(true_coords.reshape(1, -1, 3))
            
            # Compute squared Kabsch RMSD loss
            if pts_pred:
                loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
                # Also compute regular Kabsch RMSD for reporting
                with torch.no_grad():
                    rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
            else:
                loss = torch.tensor(0.0, device=device)
                rmsd = torch.tensor(0.0, device=device)
            
            # Backward pass (only decoder parameters will update)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store metrics
            metrics[model_type] = {'loss': loss.item(), 'rmsd': rmsd.item()}
    
    return metrics

def validate_step(models: Dict[str, Autoencoder], batch: MaskedBatch, model_cfg: ModelConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Perform a single validation step for all models with the same batch."""
    metrics = {} # Store metrics for each model
    
    if model_cfg.stage == "stage_1":
        # Stage 1: Masked coordinate reconstruction
        B, L, H, _ = batch.masked_data['coords'].shape
        
        # Create coord_mask for GA/RA models (valid positions that are not masked)
        unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
        assert unmasked_elements.any(dim=1).all()
        
        # Evaluate each model on the same batch
        for model_type, model in models.items():
            model.eval()
            
            with torch.no_grad():
                # Forward pass - use only first 3 atoms for standard coordinates
                three_atoms = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]

                if model_type in ("GA", "RA"): x_rec, _ = model(three_atoms, batch.masked_data['coords'], unmasked_elements)
                else: x_rec, _ = model(three_atoms)
                
                # Compute loss on unmasked positions
                # Extract valid (unmasked) coordinates
                pts_pred = []; pts_true = []
                for batch_idx in range(B):
                    real_residues = torch.arange(L, device=device)[unmasked_elements[batch_idx]]
                    pred_coords = x_rec[batch_idx][real_residues]  # [M, 3, 3]
                    true_coords = batch.masked_data['coords'][batch_idx, real_residues, :3, :]  # [M, 3, 3] - only first 3 atoms!

                    # Flatten to [1, M*3, 3]
                    pts_pred.append(pred_coords.reshape(1, -1, 3))
                    pts_true.append(true_coords.reshape(1, -1, 3))
                
                # Compute squared Kabsch RMSD loss and regular RMSD
                if pts_pred:
                    loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
                    rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
                else:
                    loss = torch.tensor(0.0, device=device)
                    rmsd = torch.tensor(0.0, device=device)
                
                # Store metrics
                metrics[model_type] = {'loss': loss.item(), 'rmsd': rmsd.item()}
    
    elif model_cfg.stage == "stage_2":
        # Stage 2: Full structure reconstruction from frozen encoder
        B, L, H, _ = batch.masked_data['coords'].shape
        
        # Create coord_mask for GA/RA models
        unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
        assert unmasked_elements.any(dim=1).all()
        
        # Evaluate each model on the same batch
        for model_type, model in models.items():
            model.eval()
            
            with torch.no_grad():
                # Forward pass through frozen encoder to get z_q
                four_atoms = batch.masked_data['coords'][:, :, :4, :]  # [B, L, 4, 3] for encoder
                three_atoms = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]
                if model_type in ("GA", "RA"): z_q, _ = model.encoder(three_atoms, four_atoms, unmasked_elements)
                else: z_q, _ = model.encoder(three_atoms)
                
                # Zero out BOS/EOS/PAD positions in z_q
                z_q[batch.beospank['coords']] = 0.0
                
                # Concatenate z_q with seq_tokens along last dimension
                # z_q: [B, L, fsq_dim], seq_tokens: [B, L] -> [B, L, 1]
                seq_tokens_float = batch.masked_data['seq'].unsqueeze(-1).float()  # [B, L, 1]
                decoder_input = torch.cat([z_q, seq_tokens_float], dim=-1)  # [B, L, fsq_dim + 1]
                
                # Decoder forward pass
                if model_type in ("GA", "RA"): x_rec = model.decoder(decoder_input, four_atoms, unmasked_elements)
                else: x_rec = model.decoder(decoder_input)
                
                # x_rec is [B, L, 14, 3] for stage 2
                # Compute loss on all valid positions (no masking in stage 2)
                pts_pred = []; pts_true = []
                for batch_idx in range(B):
                    real_residues = torch.arange(L, device=device)[unmasked_elements[batch_idx]]
                    pred_coords = x_rec[batch_idx][real_residues]  # [M, 14, 3] 
                    true_coords = batch.masked_data['coords'][batch_idx][real_residues]  # [M, 14, 3]

                    # Flatten to [1, M*14, 3]
                    pts_pred.append(pred_coords.reshape(1, -1, 3))
                    pts_true.append(true_coords.reshape(1, -1, 3))
                
                # Compute squared Kabsch RMSD loss
                if pts_pred:
                    loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
                    # Also compute regular Kabsch RMSD for reporting
                    with torch.no_grad():
                        rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
                else:
                    loss = torch.tensor(0.0, device=device)
                    rmsd = torch.tensor(0.0, device=device)
                
                # Store metrics
                metrics[model_type] = {'loss': loss.item(), 'rmsd': rmsd.item()}
    
    return metrics

def main():
    # Initialize configurations
    model_cfg = ModelConfig()
    diffusion_cfg = DiffusionConfig()
    train_cfg = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Validate model types
    valid_types = {"SA", "GA", "RA", "SC"}
    for mt in train_cfg.model_types:
        if mt not in valid_types:
            raise ValueError(f"Invalid model type: {mt}. Must be one of {valid_types}")
    
    # Arrays to store validation metrics for all models across iterations
    all_metrics = {
        model_type: {
            'val_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
            'val_rmsd': np.zeros((train_cfg.num_iter, train_cfg.max_epochs))
        }
        for model_type in train_cfg.model_types
    }
    
    print(f"Starting training with {train_cfg.num_iter} iterations")
    print(f"Training models: {train_cfg.model_types}")
    print(f"Stage: {model_cfg.stage}")
    if model_cfg.stage == "stage_1": print(f"Using masking strategy: {train_cfg.masking_strategy}")
    elif model_cfg.stage == "stage_2": print(f"Will load encoders from stage 1 checkpoints")
    
    # Determine dataset mode based on stage
    if model_cfg.stage == "stage_1": dataset_mode = "backbone"  # 4 atoms: N, CA, C, CB
    elif model_cfg.stage == "stage_2": dataset_mode = "side_chain"  # 14 atoms
    
    # -------------------- Iteration loop -------------------- #
    for iteration in range(train_cfg.num_iter):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{train_cfg.num_iter}")
        print(f"{'='*60}\n")
        
        # Create all models with fixed seed for this iteration
        print(f"Creating {len(train_cfg.model_types)} models for iteration {iteration + 1}...")
        torch.manual_seed(train_cfg.reference_model_seed + iteration)
        
        # Create models
        models = {}; optimizers = {}
        for model_type in train_cfg.model_types:
            models[model_type] = create_model_with_config(model_type, model_cfg, device)
            
            if model_cfg.stage == "stage_1":
                # Stage 1: optimize all parameters
                optimizers[model_type] = AdamW(models[model_type].parameters(), lr=train_cfg.learning_rate)
            
            # Load encoder weights and freeze for stage 2
            elif model_cfg.stage == "stage_2":
                # Construct encoder checkpoint path based on model type and iteration
                encoder_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_stage_1_iter{iteration+1}.pt"
                
                if encoder_checkpoint_path.exists():
                    checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
                    # Load encoder state dict
                    encoder_state = {k.removeprefix('encoder.'): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
                    models[model_type].encoder.load_state_dict(encoder_state)
                    print(f"Loaded encoder weights for {model_type} from {encoder_checkpoint_path}")
                else:
                    raise ValueError(f"No stage 1 checkpoint found for {model_type}. Please train stage 1 first.")
                
                # Freeze encoder parameters
                for param in models[model_type].encoder.parameters():
                    param.requires_grad = False
                
                # Only optimize decoder parameters
                optimizers[model_type] = AdamW(models[model_type].decoder.parameters(), lr=train_cfg.learning_rate)

        # Ensure consistent parameter initialization across architectures (stage 1 only)
        if model_cfg.stage == "stage_1":
            print(f"Ensuring all models have identical parameters...")
            ensure_identical_parameters_all_models(models, train_cfg.reference_model_seed + iteration)
        
        # Print parameter count (only on first iteration)
        if iteration == 0:
            for model_type, model in models.items():
                if model_cfg.stage == "stage_1":
                    total_params = sum(p.numel() for p in model.parameters())
                else:
                    total_params = sum(p.numel() for p in model.decoder.parameters())
                print(f"{model_type} {'total' if model_cfg.stage == 'stage_1' else 'decoder'} parameters: {total_params:,}")
        
        # -------------------- Data loading -------------------- #
        # Set seed for dataset split AND masking to ensure consistency
        data_seed = train_cfg.reference_model_seed + iteration * 1000
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        random.seed(data_seed)
        
        dataset = ProteinDataset(train_cfg.data_dir, mode=dataset_mode, max_length=model_cfg.max_len - 2) # Reserve 2 positions for BOS/EOS
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders with fixed seed for consistent masking
        g_train = torch.Generator()
        g_train.manual_seed(data_seed)
        g_val = torch.Generator()
        g_val.manual_seed(data_seed + 5000)
        
        # Use different masking strategies for stage 1 vs stage 2
        if model_cfg.stage == "stage_1":
            stage_1_tracks = {'seq': False, 'struct': False, 'coords': True}
            min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
            train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, stage_1_tracks, device, diffusion_cfg=diffusion_cfg, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
            val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, stage_1_tracks, device, diffusion_cfg=diffusion_cfg, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)

        elif model_cfg.stage == "stage_2":  # stage_2 - no masking
            stage_2_tracks = {'seq': True, 'struct': False, 'coords': True}
            train_loader = NoMaskDataLoader(train_ds, model_cfg, train_cfg, stage_2_tracks, device, batch_size=train_cfg.batch_size, shuffle=True, generator=g_train, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)
            val_loader = NoMaskDataLoader(val_ds, model_cfg, train_cfg, stage_2_tracks, device, batch_size=train_cfg.batch_size, shuffle=False, generator=g_val, worker_init_fn=worker_init_fn, min_unmasked=min_unmasked)

        
        # Initialize tracking for each model
        history = {model_type: {'train_loss': [], 'train_rmsd': [], 'val_loss': [], 'val_rmsd': []} for model_type in train_cfg.model_types}
        
        # -------------------- Training loop -------------------- #
        for epoch in range(train_cfg.max_epochs):
            # Training metrics accumulators
            train_metrics_sum = {model_type: {'loss': 0.0, 'rmsd': 0.0} for model_type in train_cfg.model_types}
            num_batches = 0

            # Training
            with tqdm(train_loader, desc=f"Iter {iteration+1}/{train_cfg.num_iter}, Epoch {epoch+1}/{train_cfg.max_epochs} [Train]", 
                     ascii=True, leave=True, ncols=150, position=0) as pbar:
                for batch_data in pbar:
                    # Skip empty/None batches
                    if batch_data is None: continue
                    
                    # Train all models on the same batch
                    batch_metrics = train_step(models, optimizers, batch_data, model_cfg, device)
                    
                    # Accumulate metrics
                    for model_type in train_cfg.model_types:
                        for key in train_metrics_sum[model_type]:
                            train_metrics_sum[model_type][key] += batch_metrics[model_type][key]
                    num_batches += 1
                    
                    # Update progress bar with metrics from all models
                    postfix = {}
                    for model_type in train_cfg.model_types:
                        postfix[f'{model_type}_loss'] = f"{batch_metrics[model_type]['loss']:.3f}"
                    pbar.set_postfix(postfix)
            
            # Calculate epoch averages for training
            for model_type in train_cfg.model_types:
                for key in train_metrics_sum[model_type]:
                    train_metrics_sum[model_type][key] /= num_batches
                
                history[model_type]['train_loss'].append(train_metrics_sum[model_type]['loss'])
                history[model_type]['train_rmsd'].append(train_metrics_sum[model_type]['rmsd'])
            
            # -------------------- Validation -------------------- #
            val_metrics_sum = {
                model_type: {'loss': 0.0, 'rmsd': 0.0}
                for model_type in train_cfg.model_types
            }
            val_num_batches = 0
            
            for batch_data in val_loader:
                # Skip empty/None batches
                if batch_data is None: continue
                
                # Validate all models on the same batch
                batch_metrics = validate_step(models, batch_data, model_cfg, device)
                
                # Accumulate metrics
                for model_type in train_cfg.model_types:
                    for key in val_metrics_sum[model_type]:
                        val_metrics_sum[model_type][key] += batch_metrics[model_type][key]
                val_num_batches += 1
            
            # Calculate epoch averages for validation
            for model_type in train_cfg.model_types:
                for key in val_metrics_sum[model_type]:
                    val_metrics_sum[model_type][key] /= val_num_batches
                
                # Store in history
                history[model_type]['val_loss'].append(val_metrics_sum[model_type]['loss'])
                history[model_type]['val_rmsd'].append(val_metrics_sum[model_type]['rmsd'])
                
                # Store in global metrics arrays
                all_metrics[model_type]['val_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss']
                all_metrics[model_type]['val_rmsd'][iteration, epoch] = val_metrics_sum[model_type]['rmsd']
            
            # Print detailed epoch summary
            print(f"\nIteration {iteration+1}, Epoch {epoch+1}/{train_cfg.max_epochs}")
            for model_type in train_cfg.model_types:
                print(f"\n{model_type}:")
                print(f"  Train - Loss (squared RMSD): {train_metrics_sum[model_type]['loss']:.4f}, "
                      f"RMSD: {train_metrics_sum[model_type]['rmsd']:.4f}")
                print(f"  Val   - Loss (squared RMSD): {val_metrics_sum[model_type]['loss']:.4f}, "
                      f"RMSD: {val_metrics_sum[model_type]['rmsd']:.4f}")
        
        # Save final checkpoints only for the first iteration
        if iteration == 0:
            for model_type, model in models.items():
                # Get the actual model config from the model
                model_cfg_dict = asdict(model.cfg)
                
                checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_{model_cfg.stage}_iter{iteration+1}_{train_cfg.masking_strategy}.pt"
                torch.save({'model_state_dict': model.state_dict(), 'model_cfg_dict': model_cfg_dict}, checkpoint_path)
            print(f"\nSaved checkpoints for iteration {iteration+1}")
    
    # -------------------- Save validation metrics to CSV -------------------- #
    for model_type in train_cfg.model_types:
        # Save validation losses
        loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_{model_cfg.stage}_val_loss_{train_cfg.masking_strategy}.csv"
        np.savetxt(loss_csv_path, all_metrics[model_type]['val_loss'], delimiter=',', 
                   header=f"Validation squared RMSD losses for {model_type} {model_cfg.stage}\nRows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})",
                   comments='# ')
        
        # Save validation RMSDs
        rmsd_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_{model_cfg.stage}_val_rmsd_{train_cfg.masking_strategy}.csv"
        np.savetxt(rmsd_csv_path, all_metrics[model_type]['val_rmsd'], delimiter=',',
                   header=f"Validation RMSDs for {model_type} {model_cfg.stage}\nRows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})",
                   comments='# ')
        
        print(f"\nSaved metrics for {model_type}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Stage: {model_cfg.stage}")
    print(f"Models trained: {train_cfg.model_types}")
    print(f"Number of iterations: {train_cfg.num_iter}")
    print(f"Number of epochs per iteration: {train_cfg.max_epochs}")
    
    for model_type in train_cfg.model_types:
        print(f"\n{model_type}:")
        print(f"  Validation RMSD:")
        print(f"    Mean final epoch: {all_metrics[model_type]['val_rmsd'][:, -1].mean():.4f} ± {all_metrics[model_type]['val_rmsd'][:, -1].std():.4f}")
        print(f"    Best single run: {all_metrics[model_type]['val_rmsd'].min():.4f}")

if __name__ == "__main__":
    main() 