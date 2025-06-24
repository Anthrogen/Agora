"""
Validation script to benchmark simple, complex, and discrete diffusion trained models.

This script evaluates models at different masking rates corresponding to specific 
discrete diffusion time indices. For each time index, we compute the corresponding
mask probability and evaluate all models with that masking rate.
"""
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random
from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.transformer import TransformerTrunk
from src.models.autoencoder import FSQEncoder
from src.data_util.dataset import ProteinDataset
from src.dataloader import DiffusionDataLoader, MaskedBatch, MaskingDataLoader
from src.losses import score_entropy_loss

# Import functions from training scripts
from train_transformer_discrete_diffusion_trunk import (
    ModelConfig as DiffusionModelConfig, DiffusionConfig,
    create_model_with_config as create_diffusion_model
)
from train_transformer_MLM_trunk import (
    ModelConfig as MLMModelConfig,
    create_model_with_config as create_mlm_model
)

@dataclass
class ValidationConfig:
    """Configuration for validation."""
    # Model type to evaluate (single selection)
    model_type: str = "C"  # Options: "SA", "GA", "RA", "C"
    
    # Time indices to evaluate (directly specified)
    time_indices: List[int] = field(default_factory=lambda: [0,9,19,29,39,49,59,69,79,89,99])
    
    # Training methods to evaluate (any combination)
    training_methods: List[str] = field(default_factory=lambda: ["simple", "complex", "diffusion"])
    
    # Data settings
    data_dir: str = "../sample_data/1k"
    batch_size: int = 4
    
    # Model paths (models in /scripts/checkpoints)
    simple_checkpoint_pattern: str = "checkpoints/{}_simple_iter1_final.pt"
    complex_checkpoint_pattern: str = "checkpoints/{}_complex_iter1_final.pt"
    diffusion_checkpoint_pattern: str = "../checkpoints/transformer_trunk/{}_discrete_diffusion_iter1_final.pt"
    
    # FSQ encoder paths (in /checkpoints, not /scripts/checkpoints)
    fsq_encoder_pattern: str = "../checkpoints/fsq/{}_stage_1_iter1_{}.pt"
    
    def __post_init__(self):
        # Validate model type
        valid_types = {"SA", "GA", "RA", "C"}
        if self.model_type not in valid_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_types}")
        
        # Validate training methods
        valid_methods = {"simple", "complex", "diffusion"}
        for method in self.training_methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid training method: {method}. Must be one of {valid_methods}")

def compute_mask_prob_for_time(t: int, diffusion_cfg: DiffusionConfig) -> float:
    """Compute mask probability for a given time index using the diffusion schedule."""
    from src.dataloader import get_noise_levels
    
    # Get noise levels
    inst_noise_levels, cumulative_noise_levels = get_noise_levels(
        diffusion_cfg.sigma_min,
        diffusion_cfg.sigma_max,
        diffusion_cfg.num_timesteps,
        schedule_type=diffusion_cfg.noise_schedule
    )
    
    # Get cumulative noise at time t
    cumulative_noise = cumulative_noise_levels[t]
    
    # Compute mask probability
    mask_prob = 1 - torch.exp(-cumulative_noise)
    
    return mask_prob.item()

def load_model_checkpoint(checkpoint_path: str, model_type: str, training_method: str, device: torch.device, 
                         fsq_encoder_pattern: str = None) -> Tuple[TransformerTrunk, Optional[FSQEncoder]]:
    """Load a model from checkpoint."""
    # Check if this is a diffusion model based on the training method
    is_diffusion = training_method == 'diffusion'
    
    # Fix pickle loading by temporarily setting ModelConfig in global namespace
    import __main__
    # Import the configs to ensure they're available in this scope
    from train_transformer_discrete_diffusion_trunk import ModelConfig as DiffusionModelConfig_local, DiffusionConfig as DiffusionConfig_local
    from train_transformer_MLM_trunk import ModelConfig as MLMModelConfig_local
    
    if is_diffusion:
        __main__.ModelConfig = DiffusionModelConfig_local
        __main__.DiffusionConfig = DiffusionConfig_local
    else: 
        __main__.ModelConfig = MLMModelConfig_local
    
    # ---------------------------------------
    # Load checkpoint weights first
    # ---------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if is_diffusion:
        cfg_obj = ckpt.get('model_config', None)
        cfg_dict = cfg_obj.__dict__ if hasattr(cfg_obj, '__dict__') else cfg_obj
        model_cfg = DiffusionModelConfig(**{k: v for k, v in cfg_dict.items() if k in DiffusionModelConfig.__dataclass_fields__})
        model = create_diffusion_model(model_type, model_cfg, device)
    else:
        cfg_obj = ckpt.get('model_config', None)
        cfg_dict = cfg_obj.__dict__ if hasattr(cfg_obj, '__dict__') else cfg_obj
        model_cfg = MLMModelConfig(**{k: v for k, v in cfg_dict.items() if k in MLMModelConfig.__dataclass_fields__})
        model = create_mlm_model(model_type, model_cfg, device)

    # Now load weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
        
    # Clean up global namespace
    if hasattr(__main__, 'ModelConfig'): 
        delattr(__main__, 'ModelConfig')
    if hasattr(__main__, 'DiffusionConfig'): 
        delattr(__main__, 'DiffusionConfig')
    
    # Load FSQ encoder using the correct training method
    fsq_encoder = None
    if fsq_encoder_pattern:
        # Map training method to the FSQ encoder suffix
        if training_method == 'diffusion':
            encoder_suffix = 'discrete_diffusion'
        else:
            encoder_suffix = training_method  # 'simple' or 'complex'
        
        encoder_checkpoint_path = fsq_encoder_pattern.format(model_type, encoder_suffix)
        
        if not os.path.exists(encoder_checkpoint_path):
            raise FileNotFoundError(f"FSQ encoder not found at: {encoder_checkpoint_path}")
        
        encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
        encoder_state = {k.removeprefix('encoder.'): v for k, v in encoder_checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        
        # Handle key mapping for Consensus model
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
        
        fsq_config = SimpleNamespace(**encoder_checkpoint['model_cfg_dict']) 
        fsq_encoder = FSQEncoder(fsq_config)
        
        fsq_encoder.load_state_dict(encoder_state)
        fsq_encoder.eval()
        fsq_encoder.requires_grad_(False)
        fsq_encoder = fsq_encoder.to(device)
    
    return model, fsq_encoder

def evaluate_mlm_model(model: TransformerTrunk, dataloader: DataLoader, model_type: str, device: torch.device) -> Dict[str, float]:
    """Evaluate an MLM model using cross-entropy loss over all valid positions."""
    total_loss = 0.0
    total_seq_loss = 0.0
    total_struct_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch using new MaskedBatch structure
            seq_x_t = batch.masked_data['seq']
            struct_x_t = batch.masked_data['struct']
            coords_x_t = batch.masked_data['coords']
            
            seq_tokens = batch.unmasked_data['seq']
            struct_tokens = batch.unmasked_data['struct']
            
            B, L = seq_x_t.shape
            
            # Use beospad mask for valid positions (excluding BOS/EOS/PAD)
            valid_mask = ~batch.beospad['seq']
            
            # Create coord_mask for GA/RA models
            unmasked_coords_elements = (~batch.masks['coords'] & ~batch.beospad['coords']).bool()
            
            # Ensure at least one position is valid in coord_mask for each sequence
            assert unmasked_coords_elements.any(dim=1).all()
            
            # Forward pass
            if model_type in ("GA", "RA"):
                outputs = model((seq_x_t, struct_x_t), coords_x_t, unmasked_coords_elements)
            else:
                outputs = model((seq_x_t, struct_x_t))
            
            seq_logits, struct_logits = outputs
            
            # Flatten tensors for loss computation
            seq_logits_flat = seq_logits.reshape(-1, seq_logits.size(-1))
            struct_logits_flat = struct_logits.reshape(-1, struct_logits.size(-1))
            seq_labels_flat = seq_tokens.reshape(-1)
            struct_labels_flat = struct_tokens.reshape(-1)
            valid_mask_flat = valid_mask.reshape(-1)
            
            # Compute cross-entropy loss over all valid positions (matching training)
            seq_loss = F.cross_entropy(
                seq_logits_flat[valid_mask_flat],
                seq_labels_flat[valid_mask_flat].long(),
                reduction='mean'
            )
            
            struct_loss = F.cross_entropy(
                struct_logits_flat[valid_mask_flat],
                struct_labels_flat[valid_mask_flat].long(),
                reduction='mean'
            )
            
            total_loss += (seq_loss + struct_loss).item()
            total_seq_loss += seq_loss.item()
            total_struct_loss += struct_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'seq_loss': total_seq_loss / num_batches,
        'struct_loss': total_struct_loss / num_batches
    }

def evaluate_diffusion_model(model: TransformerTrunk, batch: MaskedBatch, timestep: int, 
                           diffusion_cfg: DiffusionConfig, model_cfg: DiffusionModelConfig,
                           model_type: str, device: torch.device) -> Dict[str, float]:
    """Evaluate a diffusion model using score entropy loss on a single batch."""
    # Unpack batch using new MaskedBatch structure
    seq_x_t = batch.masked_data['seq']
    struct_x_t = batch.masked_data['struct']
    coords_x_t = batch.masked_data['coords']
    
    seq_x_0 = batch.unmasked_data['seq']
    struct_x_0 = batch.unmasked_data['struct']
    
    B, L = seq_x_t.shape
    
    # Create timestep tensors
    timestep_indices = torch.full((B,), timestep, dtype=torch.long, device=device)
    timesteps = timestep_indices.float().unsqueeze(-1)
    
    # Get noise levels for this timestep
    from src.dataloader import get_noise_levels
    inst_noise_levels, cumulative_noise_levels = get_noise_levels(
        diffusion_cfg.sigma_min,
        diffusion_cfg.sigma_max,
        diffusion_cfg.num_timesteps,
        schedule_type=diffusion_cfg.noise_schedule
    )
    cumulative_noise = cumulative_noise_levels[timestep].unsqueeze(0).unsqueeze(1).to(device)
    inst_noise = inst_noise_levels[timestep].unsqueeze(0).unsqueeze(1).to(device)
    
    # Use beospad mask for valid positions
    valid_mask = batch.beospad['seq']
    
    # Create coord_mask for GA/RA models
    unmasked_coords_elements = (~batch.masks['coords'] & ~batch.beospad['coords']).bool()
    
    # Ensure at least one position is valid in coord_mask for each sequence
    assert unmasked_coords_elements.any(dim=1).all()
    
    with torch.no_grad():
        # Forward pass
        if model_type in ("GA", "RA"):
            outputs = model((seq_x_t, struct_x_t), coords_x_t, unmasked_coords_elements, timesteps)
        else:
            outputs = model((seq_x_t, struct_x_t), timesteps=timesteps)
        
        seq_logits, struct_logits = outputs
        
        # Compute score entropy loss
        seq_loss = score_entropy_loss(
            seq_logits, seq_x_0, seq_x_t, cumulative_noise, inst_noise,
            diffusion_cfg.seq_absorb_token, valid_mask=valid_mask
        )
        
        struct_loss = score_entropy_loss(
            struct_logits, struct_x_0, struct_x_t, cumulative_noise, inst_noise,
            diffusion_cfg.struct_absorb_token, valid_mask=batch.beospad['coords']
        )
        
        total_loss = seq_loss + struct_loss
    
    return {
        'loss': total_loss.item(),
        'seq_loss': seq_loss.item(),
        'struct_loss': struct_loss.item()
    }

# Custom validation dataloader classes
class FixedTimestepDiffusionDataLoader(DiffusionDataLoader):
    """DiffusionDataLoader that uses a fixed timestep for validation."""
    
    def __init__(self, dataset, model_cfg, train_cfg, diffusion_cfg, tracks, fixed_timestep: int,
                 fsq_encoder=None, device=None, min_unmasked=None, **kwargs):
        super().__init__(dataset, model_cfg, train_cfg, diffusion_cfg, tracks, 
                        fsq_encoder=fsq_encoder, device=device, min_unmasked=min_unmasked, **kwargs)
        self.fixed_timestep = fixed_timestep
    
    def sample_masks(self, batch):
        # Use fixed timestep instead of random sampling
        timestep_indices = torch.full((batch.B,), self.fixed_timestep, dtype=torch.long, device=self.device)
        
        # Get corresponding noise levels
        cumulative_noise_level = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        inst_noise_levels = self.inst_noise_levels[timestep_indices].unsqueeze(1)

        batch.metadata = {'timestep_indices': timestep_indices, 'cumulative_noise': cumulative_noise_level, 'inst_noise': inst_noise_levels}

        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:
            mask_prob = 1 - torch.exp(-cumulative_noise_level)
            mask_prob_expanded = mask_prob.expand(batch.B, batch.L)
            desired_masks = torch.rand(batch.B, batch.L, device=self.device) < mask_prob_expanded
            desired_masks = desired_masks.bool()

            batch.masks[track] = desired_masks

class FixedMaskProbDataLoader(MaskingDataLoader):
    """MaskingDataLoader that uses a fixed mask probability for MLM validation."""
    
    def __init__(self, dataset, model_cfg, train_cfg, tracks, fixed_mask_prob: float,
                 fsq_encoder=None, device=None, min_unmasked=None, **kwargs):
        super().__init__(dataset, model_cfg, train_cfg, tracks,
                        fsq_encoder=fsq_encoder, device=device, min_unmasked=min_unmasked, **kwargs)
        self.fixed_mask_prob = fixed_mask_prob
    
    def sample_masks(self, batch):
        for track in [t for t in batch.tracks if (batch.tracks[t] and t != 'struct')]:
            # Create masks with fixed probability
            mask = torch.rand(batch.B, batch.L, device=self.device) < self.fixed_mask_prob
            batch.masks[track] = mask.bool()

def main(custom_config: Optional[ValidationConfig] = None):
    # Initialize configuration
    val_cfg = custom_config if custom_config is not None else ValidationConfig()
    mlm_model_cfg = MLMModelConfig()
    diffusion_model_cfg = DiffusionModelConfig()
    diffusion_cfg = DiffusionConfig()
    
    # Create dummy training config for dataloader compatibility
    @dataclass
    class DummyTrainingConfig:
        masking_strategy: str = "simple"
        
    train_cfg = DummyTrainingConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset (use max_len from either config, they should be the same)
    dataset = ProteinDataset(val_cfg.data_dir, max_length=mlm_model_cfg.max_len - 2)
    
    print(f"\nValidation dataset size: {len(dataset)}")
    print(f"Time indices to evaluate: {val_cfg.time_indices}")
    print(f"Model type to evaluate: {val_cfg.model_type}")
    print(f"Training methods to evaluate: {val_cfg.training_methods}")
    
    # Results storage
    results = {
        'time_index': [],
        'mask_prob': [],
        'model_type': [],
        'training_method': [],
        'loss': [],
        'seq_loss': [],
        'struct_loss': []
    }
    
    # Evaluate each time index
    for t_idx in val_cfg.time_indices:
        # Compute mask probability for this time index
        mask_prob = compute_mask_prob_for_time(t_idx, diffusion_cfg)
        print(f"\n{'='*60}")
        print(f"Time index t={t_idx}, mask probability={mask_prob:.3f}")
        print(f"{'='*60}")
        
        print(f"\nEvaluating {val_cfg.model_type} models...")
        
        # Evaluate all configured training methods
        for training_method in val_cfg.training_methods:
            # Get checkpoint path
            checkpoint_pattern = f"{training_method}_checkpoint_pattern"
            checkpoint_path = getattr(val_cfg, checkpoint_pattern).format(val_cfg.model_type)
            
            if not os.path.exists(checkpoint_path):
                print(f"  Checkpoint not found: {checkpoint_path}")
                continue
            
            print(f"  Loading {training_method} model from {checkpoint_path}")
            
            # Load model and FSQ encoder
            model, fsq_encoder = load_model_checkpoint(checkpoint_path, val_cfg.model_type, training_method, device, val_cfg.fsq_encoder_pattern)
            
            # Use appropriate model config for dataloader
            if training_method in ['simple', 'complex']:
                model_cfg_for_loader = mlm_model_cfg
            else:
                model_cfg_for_loader = diffusion_model_cfg
            
            # Create dataloader with fixed mask probability
            if training_method == 'diffusion':
                # Use custom fixed timestep dataloader for diffusion models
                diffusion_tracks = {'seq': True, 'struct': True, 'coords': True}
                min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
                
                val_loader = FixedTimestepDiffusionDataLoader(
                    dataset, model_cfg_for_loader, train_cfg, diffusion_cfg, diffusion_tracks,
                    fixed_timestep=t_idx, fsq_encoder=fsq_encoder, device=device, 
                    min_unmasked=min_unmasked, batch_size=val_cfg.batch_size, shuffle=False
                )
            else:
                # For MLM models, use custom fixed mask probability dataloader
                diffusion_tracks = {'seq': True, 'struct': True, 'coords': False}
                min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
                
                val_loader = FixedMaskProbDataLoader(
                    dataset, model_cfg_for_loader, train_cfg, diffusion_tracks,
                    fixed_mask_prob=mask_prob, fsq_encoder=fsq_encoder, device=device,
                    min_unmasked=min_unmasked, batch_size=val_cfg.batch_size, shuffle=False
                )
            
            # Evaluate based on training method
            if training_method in ['simple', 'complex']:
                # MLM models use cross-entropy loss
                metrics = evaluate_mlm_model(model, val_loader, val_cfg.model_type, device)
            else:
                # Diffusion models use score entropy loss
                all_metrics = {'loss': 0.0, 'seq_loss': 0.0, 'struct_loss': 0.0}
                num_batches = 0
                
                for batch in val_loader:
                    batch_metrics = evaluate_diffusion_model(
                        model, batch, t_idx, diffusion_cfg, diffusion_model_cfg, val_cfg.model_type, device
                    )
                    for key in all_metrics:
                        all_metrics[key] += batch_metrics[key]
                    num_batches += 1
                
                # Average metrics
                metrics = {key: all_metrics[key] / num_batches for key in all_metrics}
            
            # Store results
            results['time_index'].append(t_idx)
            results['mask_prob'].append(mask_prob)
            results['model_type'].append(val_cfg.model_type)
            results['training_method'].append(training_method)
            results['loss'].append(metrics['loss'])
            results['seq_loss'].append(metrics['seq_loss'])
            results['struct_loss'].append(metrics['struct_loss'])
            
            # Print results
            loss_type = "Score Entropy" if training_method == 'diffusion' else "Cross Entropy"
            print(f"    {training_method.capitalize()} - {loss_type} Loss: {metrics['loss']:.4f} "
                  f"(Seq: {metrics['seq_loss']:.4f}, Struct: {metrics['struct_loss']:.4f})")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = "validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Create visualization plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors and markers for each training method
    method_styles = {
        'simple': {'color': 'blue', 'marker': 'o', 'label': 'Simple MLM'},
        'complex': {'color': 'green', 'marker': 's', 'label': 'Complex MLM'},
        'diffusion': {'color': 'red', 'marker': '^', 'label': 'Discrete Diffusion'}
    }
    
    # Plot 1: Mask Percentage vs Sequence Loss
    ax1.set_title(f'Mask Percentage vs Sequence Loss ({val_cfg.model_type} Model)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mask Percentage (%)', fontsize=12)
    ax1.set_ylabel('Sequence Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for training_method in val_cfg.training_methods:
        method_data = results_df[results_df['training_method'] == training_method]
        if not method_data.empty:
            mask_percentages = method_data['mask_prob'].values * 100  # Convert to percentage
            seq_losses = method_data['seq_loss'].values
            
            style = method_styles.get(training_method, {'color': 'black', 'marker': 'x', 'label': training_method})
            ax1.plot(mask_percentages, seq_losses, 
                    color=style['color'], marker=style['marker'], 
                    linewidth=2, markersize=8, label=style['label'])
    
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 100)
    
    # Plot 2: Mask Percentage vs Structure Loss
    ax2.set_title(f'Mask Percentage vs Structure Loss ({val_cfg.model_type} Model)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mask Percentage (%)', fontsize=12)
    ax2.set_ylabel('Structure Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for training_method in val_cfg.training_methods:
        method_data = results_df[results_df['training_method'] == training_method]
        if not method_data.empty:
            mask_percentages = method_data['mask_prob'].values * 100  # Convert to percentage
            struct_losses = method_data['struct_loss'].values
            
            style = method_styles.get(training_method, {'color': 'black', 'marker': 'x', 'label': training_method})
            ax2.plot(mask_percentages, struct_losses, 
                    color=style['color'], marker=style['marker'], 
                    linewidth=2, markersize=8, label=style['label'])
    
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, 100)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = f"mask_percentage_vs_losses_{val_cfg.model_type}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Average losses across all time indices")
    print("="*80)
    
    print(f"\n{val_cfg.model_type}:")
    for training_method in ['simple', 'complex', 'diffusion']:
        mask = (results_df['model_type'] == val_cfg.model_type) & (results_df['training_method'] == training_method)
        if mask.any():
            avg_loss = results_df[mask]['loss'].mean()
            loss_type = "Score Entropy" if training_method == 'diffusion' else "Cross Entropy"
            print(f"  {training_method.capitalize():10s} - Avg {loss_type} Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main() 