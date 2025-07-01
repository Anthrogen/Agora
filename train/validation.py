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
from src.dataset import ProteinDataset
from src.dataloader import _get_training_dataloader, MaskedBatch
from src.losses import score_entropy_loss
from src.dataloader import _get_noise_levels

# Import functions from training scripts
from discrete_diffusion_step import (
    ModelConfig as DiffusionModelConfig, DiffusionConfig, 
    create_model_with_config as create_diffusion_model,
    discrete_diffusion_step
)
from mlm_step import (
    ModelConfig as MLMModelConfig,
    mlm_step,
    create_model_with_config as create_mlm_model
)

@dataclass
class TrainingConfig:
    """Configuration for validation."""
    # Model type to evaluate (single selection)
    model_type: str = "SC"  # Options: "SA", "GA", "RA", "SC"
    
    # Masking strategy and parameters
    masking_strategy: str = "simple"
    mask_prob_seq: float = None
    mask_prob_coords: float = None
    
    # Time indices to evaluate (directly specified)
    time_indices: List[int] = field(default_factory=lambda: [0,9,19,29,39,49,59,69,79,89,99])
    
    # Training methods to evaluate (any combination)
    training_methods: List[str] = field(default_factory=lambda: ["simple", "complex", "discrete_diffusion"])
    
    # Data settings
    data_dir: str = "../sample_data/1k.csv"
    batch_size: int = 4

    # Loss weights
    seq_loss_weight: float = 1.0
    struct_loss_weight: float = 1.0

    # Cross-entropy loss function: which elements should contribute to the loss?
    # "masked": only masked positions
    # "non_beospank": all non-BOS/EOS/PAD positions, including masks
    # "non_special": all non-special tokens, including masks
    ce_loss_function_elements: str = "masked"
    
    # Model paths (models in /scripts/checkpoints)
    simple_checkpoint_pattern: str = "../checkpoints/transformer_trunk/{}_simple_iter1_final.pt"
    complex_checkpoint_pattern: str = "../checkpoints/transformer_trunk/{}_complex_iter1_final.pt"
    discrete_diffusion_checkpoint_pattern: str = "../checkpoints/transformer_trunk/{}_discrete_diffusion_iter1_final.pt"
    
    # FSQ encoder paths (in /checkpoints, not /scripts/checkpoints)
    fsq_encoder_pattern: str = "../checkpoints/fsq/{}_stage_1_iter1_{}.pt"
    
    def __post_init__(self):
        # Validate model type
        valid_types = {"SA", "GA", "RA", "SC"}
        if self.model_type not in valid_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_types}")
        
        # Validate training methods
        valid_methods = {"simple", "complex", "discrete_diffusion"}
        for method in self.training_methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid training method: {method}. Must be one of {valid_methods}")

def compute_mask_prob_for_time(t: int, diffusion_cfg: DiffusionConfig) -> float:
    """Compute mask probability for a given time index using the diffusion schedule."""
    # Get noise levels
    inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
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
    is_diffusion = training_method == 'discrete_diffusion'
    
    # Fix pickle loading by temporarily setting ModelConfig in global namespace
    import __main__
    # Import the configs to ensure they're available in this scope
    from discrete_diffusion_step import ModelConfig as DiffusionModelConfig_local, DiffusionConfig as DiffusionConfig_local
    from mlm_step import ModelConfig as MLMModelConfig_local
    
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
        if training_method == 'discrete_diffusion':
            encoder_suffix = 'discrete_diffusion'
        else:
            encoder_suffix = training_method  # 'simple' or 'complex'
        
        encoder_checkpoint_path = fsq_encoder_pattern.format(model_type, encoder_suffix)
        
        if not os.path.exists(encoder_checkpoint_path):
            raise FileNotFoundError(f"FSQ encoder not found at: {encoder_checkpoint_path}")
        
        encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device, weights_only=False)
        encoder_state = {k.removeprefix('encoder.'): v for k, v in encoder_checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        fsq_config = SimpleNamespace(**encoder_checkpoint['model_cfg_dict']) 
        fsq_encoder = FSQEncoder(fsq_config)
        
        fsq_encoder.load_state_dict(encoder_state)
        fsq_encoder.eval()
        fsq_encoder.requires_grad_(False)
        fsq_encoder = fsq_encoder.to(device)
    
    return model, fsq_encoder

def evaluate_mlm_model(model: TransformerTrunk, batch: MaskedBatch, model_type: str, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> Dict[str, float]:
    """Evaluate an MLM model using cross-entropy loss on a single batch."""
    with torch.no_grad():
        retval = mlm_step({model_type : model}, optimizer=None, batch=batch, model_cfg=model_cfg, train_cfg=train_cfg, train_mode=False)
        return retval[model_type]

def evaluate_diffusion_model(model: TransformerTrunk, batch: MaskedBatch, model_type: str, timestep: int, model_cfg: ModelConfig, diffusion_cfg: DiffusionConfig,
                           train_cfg: TrainingConfig, device: torch.device) -> Dict[str, float]:
    """Evaluate a diffusion model using score entropy loss on a single batch."""

    B = batch.masked_data['seq'].shape[0]
    psuedo_timestep_indices = torch.full((B,), timestep, dtype=torch.long)

    inst_noise_levels, cumulative_noise_levels = _get_noise_levels(
        diffusion_cfg.sigma_min,
        diffusion_cfg.sigma_max,
        diffusion_cfg.num_timesteps,
        schedule_type=diffusion_cfg.noise_schedule
    )
    pseudo_cumulative_noise = cumulative_noise_levels[psuedo_timestep_indices].unsqueeze(1)
    pseudo_inst_noise = inst_noise_levels[psuedo_timestep_indices].unsqueeze(1)

    batch.metadata['pseudo_timestep_indices'] = psuedo_timestep_indices.to(device)
    batch.metadata['pseudo_cumulative_noise'] = pseudo_cumulative_noise.to(device)
    batch.metadata['pseudo_inst_noise'] = pseudo_inst_noise.to(device)

    with torch.no_grad():
        retval = discrete_diffusion_step({model_type: model}, optimizer=None, batch=batch, model_cfg=model_cfg, train_cfg=train_cfg, train_mode=False)
        return retval[model_type]


def main(custom_config: Optional[TrainingConfig] = None):
    # Initialize configuration
    train_cfg = custom_config if custom_config is not None else TrainingConfig()
    mlm_model_cfg = MLMModelConfig()
    diffusion_model_cfg = DiffusionModelConfig()
    diffusion_cfg = DiffusionConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset (use max_len from either config, they should be the same)
    dataset = ProteinDataset(train_cfg.data_dir, max_length=mlm_model_cfg.max_len - 2)
    
    print(f"\nValidation dataset size: {len(dataset)}")
    print(f"Time indices to evaluate: {train_cfg.time_indices}")
    print(f"Model type to evaluate: {train_cfg.model_type}")
    print(f"Training methods to evaluate: {train_cfg.training_methods}")
    
    # Results storage
    results = {'time_index': [], 'mask_prob': [], 'model_type': [], 'training_method': [], 'loss': [], 'seq_loss': [], 'struct_loss': []}
    
    # Evaluate each time index
    for t_idx in train_cfg.time_indices:
        # Compute mask probability for this time index
        mask_prob = compute_mask_prob_for_time(t_idx, diffusion_cfg)
        print(f"\n{'='*60}")
        print(f"Time index t={t_idx}, mask probability={mask_prob:.3f}")
        print(f"{'='*60}")
        print(f"\nEvaluating {train_cfg.model_type} models...")
        
        # Set mask probabilities for this time index
        train_cfg.mask_prob_seq = mask_prob
        train_cfg.mask_prob_coords = mask_prob
        
        tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
        
        # Evaluate all configured training methods
        for training_method in train_cfg.training_methods:
            # Get checkpoint path
            checkpoint_pattern = f"{training_method}_checkpoint_pattern"
            checkpoint_path = getattr(train_cfg, checkpoint_pattern).format(train_cfg.model_type)
            
            if not os.path.exists(checkpoint_path):
                print(f"  Checkpoint not found: {checkpoint_path}")
                continue
            
            print(f"  Loading {training_method} model from {checkpoint_path}")
            
            # Load model and FSQ encoder
            model, fsq_encoder = load_model_checkpoint(checkpoint_path, train_cfg.model_type, training_method, device, train_cfg.fsq_encoder_pattern)
            
            # Use appropriate model config
            if training_method in ['simple', 'complex']: model_cfg = mlm_model_cfg
            else: model_cfg = diffusion_model_cfg
            
            # Add generator for reproducible validation
            val_generator = torch.Generator()
            val_generator.manual_seed(42)  # or any fixed seed

            val_loader = _get_training_dataloader(dataset=dataset, model_cfg=model_cfg, train_cfg=train_cfg, tracks=tracks, device=device, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder, shuffle=False, batch_size=train_cfg.batch_size, generator=val_generator)
            
            # Evaluate model using unified approach - loop over batches
            all_metrics = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0}
            num_batches = 0
            
            for batch in val_loader:
                # Call appropriate evaluation function based on training method
                if training_method in ['simple', 'complex']:
                    # MLM models use cross-entropy loss
                    batch_metrics = evaluate_mlm_model(model, batch, train_cfg.model_type, model_cfg, train_cfg)
                else:
                    # Discrete diffusion models use score entropy loss
                    batch_metrics = evaluate_diffusion_model(model, batch, train_cfg.model_type, t_idx, model_cfg, diffusion_cfg, train_cfg, device)
                
                # Accumulate metrics
                for key in all_metrics: all_metrics[key] += batch_metrics[key]
                num_batches += 1
            
            # Average metrics
            metrics = {key: all_metrics[key] / num_batches for key in all_metrics}
            
            # Store results
            results['time_index'].append(t_idx)
            results['mask_prob'].append(mask_prob)
            results['model_type'].append(train_cfg.model_type)
            results['training_method'].append(training_method)
            results['loss'].append(metrics['loss'])
            results['seq_loss'].append(metrics['loss_seq'])
            results['struct_loss'].append(metrics['loss_struct'])
            
            # Print results
            loss_type = "Score Entropy" if training_method == 'discrete_diffusion' else "Cross Entropy"
            print(f"    {training_method.capitalize()} - {loss_type} Loss: {metrics['loss']:.4f} (Seq: {metrics['loss_seq']:.4f}, Struct: {metrics['loss_struct']:.4f})")
    
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
        'discrete_diffusion': {'color': 'red', 'marker': '^', 'label': 'Discrete Diffusion'}
    }
    
    # Plot 1: Mask Percentage vs Sequence Loss
    ax1.set_title(f'Mask Percentage vs Sequence Loss ({train_cfg.model_type} Model)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mask Percentage (%)', fontsize=12)
    ax1.set_ylabel('Sequence Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for training_method in train_cfg.training_methods:
        method_data = results_df[results_df['training_method'] == training_method]
        if not method_data.empty:
            mask_percentages = method_data['mask_prob'].values * 100  # Convert to percentage
            seq_losses = method_data['seq_loss'].values
            
            style = method_styles.get(training_method, {'color': 'black', 'marker': 'x', 'label': training_method})
            ax1.plot(mask_percentages, seq_losses, color=style['color'], marker=style['marker'], linewidth=2, markersize=8, label=style['label'])
    
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 100)
    
    # Plot 2: Mask Percentage vs Structure Loss
    ax2.set_title(f'Mask Percentage vs Structure Loss ({train_cfg.model_type} Model)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mask Percentage (%)', fontsize=12)
    ax2.set_ylabel('Structure Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for training_method in train_cfg.training_methods:
        method_data = results_df[results_df['training_method'] == training_method]
        if not method_data.empty:
            mask_percentages = method_data['mask_prob'].values * 100  # Convert to percentage
            struct_losses = method_data['struct_loss'].values
            
            style = method_styles.get(training_method, {'color': 'black', 'marker': 'x', 'label': training_method})
            ax2.plot(mask_percentages, struct_losses, color=style['color'], marker=style['marker'], linewidth=2, markersize=8, label=style['label'])
    
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, 100)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = f"mask_percentage_vs_losses_{train_cfg.model_type}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Average losses across all time indices")
    print("="*80)
    
    print(f"\n{train_cfg.model_type}:")
    for training_method in ['simple', 'complex', 'discrete_diffusion']:
        mask = (results_df['model_type'] == train_cfg.model_type) & (results_df['training_method'] == training_method)
        if mask.any():
            avg_loss = results_df[mask]['loss'].mean()
            loss_type = "Score Entropy" if training_method == 'discrete_diffusion' else "Cross Entropy"
            method_display = training_method.capitalize()
            print(f"  {method_display:17s} - Avg {loss_type} Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main() 