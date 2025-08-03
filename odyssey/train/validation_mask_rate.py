import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional
import random
from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.autoencoder import Autoencoder
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import _get_training_dataloader, MaskedBatch, worker_init_fn, SimpleDataLoader, MaskingDataLoader
from odyssey.src.dataloader import _get_noise_levels
from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.configurations import *
from odyssey.src.tokenizer import CorruptionMode

from mlm_step import mlm_step
from discrete_diffusion_step import discrete_diffusion_step
from fsq_step import stage_1_step, stage_2_step

class CustomSimpleDataLoader(MaskingDataLoader):
    """SimpleDataLoader that allows custom corruption mode."""
    def __init__(self, dataset, model_cfg, train_cfg, tracks, device, corruption_mode=CorruptionMode.MASK, autoencoder=None, min_unmasked=None, **kwargs):
        if min_unmasked is None:
            min_unmasked = {'seq': 0, 'coords': 1}
        
        super(CustomSimpleDataLoader, self).__init__(dataset, corruption_mode, model_cfg, train_cfg, tracks, device, autoencoder=autoencoder, min_unmasked=min_unmasked, **kwargs)

        assert isinstance(train_cfg.mask_config, SimpleMaskConfig)
        # Use mask_prob_seq as default for sequence-like tracks, mask_prob_struct for structure-like tracks
        self.simple_mask_prob = {
            'seq': train_cfg.mask_config.mask_prob_seq, 
            'coords': train_cfg.mask_config.mask_prob_struct, 
            'ss8': train_cfg.mask_config.mask_prob_seq, 
            'sasa': train_cfg.mask_config.mask_prob_seq, 
            'plddt': train_cfg.mask_config.mask_prob_seq, 
            'domains': train_cfg.mask_config.mask_prob_seq
        }

    def sample_masks(self, tracks, batch_len):
        masks = {}
        for track in [t for t in tracks if (tracks[t] and t != 'struct' and t != 'orthologous_groups' and t != 'semantic_description')]:
            # Create masks with fixed probabilities
            mask = torch.rand(batch_len, self.L, device=self.device) < self.simple_mask_prob[track]
            masks[track] = mask.bool()

        return masks, None

def compute_mask_prob_for_time(t: int, diffusion_cfg: DiffusionMaskConfig) -> float:
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

def evaluate_mlm_model(model: TransformerTrunk, batch: MaskedBatch, model_type: str, model_cfg: TransformerConfig, train_cfg: TrainingConfig) -> Dict[str, float]:
    """Evaluate an MLM model using cross-entropy loss on a single batch."""
    with torch.no_grad():
        retval = mlm_step(model, optimizer=None, scheduler=None, batch=batch, model_cfg=model_cfg, train_cfg=train_cfg, train_mode=False)
        return retval

def evaluate_diffusion_model(model: TransformerTrunk, batch: MaskedBatch, model_type: str, timestep: int, model_cfg: TransformerConfig, diffusion_cfg: DiffusionMaskConfig,
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
        retval = discrete_diffusion_step(model, optimizer=None, scheduler=None, batch=batch, model_cfg=model_cfg, train_cfg=train_cfg, train_mode=False)
        return retval

def validate(path_to_model_checkpoint_absorb: str,
             path_to_model_checkpoint_uniform: str, 
             path_to_model_checkpoint_simple: str, 
             path_to_model_checkpoint_complex: str,
             time_steps: List[int],
             validation_batch_size: int):
    """Validate four models across different mask rates corresponding to time steps."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ###########################################################################
    #  Load all models and their configs
    ###########################################################################
    print("Loading models...")
    
    # Load absorb discrete diffusion model and its autoencoder
    absorb_model, absorb_model_cfg, absorb_train_cfg = load_model_from_checkpoint(path_to_model_checkpoint_absorb, device)
    absorb_autoencoder, absorb_autoencoder_model_cfg, absorb_autoencoder_train_cfg = load_model_from_checkpoint(absorb_model_cfg.autoencoder_path, device)
    
    # Load uniform discrete diffusion model and its autoencoder
    uniform_model, uniform_model_cfg, uniform_train_cfg = load_model_from_checkpoint(path_to_model_checkpoint_uniform, device)
    uniform_autoencoder, uniform_autoencoder_model_cfg, uniform_autoencoder_train_cfg = load_model_from_checkpoint(uniform_model_cfg.autoencoder_path, device)
    
    # Load simple MLM model and its autoencoder
    simple_model, simple_model_cfg, simple_train_cfg = load_model_from_checkpoint(path_to_model_checkpoint_simple, device)
    simple_autoencoder, simple_autoencoder_model_cfg, simple_autoencoder_train_cfg = load_model_from_checkpoint(simple_model_cfg.autoencoder_path, device)
    
    # Load complex MLM model and its autoencoder
    complex_model, complex_model_cfg, complex_train_cfg = load_model_from_checkpoint(path_to_model_checkpoint_complex, device)
    complex_autoencoder, complex_autoencoder_model_cfg, complex_autoencoder_train_cfg = load_model_from_checkpoint(complex_model_cfg.autoencoder_path, device)
    
    # Call post_init after loading configs from checkpoint to set global and local annotation information
    # When deserializing from checkpoint, __post_init__ is not called, so vocab sizes aren't computed
    absorb_model_cfg.__post_init__()
    absorb_train_cfg.__post_init__()
    absorb_autoencoder_model_cfg.__post_init__()
    absorb_autoencoder_train_cfg.__post_init__()
    
    uniform_model_cfg.__post_init__()
    uniform_train_cfg.__post_init__()
    uniform_autoencoder_model_cfg.__post_init__()
    uniform_autoencoder_train_cfg.__post_init__()
    
    simple_model_cfg.__post_init__()
    simple_train_cfg.__post_init__()
    simple_autoencoder_model_cfg.__post_init__()
    simple_autoencoder_train_cfg.__post_init__()
    
    complex_model_cfg.__post_init__()
    complex_train_cfg.__post_init__()
    complex_autoencoder_model_cfg.__post_init__()
    complex_autoencoder_train_cfg.__post_init__()
    
    # Store models in a list for easier iteration
    models = [
        ("absorb", absorb_model, absorb_model_cfg, absorb_train_cfg, absorb_autoencoder),
        ("uniform", uniform_model, uniform_model_cfg, uniform_train_cfg, uniform_autoencoder),
        ("simple", simple_model, simple_model_cfg, simple_train_cfg, simple_autoencoder),
        ("complex", complex_model, complex_model_cfg, complex_train_cfg, complex_autoencoder)
    ]
    
    # Get diffusion config from absorb model (both should have same noise schedule parameters)
    diffusion_cfg = absorb_train_cfg.mask_config
    
    ###########################################################################
    #  Data setup
    ###########################################################################
    dataset_mode = "backbone"
    data_dir = "/workspace/demo/Odyssey/sample_data/3k.csv"
    
    # Use comprehensive tracks like in generate.py
    tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
    min_unmasked = {'seq': 0, 'coords': 1}
    
    # Use the reference model seed for consistency
    data_seed = absorb_model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)
    
    # Create dataset and validation split
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000)
    val_ds = ProteinDataset(data_dir, mode=dataset_mode, max_length=absorb_model_cfg.max_len - 2, max_length_orthologous_groups=absorb_model_cfg.max_len_orthologous_groups - 2, max_length_semantic_description=absorb_model_cfg.max_len_semantic_description - 2)
    
    ###########################################################################
    #  Validation across time steps
    ###########################################################################
    results = {
        "time_index": [],
        "mask_prob": [],
        "model_type": [],
        "training_method": [],
        "loss": [],
        "seq_loss": [],
        "struct_loss": []
    }
    
    for timestep in time_steps:
        # Compute mask probability for this timestep
        mask_prob = compute_mask_prob_for_time(timestep, diffusion_cfg)
        print(f"\nTimestep {timestep}: mask_prob = {mask_prob:.3f}")
        
        # Create a simple mask config with this mask probability for all models
        simple_mask_cfg = SimpleMaskConfig(
            mask_prob_seq=mask_prob,
            mask_prob_struct=mask_prob
        )
        
        # Evaluate all models on this dataloader
        for model_name, model, model_cfg, train_cfg, autoencoder in models:
            model.eval()
            autoencoder.eval()
            
            # Update training config with simple mask config for consistent mask probabilities
            temp_train_cfg = replace(train_cfg, mask_config=simple_mask_cfg)
            
            # Create dataloader with appropriate corruption mode
            if model_name == 'uniform':
                # For uniform diffusion, we need uniform corruption but with simple masking probabilities
                val_loader = CustomSimpleDataLoader(
                    val_ds, 
                    model_cfg,
                    temp_train_cfg,
                    tracks,
                    device,
                    corruption_mode=CorruptionMode.UNIFORM,
                    autoencoder=autoencoder,
                    min_unmasked=min_unmasked,
                    batch_size=validation_batch_size,
                    shuffle=False,
                    generator=g_val,
                    worker_init_fn=worker_init_fn
                )
            else:
                # For all other models, use standard simple dataloader (mask corruption)
                val_loader = CustomSimpleDataLoader(
                    val_ds, 
                    model_cfg,
                    temp_train_cfg,
                    tracks,
                    device,
                    corruption_mode=CorruptionMode.MASK,
                    autoencoder=autoencoder,
                    min_unmasked=min_unmasked,
                    batch_size=validation_batch_size,
                    shuffle=False,
                    generator=g_val,
                    worker_init_fn=worker_init_fn
                )
            
            # Initialize accumulators for this model and timestep
            all_metrics_sum = {}
            all_metrics_count = {}
            
            for batch in tqdm(val_loader, desc=f"Validating {model_name} at timestep {timestep}", leave=True, ncols=100):
                if batch is None:
                    continue
                
                # Call appropriate evaluation function based on training method
                if model_name in ['simple', 'complex']:
                    # MLM models use cross-entropy loss with temp config
                    batch_metrics = evaluate_mlm_model(
                        model, batch, model_name, model_cfg, temp_train_cfg
                    )
                elif model_name in ['absorb', 'uniform']:
                    # Discrete diffusion models use score entropy loss with original config
                    # (need original config for corruption_mode in loss function selection)
                    batch_metrics = evaluate_diffusion_model(
                        model, batch, model_name, timestep,
                        model_cfg, train_cfg.mask_config, train_cfg, device
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_name}")
                
                # Accumulate metrics (now handling (value, count) tuples)
                for key, (value, count) in batch_metrics.items():
                    if key not in all_metrics_sum:
                        all_metrics_sum[key] = 0.0
                        all_metrics_count[key] = 0
                    all_metrics_sum[key] += value * count
                    all_metrics_count[key] += count
            
            # Average metrics
            metrics = {key: all_metrics_sum[key] / all_metrics_count[key] for key in all_metrics_sum.keys()}
            
            # Store results
            results['time_index'].append(timestep)
            results['mask_prob'].append(mask_prob)
            results['model_type'].append(model_cfg.first_block_cfg.initials())
            results['training_method'].append(model_name)
            results['loss'].append(metrics['loss'])
            results['seq_loss'].append(metrics['loss_seq'])
            results['struct_loss'].append(metrics['loss_struct'])
            
            # Print results
            loss_type = "Score Entropy" if model_name in ['absorb', 'uniform'] else "Cross Entropy"
            print(f"    {model_name.capitalize()} - {loss_type} Loss: {metrics['loss']:.4f} (Seq: {metrics['loss_seq']:.4f}, Struct: {metrics['loss_struct']:.4f})")
    
    ###########################################################################
    #  Save and plot results
    ###########################################################################
    results_df = pd.DataFrame(results)
    
    # Create visualization plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors and markers for each training method
    method_styles = {
        'simple': {'color': 'blue', 'marker': 'o', 'label': 'Simple MLM'},
        'complex': {'color': 'green', 'marker': 's', 'label': 'Complex MLM'},
        'absorb': {'color': 'red', 'marker': '^', 'label': 'Discrete Diffusion (Absorb)'},
        'uniform': {'color': 'orange', 'marker': 'v', 'label': 'Discrete Diffusion (Uniform)'}
    }
    
    # Get unique model types
    model_types = results_df['model_type'].unique()
    model_type_label = "/".join(model_types) if len(model_types) > 1 else model_types[0]
    
    # Plot 1: Mask Percentage vs Sequence Loss
    ax1.set_title(f'Mask Percentage vs Sequence Loss ({model_type_label} Model)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mask Percentage (%)', fontsize=12)
    ax1.set_ylabel('Sequence Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for training_method in ['simple', 'complex', 'absorb', 'uniform']:
        method_data = results_df[results_df['training_method'] == training_method]
        if not method_data.empty:
            mask_percentages = method_data['mask_prob'].values * 100  # Convert to percentage
            seq_losses = method_data['seq_loss'].values
            
            style = method_styles.get(training_method, {'color': 'black', 'marker': 'x', 'label': training_method})
            ax1.plot(mask_percentages, seq_losses, color=style['color'], marker=style['marker'], 
                    linewidth=2, markersize=8, label=style['label'])
    
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 100)
    
    # Plot 2: Mask Percentage vs Structure Loss
    ax2.set_title(f'Mask Percentage vs Structure Loss ({model_type_label} Model)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mask Percentage (%)', fontsize=12)
    ax2.set_ylabel('Structure Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for training_method in ['simple', 'complex', 'absorb', 'uniform']:
        method_data = results_df[results_df['training_method'] == training_method]
        if not method_data.empty:
            mask_percentages = method_data['mask_prob'].values * 100  # Convert to percentage
            struct_losses = method_data['struct_loss'].values
            
            style = method_styles.get(training_method, {'color': 'black', 'marker': 'x', 'label': training_method})
            ax2.plot(mask_percentages, struct_losses, color=style['color'], marker=style['marker'], 
                    linewidth=2, markersize=8, label=style['label'])
    
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, 100)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = f"mask_percentage_vs_losses_{model_type_label}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Average losses across all time indices")
    print("="*80)
    
    for model_type in model_types:
        print(f"\n{model_type}:")
        for training_method in ['simple', 'complex', 'absorb', 'uniform']:
            mask = (results_df['model_type'] == model_type) & (results_df['training_method'] == training_method)
            if mask.any():
                avg_loss = results_df[mask]['loss'].mean()
                avg_seq_loss = results_df[mask]['seq_loss'].mean()
                avg_struct_loss = results_df[mask]['struct_loss'].mean()
                loss_type = "Score Entropy" if training_method in ['absorb', 'uniform'] else "Cross Entropy"
                method_display = training_method.capitalize()
                print(f"  {method_display:17s} - Avg {loss_type} Loss: {avg_loss:.4f} (Seq: {avg_seq_loss:.4f}, Struct: {avg_struct_loss:.4f})")
    
    return results_df

if __name__ == "__main__":
    path_to_model_checkpoint_absorb = f'/workspace/demo/Odyssey/checkpoints/transformer_trunk/discrete_diffusion_absorb_config/discrete_diffusion_absorb_config_000/checkpoint_step_45924.pt'
    path_to_model_checkpoint_uniform = f'/workspace/demo/Odyssey/checkpoints/transformer_trunk/discrete_diffusion_uniform_config/discrete_diffusion_uniform_config_000/checkpoint_step_45924.pt'
    path_to_model_checkpoint_simple = f'/workspace/demo/Odyssey/checkpoints/transformer_trunk/mlm_simple_config/mlm_simple_config_000/checkpoint_step_34443.pt'
    path_to_model_checkpoint_complex = f'/workspace/demo/Odyssey/checkpoints/transformer_trunk/mlm_complex_config/mlm_complex_config_000/checkpoint_step_34443.pt'
    time_steps = [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89]
    validation_batch_size = 4
    validate(path_to_model_checkpoint_absorb, path_to_model_checkpoint_uniform, path_to_model_checkpoint_simple, 
             path_to_model_checkpoint_complex, time_steps, validation_batch_size) 