"""
The purpose of this module is to organize the models in the checkpoints/ directory.
It provides methods of loading and saving models to and from checkpoints/.
"""

import torch
import numpy as np
import os
from typing import Dict
from dataclasses import replace
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.blocks import StandardTransformerBlock
from odyssey.src.models.autoencoder import Autoencoder, FSQEncoder
from odyssey.src.configurations import *

def ensure_identical_parameters_transformers(models: Dict[str, TransformerTrunk], seed: int):
    """Ensure all models have identical parameters where possible.
    
    Strategy:
    1. Set random seed for reproducible initialization
    2. Copy shared embeddings and output layers from first model
    3. For transformer layers beyond the first (which is architecture-specific),
       copy entire StandardTransformerBlock state dicts when both models use them
    
    Args:
        models: Dictionary mapping model type to model instance
        seed: Random seed for consistent initialization
    """
    if len(models) == 0: 
        return
    
    torch.manual_seed(seed)
    ref_model = next(iter(models.values()))
    
    with torch.no_grad():
        for model_type, model in models.items():
            if model is not ref_model:
                # Copy embeddings
                model.seq_embed.load_state_dict(ref_model.seq_embed.state_dict())
                model.struct_embed.load_state_dict(ref_model.struct_embed.state_dict())
                model.ss8_embed.load_state_dict(ref_model.ss8_embed.state_dict())
                model.sasa_embed.load_state_dict(ref_model.sasa_embed.state_dict())
                model.plddt_embed.load_state_dict(ref_model.plddt_embed.state_dict())
                
                # Copy output layers
                model.final_norm.load_state_dict(ref_model.final_norm.state_dict())
                model.seq_logits.load_state_dict(ref_model.seq_logits.state_dict())
                model.struct_logits.load_state_dict(ref_model.struct_logits.state_dict())
                
                # For layers beyond the first (which is architecture-specific),
                # copy entire StandardTransformerBlocks when both models use them
                for i in range(1, min(len(model.layers), len(ref_model.layers))):
                    if (isinstance(model.layers[i], StandardTransformerBlock) and 
                        isinstance(ref_model.layers[i], StandardTransformerBlock)):
                        model.layers[i].load_state_dict(ref_model.layers[i].state_dict())

def ensure_identical_parameters_autoencoders(models: Dict[str, Autoencoder], seed: int):
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

def load_model_from_empty(model_cfg, device):
    """
    Create a model, synchronized with a baseline Self-Attention model.
    
    Args:
        model_cfg: Model configuration object
        device: Device to place models on
        
    Returns:
        model: The target model (either SA or synchronized with SA)
    """
    # Set seed for model creation
    torch.manual_seed(model_cfg.reference_model_seed)
    
    # Determine constructor and sync function based on model type
    desired_constructor = Autoencoder if isinstance(model_cfg, FSQConfig) else TransformerTrunk
    sync_function = ensure_identical_parameters_autoencoders if isinstance(model_cfg, FSQConfig) else ensure_identical_parameters_transformers

    # Create SA reference model
    # model_cfg_sa = replace(model_cfg, first_block_cfg=SelfAttentionConfig())
    model_cfg_sa = model_cfg.make_copy()
    model_cfg_sa.first_block_cfg = SelfAttentionConfig()
    
    model_sa = desired_constructor(model_cfg_sa).to(device)

    if isinstance(model_cfg.first_block_cfg, SelfAttentionConfig): model = model_sa
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

    return model

def save_model_checkpoint(path, model, model_cfg, train_cfg, optimizer):
    # Save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_cfg,
        'train_config': train_cfg,
        'model_config_dict': model_cfg.to_dict(),  # Backup dictionary
        'training_config_dict': train_cfg.to_dict()  # Backup dictionary
    }, path)


def load_autoencoder_from_checkpoint(model_path, device, freeze=True):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint['model_config']
    train_cfg = checkpoint['train_config']
    
    assert isinstance(model_cfg, FSQConfig)

    model = Autoencoder(model_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])

    if freeze:
        model.eval()
        model.requires_grad_(False)

    model = model.to(device)
    
    return model, model_cfg, train_cfg


def load_model_from_checkpoint(model_path, device, freeze=True):
    """
    Load model from checkpoint. Handles both FSQ encoders and full models.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to place the model on
        freeze: Whether to freeze the model parameters
        
    Returns:
        model: Loaded model (FSQ encoder for Autoencoder configs, full model for others)
    """
    # Load checkpoint with dynamic path based on model type
    # TODO also, we should be using os.path.join rather than / wherever possible.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = checkpoint['model_config']
    train_cfg = checkpoint['train_config']

    constructor = FSQEncoder if isinstance(model_cfg, FSQConfig) else TransformerTrunk
    
    # Determine model type based on config instance
    if isinstance(model_cfg, FSQConfig):
        # FSQConfig - Load FSQ encoder from Autoencoder checkpoint
        encoder_state = {k.removeprefix('encoder.'): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
    elif isinstance(model_cfg, TrunkConfig):
        encoder_state = checkpoint['model_state_dict']
    else:
        raise ValueError(f"Unknown model config type: {type(model_cfg).__name__}. Expected FSQConfig or TrunkConfig.")

    model = constructor(model_cfg)
    if isinstance(model_cfg, FSQConfig):
        model.load_state_dict(encoder_state)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {type(model_cfg).__name__} model weights from: {model_path}")

    if freeze:
        model.eval()
        model.requires_grad_(False)

    model = model.to(device)
    
    return model, model_cfg, train_cfg

def save_summary_history(arrays_list, output_file, name_prefix="name", header_list=None):
    """
    arrays_list: List of numpy arrays
    output_file: Path to output CSV file
    name_prefix: Prefix for super-headers
    header_list: List of lists containing column headers for each array
    """
    if arrays_list is None:
        return

    for idx in range(len(arrays_list)):
        arrays_list[idx] = np.array(arrays_list[idx])
    
    # Calculate dimensions
    max_rows = max(arr.shape[0] for arr in arrays_list)
    total_cols = sum(arr.shape[1] if arr.ndim > 1 else 1 for arr in arrays_list)
    
    # Create giant array with NaN
    giant_array = np.full((max_rows, total_cols), np.nan)
    
    # Fill data and create headers
    col_offset = 0
    superheaders = []
    column_headers = []
    
    for i, arr in enumerate(arrays_list):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        
        rows, cols = arr.shape
        giant_array[:rows, col_offset:col_offset+cols] = arr
        
        # Superheader
        superheaders.append(f"{name_prefix}_{i:03d}")
        superheaders.extend([''] * (cols - 1))
        
        # Column headers
        if header_list is not None and i < len(header_list):
            headers = header_list[i]
            # Ensure we have the right number of headers
            if len(headers) < cols:
                headers = headers + [''] * (cols - len(headers))
            column_headers.extend(headers[:cols])
        else:
            # Default headers if not provided
            column_headers.extend([f'col_{j}' for j in range(cols)])
        
        col_offset += cols
    
    # Save to CSV
    with open(output_file, 'w') as f:
        # Write superheader
        f.write(','.join(superheaders) + '\n')
        
        # Write column headers if provided
        if header_list:
            f.write(','.join(column_headers) + '\n')
        
        # Write data rows
        for row in giant_array:
            # Convert row to strings, replacing nan with empty string
            row_str = ['' if np.isnan(val) else str(val) for val in row]
            f.write(','.join(row_str) + '\n')