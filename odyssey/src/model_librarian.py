"""
The purpose of this module is to organize the models in the checkpoints/ directory.
It provides methods of loading and saving models to and from checkpoints/.
"""

import torch
import os
from typing import Dict
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.blocks import StandardTransformerBlock
from odyssey.src.models.autoencoder import Autoencoder


def ensure_identical_parameters_all_models(models: Dict[str, TransformerTrunk | Autoencoder], seed: int):
    """
    I write this function in the hopes that we minimally disrupt this module's interface when we unify the below code.
    I think that there should be a registration system for the trainable parameters of models.
    """
    if len(models) == 0: return

    first_key = list(models.keys())[0]

    if isinstance(models[first_key], TransformerTrunk):
        ensure_identical_parameters_all_transformers(models, seed)
    elif isinstance(models[first_key], Autoencoder):
        ensure_identical_parameters_all_autoencoders(models, seed)
    else:
        raise ValueError(f"Model type {type(models[first_key])} is neither a TransformerTrunk nor an Autoencoder")

def ensure_identical_parameters_all_autoencoders(models: Dict[str, Autoencoder], seed: int):
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
                #TODO: this needs to be in the model.
                #e.g. model.load(state_dictionary) and it can unpack what it needs internally.
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

def ensure_identical_parameters_all_transformers(models: Dict[str, TransformerTrunk], seed: int):
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

