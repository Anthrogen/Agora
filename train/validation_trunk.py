"""
Validation script for evaluating models trained with different masking strategies.

This script:
1. Loads all files from a specified path as validation dataset
2. Evaluates a single model type (SA/GA/RA/C) trained with different masking strategies
3. Computes both score entropy loss (from discrete diffusion) and cross entropy loss (from MLM)
4. Uses discrete diffusion dataloader for all evaluations
"""
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import random
from types import SimpleNamespace

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.transformer import TransformerTrunk
from src.models.autoencoder_trunk import FSQEncoder
from src.dataset_trunk import ProteinBackboneDataset
from src.dataloader_trunk import DiffusionDataLoader

# Import utilities from train_transformer_discrete_diffusion
from train_transformer_discrete_diffusion import (ModelConfig, DiffusionConfig, score_entropy_loss, 
    calculate_accuracy, create_model_with_config as create_diffusion_model)

# Import utilities from train_transformer_MLM
from train_transformer_MLM import (create_model_with_config as create_mlm_model)

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ValidationConfig:
    """Validation configuration."""
    model_type: str = "SA"  # Single model type: one of ["SA", "GA", "RA", "C"]
    masking_strategies: List[str] = field(default_factory=lambda: ["simple", "complex", "discrete_diffusion"])
    
    # Data paths
    validation_data_dir: str = "../data/sample_training_data"  # Path containing all validation files
    checkpoint_dir: str = "checkpoints"  # Directory containing model checkpoints
    
    # Evaluation settings
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    output_file: str = "validation_results.txt"

# --------------------------------------------------------------------------- #
#  Model Creation Helper                                                       #
# --------------------------------------------------------------------------- #
def create_model_for_masking_strategy(model_type: str, masking_strategy: str, 
                                     model_cfg: ModelConfig, device: torch.device) -> TransformerTrunk:
    """Create a model with the appropriate configuration for the masking strategy."""
    if masking_strategy == "discrete_diffusion":
        # For discrete diffusion, use AdaLN (adaptive layer norm for time conditioning)
        return create_diffusion_model(model_type, model_cfg, device)
    else:
        # For simple/complex masking, create model without AdaLN
        return create_mlm_model(model_type, model_cfg, device)

# --------------------------------------------------------------------------- #
#  Cross Entropy Loss Computation                                              #
# --------------------------------------------------------------------------- #
def compute_cross_entropy_loss(seq_logits: torch.Tensor, struct_logits: torch.Tensor,
                             seq_labels: torch.Tensor, struct_labels: torch.Tensor,
                             seq_mask: torch.Tensor, struct_mask: torch.Tensor,
                             device: torch.device) -> Tuple[float, float]:
    """Compute cross entropy loss for sequence and structure predictions."""
    # Flatten tensors
    seq_logits_flat = seq_logits.view(-1, seq_logits.size(-1))
    struct_logits_flat = struct_logits.view(-1, struct_logits.size(-1))
    seq_labels_flat = seq_labels.view(-1)
    struct_labels_flat = struct_labels.view(-1)
    seq_mask_flat = seq_mask.view(-1)
    struct_mask_flat = struct_mask.view(-1)
    
    # Compute cross entropy only on masked positions
    loss_seq = F.cross_entropy(seq_logits_flat[seq_mask_flat], seq_labels_flat[seq_mask_flat].long()) if seq_mask_flat.any() else torch.tensor(0.0, device=device)
    loss_struct = F.cross_entropy(struct_logits_flat[struct_mask_flat], struct_labels_flat[struct_mask_flat].long()) if struct_mask_flat.any() else torch.tensor(0.0, device=device)
    
    return loss_seq.item(), loss_struct.item()

# --------------------------------------------------------------------------- #
#  Validation Function                                                         #
# --------------------------------------------------------------------------- #
def validate_model(model: TransformerTrunk, dataloader: DiffusionDataLoader, 
                  model_cfg: ModelConfig, diffusion_cfg: DiffusionConfig,
                  model_type: str, device: torch.device, 
                  masking_strategy: str) -> Dict[str, float]:
    """Validate a model using the appropriate loss for its training strategy."""
    model.eval()
    
    # Determine if we're using time conditioning (only for discrete diffusion)
    use_time_conditioning = (masking_strategy == "discrete_diffusion")
    
    # Metrics accumulators
    metrics = {'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_accuracy': 0.0, 'struct_accuracy': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating", leave=False):
            # Unpack batch data
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
            coord_mask = coord_mask & (~struct_mask)
            
            # Handle all-masked sequences
            all_masked = ~coord_mask.any(dim=1)
            if all_masked.any():
                for idx in torch.where(all_masked)[0]:
                    coords[idx, 1, 0, :] = torch.tensor([0., 0., 0.], device=device)  # N
                    coords[idx, 1, 1, :] = torch.tensor([1.5, 0., 0.], device=device)  # CA  
                    coords[idx, 1, 2, :] = torch.tensor([2.4, 1.3, 0.], device=device)  # C
                    coord_mask[idx, 1] = True
            
            # Prepare timesteps
            timesteps = timestep_indices.float().unsqueeze(-1) if use_time_conditioning else None
            
            # Forward pass
            inputs = (seq_x_t, struct_x_t)
            if model_type in ("GA", "RA"):
                if use_time_conditioning:
                    outputs = model(inputs, coords, coord_mask=coord_mask, timesteps=timesteps)
                else:
                    outputs = model(inputs, coords, coord_mask=coord_mask)
            else:
                if use_time_conditioning:
                    outputs = model(inputs, timesteps=timesteps)
                else:
                    outputs = model(inputs)
            
            seq_logits, struct_logits = outputs
            
            # Compute appropriate loss based on masking strategy
            if masking_strategy == "discrete_diffusion":
                # Compute score entropy losses for discrete diffusion
                loss_seq = score_entropy_loss(seq_logits, seq_x_0, seq_x_t, cumulative_noise_levels, inst_noise, diffusion_cfg.seq_absorb_token, validation=True)
                loss_struct = score_entropy_loss(struct_logits, struct_x_0, struct_x_t, cumulative_noise_levels, inst_noise, diffusion_cfg.struct_absorb_token, validation=True)
            else:
                # Compute cross entropy losses for simple/complex
                loss_seq, loss_struct = compute_cross_entropy_loss(seq_logits, struct_logits, seq_x_0, struct_x_0, seq_mask, struct_mask, device)
            
            # Calculate accuracies (same for all strategies)
            seq_acc = calculate_accuracy(seq_logits.view(-1, model_cfg.seq_vocab), seq_x_0.view(-1), diffusion_cfg.seq_absorb_token)
            struct_acc = calculate_accuracy(struct_logits.view(-1, model_cfg.struct_vocab), struct_x_0.view(-1), diffusion_cfg.struct_absorb_token)
            
            # Accumulate metrics
            metrics['loss_seq'] += loss_seq.item() if torch.is_tensor(loss_seq) else loss_seq
            metrics['loss_struct'] += loss_struct.item() if torch.is_tensor(loss_struct) else loss_struct
            metrics['seq_accuracy'] += seq_acc
            metrics['struct_accuracy'] += struct_acc
            num_batches += 1
    
    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches
    
    return metrics

# --------------------------------------------------------------------------- #
#  Main Validation Loop                                                        #
# --------------------------------------------------------------------------- #
def main():
    # Initialize configuration
    val_cfg = ValidationConfig()
    model_cfg = ModelConfig()
    diffusion_cfg = DiffusionConfig()
    
    device = torch.device(val_cfg.device)
    
    print(f"Validation Configuration:")
    print(f"  Model type: {val_cfg.model_type}")
    print(f"  Masking strategies: {val_cfg.masking_strategies}")
    print(f"  Validation data directory: {val_cfg.validation_data_dir}")
    print(f"  Device: {device}")
    print()
    
    # Load validation dataset (all files in the directory)
    dataset = ProteinBackboneDataset(val_cfg.validation_data_dir, max_length=model_cfg.max_len - 2)
    print(f"Loaded {len(dataset)} validation samples")
    
    # Results storage
    results = {}
    
    # Validate each masking strategy
    for masking_strategy in val_cfg.masking_strategies:
        print(f"\n{'='*60}")
        print(f"Evaluating {val_cfg.model_type} with {masking_strategy} masking")
        print(f"{'='*60}")
        
        # Load FSQ encoder for this masking strategy
        encoder_checkpoint_path = f"../checkpoints/{val_cfg.model_type}_stage_1_iter1_{masking_strategy}.pt"
        checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
        encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        fsq_config = SimpleNamespace(**checkpoint['model_cfg_dict'])
        fsq_encoder = FSQEncoder(fsq_config)
        
        # Handle key mapping for Consensus model
        if val_cfg.model_type == "C":
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
        fsq_encoder.eval()
        fsq_encoder.requires_grad_(False)
        fsq_encoder = fsq_encoder.to(device)
        print(f"Loaded FSQ encoder from: {encoder_checkpoint_path}")
        
        # Create dataloader with this masking strategy's encoder
        g_val = torch.Generator()
        g_val.manual_seed(42)  # Fixed seed for reproducibility
        
        val_loader = DiffusionDataLoader(
            dataset, fsq_encoder=fsq_encoder, model_cfg=model_cfg, diffusion_cfg=diffusion_cfg,
            device=device, batch_size=val_cfg.batch_size, shuffle=False, generator=g_val
        )
        
        # Determine checkpoint path - same pattern for all masking strategies
        checkpoint_path = Path(val_cfg.checkpoint_dir) / f"{val_cfg.model_type}_{masking_strategy}_iter1_final.pt"
        
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = create_model_for_masking_strategy(val_cfg.model_type, masking_strategy, model_cfg, device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # For older checkpoints that might store the state directly
            model.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
        
        # Validate
        metrics = validate_model(model, val_loader, model_cfg, diffusion_cfg, val_cfg.model_type, device, masking_strategy)
        
        # Store results
        results[masking_strategy] = metrics
        
        # Print results
        print(f"\nResults for {masking_strategy}:")
        print(f"  Loss:")
        print(f"    - Sequence: {metrics['loss_seq']:.4f}")
        print(f"    - Structure: {metrics['loss_struct']:.4f}")
        print(f"  Accuracy:")
        print(f"    - Sequence: {metrics['seq_accuracy']:.4f}")
        print(f"    - Structure: {metrics['struct_accuracy']:.4f}")
    
    # Save results to file
    output_path = Path(val_cfg.output_file)
    with open(output_path, 'w') as f:
        f.write(f"Validation Results\n")
        f.write(f"Model Type: {val_cfg.model_type}\n")
        f.write(f"Dataset: {val_cfg.validation_data_dir}\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        f.write(f"\n")
        
        for masking_strategy, metrics in results.items():
            f.write(f"\n{masking_strategy.upper()} MASKING:\n")
            f.write(f"  Loss:\n")
            f.write(f"    - Sequence: {metrics['loss_seq']:.4f}\n")
            f.write(f"    - Structure: {metrics['loss_struct']:.4f}\n")
            f.write(f"  Accuracy:\n")
            f.write(f"    - Sequence: {metrics['seq_accuracy']:.4f}\n")
            f.write(f"    - Structure: {metrics['struct_accuracy']:.4f}\n")
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Masking':<20} {'Loss Type':<15} {'Seq Acc':<10} {'Struct Acc':<10}")
    print("-" * 67)
    for masking_strategy, metrics in results.items():
        loss_type = "Score Entropy" if masking_strategy == "discrete_diffusion" else "Cross Entropy"
        print(f"{masking_strategy:<20} {loss_type:<15} {metrics['seq_accuracy']:<10.4f} {metrics['struct_accuracy']:<10.4f}")

if __name__ == "__main__":
    main() 