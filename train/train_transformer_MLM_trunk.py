"""
Train the unified transformer model with masked language modeling.
Refactored version with masking logic moved to DataLoader.

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

Masking Strategy:
- All architectures use IDENTICAL masking patterns for each iteration by using
  a fixed seed for the DataLoader workers
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
from types import SimpleNamespace

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.transformer import TransformerTrunk, StandardTransformerBlock
from src.models.autoencoder import FSQEncoder
from src.dataloader import _get_training_dataloader, MaskedBatch
from src.dataset import ProteinDataset
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from src.losses import cross_entropy_loss, calculate_accuracy

# --------------------------------------------------------------------------- #
#  Configurations                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128 # 768  # Model dimensions
    n_heads: int = 1 # 12
    n_layers: int = 3 # 12
    seq_vocab: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)  # Sequence tokens + special tokens
    struct_vocab: int = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
    max_len: int = 2048
    dropout: float = 0.1  # Other architecture params
    ff_mult: int = 4
    ff_hidden_dim: int = d_model * ff_mult
    
    # Consensus-specific parameters
    consensus_num_iterations: int = 1 # Number of Consensus gradient iterations
    consensus_connectivity_type: str = "local_window"  # "local_window" or "top_w"
    consensus_w: int = 2  # Window size for local_window, or w value for top_w
    consensus_r: int = 8  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim: int = 24  # Hidden dim for edge networks



@dataclass
class TrainingConfig:
    """Training process configuration."""
    model_type: str = "SC"  # Model to train - can be "SA", "GA", "RA", or "SC"
    batch_size: int = 4  # Training hyperparameters
    max_epochs: int = 50
    learning_rate: float = 1e-5
    num_iter: int = 3  # Number of iterations to repeat training
    masking_strategy: str = "simple" # Masking strategy: 'simple' or 'complex'
    
    if masking_strategy == "simple":
        seq_loss_weight: float = 1.0  # sequence loss weight - simple: 1.0
        struct_loss_weight: float = 1.0 # structure loss weight - simple: 1.0
        mask_prob_seq: float = 0.2 # Masking probability for sequence tokens
        mask_prob_coords: float = 0.2 # Masking probability for structure tokens
    elif masking_strategy == "complex":
        seq_loss_weight: float = 1.0  # sequence loss weight - complex: 1.0
        struct_loss_weight: float = 0.5  # structure loss weight - complex: 0.5

    # Cross-entropy loss function: which elements should contribute to the loss?
    # "masked": only masked positions
    # "non_beospank": all non-BOS/EOS/PAD positions, including masks
    # "non_special": all non-special tokens, including masks
    ce_loss_function_elements: str = "masked"

    data_dir: str = "../sample_data/1k.csv"  # Data paths
    checkpoint_dir: str = "../checkpoints/transformer_trunk"  # Checkpointing
    reference_model_seed: int = 22 # Reference model seed for consistent parameter initialization across architectures

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
    return TransformerTrunk(config).to(device)

def ensure_identical_parameters_all_models(models: Dict[str, TransformerTrunk], seed: int):
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

# --------------------------------------------------------------------------- #
#  Training utilities                                                          #
# --------------------------------------------------------------------------- #
def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def mlm_step(model: TransformerTrunk, optimizer: torch.optim.Optimizer, batch: MaskedBatch, model_cfg: ModelConfig, train_cfg: TrainingConfig, train_mode: bool = True) -> Dict[str, float]:
    """Perform a single MLM step with train/validation mode."""
    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    mask_seq, mask_struct, mask_coords= batch.masks['seq'], batch.masks['struct'], batch.masks['coords']
    seq_tokens, struct_tokens = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    B, L = masked_seq.shape

    assert torch.all(mask_coords == mask_struct), f"mask_coords and mask_struct differ in some positions:\nmask_coords:\n{mask_coords}\nmask_struct:\n{mask_struct}"

    # Create coord_mask for GA/RA models
    nonspecial_elements_coords = (~batch.beospank['coords']) & (~mask_struct)
    # We need one non-special element in coords for GA/RA models.
    assert nonspecial_elements_coords.any(dim=1).all()
    
    inputs = (masked_seq, masked_struct) # Prepare model input
    model.train(train_mode)
    
    with torch.set_grad_enabled(train_mode):
        # Forward pass
        if train_cfg.model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, nonspecial_elements_coords)
        else: outputs = model(inputs)
        seq_logits, struct_logits = outputs

        if train_cfg.ce_loss_function_elements == "masked":
            loss_elements_seq = batch.masks['seq']
            loss_elements_struct = batch.masks['struct']
        elif train_cfg.ce_loss_function_elements == "non_beospank":
            # Compute loss over all non-BOS/EOS/PAD positions, including masks.
            loss_elements_seq = ~batch.beospank['seq']
            loss_elements_struct = ~batch.beospank['struct']
        elif train_cfg.ce_loss_function_elements == "non_special":
            loss_elements_seq = ~batch.beospank['seq'] & ~batch.masks['seq']
            loss_elements_struct = ~batch.beospank['struct'] & ~batch.masks['struct']
        else:
            raise ValueError(f"What is {train_cfg.ce_loss_function_elements}?")

        loss_seq = cross_entropy_loss(seq_logits, batch.unmasked_data['seq'], loss_elements_seq)
        loss_struct = cross_entropy_loss(struct_logits, batch.unmasked_data['struct'], loss_elements_struct)

        loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct

        seq_acc = calculate_accuracy(seq_logits, batch.unmasked_data['seq'], loss_elements_seq)
        struct_acc = calculate_accuracy(struct_logits, batch.unmasked_data['struct'], loss_elements_struct)
        
        if train_mode:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Return metrics
        return {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item(), 'seq_acc': seq_acc, 'struct_acc': struct_acc}

def main():
    # Initialize configurations
    model_cfg, train_cfg = ModelConfig(), TrainingConfig()
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
    
    print(f"Starting training with {train_cfg.num_iter} iterations")
    print(f"Training model: {train_cfg.model_type}")
    print(f"Using masking strategy: {train_cfg.masking_strategy}")
    
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
        #TODO: make this configurable; use os.path.join
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
        # Set seed for dataset split AND masking to ensure consistency
        data_seed = train_cfg.reference_model_seed + iteration * 1000
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        random.seed(data_seed)
        
        dataset = ProteinDataset(train_cfg.data_dir, max_length=model_cfg.max_len - 2, verbose=False) # Reserve 2 positions for BOS/EOS
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders with fixed seed for consistent masking (Use Consensus model for all models)
        g_train = torch.Generator()
        g_train.manual_seed(data_seed)
        g_val = torch.Generator()
        g_val.manual_seed(data_seed + 5000)
        
        tracks = {'seq': True, 'struct': True, 'coords': True}
        min_unmasked = {'seq': 0, 'struct': 0, 'coords': 1}
        train_loader = _get_training_dataloader(train_ds, model_cfg, train_cfg, tracks, device, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder, shuffle=True, batch_size=train_cfg.batch_size, generator=g_train, worker_init_fn=worker_init_fn)
        val_loader = _get_training_dataloader(val_ds, model_cfg, train_cfg, tracks, device, min_unmasked=min_unmasked, fsq_encoder=fsq_encoder, shuffle=False, batch_size=train_cfg.batch_size, generator=g_val, worker_init_fn=worker_init_fn)
        
        # Initialize tracking
        history = {'train_loss': [], 'train_seq_acc': [], 'train_struct_acc': [], 'val_loss': [], 'val_seq_acc': [], 'val_struct_acc': [], 'val_seq_loss': [], 'val_struct_loss': []}
        
        # -------------------- Training loop -------------------- #
        for epoch in range(train_cfg.max_epochs):
            # Training metrics accumulators
            train_metrics_sum = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0}
            train_metrics_num_batches = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0}

            # Training
            with tqdm(train_loader, desc=f"Iter {iteration+1}/{train_cfg.num_iter}, Epoch {epoch+1}/{train_cfg.max_epochs} [{train_cfg.model_type} Train]", 
                     ascii=True, leave=True, ncols=150, position=0) as pbar:
                for batch in pbar:
                    # Skip empty/None batches
                    if batch is None: continue
                    
                    # Train single model on batch
                    batch_metrics = mlm_step(model, optimizer, batch, model_cfg, train_cfg, train_mode=True)
                    
                    # Accumulate metrics
                    for key in train_metrics_sum:
                        if batch_metrics[key] is not None: # Some metrics (e.g. accuracy) may be None if there are no loss_elements.
                            train_metrics_sum[key] += batch_metrics[key]
                            train_metrics_num_batches[key] += 1
                    
                    # Update progress bar
                    pbar.set_postfix({f'{train_cfg.model_type}_loss': f"{batch_metrics['loss']:.3f}"})
            
            # Calculate epoch averages for training
            for key in train_metrics_sum: train_metrics_sum[key] /= train_metrics_num_batches[key]
            
            history['train_loss'].append(train_metrics_sum['loss'])
            history['train_seq_acc'].append(train_metrics_sum['seq_acc'])
            history['train_struct_acc'].append(train_metrics_sum['struct_acc'])
            
            # -------------------- Validation -------------------- #
            val_metrics_sum = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0}
            val_metrics_num_batches = {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0}
            
            for batch in val_loader:
                # Skip empty/None batches
                if batch is None: continue
                
                # Validate single model on batch
                batch_metrics = mlm_step(model, optimizer, batch, model_cfg, train_cfg, train_mode=False)
                
                # Accumulate metrics
                for key in val_metrics_sum:
                    if batch_metrics[key] is not None: # Some metrics (e.g. accuracy) may be None if there are no loss_elements.
                        val_metrics_sum[key] += batch_metrics[key]
                        val_metrics_num_batches[key] += 1
            
            # Calculate epoch averages for validation
            for key in val_metrics_sum: val_metrics_sum[key] /= val_metrics_num_batches[key]
            
            # Store in history
            history['val_loss'].append(val_metrics_sum['loss'])
            history['val_seq_acc'].append(val_metrics_sum['seq_acc'])
            history['val_struct_acc'].append(val_metrics_sum['struct_acc'])
            history['val_seq_loss'].append(val_metrics_sum['loss_seq'])
            history['val_struct_loss'].append(val_metrics_sum['loss_struct'])
            
            # Store in global metrics arrays
            all_metrics['val_loss'][iteration, epoch] = val_metrics_sum['loss']
            all_metrics['val_seq_loss'][iteration, epoch] = val_metrics_sum['loss_seq']
            all_metrics['val_struct_loss'][iteration, epoch] = val_metrics_sum['loss_struct']
            
            # Print detailed epoch summary
            print(f"\nIteration {iteration+1}, Epoch {epoch+1}/{train_cfg.max_epochs} - {train_cfg.model_type}:")
            # Training metrics
            print(f"  Train:")
            print(f"    Loss: {train_metrics_sum['loss']:.4f} (Seq: {train_metrics_sum['loss_seq']:.4f}, Struct: {train_metrics_sum['loss_struct']:.4f})")
            print(f"    Acc:  Seq: {train_metrics_sum['seq_acc']:.4f}, Struct: {train_metrics_sum['struct_acc']:.4f}")
            
            # Validation metrics
            print(f"  Val:")
            print(f"    Loss: {val_metrics_sum['loss']:.4f} (Seq: {val_metrics_sum['loss_seq']:.4f}, Struct: {val_metrics_sum['loss_struct']:.4f})")
            print(f"    Acc:  Seq: {val_metrics_sum['seq_acc']:.4f}, Struct: {val_metrics_sum['struct_acc']:.4f}")
                    
        # Save final checkpoints only for the first iteration
        if iteration == 0:
            final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_{train_cfg.masking_strategy}_iter{iteration+1}_final.pt"
            torch.save({
                'iteration': iteration + 1,
                'epoch': train_cfg.max_epochs,
                'model_type': train_cfg.model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'model_config': model_cfg,
            }, final_checkpoint_path)
            print(f"\nSaved final checkpoint for {train_cfg.model_type} iteration {iteration+1}")
        else:
            print(f"\nSkipping checkpoint save for {train_cfg.model_type} iteration {iteration+1} (only saving iteration 1)")
    
    # -------------------- Save validation metrics to CSV -------------------- #
    # Save validation losses
    loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_{train_cfg.masking_strategy}_val_loss.csv"
    np.savetxt(loss_csv_path, all_metrics['val_loss'], delimiter=',',
               header=f"Validation losses for {train_cfg.model_type} ({train_cfg.masking_strategy})\n"
                     f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
    
    # Save sequence losses
    seq_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_{train_cfg.masking_strategy}_seq_val_loss.csv"
    np.savetxt(seq_loss_csv_path, all_metrics['val_seq_loss'], delimiter=',', 
               header=f"Sequence validation losses for {train_cfg.model_type} ({train_cfg.masking_strategy})\n"
                     f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
    
    # Save structure losses
    struct_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{train_cfg.model_type}_{train_cfg.masking_strategy}_struct_val_loss.csv"
    np.savetxt(struct_loss_csv_path, all_metrics['val_struct_loss'], delimiter=',',
               header=f"Structure validation losses for {train_cfg.model_type} ({train_cfg.masking_strategy})\n"
                     f"Rows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
    
    print(f"\nSaved metrics for {train_cfg.model_type}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Model trained: {train_cfg.model_type}")
    print(f"Number of iterations: {train_cfg.num_iter}")
    print(f"Number of epochs per iteration: {train_cfg.max_epochs}")
    
    print(f"\n{train_cfg.model_type}:")
    print(f"  Total validation loss:")
    print(f"    Mean final epoch: {all_metrics['val_loss'][:, -1].mean():.4f} ± {all_metrics['val_loss'][:, -1].std():.4f}")
    print(f"    Best single run: {all_metrics['val_loss'].min():.4f}")
    print(f"  Sequence validation loss:")
    print(f"    Mean final epoch: {all_metrics['val_seq_loss'][:, -1].mean():.4f} ± {all_metrics['val_seq_loss'][:, -1].std():.4f}")
    print(f"    Best single run: {all_metrics['val_seq_loss'].min():.4f}")
    print(f"  Structure validation loss:")
    print(f"    Mean final epoch: {all_metrics['val_struct_loss'][:, -1].mean():.4f} ± {all_metrics['val_struct_loss'][:, -1].std():.4f}")
    print(f"    Best single run: {all_metrics['val_struct_loss'].min():.4f}")

if __name__ == "__main__":
    main() 