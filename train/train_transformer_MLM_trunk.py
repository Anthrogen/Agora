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
    consensus_num_iterations: int = 5 # Number of Consensus gradient iterations
    consensus_connectivity_type: str = "local_window"  # "local_window" or "top_w"
    consensus_w: int = 2  # Window size for local_window, or w value for top_w
    consensus_r: int = 8  # Rank of Lambda_ij matrices
    consensus_edge_hidden_dim: int = 24  # Hidden dim for edge networks

@dataclass
class TrainingConfig:
    """Training process configuration."""
    model_types: List[str] = field(default_factory=lambda: ["SA", "GA", "RA", "C"]) # Models to train - can be any subset of ["SA", "GA", "RA", "C"]
    batch_size: int = 4  # Training hyperparameters
    max_epochs: int = 25
    learning_rate: float = 1e-5
    num_iter: int = 5  # Number of iterations to repeat training
    masking_strategy: str = "simple" # Masking strategy: 'simple' or 'complex'
    
    if masking_strategy == "simple":
        seq_loss_weight: float = 1.0  # sequence loss weight - simple: 1.0
        struct_loss_weight: float = 1.0 # structure loss weight - simple: 1.0
        mask_prob_seq: float = 0.2 # Masking probability for sequence tokens
        mask_prob_coords: float = 0.2 # Masking probability for structure tokens
    elif masking_strategy == "complex":
        seq_loss_weight: float = 1.0  # sequence loss weight - complex: 1.0
        struct_loss_weight: float = 0.5  # structure loss weight - complex: 0.5

    data_dir: str = "../sample_data/1k/"  # Data paths
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
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Calculate accuracy for masked positions only."""
    if not mask.any(): return 0.0
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()

def worker_init_fn(worker_id):
    """Initialize each worker with a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_step(models: Dict[str, TransformerTrunk], optimizers: Dict[str, torch.optim.Optimizer], batch: MaskedBatch, train_cfg: TrainingConfig, model_cfg: ModelConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Perform a single training step for all models with the same batch."""
    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    mask_seq, mask_struct, mask_coords= batch.masks['seq'], batch.masks['struct'], batch.masks['coords']
    seq_tokens, struct_tokens = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    B, L = masked_seq.shape

    assert torch.all(mask_coords == mask_struct), f"mask_coords and mask_struct differ in some positions:\nmask_coords:\n{mask_coords}\nmask_struct:\n{mask_struct}"

    # Create coord_mask for GA/RA models
    nonspecial_elements_coords = (~batch.beospad['coords']) & (~mask_struct)
    # We need one non-special element in coords for GA/RA models.
    assert nonspecial_elements_coords.any(dim=1).all()
    
    inputs = (masked_seq, masked_struct) # Prepare model input
    metrics = {} # Store metrics for each model
    
    # Train each model on the same batch
    for model_type, model in models.items():
        model.train()
        optimizer = optimizers[model_type]
        
        # Forward pass
        if model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, nonspecial_elements_coords)
        else: outputs = model(inputs)
        seq_logits, struct_logits = outputs

        
        # Flatten for loss computation
        #TODO: double-check the relative weighting between residues of different proteins within a common batch.
        seq_logits_flat = seq_logits.view(-1, model_cfg.seq_vocab)
        struct_logits_flat = struct_logits.view(-1, model_cfg.struct_vocab)
        seq_labels_flat = seq_tokens.view(-1)
        struct_labels_flat = struct_tokens.view(-1)
        seq_mask_flat = mask_seq.view(-1)
        struct_mask_flat = mask_struct.view(-1)
        
        # # Compute losses
        # loss_seq = F.cross_entropy(seq_logits_flat[seq_mask_flat], seq_labels_flat[seq_mask_flat].long()) if seq_mask_flat.any() else torch.tensor(0.0, device=device)
        # loss_struct = F.cross_entropy(struct_logits_flat[struct_mask_flat], struct_labels_flat[struct_mask_flat].long()) if struct_mask_flat.any() else torch.tensor(0.0, device=device)

        # Compute losses
        if seq_mask_flat.any():
            loss_seq = F.cross_entropy(seq_logits_flat[seq_mask_flat], seq_labels_flat[seq_mask_flat].long())
        else:
            # Use a small regularization loss to maintain gradient flow
            loss_seq = 0.0 * seq_logits.sum()
            print(f"Warning: No sequence positions masked for {model_type}")
        
        if struct_mask_flat.any():
            loss_struct = F.cross_entropy(struct_logits_flat[struct_mask_flat], struct_labels_flat[struct_mask_flat].long())
        else:
            # Use a small regularization loss to maintain gradient flow
            loss_struct = 0.0 * struct_logits.sum()
            print(f"Warning: No structure positions masked for {model_type}")

        loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
        

        # Calculate accuracies
        seq_acc = calculate_accuracy(seq_logits_flat, seq_labels_flat, seq_mask_flat)
        struct_acc = calculate_accuracy(struct_logits_flat, struct_labels_flat, struct_mask_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store metrics
        metrics[model_type] = {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item(), 'seq_acc': seq_acc, 'struct_acc': struct_acc}
    
    return metrics

def validate_step(models: Dict[str, TransformerTrunk], batch: MaskedBatch, train_cfg: TrainingConfig, model_cfg: ModelConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Perform a single validation step for all models with the same batch."""
    masked_seq, masked_struct, masked_coords = batch.masked_data['seq'], batch.masked_data['struct'], batch.masked_data['coords']
    mask_seq, mask_struct = batch.masks['seq'], batch.masks['struct']
    seq_tokens, struct_tokens = batch.unmasked_data['seq'], batch.unmasked_data['struct']
    B, L = masked_seq.shape

    # Create coord_mask for GA/RA models
    nonspecial_elements = (~batch.beospad['coords']) & (~mask_struct) # boolean tensor of shape (B,L), True for all positions corresponding to non-BOS/EOS/PAD and non-MASK tokens.
    # We need one non-special element in coords for GA/RA models.
    assert nonspecial_elements.any(dim=1).all()
    
    inputs = (masked_seq, masked_struct) # Prepare model input
    metrics = {} # Store metrics for each model
    
    # Evaluate each model on the same batch
    for model_type, model in models.items():
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            if model_type in ("GA", "RA"): outputs = model(inputs, masked_coords, nonspecial_elements)
            else: outputs = model(inputs)  
            seq_logits, struct_logits = outputs
            
            # Flatten for loss computation
            seq_logits_flat = seq_logits.view(-1, model_cfg.seq_vocab)
            struct_logits_flat = struct_logits.view(-1, model_cfg.struct_vocab)
            seq_labels_flat = seq_tokens.view(-1)
            struct_labels_flat = struct_tokens.view(-1)
            seq_mask_flat = mask_seq.view(-1)
            struct_mask_flat = mask_struct.view(-1)
            
            # Compute losses
            loss_seq = F.cross_entropy(seq_logits_flat[seq_mask_flat], seq_labels_flat[seq_mask_flat].long()) if seq_mask_flat.any() else torch.tensor(0.0, device=device)
            loss_struct = F.cross_entropy(struct_logits_flat[struct_mask_flat], struct_labels_flat[struct_mask_flat].long()) if struct_mask_flat.any() else torch.tensor(0.0, device=device)
            loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct
            
            # Calculate accuracies
            seq_acc = calculate_accuracy(seq_logits_flat, seq_labels_flat, seq_mask_flat)
            struct_acc = calculate_accuracy(struct_logits_flat, struct_labels_flat, struct_mask_flat)
            
            # Store metrics
            metrics[model_type] = {'loss': loss.item(), 'loss_seq': loss_seq.item(), 'loss_struct': loss_struct.item(), 'seq_acc': seq_acc, 'struct_acc': struct_acc}
    
    return metrics

def main():
    # Initialize configurations
    model_cfg, train_cfg = ModelConfig(), TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Validate model types
    valid_types = {"SA", "GA", "RA", "C"}
    for mt in train_cfg.model_types:
        if mt not in valid_types:
            raise ValueError(f"Invalid model type: {mt}. Must be one of {valid_types}")
    
    # Arrays to store validation metrics for all models across iterations: (model_type, num_iter, max_epochs)
    all_metrics = {
        model_type: {'val_seq_acc': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)), 'val_struct_acc': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),
                    'val_seq_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs)),'val_struct_loss': np.zeros((train_cfg.num_iter, train_cfg.max_epochs))}
        for model_type in train_cfg.model_types
    }
    
    print(f"Starting training with {train_cfg.num_iter} iterations")
    print(f"Training models: {train_cfg.model_types}")
    print(f"Using masking strategy: {train_cfg.masking_strategy}")
    
    # -------------------- Iteration loop -------------------- #
    for iteration in range(train_cfg.num_iter):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{train_cfg.num_iter}")
        print(f"{'='*60}\n")
        
        # Create all models with fixed seed for this iteration
        print(f"Creating {len(train_cfg.model_types)} models for iteration {iteration + 1}...")
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
            #TODO: make this configurable; use os.path.join
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
        
        # Ensure consistent parameter initialization across architectures
        print(f"Ensuring all models have identical parameters...")
        ensure_identical_parameters_all_models(models, train_cfg.reference_model_seed + iteration)
        
        # Print parameter count (only on first iteration)
        if iteration == 0:
            for model_type, model in models.items():
                total_params = sum(p.numel() for p in model.parameters())
                print(f"{model_type} total parameters: {total_params:,}")
        
        # -------------------- Data loading -------------------- #
        # Set seed for dataset split AND masking to ensure consistency
        data_seed = train_cfg.reference_model_seed + iteration * 1000
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        random.seed(data_seed)
        
        dataset = ProteinDataset(train_cfg.data_dir, max_length=model_cfg.max_len - 2) # Reserve 2 positions for BOS/EOS
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
        assert fsq_encoders[train_cfg.model_types[-1]] is not None
        train_loader = _get_training_dataloader(train_ds, 
                                                model_cfg, 
                                                train_cfg, 
                                                tracks, 
                                                device, 
                                                min_unmasked=min_unmasked, 
                                                fsq_encoder=fsq_encoders[train_cfg.model_types[-1]],
                                                shuffle=True, 
                                                batch_size=train_cfg.batch_size,
                                                generator=g_train, 
                                                worker_init_fn=worker_init_fn)

        val_loader = _get_training_dataloader(val_ds, 
                                                model_cfg, 
                                                train_cfg, 
                                                tracks, 
                                                device, 
                                                min_unmasked=min_unmasked, 
                                                fsq_encoder=fsq_encoders[train_cfg.model_types[-1]],
                                                shuffle=False, 
                                                batch_size=train_cfg.batch_size,
                                                generator=g_val, 
                                                worker_init_fn=worker_init_fn)
        
        # Initialize tracking for each model
        history = {
            model_type: {'train_loss': [], 'train_seq_acc': [], 'train_struct_acc': [],
                'val_loss': [], 'val_seq_acc': [], 'val_struct_acc': [], 'val_seq_loss': [], 'val_struct_loss': []
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
                     ascii=True, leave=True, ncols=150, position=0) as pbar:
                for batch in pbar:
                    # Train all models on the same batch
                    batch_metrics = train_step(models, optimizers, batch, train_cfg, model_cfg, device)
                    
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
                history[model_type]['train_seq_acc'].append(train_metrics_sum[model_type]['seq_acc'])
                history[model_type]['train_struct_acc'].append(train_metrics_sum[model_type]['struct_acc'])
            
            # -------------------- Validation -------------------- #
            val_metrics_sum = {
                model_type: {'loss': 0.0, 'loss_seq': 0.0, 'loss_struct': 0.0, 'seq_acc': 0.0, 'struct_acc': 0.0}
                for model_type in train_cfg.model_types
            }
            val_num_batches = 0
            
            for batch in val_loader:
                # Validate all models on the same batch
                batch_metrics = validate_step(models, batch, train_cfg, model_cfg, device)
                
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
                history[model_type]['val_seq_acc'].append(val_metrics_sum[model_type]['seq_acc'])
                history[model_type]['val_struct_acc'].append(val_metrics_sum[model_type]['struct_acc'])
                history[model_type]['val_seq_loss'].append(val_metrics_sum[model_type]['loss_seq'])
                history[model_type]['val_struct_loss'].append(val_metrics_sum[model_type]['loss_struct'])
                
                # Store in global metrics arrays
                all_metrics[model_type]['val_seq_acc'][iteration, epoch] = val_metrics_sum[model_type]['seq_acc']
                all_metrics[model_type]['val_struct_acc'][iteration, epoch] = val_metrics_sum[model_type]['struct_acc']
                all_metrics[model_type]['val_seq_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss_seq']
                all_metrics[model_type]['val_struct_loss'][iteration, epoch] = val_metrics_sum[model_type]['loss_struct']
            
            # Print detailed epoch summary
            print(f"\nIteration {iteration+1}, Epoch {epoch+1}/{train_cfg.max_epochs}")
            for model_type in train_cfg.model_types:
                print(f"\n{model_type}:")
                print(f"  Train - Loss: {train_metrics_sum[model_type]['loss']:.4f} "
                      f"(Seq: {train_metrics_sum[model_type]['loss_seq']:.4f}, "
                      f"Struct: {train_metrics_sum[model_type]['loss_struct']:.4f})")
                print(f"          Acc:  Seq: {train_metrics_sum[model_type]['seq_acc']:.4f}, "
                      f"Struct: {train_metrics_sum[model_type]['struct_acc']:.4f}")
                print(f"  Val   - Loss: {val_metrics_sum[model_type]['loss']:.4f} "
                      f"(Seq: {val_metrics_sum[model_type]['loss_seq']:.4f}, "
                      f"Struct: {val_metrics_sum[model_type]['loss_struct']:.4f})")
                print(f"          Acc:  Seq: {val_metrics_sum[model_type]['seq_acc']:.4f}, "
                      f"Struct: {val_metrics_sum[model_type]['struct_acc']:.4f}")
                    
        # Save final checkpoints only for the first iteration
        if iteration == 0:
            for model_type in train_cfg.model_types:
                final_checkpoint_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_{train_cfg.masking_strategy}_iter{iteration+1}_final.pt"
                torch.save({
                    'iteration': iteration + 1,
                    'epoch': train_cfg.max_epochs,
                    'model_type': model_type,
                    'model_state_dict': models[model_type].state_dict(),
                    'optimizer_state_dict': optimizers[model_type].state_dict(),
                    'history': history[model_type],
                    'model_cfg': model_cfg,
                }, final_checkpoint_path)
            print(f"\nSaved final checkpoints for iteration {iteration+1}")
        else:
            print(f"\nSkipping checkpoint save for iteration {iteration+1} (only saving iteration 1)")
    
    # -------------------- Save validation metrics to CSV -------------------- #
    for model_type in train_cfg.model_types:
        # Save sequence validation accuracies
        seq_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_sequence_val_acc.csv"
        np.savetxt(seq_csv_path, all_metrics[model_type]['val_seq_acc'], delimiter=',', 
                   header=f"Validation sequence accuracies for {model_type}\nRows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        # Save structure validation accuracies
        struct_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_structure_val_acc.csv"
        np.savetxt(struct_csv_path, all_metrics[model_type]['val_struct_acc'], delimiter=',',
                   header=f"Validation structure accuracies for {model_type}\nRows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        # Save sequence validation losses
        seq_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_sequence_val_loss.csv"
        np.savetxt(seq_loss_csv_path, all_metrics[model_type]['val_seq_loss'], delimiter=',', 
                   header=f"Validation sequence losses for {model_type}\nRows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        # Save structure validation losses
        struct_loss_csv_path = Path(train_cfg.checkpoint_dir) / f"{model_type}_structure_val_loss.csv"
        np.savetxt(struct_loss_csv_path, all_metrics[model_type]['val_struct_loss'], delimiter=',',
                   header=f"Validation structure losses for {model_type}\nRows: iterations ({train_cfg.num_iter}), Columns: epochs ({train_cfg.max_epochs})", comments='# ')
        
        print(f"\nSaved metrics for {model_type}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Models trained: {train_cfg.model_types}")
    print(f"Number of iterations: {train_cfg.num_iter}")
    print(f"Number of epochs per iteration: {train_cfg.max_epochs}")
    
    for model_type in train_cfg.model_types:
        print(f"\n{model_type}:")
        print(f"  Sequence validation accuracy:")
        print(f"    Mean final epoch: {all_metrics[model_type]['val_seq_acc'][:, -1].mean():.4f} ± {all_metrics[model_type]['val_seq_acc'][:, -1].std():.4f}")
        print(f"    Best single run: {all_metrics[model_type]['val_seq_acc'].max():.4f}")
        print(f"  Structure validation accuracy:")
        print(f"    Mean final epoch: {all_metrics[model_type]['val_struct_acc'][:, -1].mean():.4f} ± {all_metrics[model_type]['val_struct_acc'][:, -1].std():.4f}")
        print(f"    Best single run: {all_metrics[model_type]['val_struct_acc'].max():.4f}")

if __name__ == "__main__":
    main() 