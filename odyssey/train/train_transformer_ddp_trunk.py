"""
Train the unified transformer model with masked language modeling.
The model uses either GeometricAttention or SelfAttention in the first layer,
with all subsequent layers using SelfAttention. The FSQ encoder and decoder
are frozen during training.

Parameter count calculation for 1M target:
- Embedding layers: seq_vocab (30) * d_model + struct_vocab (4379) * d_model
  For d_model = 128: (30 + 4379) * 128 = ~564K
- Attention layers (per block): 4 * d_model * d_model (Q,K,V,O matrices) = 4 * 128 * 128 = ~66K
  FF layer: 2 * d_model * 2 * d_model = ~65K, Total per block: ~131K
- For 3 layers: ~393K + 564K embeddings ≈ 957K parameters (~1M)
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
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.quantizer.fsq import FSQ
from trained_fsq.fsq_autoencoder_GA_var3 import FSQEncoder, FSQDecoder
from odyssey.src.data_loader import (
    ProteinBackboneDataset, AMINO_ACID_VOCAB, STRUCTURE_SPECIAL_TOKENS,
    STRUCTURE_VOCAB_SIZE, add_structure_special_tokens
)

# --------------------------------------------------------------------------- #
#  Configurations                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128  # Model dimensions (reduced from 768)
    n_heads: int = 8    # Number of attention heads (reduced from 12)
    n_layers: int = 3   # Number of transformer layers (reduced from 12)
    # Geometric Attention = GA,
    # Self Attention = SA,
    # Reflexive Attention = RA
    first_layer: str = "SA"  
    
    seq_vocab: int = 30    # 20 canonical + 5 special + 4 non-standard + 1 ambiguous (Odyssey spec)
    struct_vocab: int = 4379  # 4375 FSQ tokens + 4 special tokens
    max_len: int = 2048
    
    dropout: float = 0.1  # Other architecture params
    ff_mult: int = 2  # Feed-forward multiplier (reduced from 4)
    ff_hidden_dim: int = d_model * ff_mult

@dataclass
class TrainingConfig:
    """Training process configuration."""
    batch_size: int = 52  # Training hyperparameters
    max_epochs: int = 100
    learning_rate: float = 1e-4
    
    masking_strategy: str = "complex"  # 'simple' or 'complex'
    mask_prob_seq: float = 0.15  # Simple masking probabilities for MLM on each track
    mask_prob_struct: float = 0.15
    
    if masking_strategy == "simple":
        seq_loss_weight: float = 1.0  # Loss weights - simple: (1.0, 1.0)
        struct_loss_weight: float = 1.0 
    elif masking_strategy == "complex":
        seq_loss_weight: float = 1.0  # Loss weights - complex: (1.0, 0.5)
        struct_loss_weight: float = 0.5 
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

    data_dir: str = "/workspace/cmu_vqvae_data"  # Data paths - changed to real dataset
    csv_file: str = "/workspace/cmu_vqvae_data/single_chain_clusters_full1.csv"  # CSV file with representative structures
    checkpoint_dir: str = "checkpoints"  # Checkpointing
    save_every: int = 5
    
    # W&B configuration
    use_wandb: bool = True
    wandb_project: str = "transformer-training-real"
    wandb_run_name: Optional[str] = None  # If None, will auto-generate with timestamp

# --------------------------------------------------------------------------- #
#  Noise Schedules for complex masking                                        #
# --------------------------------------------------------------------------- #
def sample_betalinear30(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample mask rates from betalinear30 distribution. 80% Beta(3,9), 20% Uniform(0,1), avg ~30%"""
    mask_rates = torch.zeros(batch_size, device=device)
    use_beta = torch.rand(batch_size) < 0.8  # Choose distribution for each batch element
    
    beta_samples = torch.distributions.Beta(3.0, 9.0).sample((batch_size,)).to(device)  # Beta(3, 9) samples
    uniform_samples = torch.rand(batch_size, device=device)  # Uniform(0, 1) samples
    mask_rates = torch.where(use_beta.to(device), beta_samples, uniform_samples)  # Combine based on use_beta
    
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values

def sample_cosine(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample mask rates from cosine distribution using sin(u * π/2) for higher density near 1."""
    u = torch.rand(batch_size, device=device)  # Sample uniform values
    mask_rates = torch.sin(u * np.pi / 2)  # Apply sine transformation
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values

def create_masked_inputs_complex(tokens: torch.Tensor, mask_rates: torch.Tensor, 
                                mask_token_id: int = AMINO_ACID_VOCAB['<MASK>']) -> tuple[torch.Tensor, torch.Tensor]:
    """Create masked tokens using complex masking with variable mask rates."""
    B, L = tokens.shape
    device = tokens.device
    
    rand_vals = torch.rand(B, L, device=device)  # Create random values for each position
    mask_rates_expanded = mask_rates.unsqueeze(1).expand(B, L)  # Expand mask_rates to match token shape
    mask = rand_vals < mask_rates_expanded  # Create mask based on mask rates
    
    # Ensure we don't mask everything - keep at least one token unmasked per sequence (prevents NaN losses)
    all_masked = mask.all(dim=1)
    if all_masked.any():
        for idx in torch.where(all_masked)[0]:  # For sequences where everything is masked, unmask a random position
            unmask_pos = torch.randint(0, L, (1,), device=device)
            mask[idx, unmask_pos] = False
    
    masked_tokens = tokens.clone()  # Apply masking
    masked_tokens[mask] = mask_token_id
    return masked_tokens, mask

def create_masked_structure_tokens_complex(tokens: torch.Tensor, mask_rates: torch.Tensor,
                                         mask_token_id: int = STRUCTURE_SPECIAL_TOKENS['<STRUCT_MASK>']) -> tuple[torch.Tensor, torch.Tensor]:
    """Create masked structure tokens using complex masking with variable mask rates."""
    return create_masked_inputs_complex(tokens, mask_rates, mask_token_id=mask_token_id)

def collate_batch(batch):
    """Custom collate_fn for DataLoader. Stacks (seq_tokens, coords, mask) along batch dimension."""
    seq_list, coord_list, mask_list = zip(*batch)  # unpack
    return torch.stack(seq_list, dim=0), torch.stack(coord_list, dim=0), torch.stack(mask_list, dim=0)  # [B, L], [B, L, 3, 3], [B, L]

# --------------------------------------------------------------------------- #
#  Training utilities                                                          #
# --------------------------------------------------------------------------- #
def create_coordinate_mask(original_lengths: torch.Tensor, total_length: int, device: torch.device) -> torch.Tensor:
    """Create mask for valid coordinate positions. Valid coordinates are positions 1 to original_length."""
    B = original_lengths.shape[0]
    coord_mask = torch.zeros(B, total_length, dtype=torch.bool, device=device)
    
    for i in range(B):
        if original_lengths[i] > 0:
            eos_pos = min(original_lengths[i] + 1, total_length - 1)  # EOS position depends on whether sequence fills buffer
            end_pos = eos_pos if eos_pos < total_length else total_length - 1  # Valid coordinates from position 1 to before EOS
            coord_mask[i, 1:end_pos] = True
    
    return coord_mask

def apply_special_token_masks(mask: torch.Tensor, original_lengths: torch.Tensor, mask_type: str = "both") -> torch.Tensor:
    """Ensure BOS/EOS tokens are never masked for MLM."""
    B, L = mask.shape
    
    for i in range(B):
        if mask_type in ("both", "bos"):  # Always protect BOS at position 0
            mask[i, 0] = False
        if mask_type in ("both", "eos"):  # Protect EOS token
            eos_pos = min(original_lengths[i] + 1, L - 1)
            mask[i, eos_pos] = False
    
    return mask

def process_batch_masking(seq_tokens: torch.Tensor, struct_tokens: torch.Tensor, original_lengths: torch.Tensor,
                         masking_strategy: str, mask_prob_seq: float = 0.15, mask_prob_struct: float = 0.15,
                         device: torch.device = None) -> tuple:
    """Unified masking logic for both training and validation. Returns: masked_seq, mask_seq, masked_struct, mask_struct"""
    if device is None:
        device = seq_tokens.device
    B = seq_tokens.shape[0]
    
    if masking_strategy == "simple":
        # Simple masking
        masked_seq, mask_seq = create_masked_inputs(seq_tokens, mask_prob_seq, mask_token_id=AMINO_ACID_VOCAB['<MASK>'])
        masked_struct, mask_struct = create_masked_inputs(struct_tokens, mask_prob_struct, mask_token_id=STRUCTURE_SPECIAL_TOKENS['<STRUCT_MASK>'])
        
        # Apply special token protection
        mask_seq = apply_special_token_masks(mask_seq, original_lengths)
        mask_struct = apply_special_token_masks(mask_struct, original_lengths)
        
        # Re-apply masks after protection
        masked_seq = seq_tokens.clone()
        masked_seq[mask_seq] = AMINO_ACID_VOCAB['<MASK>']
        masked_struct = struct_tokens.clone()
        masked_struct[mask_struct] = STRUCTURE_SPECIAL_TOKENS['<STRUCT_MASK>']
        
    elif masking_strategy == "complex":
        # Complex masking with noise schedules
        seq_mask_rates = sample_betalinear30(B, device)
        masked_seq, mask_seq = create_masked_inputs_complex(seq_tokens, seq_mask_rates, AMINO_ACID_VOCAB['<MASK>'])
        
        struct_mask_rates = sample_cosine(B, device)
        masked_struct, mask_struct = create_masked_structure_tokens_complex(struct_tokens, struct_mask_rates, STRUCTURE_SPECIAL_TOKENS['<STRUCT_MASK>'])
        
        # Apply special token protection
        mask_seq = apply_special_token_masks(mask_seq, original_lengths)
        mask_struct = apply_special_token_masks(mask_struct, original_lengths)
        
        # Re-apply masks after protection
        masked_seq = seq_tokens.clone()
        masked_seq[mask_seq] = AMINO_ACID_VOCAB['<MASK>']
        masked_struct = struct_tokens.clone()
        masked_struct[mask_struct] = STRUCTURE_SPECIAL_TOKENS['<STRUCT_MASK>']
        
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")
    
    return masked_seq, mask_seq, masked_struct, mask_struct

def create_masked_inputs(seq_tokens: torch.Tensor, mask_prob: Optional[float] = None, mask: Optional[torch.Tensor] = None, 
                        mask_token_id: int = AMINO_ACID_VOCAB['<MASK>']):
    """Create masked sequence for MLM training (simple strategy)."""
    assert not (mask is None and mask_prob is None)

    if mask is None:
        mask = torch.rand_like(seq_tokens.float()) < mask_prob

    masked_tokens = seq_tokens.clone()
    masked_tokens[mask] = mask_token_id  # Use specified mask token ID
    return masked_tokens, mask

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Calculate accuracy for masked positions only."""
    if not mask.any():
        return 0.0
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()

def main():
    # Initialize distributed training
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    # Initialize configurations
    model_cfg, train_cfg = ModelConfig(), TrainingConfig()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Create directories (only on rank 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        train_cfg.checkpoint_dir = f"{train_cfg.checkpoint_dir}/{timestamp}-transformer-mlm"
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
        
        # Initialize W&B (only on rank 0)
        if train_cfg.use_wandb:
            run_name = train_cfg.wandb_run_name or f"transformer-{model_cfg.first_layer}-{timestamp}"
            
            # Set attention type tag based on first layer
            attention_tags = []
            if model_cfg.first_layer == "SA":
                attention_tags.append("Self-Attention")
            elif model_cfg.first_layer == "GA":
                attention_tags.append("Geometric-Attention")
            elif model_cfg.first_layer == "RA":
                attention_tags.append("Reflexive-Attention")
            
            wandb.init(
                project=train_cfg.wandb_project,
                name=run_name,
                tags=attention_tags,
                config={
                    # Model config
                    "model_type": "transformer",
                    "first_layer": model_cfg.first_layer,
                    "d_model": model_cfg.d_model,
                    "n_heads": model_cfg.n_heads,
                    "n_layers": model_cfg.n_layers,
                    "seq_vocab": model_cfg.seq_vocab,
                    "struct_vocab": model_cfg.struct_vocab,
                    "max_len": model_cfg.max_len,
                    "dropout": model_cfg.dropout,
                    "ff_mult": model_cfg.ff_mult,
                    
                    # Training config
                    "batch_size": train_cfg.batch_size,
                    "max_epochs": train_cfg.max_epochs,
                    "learning_rate": train_cfg.learning_rate,
                    "masking_strategy": train_cfg.masking_strategy,
                    "mask_prob_seq": train_cfg.mask_prob_seq,
                    "mask_prob_struct": train_cfg.mask_prob_struct,
                    "seq_loss_weight": train_cfg.seq_loss_weight,
                    "struct_loss_weight": train_cfg.struct_loss_weight,
                    
                    # System config
                    "device": str(device),
                    "world_size": world_size,
                    "data_source": train_cfg.csv_file,
                }
            )
    
    # Load FSQ encoder (frozen)
    fsq_encoder = FSQEncoder()
    fsq_encoder.load_state_dict(torch.load("../trained_fsq/encoder_epoch_470.pt", map_location=device))
    fsq_encoder.eval()
    fsq_encoder.requires_grad_(False)
    fsq_encoder = fsq_encoder.to(device)
    
    # Initialize transformer with model config
    model = TransformerTrunk(model_cfg).to(device)
    
    # Wrap model with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Print parameter count (only on rank 0)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Using masking strategy: {train_cfg.masking_strategy}")
        print(f"Using first layer type: {model_cfg.first_layer}")
        print(f"World size: {world_size}")
        
        # Log model to W&B
        if train_cfg.use_wandb:
            wandb.watch(model, log="all", log_freq=100)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate)
    
    # -------------------- Data loading -------------------- #
    dataset = ProteinBackboneDataset(
        train_cfg.data_dir,
        use_cbeta=(model_cfg.first_layer == "RA"),
        max_length=model_cfg.max_len
    )
    val_size = max(1, int(0.1 * len(dataset)))  # Split into train / val (90/10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Create distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders with appropriate samplers
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        collate_fn=collate_batch, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=train_cfg.batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        collate_fn=collate_batch, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize tracking lists
    train_loss_history = []
    val_loss_history = []
    train_seq_acc_history = []
    train_struct_acc_history = []
    val_seq_acc_history = []
    val_struct_acc_history = []
    
    # -------------------- Training loop -------------------- #
    for epoch in range(train_cfg.max_epochs):
        # Set epoch for distributed sampler
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0.0
        total_seq_loss = 0.0
        total_struct_loss = 0.0
        total_seq_acc = 0.0
        total_struct_acc = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move everything to device
            coords, mask, seq_tokens = batch[1].to(device), batch[2].to(device), batch[0].to(device)
            B = seq_tokens.shape[0]
        
            # Get structure embeddings from frozen FSQ encoder
            with torch.no_grad():
                # FSQ encoder expects [B, L, 3, 3] - only use N, CA, C atoms
                coords_for_fsq = coords[:, :, :3, :] if coords.shape[2] == 4 else coords
                struct_tokens = fsq_encoder.encode_to_tokens(coords_for_fsq).squeeze(-1)  # Remove last dim [32, 2048, 1] -> [32, 2048]
                original_lengths = mask.sum(dim=1)  # Store original lengths before adding special tokens
                struct_tokens[~mask] = STRUCTURE_SPECIAL_TOKENS['<STRUCT_PAD>']  # Replace padding positions with STRUCT_PAD token
                
                # Add BOS/EOS tokens to structure sequences
                struct_tokens_np = struct_tokens.cpu().numpy()
                struct_tokens_with_special = []
                for i in range(B):
                    seq_len = original_lengths[i].item()  # Only pass the valid tokens (not padding) to add_structure_special_tokens
                    valid_tokens = struct_tokens_np[i, :seq_len] if seq_len > 0 else np.array([], dtype=np.int64)
                    tokens_with_special = add_structure_special_tokens(valid_tokens, seq_tokens.shape[1])  # Use same length as sequence
                    struct_tokens_with_special.append(tokens_with_special)
                struct_tokens = torch.tensor(np.array(struct_tokens_with_special), device=device)
                
                # Pad coordinates to account for BOS token at position 0
                # Original coords[i] should now be at position i+1 to align with shifted tokens
                if model_cfg.first_layer == "RA":
                    # For ReflexiveAttention, coords shape is [B, L, 4, 3]
                    coords_padded = torch.zeros(B, seq_tokens.shape[1], 4, 3, device=device)
                    coords_padded[:, 1:, :, :] = coords[:, :seq_tokens.shape[1]-1, :, :]  # Shift coordinates right by 1
                else:
                    # For GeometricAttention, coords shape is [B, L, 3, 3]
                    coords_padded = torch.zeros(B, seq_tokens.shape[1], 3, 3, device=device)
                    coords_padded[:, 1:, :, :] = coords[:, :seq_tokens.shape[1]-1, :, :]  # Shift coordinates right by 1
                coords = coords_padded

            masked_seq, mask_seq, masked_struct, mask_struct = process_batch_masking(
                seq_tokens, struct_tokens, original_lengths, train_cfg.masking_strategy, train_cfg.mask_prob_seq, train_cfg.mask_prob_struct, device
            )

            # Create coordinate mask that properly handles BOS/EOS tokens
            # BOS is at position 0, original tokens at 1 to original_length, EOS at original_length+1
            coord_mask = create_coordinate_mask(original_lengths, seq_tokens.shape[1], device)
            
            inputs = (masked_seq, masked_struct)  # Prepare model input as tuple (seq, struct)
            
            # Forward pass through model returns tuple
            if model_cfg.first_layer in ("GA", "RA"):
                outputs = model(inputs, coords, coord_mask)
            else:
                outputs = model(inputs)

            seq_logits, struct_logits = outputs

            # --- MLM loss --------------------------------------------------
            # Flatten for easy masking
            seq_logits_flat, struct_logits_flat = seq_logits.view(-1, model_cfg.seq_vocab), struct_logits.view(-1, model_cfg.struct_vocab)
            seq_labels_flat, struct_labels_flat = seq_tokens.view(-1), struct_tokens.view(-1)
            seq_mask_flat, struct_mask_flat = mask_seq.view(-1), mask_struct.view(-1)

            # Compute losses only if there are masked tokens
            loss_seq = F.cross_entropy(seq_logits_flat[seq_mask_flat], seq_labels_flat[seq_mask_flat].long()) if seq_mask_flat.any() else torch.tensor(0.0, device=device)
            loss_struct = F.cross_entropy(struct_logits_flat[struct_mask_flat], struct_labels_flat[struct_mask_flat].long()) if struct_mask_flat.any() else torch.tensor(0.0, device=device)

            loss = train_cfg.seq_loss_weight * loss_seq + train_cfg.struct_loss_weight * loss_struct

            # Calculate accuracies
            seq_acc = calculate_accuracy(seq_logits_flat, seq_labels_flat, seq_mask_flat)
            struct_acc = calculate_accuracy(struct_logits_flat, struct_labels_flat, struct_mask_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_seq_loss += loss_seq.item()
            total_struct_loss += loss_struct.item()
            total_seq_acc += seq_acc
            total_struct_acc += struct_acc
            num_batches += 1
            
            # Log batch metrics to W&B (only on rank 0)
            # Commented out to track only by epochs instead of steps
            # if rank == 0 and train_cfg.use_wandb and batch_idx % 10 == 0:
            #     wandb.log({
            #         "batch/train_loss": loss.item(),
            #         "batch/seq_loss": loss_seq.item(),
            #         "batch/struct_loss": loss_struct.item(),
            #         "batch/seq_accuracy": seq_acc,
            #         "batch/struct_accuracy": struct_acc,
            #         "batch/learning_rate": optimizer.param_groups[0]["lr"],
            #     }, step=epoch * len(train_loader) + batch_idx)
            
            # Print progress every 10 batches (only on rank 0)
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{train_cfg.max_epochs} - Batch {batch_idx}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Seq Acc: {seq_acc:.4f} - Struct Acc: {struct_acc:.4f}")
                  
        # Calculate average training loss
        train_loss_epoch = total_train_loss / num_batches
        train_seq_loss_epoch = total_seq_loss / num_batches
        train_struct_loss_epoch = total_struct_loss / num_batches
        train_seq_acc_epoch = total_seq_acc / num_batches
        train_struct_acc_epoch = total_struct_acc / num_batches
        
        # Aggregate loss across all processes
        if world_size > 1:
            metrics_tensor = torch.tensor([train_loss_epoch, train_seq_loss_epoch, train_struct_loss_epoch, 
                                         train_seq_acc_epoch, train_struct_acc_epoch], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
            train_loss_epoch, train_seq_loss_epoch, train_struct_loss_epoch, train_seq_acc_epoch, train_struct_acc_epoch = metrics_tensor.tolist()
        
        train_loss_history.append(train_loss_epoch)
        train_seq_acc_history.append(train_seq_acc_epoch)
        train_struct_acc_history.append(train_struct_acc_epoch)
        
        # -------------------- Validation -------------------- #
        model.eval()
        val_loss_total = 0.0
        val_seq_loss_total = 0.0
        val_struct_loss_total = 0.0
        val_seq_acc_total = 0.0
        val_struct_acc_total = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                coords, mask, seq_tokens = batch[1].to(device), batch[2].to(device), batch[0].to(device)
                B = seq_tokens.shape[0]
            
                # Get structure embeddings from frozen FSQ encoder
                # FSQ encoder expects [B, L, 3, 3] - only use N, CA, C atoms
                coords_for_fsq = coords[:, :, :3, :] if coords.shape[2] == 4 else coords
                struct_tokens = fsq_encoder.encode_to_tokens(coords_for_fsq).squeeze(-1)  # Remove last dim [32, 2048, 1] -> [32, 2048]
                original_lengths = mask.sum(dim=1)  # Store original lengths before adding special tokens
                struct_tokens[~mask] = STRUCTURE_SPECIAL_TOKENS['<STRUCT_PAD>']  # Replace padding positions with STRUCT_PAD token
                
                # Add BOS/EOS tokens to structure sequences
                struct_tokens_np = struct_tokens.cpu().numpy()
                struct_tokens_with_special = []
                for i in range(B):
                    seq_len = original_lengths[i].item()  # Only pass the valid tokens (not padding) to add_structure_special_tokens
                    valid_tokens = struct_tokens_np[i, :seq_len] if seq_len > 0 else np.array([], dtype=np.int64)
                    tokens_with_special = add_structure_special_tokens(valid_tokens, seq_tokens.shape[1])  # Use same length as sequence
                    struct_tokens_with_special.append(tokens_with_special)
                struct_tokens = torch.tensor(np.array(struct_tokens_with_special), device=device)
                
                # Pad coordinates to account for BOS token at position 0
                # Original coords[i] should now be at position i+1 to align with shifted tokens
                if model_cfg.first_layer == "RA":
                    # For ReflexiveAttention, coords shape is [B, L, 4, 3]
                    coords_padded = torch.zeros(B, seq_tokens.shape[1], 4, 3, device=device)
                    coords_padded[:, 1:, :, :] = coords[:, :seq_tokens.shape[1]-1, :, :]  # Shift coordinates right by 1
                else:
                    # For GeometricAttention, coords shape is [B, L, 3, 3]
                    coords_padded = torch.zeros(B, seq_tokens.shape[1], 3, 3, device=device)
                    coords_padded[:, 1:, :, :] = coords[:, :seq_tokens.shape[1]-1, :, :]  # Shift coordinates right by 1
                coords = coords_padded

                masked_seq, mask_seq, masked_struct, mask_struct = process_batch_masking(
                    seq_tokens, struct_tokens, original_lengths, train_cfg.masking_strategy, train_cfg.mask_prob_seq, train_cfg.mask_prob_struct, device
                )

                # Create coordinate mask that properly handles BOS/EOS tokens
                # BOS is at position 0, original tokens at 1 to original_length, EOS at original_length+1
                coord_mask = create_coordinate_mask(original_lengths, seq_tokens.shape[1], device)
                
                inputs = (masked_seq, masked_struct)  # Prepare model input as tuple (seq, struct)
                
                # Forward pass through model returns tuple
                if model_cfg.first_layer in ("GA", "RA"):
                    outputs = model(inputs, coords, coord_mask)
                else:
                    outputs = model(inputs)

                seq_logits, struct_logits = outputs

                # Flatten
                seq_logits_flat, struct_logits_flat = seq_logits.view(-1, model_cfg.seq_vocab), struct_logits.view(-1, model_cfg.struct_vocab)
                seq_labels_flat, struct_labels_flat = seq_tokens.view(-1), struct_tokens.view(-1)
                seq_mask_flat, struct_mask_flat = mask_seq.view(-1), mask_struct.view(-1)

                # Compute losses only if there are masked tokens
                val_loss_seq = F.cross_entropy(seq_logits_flat[seq_mask_flat], seq_labels_flat[seq_mask_flat].long()) if seq_mask_flat.any() else torch.tensor(0.0, device=device)
                val_loss_struct = F.cross_entropy(struct_logits_flat[struct_mask_flat], struct_labels_flat[struct_mask_flat].long()) if struct_mask_flat.any() else torch.tensor(0.0, device=device)
                
                val_loss = train_cfg.seq_loss_weight * val_loss_seq + train_cfg.struct_loss_weight * val_loss_struct
                val_seq_acc = calculate_accuracy(seq_logits_flat, seq_labels_flat, seq_mask_flat)
                val_struct_acc = calculate_accuracy(struct_logits_flat, struct_labels_flat, struct_mask_flat)
                
                val_loss_total += val_loss.item()
                val_seq_loss_total += val_loss_seq.item()
                val_struct_loss_total += val_loss_struct.item()
                val_seq_acc_total += val_seq_acc
                val_struct_acc_total += val_struct_acc
                val_num_batches += 1

        # Calculate average validation loss
        val_loss_epoch = val_loss_total / val_num_batches
        val_seq_loss_epoch = val_seq_loss_total / val_num_batches
        val_struct_loss_epoch = val_struct_loss_total / val_num_batches
        val_seq_acc_epoch = val_seq_acc_total / val_num_batches
        val_struct_acc_epoch = val_struct_acc_total / val_num_batches
        
        # Aggregate validation loss across all processes
        if world_size > 1:
            val_metrics_tensor = torch.tensor([val_loss_epoch, val_seq_loss_epoch, val_struct_loss_epoch,
                                             val_seq_acc_epoch, val_struct_acc_epoch], device=device)
            dist.all_reduce(val_metrics_tensor, op=dist.ReduceOp.AVG)
            val_loss_epoch, val_seq_loss_epoch, val_struct_loss_epoch, val_seq_acc_epoch, val_struct_acc_epoch = val_metrics_tensor.tolist()
        
        val_loss_history.append(val_loss_epoch)
        val_seq_acc_history.append(val_seq_acc_epoch)
        val_struct_acc_history.append(val_struct_acc_epoch)
        
        # Log epoch metrics to W&B (only on rank 0)
        if rank == 0 and train_cfg.use_wandb:
            wandb.log({
                "epoch/train_loss": train_loss_epoch,
                "epoch/train_seq_loss": train_seq_loss_epoch,
                "epoch/train_struct_loss": train_struct_loss_epoch,
                "epoch/train_seq_accuracy": train_seq_acc_epoch,
                "epoch/train_struct_accuracy": train_struct_acc_epoch,
                "epoch/val_loss": val_loss_epoch,
                "epoch/val_seq_loss": val_seq_loss_epoch,
                "epoch/val_struct_loss": val_struct_loss_epoch,
                "epoch/val_seq_accuracy": val_seq_acc_epoch,
                "epoch/val_struct_accuracy": val_struct_acc_epoch,
                "epoch/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }, step=epoch + 1)
        
        # Print epoch summary (only on rank 0)
        if rank == 0:
            print(f"Epoch {epoch+1}/{train_cfg.max_epochs} -- "
                  f"Train Loss: {train_loss_epoch:.4f} (Seq: {train_seq_loss_epoch:.4f}, Struct: {train_struct_loss_epoch:.4f}) | "
                  f"Val Loss: {val_loss_epoch:.4f} (Seq: {val_seq_loss_epoch:.4f}, Struct: {val_struct_loss_epoch:.4f})")
            print(f"  Train Acc - Seq: {train_seq_acc_epoch:.4f}, Struct: {train_struct_acc_epoch:.4f} | "
                  f"Val Acc - Seq: {val_seq_acc_epoch:.4f}, Struct: {val_struct_acc_epoch:.4f}")
        
        # Save checkpoint (only on rank 0)
        if rank == 0 and (epoch + 1) % train_cfg.save_every == 0:
            checkpoint_path = Path(train_cfg.checkpoint_dir) / f"checkpoint_{epoch+1}.pt"
            # Handle DDP state dict
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_epoch,
                'val_loss': val_loss_epoch,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'train_seq_acc_history': train_seq_acc_history,
                'train_struct_acc_history': train_struct_acc_history,
                'val_seq_acc_history': val_seq_acc_history,
                'val_struct_acc_history': val_struct_acc_history,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Log checkpoint to W&B
            if train_cfg.use_wandb:
                wandb.save(str(checkpoint_path))
    
    # Final checkpoint and plotting (only on rank 0)
    if rank == 0:
        # Save final checkpoint
        final_checkpoint_path = Path(train_cfg.checkpoint_dir) / "checkpoint_final.pt"
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save({
            'epoch': train_cfg.max_epochs,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_epoch,
            'val_loss': val_loss_epoch,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_seq_acc_history': train_seq_acc_history,
            'train_struct_acc_history': train_struct_acc_history,
            'val_seq_acc_history': val_seq_acc_history,
            'val_struct_acc_history': val_struct_acc_history,
        }, final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")
        
        if train_cfg.use_wandb:
            wandb.save(str(final_checkpoint_path))
        
        # Plot training curves
        epochs = range(1, len(train_loss_history) + 1)
        
        # Loss curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_history, label='Train Loss', linewidth=2)
        plt.plot(epochs, val_loss_history, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_seq_acc_history, label='Train Seq Acc', linewidth=2)
        plt.plot(epochs, train_struct_acc_history, label='Train Struct Acc', linewidth=2)
        plt.plot(epochs, val_seq_acc_history, label='Val Seq Acc', linewidth=2, linestyle='--')
        plt.plot(epochs, val_struct_acc_history, label='Val Struct Acc', linewidth=2, linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Sequence and Structure Prediction Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(train_cfg.checkpoint_dir) / "training_curves.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved training curves to {plot_path}")
        
        if train_cfg.use_wandb:
            wandb.log({"training_curves": wandb.Image(str(plot_path))})
        
        # Log final summary statistics
        if train_cfg.use_wandb:
            wandb.run.summary["final_train_loss"] = train_loss_epoch
            wandb.run.summary["final_val_loss"] = val_loss_epoch
            wandb.run.summary["final_train_seq_acc"] = train_seq_acc_epoch
            wandb.run.summary["final_train_struct_acc"] = train_struct_acc_epoch
            wandb.run.summary["final_val_seq_acc"] = val_seq_acc_epoch
            wandb.run.summary["final_val_struct_acc"] = val_struct_acc_epoch
            wandb.run.summary["best_val_loss"] = min(val_loss_history)
            wandb.run.summary["best_val_seq_acc"] = max(val_seq_acc_history)
            wandb.run.summary["best_val_struct_acc"] = max(val_struct_acc_history)
            wandb.run.summary["total_params"] = sum(p.numel() for p in model.parameters())
            wandb.run.summary["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Finish W&B run
            wandb.finish()
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
