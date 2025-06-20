import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from Odyssey.src.data_util.tokenizer_bos_eos_pad import SequenceTokenizer, StructureTokenizer
import math

# --------------------------------------------------------------------------- #
#  Noise Schedules                                        #
# --------------------------------------------------------------------------- #
def get_noise_levels(s_min, s_max, T):
    """Generate instantaneous and cumulative noise levels for discrete diffusion.
    
    Args:
        s_min: Minimum noise level (sigma_min)
        s_max: Maximum noise level (sigma_max)
        T: Number of timesteps
        
    Returns:
        inst_noise_levels: Tensor of shape (T,) with instantaneous noise at each timestep
        cumulative_noise_levels: Tensor of shape (T,) with cumulative noise up to each timestep
    """
    t = torch.arange(T, dtype=torch.float32)
    # Geometric schedule for instantaneous noise
    inst_noise_levels = s_min**(1-t/T) * s_max**(t/T)
    
    # Cumulative noise: integral of instantaneous noise
    # For geometric schedule: ∫_0^t σ(s) ds
    cumulative_noise_levels = (math.log(s_max/s_min)/T) * (inst_noise_levels - inst_noise_levels[0])
    
    return inst_noise_levels, cumulative_noise_levels
def sample_betalinear30(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from betalinear30 distribution. 80% Beta(3,9), 20% Uniform(0,1), avg ~30%"""
    mask_rates = torch.zeros(batch_size, device=device)
    use_beta = torch.rand(batch_size) < 0.8  # Choose distribution for each batch element
    
    beta_samples = torch.distributions.Beta(3.0, 9.0).sample((batch_size,)).to(device)  # Beta(3, 9) samples
    uniform_samples = torch.rand(batch_size, device=device)  # Uniform(0, 1) samples
    mask_rates = torch.where(use_beta.to(device), beta_samples, uniform_samples)  # Combine based on use_beta
    
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values

def sample_cosine(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from cosine distribution using sin(u * π/2) for higher density near 1."""
    u = torch.rand(batch_size, device=device)  # Sample uniform values
    mask_rates = torch.sin(u * np.pi / 2)  # Apply sine transformation
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values

# --------------------------------------------------------------------------- #
#  Custom DataLoader for Discrete Diffusion                                   #
# --------------------------------------------------------------------------- #
class DiffusionDataLoader(DataLoader):
    """DataLoader that applies discrete diffusion noise process during batch collation."""
    
    def __init__(self, dataset, fsq_encoder, model_cfg, diffusion_cfg, device=None, **kwargs):
        self.fsq_encoder = fsq_encoder
        self.model_cfg = model_cfg
        self.diffusion_cfg = diffusion_cfg
        self.device = device if device is not None else torch.device("cpu")
        
        # Pre-compute noise levels
        self.inst_noise_levels, self.cumulative_noise_levels = get_noise_levels(
            diffusion_cfg.sigma_min, 
            diffusion_cfg.sigma_max, 
            diffusion_cfg.num_timesteps
        )
        # Move to device
        self.inst_noise_levels = self.inst_noise_levels.to(self.device)
        self.cumulative_noise_levels = self.cumulative_noise_levels.to(self.device)
        
        # Override collate_fn with our custom one
        kwargs['collate_fn'] = self._collate_with_diffusion
        super().__init__(dataset, **kwargs)

        # Initialize tokenizers
        self.sequence_tokenizer = SequenceTokenizer(self.model_cfg.max_len - 2)
        self.structure_tokenizer = StructureTokenizer(self.model_cfg.max_len - 2, self.fsq_encoder)
    
    def apply_discrete_noise(self, x_0, absorb_token, cumulative_noise_level):
        """Apply Q_absorb noise process to get x_t from x_0.
        
        Args:
            x_0: Original tokens of shape (B, L)
            absorb_token: Token index for absorbing state
            cumulative_noise_level: Cumulative noise level, shape (B, 1)
            
        Returns:
            x_t: Noisy tokens of shape (B, L)
            masks: Boolean mask of noised positions
        """
        B, L = x_0.shape
        device = x_0.device
        
        # Probability of transitioning to absorbing state
        # p(mask) = 1 - exp(-cumulative_noise_level)
        mask_prob = 1 - torch.exp(-cumulative_noise_level)  # [B, 1]
        mask_prob_expanded = mask_prob.expand(B, L)  # Sample masks for each position [B, L]
        masks = torch.rand(B, L, device=device) < mask_prob_expanded
        
        # Apply masks: replace with absorbing token where mask is True
        x_t = x_0.clone()
        x_t[masks] = absorb_token
        
        return x_t, masks
    
    def _collate_with_diffusion(self, batch):
        """Custom collate function that applies discrete diffusion noise."""
        # Unpack batch from ProteinBackboneDataset
        seq_list, coord_list, length_list = zip(*batch)
        
        # Tokenize sequences
        tokenized_seqs = []
        for seq in seq_list:
            tokens = self.sequence_tokenizer.tokenize(seq)
            tokenized_seqs.append(tokens)
        seq_tokens = torch.stack(tokenized_seqs, dim=0).to(self.device)
        
        # Tokenize structures
        tokenized_structs = []
        tokenized_coords = []
        for coords in coord_list:
            padded_coords, struct_tokens = self.structure_tokenizer.tokenize(coords)
            tokenized_coords.append(padded_coords)
            tokenized_structs.append(struct_tokens)
        struct_tokens = torch.stack(tokenized_structs, dim=0).to(self.device)
        coords = torch.stack(tokenized_coords, dim=0).to(self.device)
        
        # Stack lengths
        lengths = torch.tensor([l.item() for l in length_list], device=self.device)
        B = len(seq_list)
        
        # Sample timestep indices uniformly from [0, T-1]
        timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (B,), device=self.device)
        
        # Get corresponding noise levels
        inst_noise = self.inst_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        cumulative_noise = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
        
        # Apply discrete noise to get x_t from x_0
        seq_x_t, seq_masks = self.apply_discrete_noise(seq_tokens, self.diffusion_cfg.seq_absorb_token, cumulative_noise)
        struct_x_t, struct_masks = self.apply_discrete_noise(struct_tokens, self.diffusion_cfg.struct_absorb_token, cumulative_noise)
        
        # Zero out coordinates at masked structure positions
        coord_mask_expanded = struct_masks.unsqueeze(-1).unsqueeze(-1)
        coord_mask_expanded = coord_mask_expanded.expand(-1, -1, coords.shape[2], coords.shape[3])
        coords_noisy = coords.clone()
        coords_noisy[coord_mask_expanded] = 0.0
        
        # Add BOS/EOS tokens
        L = seq_tokens.shape[1]
        
        # Get special token IDs
        seq_bos = self.sequence_tokenizer.mapping['BOS']
        seq_eos = self.sequence_tokenizer.mapping['EOS']
        struct_bos = self.structure_tokenizer.special['BOS']
        struct_eos = self.structure_tokenizer.special['EOS']
        
        # Create tensors with BOS/EOS
        seq_x_0_with_special = torch.cat([
            torch.full((B, 1), seq_bos, dtype=torch.long, device=self.device),
            seq_tokens,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        seq_x_t_with_special = torch.cat([
            torch.full((B, 1), seq_bos, dtype=torch.long, device=self.device),
            seq_x_t,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        struct_x_0_with_special = torch.cat([
            torch.full((B, 1), struct_bos, dtype=torch.long, device=self.device),
            struct_tokens,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        struct_x_t_with_special = torch.cat([
            torch.full((B, 1), struct_bos, dtype=torch.long, device=self.device),
            struct_x_t,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        coords_with_special = torch.cat([
            torch.zeros((B, 1, coords_noisy.shape[2], coords_noisy.shape[3]), 
                       dtype=coords_noisy.dtype, device=self.device),
            coords_noisy,
            torch.zeros((B, 1, coords_noisy.shape[2], coords_noisy.shape[3]), 
                       dtype=coords_noisy.dtype, device=self.device)
        ], dim=1)
        
        # Create masks with BOS/EOS padding
        mask_seq_with_special = torch.cat([
            torch.zeros((B, 1), dtype=seq_masks.dtype, device=self.device),
            seq_masks,
            torch.zeros((B, 1), dtype=seq_masks.dtype, device=self.device)
        ], dim=1)
        
        mask_struct_with_special = torch.cat([
            torch.zeros((B, 1), dtype=struct_masks.dtype, device=self.device),
            struct_masks,
            torch.zeros((B, 1), dtype=struct_masks.dtype, device=self.device)
        ], dim=1)
        
        # Set EOS tokens at appropriate positions
        eos_positions = (lengths + 1).long()
        batch_indices = torch.arange(B, device=self.device)
        seq_x_0_with_special[batch_indices, eos_positions] = seq_eos
        seq_x_t_with_special[batch_indices, eos_positions] = seq_eos
        struct_x_0_with_special[batch_indices, eos_positions] = struct_eos
        struct_x_t_with_special[batch_indices, eos_positions] = struct_eos
        
        # Return: (x_t, x_0, masks, timesteps, cumulative_noise_levels, inst_noise_levels, lengths, coords)
        return (
            (seq_x_t_with_special, struct_x_t_with_special),  # x_t (noisy inputs)
            (seq_x_0_with_special, struct_x_0_with_special),  # x_0 (original tokens)
            (mask_seq_with_special, mask_struct_with_special),  # masks indicating noised positions
            timestep_indices,                                  # timestep indices
            cumulative_noise,                                  # cumulative noise levels
            inst_noise,                                        # instantaneous noise levels
            lengths,                                           # sequence lengths
            coords_with_special                                # coordinates
        )

# --------------------------------------------------------------------------- #
#  Custom DataLoader with Masking                                             #
# --------------------------------------------------------------------------- #
class MLMDataLoader(DataLoader):
    """DataLoader that applies masking strategy during batch collation."""
    
    def __init__(self, dataset, fsq_encoder, model_cfg, masking_strategy="simple", 
                 mask_prob_seq=0.15, mask_prob_struct=0.15, device=None, **kwargs):
        self.fsq_encoder = fsq_encoder
        self.model_cfg = model_cfg
        self.masking_strategy = masking_strategy
        self.mask_prob_seq = mask_prob_seq
        self.mask_prob_struct = mask_prob_struct
        self.device = device if device is not None else torch.device("cpu")
        
        # Override collate_fn with our custom one
        kwargs['collate_fn'] = self._collate_with_masking
        super().__init__(dataset, **kwargs)

        self.sequence_tokenizer = SequenceTokenizer(self.model_cfg.max_len - 2)  # Account for BOS/EOS
        self.structure_tokenizer = StructureTokenizer(self.model_cfg.max_len - 2, self.fsq_encoder)  # Account for BOS/EOS

    def create_masked_inputs_simple(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create masked tokens using simple masking strategy.
        
        Args:
            batch: Tuple of (seq_tokens, coords, struct_tokens)
            
        Returns:
            Tuple of (masked_seq_tokens, masked_coords, masked_struct_tokens, mask_seq, mask_struct)
        """
        seq_tokens, coords, struct_tokens = batch
        B, L = seq_tokens.shape
        
        # Create masks with fixed probabilities
        mask_seq = torch.rand(B, L, device=seq_tokens.device) < self.mask_prob_seq
        mask_struct = torch.rand(B, L, device=struct_tokens.device) < self.mask_prob_struct
        
        # Apply masks using the apply_mask method
        masked_seq_tokens, masked_coords, masked_struct_tokens = self.apply_mask(
            batch, mask_seq, mask_struct
        )
        
        return masked_seq_tokens, masked_coords, masked_struct_tokens, mask_seq, mask_struct
    
    def create_masked_inputs_complex(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create masked tokens using complex masking with variable mask rates.
        
        Samples different mask rates for each protein in the batch, separately for
        sequence and structure. Within each protein, masking is IID Bernoulli.
        
        Args:
            batch: Tuple of (seq_tokens, coords, struct_tokens)
            
        Returns:
            Tuple of (masked_seq_tokens, masked_coords, masked_struct_tokens, mask_seq, mask_struct)
        """
        seq_tokens, coords, struct_tokens = batch
        B, L = seq_tokens.shape
        
        # Sample mask rates for each protein in batch
        seq_mask_rates = sample_betalinear30(B, seq_tokens.device)  # [B]
        struct_mask_rates = sample_cosine(B, struct_tokens.device)  # [B]
        
        # Create masks with per-protein rates
        # Expand rates to [B, L] and sample IID Bernoulli
        seq_mask_rates_expanded = seq_mask_rates.unsqueeze(1).expand(B, L)
        struct_mask_rates_expanded = struct_mask_rates.unsqueeze(1).expand(B, L)
        
        mask_seq = torch.rand(B, L, device=seq_tokens.device) < seq_mask_rates_expanded
        mask_struct = torch.rand(B, L, device=struct_tokens.device) < struct_mask_rates_expanded
        
        # Apply masks using the apply_mask method
        masked_seq_tokens, masked_coords, masked_struct_tokens = self.apply_mask(
            batch, mask_seq, mask_struct
        )
        
        return masked_seq_tokens, masked_coords, masked_struct_tokens, mask_seq, mask_struct

    def apply_mask(self, batch, seq_mask, struct_mask):
        """
        Given a boolean sequence mask and a boolean structure mask,
          apply the mask as appropriate WITHOUT overwriting padding tokens.
        """
        # Unpack batch
        seq_tokens, coords, struct_tokens = batch
        
        # Get mask and pad tokens
        seq_mask_token = self.sequence_tokenizer.mapping['MASK']
        seq_pad_token = self.sequence_tokenizer.mapping['PAD']
        
        struct_mask_token = self.structure_tokenizer.special['MASK']
        struct_pad_token = self.structure_tokenizer.special['PAD']
        
        # Identify non-padding positions
        seq_not_pad = (seq_tokens != seq_pad_token)
        struct_not_pad = (struct_tokens != struct_pad_token)
        
        # Apply masks only to non-padding positions
        seq_mask_final = seq_mask & seq_not_pad
        struct_mask_final = struct_mask & struct_not_pad
        
        # Clone tensors to avoid modifying originals
        masked_seq_tokens = seq_tokens.clone()
        masked_struct_tokens = struct_tokens.clone()
        masked_coords = coords.clone()
        
        # Apply masking:
        # 1. Sequence tokens: replace with MASK token where seq_mask is True (excluding pads)
        masked_seq_tokens[seq_mask_final] = seq_mask_token
        
        # 2. Structure tokens: replace with MASK token where struct_mask is True (excluding pads)
        masked_struct_tokens[struct_mask_final] = struct_mask_token

        # 3. Coordinates: zero out where struct_mask is True
        # Note: We use the original struct_mask here, not struct_mask_final,
        # because coordinates don't have a separate padding token
        coord_mask_expanded = struct_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        coord_mask_expanded = coord_mask_expanded.expand(-1, -1, coords.shape[2], coords.shape[3])  # [B, L, H, 3]
        masked_coords[coord_mask_expanded] = 0.0
        
        return masked_seq_tokens, masked_coords, masked_struct_tokens

    def _collate_with_masking(self, batch):
        """Custom collate function that applies masking.
        
        Workflow:
        1. Receive batch from ProteinBackboneDataset
        2. Tokenize sequences and structures
        3. Apply masking based on strategy
        4. Return masked data (only mask and pad special tokens, no BOS/EOS)
        """
        # Unpack batch from ProteinBackboneDataset
        # Dataset returns (seq, coords, length) for each protein
        seq_list, coord_list, length_list = zip(*batch)
        
        # Tokenize sequences
        tokenized_seqs = []
        for seq in seq_list:
            tokens = self.sequence_tokenizer.tokenize(seq)
            tokenized_seqs.append(tokens)
        seq_tokens = torch.stack(tokenized_seqs, dim=0).to(self.device)
        
        # Tokenize structures (FSQ encoding happens inside tokenizer)
        # StructureTokenizer returns (padded_coords, padded_struct_tokens)
        tokenized_structs = []
        tokenized_coords = []
        for coords in coord_list:
            padded_coords, struct_tokens = self.structure_tokenizer.tokenize(coords)
            tokenized_coords.append(padded_coords)
            tokenized_structs.append(struct_tokens)
        struct_tokens = torch.stack(tokenized_structs, dim=0).to(self.device)
        coords = torch.stack(tokenized_coords, dim=0).to(self.device)
        
        # Stack lengths
        lengths = torch.tensor([l.item() for l in length_list], device=self.device)
        
        # Create batch tuple for masking
        batch_tuple = (seq_tokens, coords, struct_tokens)
        
        # Apply masking based on strategy
        if self.masking_strategy == "simple":
            masked_seq, masked_coords, masked_struct, mask_seq, mask_struct = self.create_masked_inputs_simple(batch_tuple)
        elif self.masking_strategy == "complex":
            masked_seq, masked_coords, masked_struct, mask_seq, mask_struct = self.create_masked_inputs_complex(batch_tuple)
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
        
        # Add BOS/EOS tokens
        B, L = masked_seq.shape
        
        # Get special token IDs
        seq_bos = self.sequence_tokenizer.mapping['BOS']
        seq_eos = self.sequence_tokenizer.mapping['EOS']
        struct_bos = self.structure_tokenizer.special['BOS']
        struct_eos = self.structure_tokenizer.special['EOS']
        
        # Create new tensors with BOS prepended and space for EOS
        masked_seq_with_special = torch.cat([
            torch.full((B, 1), seq_bos, dtype=torch.long, device=self.device),
            masked_seq,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        masked_struct_with_special = torch.cat([
            torch.full((B, 1), struct_bos, dtype=torch.long, device=self.device),
            masked_struct,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        masked_coords_with_special = torch.cat([
            torch.zeros((B, 1, masked_coords.shape[2], masked_coords.shape[3]), dtype=masked_coords.dtype, device=self.device),
            masked_coords,
            torch.zeros((B, 1, masked_coords.shape[2], masked_coords.shape[3]), dtype=masked_coords.dtype, device=self.device)
        ], dim=1)
        
        # Shift masks to align with BOS/EOS shifted data
        mask_seq_with_special = torch.cat([
            torch.zeros((B, 1), dtype=mask_seq.dtype, device=self.device),
            mask_seq,
            torch.zeros((B, 1), dtype=mask_seq.dtype, device=self.device)
        ], dim=1)
        
        mask_struct_with_special = torch.cat([
            torch.zeros((B, 1), dtype=mask_struct.dtype, device=self.device),
            mask_struct,
            torch.zeros((B, 1), dtype=mask_struct.dtype, device=self.device)
        ], dim=1)
        
        # Set EOS tokens at the appropriate positions (lengths + 1 for BOS offset)
        eos_positions = (lengths + 1).long()
        batch_indices = torch.arange(B, device=self.device)
        masked_seq_with_special[batch_indices, eos_positions] = seq_eos
        masked_struct_with_special[batch_indices, eos_positions] = struct_eos
        
        # Create original tokens with BOS/EOS for loss computation
        seq_tokens_with_special = torch.cat([
            torch.full((B, 1), seq_bos, dtype=torch.long, device=self.device),
            seq_tokens,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        struct_tokens_with_special = torch.cat([
            torch.full((B, 1), struct_bos, dtype=torch.long, device=self.device),
            struct_tokens,
            torch.zeros((B, 1), dtype=torch.long, device=self.device)
        ], dim=1)
        
        # Set EOS tokens for original tokens
        seq_tokens_with_special[batch_indices, eos_positions] = seq_eos
        struct_tokens_with_special[batch_indices, eos_positions] = struct_eos
        
        # Return as tuple: (inputs, masks, lengths, original_tokens)
        # inputs: (masked_seq, masked_struct, masked_coords)
        # masks: (mask_seq, mask_struct)
        # lengths: number of residues per protein
        # original_tokens: (seq_tokens, struct_tokens) for loss computation
        return (
            (masked_seq_with_special, masked_struct_with_special, masked_coords_with_special),  # inputs
            (mask_seq_with_special, mask_struct_with_special),                                   # masks
            lengths,                                                                              # lengths
            (seq_tokens_with_special, struct_tokens_with_special)  # original tokens for loss computation
        )