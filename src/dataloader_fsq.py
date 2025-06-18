import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from src.data_util.tokenization_padding import SequenceTokenizer
import math

# --------------------------------------------------------------------------- #
#  Noise Schedules for complex masking                                        #
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
    
    def __init__(self, dataset, model_cfg, diffusion_cfg, device=None, **kwargs):
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
    
    def apply_discrete_noise(self, coords, cumulative_noise_level):
        """Apply discrete noise to coordinates by zeroing out noised positions.
        
        Args:
            coords: Original coordinates of shape (B, L, H, 3)
            cumulative_noise_level: Cumulative noise level, shape (B, 1)
            
        Returns:
            noisy_coords: Coordinates with noised positions zeroed out
            masks: Boolean mask of noised positions (B, L)
        """
        B, L, H, _ = coords.shape
        device = coords.device
        
        # Probability of transitioning to absorbing state
        # p(mask) = 1 - exp(-cumulative_noise_level)
        mask_prob = 1 - torch.exp(-cumulative_noise_level)  # [B, 1]
        mask_prob_expanded = mask_prob.expand(B, L)  # [B, L]
        masks = torch.rand(B, L, device=device) < mask_prob_expanded
        
        # Zero out coordinates at masked positions
        noisy_coords = coords.clone()
        mask_expanded = masks.unsqueeze(-1).unsqueeze(-1).expand(B, L, H, 3)
        noisy_coords[mask_expanded] = 0.0
        
        return noisy_coords, masks
    
    def _collate_with_diffusion(self, batch):
        """Custom collate function that applies discrete diffusion noise."""
        # Unpack batch from ProteinBackboneDataset
        seq_list, coord_list, length_list = zip(*batch)
        
        # Stack lengths
        lengths = torch.tensor([l.item() for l in length_list], device=self.device)
        B = len(seq_list)
        
        if self.model_cfg.stage == "stage_1":
            # Stage 1: Only work with coordinates, apply discrete noise
            tokenized_coords = []
            for coords in coord_list:
                M = coords.shape[0]
                H = coords.shape[1]
                # Zero-pad coords
                num_pad = (self.model_cfg.max_len - 2) - M
                if num_pad > 0:
                    padded_coords = torch.cat([coords, torch.zeros(num_pad, H, 3)], dim=0)
                else:
                    padded_coords = coords[:self.model_cfg.max_len - 2]
                tokenized_coords.append(padded_coords)
            coords = torch.stack(tokenized_coords, dim=0).to(self.device)
            
            # Sample timestep indices uniformly from [0, T-1]
            timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (B,), device=self.device)
            # Get corresponding noise levels
            cumulative_noise = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
            
            # Apply discrete noise to coordinates
            noisy_coords, noise_masks = self.apply_discrete_noise(coords, cumulative_noise)
            
            # Add BOS/EOS padding to coordinates
            noisy_coords_with_special = torch.cat([
                torch.zeros((B, 1, noisy_coords.shape[2], noisy_coords.shape[3]), 
                           dtype=noisy_coords.dtype, device=self.device),
                noisy_coords,
                torch.zeros((B, 1, noisy_coords.shape[2], noisy_coords.shape[3]), 
                           dtype=noisy_coords.dtype, device=self.device)
            ], dim=1)
            
            # Also need to adjust the mask to account for BOS/EOS
            noise_masks_with_special = torch.cat([
                torch.zeros((B, 1), dtype=noise_masks.dtype, device=self.device),
                noise_masks,
                torch.zeros((B, 1), dtype=noise_masks.dtype, device=self.device)
            ], dim=1)
            
            # Return for stage_1: (noisy_coords, noise_masks, lengths, timesteps, noise_levels)
            return (noisy_coords_with_special, noise_masks_with_special, lengths)
        
        elif self.model_cfg.stage == "stage_2":
            # Stage 2: Tokenize sequences, return raw coordinates (no structure tokenization)            
            # Tokenize sequences
            tokenized_seqs = []
            for seq in seq_list:
                tokens = self.sequence_tokenizer.tokenize(seq)
                tokenized_seqs.append(tokens)
            seq_tokens = torch.stack(tokenized_seqs, dim=0).to(self.device)
            
            # Pad coordinates (no tokenization needed - encoder will handle structure encoding)
            tokenized_coords = []
            for coords in coord_list:
                M = coords.shape[0]
                H = coords.shape[1]
                # Zero-pad coords
                num_pad = (self.model_cfg.max_len - 2) - M
                if num_pad > 0:
                    padded_coords = torch.cat([coords, torch.zeros(num_pad, H, 3)], dim=0)
                else:
                    padded_coords = coords[:self.model_cfg.max_len - 2]
                tokenized_coords.append(padded_coords)
            coords = torch.stack(tokenized_coords, dim=0).to(self.device)
            
            # Sample timestep indices uniformly from [0, T-1]
            timestep_indices = torch.randint(0, self.diffusion_cfg.num_timesteps, (B,), device=self.device)
            # Get corresponding noise levels
            cumulative_noise = self.cumulative_noise_levels[timestep_indices].unsqueeze(1)  # [B, 1]
            
            # Get special token IDs for sequences
            seq_bos = self.sequence_tokenizer.mapping['BOS']
            seq_eos = self.sequence_tokenizer.mapping['EOS']
            
            # Create tensors with BOS/EOS for sequences
            seq_tokens_with_special = torch.cat([
                torch.full((B, 1), seq_bos, dtype=torch.long, device=self.device),
                seq_tokens,
                torch.zeros((B, 1), dtype=torch.long, device=self.device)
            ], dim=1)
            
            # Add BOS/EOS padding to coordinates
            coords_with_special = torch.cat([
                torch.zeros((B, 1, coords.shape[2], coords.shape[3]), 
                           dtype=coords.dtype, device=self.device),
                coords,
                torch.zeros((B, 1, coords.shape[2], coords.shape[3]), 
                           dtype=coords.dtype, device=self.device)
            ], dim=1)
            
            # Set EOS tokens at appropriate positions
            eos_positions = (lengths + 1).long()
            batch_indices = torch.arange(B, device=self.device)
            seq_tokens_with_special[batch_indices, eos_positions] = seq_eos
            
            # Return for stage_2: (seq_tokens, coords, lengths)
            return (seq_tokens_with_special, coords_with_special, lengths)
        
        else:
            raise ValueError(f"Unknown stage: {self.model_cfg.stage}")

# --------------------------------------------------------------------------- #
#  Custom DataLoader with Masking                                             #
# --------------------------------------------------------------------------- #
class MLMDataLoader(DataLoader):
    """DataLoader that applies masking strategy during batch collation."""
    
    def __init__(self, dataset, model_cfg, masking_strategy="simple", mask_prob_struct=0.15, device=None, **kwargs):
        self.model_cfg = model_cfg
        self.masking_strategy = masking_strategy
        self.mask_prob_struct = mask_prob_struct
        self.device = device if device is not None else torch.device("cpu")
        
        # Override collate_fn with our custom one
        kwargs['collate_fn'] = self._collate_with_masking
        super().__init__(dataset, **kwargs)

        self.sequence_tokenizer = SequenceTokenizer(self.model_cfg.max_len - 2)  # Account for BOS/EOS

    def create_masked_inputs_simple(self, batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked tokens using simple masking strategy.
        
        Args:
            batch: Tuple of (coords,) for stage_1
            
        Returns:
            Tuple of (masked_coords, mask_struct)
        """
        coords = batch[0] if isinstance(batch, tuple) else batch
        B, L, H, _ = coords.shape
        
        # Create masks with fixed probabilities
        mask_struct = torch.rand(B, L, device=coords.device) < self.mask_prob_struct
        
        # Apply masks by zeroing out coordinates
        masked_coords = coords.clone()
        # Expand mask to match coordinate dimensions [B, L] -> [B, L, H, 3]
        mask_expanded = mask_struct.unsqueeze(-1).unsqueeze(-1).expand(B, L, H, 3)
        masked_coords[mask_expanded] = 0.0
        
        return masked_coords, mask_struct
    
    def create_masked_inputs_complex(self, batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked tokens using complex masking with variable mask rates.
        
        Samples different mask rates for each protein in the batch for structure.
        Within each protein, masking is IID Bernoulli.
        
        Args:
            batch: Tuple of (coords,) for stage_1
            
        Returns:
            Tuple of (masked_coords, mask_struct)
        """
        coords = batch[0] if isinstance(batch, tuple) else batch
        B, L, H, _ = coords.shape
        
        # Sample mask rates for each protein in batch
        struct_mask_rates = sample_cosine(B, coords.device)  # [B]
        
        # Create masks with per-protein rates
        # Expand rates to [B, L] and sample IID Bernoulli
        struct_mask_rates_expanded = struct_mask_rates.unsqueeze(1).expand(B, L)
        
        mask_struct = torch.rand(B, L, device=coords.device) < struct_mask_rates_expanded
        
        # Apply masks by zeroing out coordinates
        masked_coords = coords.clone()
        # Expand mask to match coordinate dimensions [B, L] -> [B, L, H, 3]
        mask_expanded = mask_struct.unsqueeze(-1).unsqueeze(-1).expand(B, L, H, 3)
        masked_coords[mask_expanded] = 0.0
        
        return masked_coords, mask_struct

    def _collate_with_masking(self, batch):
        """Custom collate function that applies masking for stage_1 or just tokenizes for stage_2.
        
        Workflow:
        1. Receive batch from ProteinBackboneDataset
        2. For stage_1: Apply coordinate masking
        3. For stage_2: Just tokenize sequences and structures (no masking)
        4. Return data with appropriate format for each stage
        """
        # Unpack batch from ProteinBackboneDataset
        # Dataset returns (seq, coords, length) for each protein
        seq_list, coord_list, length_list = zip(*batch)
        
        if self.model_cfg.stage == "stage_1":
            # Stage 1: Only work with coordinates, apply masking
            tokenized_coords = []
            for coords in coord_list:
                M = coords.shape[0]
                H = coords.shape[1]
                # Zero-pad coords
                num_pad = (self.model_cfg.max_len - 2) - M  # Use max_len - 2 to account for BOS/EOS
                if num_pad > 0:
                    padded_coords = torch.cat([coords, torch.zeros(num_pad, H, 3)], dim=0)
                else:
                    padded_coords = coords[:self.model_cfg.max_len - 2]  # Use max_len - 2 to account for BOS/EOS
                tokenized_coords.append(padded_coords)
            coords = torch.stack(tokenized_coords, dim=0).to(self.device)

            # Create batch tuple for masking
            batch_tuple = (coords,)

            # Apply masking based on strategy
            if self.masking_strategy == "simple":
                masked_coords, mask_struct = self.create_masked_inputs_simple(batch_tuple)
            elif self.masking_strategy == "complex":
                masked_coords, mask_struct = self.create_masked_inputs_complex(batch_tuple)
            else:
                raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
            
            # Stack lengths
            lengths = torch.tensor([l.item() for l in length_list], device=self.device)
            
            # Add BOS/EOS padding to masked coordinates
            B = masked_coords.shape[0]
            masked_coords_with_special = torch.cat([
                torch.zeros((B, 1, masked_coords.shape[2], masked_coords.shape[3]), dtype=masked_coords.dtype, device=self.device),
                masked_coords,
                torch.zeros((B, 1, masked_coords.shape[2], masked_coords.shape[3]), dtype=masked_coords.dtype, device=self.device)
            ], dim=1)
            
            # Also need to adjust the mask to account for BOS/EOS
            mask_struct_with_special = torch.cat([
                torch.zeros((B, 1), dtype=mask_struct.dtype, device=self.device),
                mask_struct,
                torch.zeros((B, 1), dtype=mask_struct.dtype, device=self.device)
            ], dim=1)
            
            # Return for stage_1: (masked_coords_with_special, mask_struct_with_special, lengths)
            return masked_coords_with_special, mask_struct_with_special, lengths

        elif self.model_cfg.stage == "stage_2":
            # Stage 2: Tokenize sequences only, return raw coordinates (no structure tokenization)
            # Tokenize sequences
            tokenized_seqs = []
            for seq in seq_list:
                tokens = self.sequence_tokenizer.tokenize(seq)
                tokenized_seqs.append(tokens)
            seq_tokens = torch.stack(tokenized_seqs, dim=0).to(self.device)
            
            # Pad coordinates (no tokenization needed - encoder will handle structure encoding)
            tokenized_coords = []
            for coords in coord_list:
                M = coords.shape[0]
                H = coords.shape[1]  # Should be 14 for stage_2
                # Zero-pad coords
                num_pad = (self.model_cfg.max_len - 2) - M  # Use max_len - 2 to account for BOS/EOS
                if num_pad > 0:
                    padded_coords = torch.cat([coords, torch.zeros(num_pad, H, 3)], dim=0)
                else:
                    padded_coords = coords[:self.model_cfg.max_len - 2]  # Use max_len - 2 to account for BOS/EOS
                tokenized_coords.append(padded_coords)
            coords = torch.stack(tokenized_coords, dim=0).to(self.device)
            
            # Stack lengths
            lengths = torch.tensor([l.item() for l in length_list], device=self.device)
            
            # Add BOS/EOS tokens and padding
            B = seq_tokens.shape[0]
            
            # Get special token IDs for sequences
            seq_bos = self.sequence_tokenizer.mapping['BOS']
            seq_eos = self.sequence_tokenizer.mapping['EOS']
            
            # Add BOS/EOS to sequence tokens
            seq_tokens_with_special = torch.cat([
                torch.full((B, 1), seq_bos, dtype=torch.long, device=self.device),
                seq_tokens,
                torch.zeros((B, 1), dtype=torch.long, device=self.device)
            ], dim=1)
            
            # Add BOS/EOS padding to coordinates
            coords_with_special = torch.cat([
                torch.zeros((B, 1, coords.shape[2], coords.shape[3]), dtype=coords.dtype, device=self.device),
                coords,
                torch.zeros((B, 1, coords.shape[2], coords.shape[3]), dtype=coords.dtype, device=self.device)
            ], dim=1)
            
            # Set EOS tokens at the appropriate positions (lengths + 1 for BOS offset)
            eos_positions = (lengths + 1).long()
            batch_indices = torch.arange(B, device=self.device)
            seq_tokens_with_special[batch_indices, eos_positions] = seq_eos
            
            # Return for stage_2: (seq_tokens, coords, lengths)
            # Note: No struct_tokens needed - encoder will produce structure representation
            return seq_tokens_with_special, coords_with_special, lengths
        
        else:
            raise ValueError(f"Unknown stage: {self.model_cfg.stage}")