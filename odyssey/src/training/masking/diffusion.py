"""Discrete diffusion masking strategy."""

import math
from typing import Dict, Optional, Literal, Tuple
import torch
import torch.nn.functional as F

from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from .base import BaseMaskingStrategy


class DiffusionMasking(BaseMaskingStrategy):
    """Discrete diffusion masking strategy."""
    
    def __init__(
        self,
        noise_schedule: Literal["linear", "inverted_u", "uniform"] = "linear",
        sigma_min: float = 0.31,
        sigma_max: float = 5.68,
        num_timesteps: int = 100,
        seq_vocab_size: int = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS),
        struct_vocab_size: int = 4375 + len(SPECIAL_TOKENS),
        seq_absorb_token: Optional[int] = None,
        struct_absorb_token: Optional[int] = None
    ):
        """
        Initialize diffusion masking strategy.
        
        Args:
            noise_schedule: Type of noise schedule
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            num_timesteps: Number of discrete timesteps
            seq_vocab_size: Vocabulary size for sequences
            struct_vocab_size: Vocabulary size for structures
            seq_absorb_token: Absorbing state token for sequences
            struct_absorb_token: Absorbing state token for structures
        """
        self.noise_schedule = noise_schedule
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_timesteps = num_timesteps
        self.seq_vocab_size = seq_vocab_size
        self.struct_vocab_size = struct_vocab_size
        
        # Set default absorbing tokens (MASK tokens)
        if seq_absorb_token is None:
            seq_absorb_token = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
        if struct_absorb_token is None:
            struct_absorb_token = SPECIAL_TOKENS.MASK.value + 4375
        
        self.seq_absorb_token = seq_absorb_token
        self.struct_absorb_token = struct_absorb_token
        
        # Precompute noise schedule
        self.sigmas = self._compute_noise_schedule()
    
    def _compute_noise_schedule(self) -> torch.Tensor:
        """Compute noise schedule sigmas."""
        t = torch.linspace(0, 1, self.num_timesteps + 1)
        
        if self.noise_schedule == "linear":
            # Linear interpolation between sigma_min and sigma_max
            sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        
        elif self.noise_schedule == "inverted_u":
            # Inverted U shape - peaks in the middle
            # Using sine function for smooth curve
            sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sin(t * math.pi)
        
        elif self.noise_schedule == "uniform":
            # Uniform noise level
            sigmas = torch.full_like(t, (self.sigma_min + self.sigma_max) / 2)
        
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
        
        return sigmas
    
    def _get_transition_matrix(
        self,
        sigma: float,
        vocab_size: int,
        absorb_token: int
    ) -> torch.Tensor:
        """
        Compute transition matrix for discrete diffusion.
        
        Args:
            sigma: Noise level
            vocab_size: Size of vocabulary
            absorb_token: Absorbing state token
            
        Returns:
            Transition matrix (vocab_size, vocab_size)
        """
        # Create transition matrix
        Q = torch.zeros(vocab_size, vocab_size)
        
        # Compute transition probabilities
        alpha = 1 / (1 + sigma)  # Probability of staying in same state
        beta = (1 - alpha) / (vocab_size - 1)  # Probability of transitioning
        
        # Fill transition matrix
        Q.fill_(beta)
        Q.fill_diagonal_(alpha)
        
        # Absorbing state stays absorbed
        Q[absorb_token, :] = 0
        Q[absorb_token, absorb_token] = 1
        
        return Q
    
    def _sample_discrete(
        self,
        tokens: torch.Tensor,
        transition_matrix: torch.Tensor,
        time_idx: int
    ) -> torch.Tensor:
        """
        Sample from discrete diffusion process.
        
        Args:
            tokens: Current tokens (B, L)
            transition_matrix: Transition matrix
            time_idx: Current time index
            
        Returns:
            Noised tokens
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        vocab_size = transition_matrix.size(0)
        
        # Convert tokens to one-hot
        tokens_onehot = F.one_hot(tokens, num_classes=vocab_size).float()  # (B, L, V)
        
        # Apply transition matrix
        # tokens_onehot: (B, L, V), transition_matrix: (V, V)
        # Result: (B, L, V)
        transition_probs = torch.matmul(tokens_onehot, transition_matrix.T)
        
        # Sample from categorical distribution
        noised_tokens = torch.multinomial(
            transition_probs.view(-1, vocab_size),
            num_samples=1
        ).view(batch_size, seq_len)
        
        return noised_tokens
    
    def apply_mask(
        self,
        seq: torch.Tensor,
        struct: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply discrete diffusion masking.
        
        Args:
            seq: Sequence tokens (B, L)
            struct: Structure tokens (B, L)
            time: Time indices (B,) - if None, samples randomly
            
        Returns:
            Dictionary with noised data and metadata
        """
        batch_size, seq_len = seq.shape
        device = seq.device
        
        # Sample time if not provided
        if time is None:
            time = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device)
        
        # Move precomputed sigmas to device
        sigmas = self.sigmas.to(device)
        
        # Initialize noised sequences
        noised_seq = seq.clone()
        noised_struct = struct.clone()
        
        # Process each time step in the batch
        for b in range(batch_size):
            t = time[b].item()
            sigma = sigmas[t].item()
            
            # Get transition matrices
            seq_Q = self._get_transition_matrix(
                sigma, self.seq_vocab_size, self.seq_absorb_token
            ).to(device)
            
            struct_Q = self._get_transition_matrix(
                sigma, self.struct_vocab_size, self.struct_absorb_token
            ).to(device)
            
            # Apply diffusion
            noised_seq[b] = self._sample_discrete(seq[b:b+1], seq_Q, t).squeeze(0)
            noised_struct[b] = self._sample_discrete(struct[b:b+1], struct_Q, t).squeeze(0)
        
        # Create masks for positions that were noised to absorbing state
        seq_mask = noised_seq == self.seq_absorb_token
        struct_mask = noised_struct == self.struct_absorb_token
        
        return {
            'masked_seq': noised_seq,
            'masked_struct': noised_struct,
            'seq_mask': seq_mask,
            'struct_mask': struct_mask,
            'original_seq': seq,
            'original_struct': struct,
            'time': time,
            'sigmas': sigmas[time]
        }
    
    def get_mask_token_seq(self) -> int:
        """Get mask token (absorbing state) for sequences."""
        return self.seq_absorb_token
    
    def get_mask_token_struct(self) -> int:
        """Get mask token (absorbing state) for structures."""
        return self.struct_absorb_token