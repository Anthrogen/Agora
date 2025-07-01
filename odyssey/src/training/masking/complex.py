"""Complex masking strategy with noise schedules."""

import math
from typing import Dict, Optional, Literal
import torch

from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from .base import BaseMaskingStrategy


class ComplexMasking(BaseMaskingStrategy):
    """Complex masking strategy with noise schedules."""
    
    def __init__(
        self,
        mask_prob: float = 0.2,
        noise_schedule: Literal["linear", "inverted_u"] = "linear",
        lambda_val: float = 1.0,
        seq_mask_token: Optional[int] = None,
        struct_mask_token: Optional[int] = None,
        max_time: int = 100
    ):
        """
        Initialize complex masking strategy.
        
        Args:
            mask_prob: Base masking probability
            noise_schedule: Type of noise schedule
            lambda_val: Lambda parameter for schedule
            seq_mask_token: Mask token for sequences
            struct_mask_token: Mask token for structures
            max_time: Maximum time steps
        """
        self.mask_prob = mask_prob
        self.noise_schedule = noise_schedule
        self.lambda_val = lambda_val
        self.max_time = max_time
        
        # Set default mask tokens
        if seq_mask_token is None:
            seq_mask_token = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
        if struct_mask_token is None:
            struct_mask_token = SPECIAL_TOKENS.MASK.value + 4375
        
        self.seq_mask_token = seq_mask_token
        self.struct_mask_token = struct_mask_token
    
    def _get_noise_rate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get noise rate based on schedule.
        
        Args:
            t: Time tensor (B,)
            
        Returns:
            Noise rate tensor (B,)
        """
        # Normalize time to [0, 1]
        normalized_t = t.float() / self.max_time
        
        if self.noise_schedule == "linear":
            # Linear schedule: rate increases linearly with time
            rate = normalized_t * self.lambda_val
        
        elif self.noise_schedule == "inverted_u":
            # Inverted U schedule: peaks in the middle
            # Using a parabola: -4(t-0.5)^2 + 1
            rate = -4 * (normalized_t - 0.5) ** 2 + 1
            rate = rate * self.lambda_val
        
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
        
        # Scale by base mask probability
        rate = rate * self.mask_prob
        
        # Clamp to [0, 1]
        return torch.clamp(rate, 0.0, 1.0)
    
    def apply_mask(
        self,
        seq: torch.Tensor,
        struct: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply complex masking with noise schedule.
        
        Args:
            seq: Sequence tokens (B, L)
            struct: Structure tokens (B, L)
            time: Time steps (B,) - if None, samples randomly
            
        Returns:
            Dictionary with masked data and metadata
        """
        batch_size, seq_len = seq.shape
        device = seq.device
        
        # Sample time if not provided
        if time is None:
            time = torch.randint(0, self.max_time, (batch_size,), device=device)
        
        # Get noise rates for each sample in batch
        noise_rates = self._get_noise_rate(time)  # (B,)
        
        # Expand noise rates to (B, 1) for broadcasting
        noise_rates = noise_rates.unsqueeze(1)
        
        # Create masks based on noise rates
        # Each position has independent probability based on the noise rate
        seq_rand = torch.rand(batch_size, seq_len, device=device)
        struct_rand = torch.rand(batch_size, seq_len, device=device)
        
        seq_mask = seq_rand < noise_rates
        struct_mask = struct_rand < noise_rates
        
        # Apply masks
        masked_seq = seq.clone()
        masked_struct = struct.clone()
        
        masked_seq[seq_mask] = self.seq_mask_token
        masked_struct[struct_mask] = self.struct_mask_token
        
        return {
            'masked_seq': masked_seq,
            'masked_struct': masked_struct,
            'seq_mask': seq_mask,
            'struct_mask': struct_mask,
            'original_seq': seq,
            'original_struct': struct,
            'time': time,
            'noise_rates': noise_rates.squeeze(1)
        }
    
    def get_mask_token_seq(self) -> int:
        """Get mask token for sequences."""
        return self.seq_mask_token
    
    def get_mask_token_struct(self) -> int:
        """Get mask token for structures."""
        return self.struct_mask_token