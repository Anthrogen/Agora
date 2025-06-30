"""Simple masking strategy implementation."""

from typing import Dict, Optional
import torch

from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from .base import BaseMaskingStrategy


class SimpleMasking(BaseMaskingStrategy):
    """Simple random masking strategy."""
    
    def __init__(
        self,
        mask_prob_seq: float = 0.2,
        mask_prob_coords: float = 0.2,
        seq_mask_token: Optional[int] = None,
        struct_mask_token: Optional[int] = None
    ):
        """
        Initialize simple masking strategy.
        
        Args:
            mask_prob_seq: Probability of masking sequence tokens
            mask_prob_coords: Probability of masking structure tokens
            seq_mask_token: Mask token for sequences
            struct_mask_token: Mask token for structures
        """
        self.mask_prob_seq = mask_prob_seq
        self.mask_prob_coords = mask_prob_coords
        
        # Set default mask tokens
        if seq_mask_token is None:
            seq_mask_token = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
        if struct_mask_token is None:
            struct_mask_token = SPECIAL_TOKENS.MASK.value + 4375
        
        self.seq_mask_token = seq_mask_token
        self.struct_mask_token = struct_mask_token
    
    def apply_mask(
        self,
        seq: torch.Tensor,
        struct: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply random masking to sequences and structures.
        
        Args:
            seq: Sequence tokens (B, L)
            struct: Structure tokens (B, L)
            
        Returns:
            Dictionary with masked sequences, structures, and masks
        """
        batch_size, seq_len = seq.shape
        device = seq.device
        
        # Create masks
        seq_mask = torch.rand(batch_size, seq_len, device=device) < self.mask_prob_seq
        struct_mask = torch.rand(batch_size, seq_len, device=device) < self.mask_prob_coords
        
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
            'original_struct': struct
        }
    
    def get_mask_token_seq(self) -> int:
        """Get mask token for sequences."""
        return self.seq_mask_token
    
    def get_mask_token_struct(self) -> int:
        """Get mask token for structures."""
        return self.struct_mask_token