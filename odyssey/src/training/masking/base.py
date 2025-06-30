"""Base masking strategy interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch


class BaseMaskingStrategy(ABC):
    """Abstract base class for masking strategies."""
    
    @abstractmethod
    def apply_mask(
        self,
        seq: torch.Tensor,
        struct: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Apply masking to sequence and structure tokens.
        
        Args:
            seq: Sequence tokens (B, L)
            struct: Structure tokens (B, L)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing masked data and masks
        """
        pass
    
    @abstractmethod
    def get_mask_token_seq(self) -> int:
        """Get mask token for sequences."""
        pass
    
    @abstractmethod
    def get_mask_token_struct(self) -> int:
        """Get mask token for structures."""
        pass