"""
Rotary Position Embeddings (RoPE) implementation.

Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
and as used in ESM3.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in RoFormer.
    
    Query and keys are rotated by theta according to their position index.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cached rotations for efficiency
        self._build_cache(max_position_embeddings)
        
    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for rotary embeddings."""
        # Create position indices
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Compute the frequencies for each position
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        
        # Create rotation embeddings by concatenating cos and sin
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        
        # Cache cos and sin values
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # [1, 1, seq_len, dim]
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])  # [1, 1, seq_len, dim]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: Optional[int] = None,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, heads, seq_len, head_dim]
            seq_len: Sequence length (if different from cached length)
            offset: Position offset for generation tasks
            
        Returns:
            Tuple of rotated (query, key) tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]
            
        # Use cached values or rebuild if sequence is longer
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
            
        # Get the relevant portions of cached cos/sin
        cos = self.cos_cached[:, :, offset : offset + seq_len, :]
        sin = self.sin_cached[:, :, offset : offset + seq_len, :]
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    This is a functional version that can be used without the RotaryEmbedding module.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine values [1, 1, seq_len, dim]
        sin: Sine values [1, 1, seq_len, dim]
        position_ids: Optional custom position indices
        
    Returns:
        Tuple of rotated (query, key) tensors
    """
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    if position_ids is not None:
        # Gather cos/sin values based on position_ids
        cos = cos.squeeze(0).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(0).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed 