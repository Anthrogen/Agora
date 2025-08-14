"""
Rotary Position Embeddings (RoPE) implementation.

Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding".
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
        
        # Compute the inverse frequencies (use float32 for precision in fp16 training)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
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
        
        # Update max_position_embeddings to reflect actual cache size
        self.max_position_embeddings = seq_len
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: Optional[torch.Tensor] = None, k: Optional[torch.Tensor] = None, 
                u_i: Optional[torch.Tensor] = None, model_type: str = "SA",
                edge_i: Optional[torch.Tensor] = None, edge_j: Optional[torch.Tensor] = None, 
                seq_len: Optional[int] = None, offset: int = 0):
        """
        Apply rotary embeddings for different model types.
        
        Args:
            q: For SA: Query tensor [batch, heads, seq_len, head_dim]
            k: For SA: Key tensor [batch, heads, seq_len, head_dim]
            u_i: For C: Source node features [batch, heads, num_edges, head_dim] (already gathered)
            model_type: Either "SA" (self-attention) or "SC" (self-consensus)
            edge_i: For C: Source node indices [batch, num_edges]
            edge_j: For C: Target node indices [batch, num_edges]
            seq_len: Sequence length (if different from cached length)
            offset: Position offset for generation tasks
            
        Returns:
            For SA: Tuple of rotated (query, key) tensors
            For C: Rotated source node features [batch, heads, num_edges, head_dim]
        """
        if model_type == "SA":
            return self._forward_self_attention(q, k, seq_len, offset)
        elif model_type == "SC":
            return self._forward_consensus(u_i, edge_i, edge_j)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'SA' or 'C'")
    
    def _forward_self_attention(self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None,offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors for self-attention."""
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
    
    def _forward_consensus(self, u_i: torch.Tensor, edge_i: torch.Tensor, edge_j: torch.Tensor) -> torch.Tensor:
        """Apply relative RoPE rotations for Consensus layer."""
        # Compute position differences
        pos_diff = edge_i - edge_j  # [B, E]
        
        # Get cos/sin tables
        cos_table = self.cos_cached.squeeze(0).squeeze(0)  # [max_len, D]
        sin_table = self.sin_cached.squeeze(0).squeeze(0)  # [max_len, D]
        
        # Handle relative positions
        abs_diff = pos_diff.abs()  # [B, E]
        sign = pos_diff.sign().unsqueeze(-1).type_as(u_i)  # [B, E, 1]
        
        cos_rel = cos_table[abs_diff]  # [B, E, D]
        sin_rel = sin_table[abs_diff] * sign  # [B, E, D]
        
        # Reshape for broadcasting with heads
        cos_rel = cos_rel.unsqueeze(1)  # [B, 1, E, D]
        sin_rel = sin_rel.unsqueeze(1)  # [B, 1, E, D]
        
        # Apply rotation
        u_i_rot = u_i * cos_rel + self._rotate_half(u_i) * sin_rel
        
        return u_i_rot