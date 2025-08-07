import torch
import torch.nn as nn
from .rope import RotaryEmbedding

class SelfAttention(nn.Module):
    """Self-attention layer with Rotary Position Embeddings (RoPE)."""
    def __init__(self, dim, heads=4, dropout=0.1, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "Dimension must be divisible by number of heads"
        
        # Separate linear layers for Q, K, V
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        x: [B, L, dim]
        mask: Optional [B, L] boolean tensor where True = valid, False = invalid/padding
        """
        assert mask is None or mask.dtype == torch.bool, "Mask must be a boolean tensor"
        B, L, C = x.shape        
                
        # Project to queries, keys, values using separate linear layers
        q = self.q(x).reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        k = self.k(x).reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        v = self.v(x).reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        
        # Apply rotary position embeddings
        q, k = self.rotary_emb(q=q, k=k, model_type="SA")
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, heads, L, L]
        
        # Apply masking if provided
        if mask is not None:
            # Mask attention scores where keys (columns) are invalid
            # Invalid positions shouldn't be attended to
            score_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_key]
            attn = attn.masked_fill(~score_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine heads and project
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        # Zero output for invalid positions
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        
        return x