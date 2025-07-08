import torch
import torch.nn as nn
from .rope import RotaryEmbedding

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention layer
    
    Standard cross-attention mechanism where queries come from target sequence
    and keys/values come from context sequence. Only the target sequence is
    updated, making it comparable to CrossConsensus.
    
    Key design:
    • Multi-head architecture: attention operates independently in each head subspace
    • Cross-attention: Q from target, K/V from context
    • Scaled dot-product attention with optional RoPE positional encoding
    • Dropout and output projection
    """

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
    
    def forward(self, target, context, target_mask=None, context_mask=None):
        """
        target: [B, L, dim] target sequence features
        context: [B, K, dim] context sequence features  
        target_mask: [B, L] boolean tensor where True = valid, False = invalid/padding
        context_mask: [B, K] boolean tensor where True = valid, False = invalid/padding
        """
        B, L, C = target.shape
        _, K, _ = context.shape
                
        # Project to queries, keys, values using separate linear layers
        q = self.q(target).reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        k = self.k(context).reshape(B, K, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, K, head_dim]
        v = self.v(context).reshape(B, K, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, K, head_dim]
        
        # Apply rotary position embeddings
        q, _ = self.rotary_emb(q=q, k=q, model_type="SA")  # Apply to q with length L
        _, k = self.rotary_emb(q=k, k=k, model_type="SA")  # Apply to k with length K
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, heads, L, K]
        
        # Apply context masking if provided (mask invalid context positions)
        if context_mask is not None:
            # Mask attention scores where context keys (columns) are invalid
            # Invalid context positions shouldn't be attended to
            score_mask = context_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, K]
            attn = attn.masked_fill(~score_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine heads and project
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        # Zero output for invalid target positions (following SelfAttention pattern)
        if target_mask is not None:
            x = x * target_mask.unsqueeze(-1).float()
        
        return x