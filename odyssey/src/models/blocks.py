from odyssey.src.attention.GeometricAttention import GeometricAttention
from odyssey.src.attention.SelfAttention import SelfAttention
from odyssey.src.attention.ReflexiveAttention import ReflexiveAttention
from odyssey.src.attention.SelfConsensus import SelfConsensus

import torch
from torch import nn
from typing import Optional
import math

##################################################################
# The following are utilities used by many transformer blocks:

class FeedForward(nn.Module):
    """Position-wise feed-forward network used in each Transformer block."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, d]
        return self.net(x)

class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization for time conditioning.
    
    Following DiT, we use AdaLN (adaptive layer normalization) which modulates
    the normalized features based on time embeddings. This allows the model to 
    adapt its behavior based on the diffusion timestep.
    
    We specifically implement AdaLN-Zero, where the MLP is initialized to output
    zeros, ensuring the diffusion model starts with the identity function.
    """
    
    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Projection from time embedding to scale and shift parameters
        self.time_proj = nn.Linear(time_embed_dim, 2 * d_model)
        
        # Initialize to zero for AdaLN-Zero (following DiT)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, d_model]
            time_emb: Time embedding of shape [B, time_embed_dim]
        Returns:
            Normalized and modulated tensor of shape [B, L, d_model]
        """
        # Normalize
        x_norm = self.norm(x)
        
        # Get scale (gamma) and shift (beta) from time embedding
        scale_shift = self.time_proj(time_emb)  # [B, 2 * d_model]
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each [B, d_model]
        
        # Add dimensions for broadcasting
        scale = scale.unsqueeze(1)  # [B, 1, d_model]
        shift = shift.unsqueeze(1)  # [B, 1, d_model]
        
        # Apply adaptive modulation
        return x_norm * (1 + scale) + shift

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time conditioning, following DiT.
    
    This implementation follows the standard diffusion model practice where
    timesteps are in the range [0, num_timesteps-1] during training.
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape [B] or [B, 1] with timestep indices
        Returns:
            Embeddings of shape [B, dim]
        """
        device = timesteps.device
        # Ensure timesteps has shape [B]
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim)
        args = timesteps[:, None].float() * freqs[None, :]  # [B, half_dim]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
        
        # Handle odd dimensions
        if self.dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings

##################################################################
# Each of the following are transformer blocks:

class ConvBlock(nn.Module):
    """1D Convolutional block with residual connection."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        """
        x: [B, C, L]
        """
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class StandardTransformerBlock(nn.Module):
    """Standard transformer block: LayerNorm + SelfAttention + LayerNorm + FeedForward."""

    def __init__(self, cfg, use_adaln: bool = False, time_embed_dim: int = None):
        super().__init__()
        self.use_adaln = use_adaln
        self.self_attn = SelfAttention(cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
        self.ff = FeedForward(cfg.d_model, hidden_dim=cfg.ff_hidden_dim, dropout=cfg.dropout)
        
        if use_adaln:
            assert time_embed_dim is not None, "time_embed_dim required when use_adaln=True"
            self.norm1 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Layer norm + self-attention + residual
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            x = x + self.self_attn(self.norm1(x, time_emb), mask=mask)
            # Layer norm + feed-forward + residual  
            x = x + self.ff(self.norm2(x, time_emb))
        else:
            x = x + self.self_attn(self.norm1(x), mask=mask)
            # Layer norm + feed-forward + residual  
            x = x + self.ff(self.norm2(x))
        return x

class GeometricTransformerBlock(nn.Module):
    """Geometric transformer block: LayerNorm + SelfAttention + LayerNorm + GeometricAttention + LayerNorm + FeedForward."""

    def __init__(self, cfg, use_adaln: bool = False, time_embed_dim: int = None):
        super().__init__()
        self.use_adaln = use_adaln
        self.self_attn = SelfAttention(cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
        self.geom_attn = GeometricAttention(cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout)
        self.ff = FeedForward(cfg.d_model, hidden_dim=cfg.ff_hidden_dim, dropout=cfg.dropout)
        
        if use_adaln:
            assert time_embed_dim is not None, "time_embed_dim required when use_adaln=True"
            self.norm1 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm3 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)
            self.norm3 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert coords is not None, "Coordinates are required for GeometricTransformerBlock"
        
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x, time_emb))
            # Layer norm + geometric attention + residual
            x = x + self.geom_attn(self.norm2(x, time_emb), coords, mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x, time_emb))
        else:
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x))
            # Layer norm + geometric attention + residual
            x = x + self.geom_attn(self.norm2(x), coords, mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x))
        return x

class ReflexiveTransformerBlock(nn.Module):
    """Reflexive transformer block: LayerNorm + SelfAttention + LayerNorm + ReflexiveAttention + LayerNorm + FeedForward."""

    def __init__(self, cfg, use_adaln: bool = False, time_embed_dim: int = None):
        super().__init__()
        self.use_adaln = use_adaln
        self.self_attn = SelfAttention(cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
        self.refl_attn = ReflexiveAttention(cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout)
        self.ff = FeedForward(cfg.d_model, hidden_dim=cfg.ff_hidden_dim, dropout=cfg.dropout)
        
        if use_adaln:
            assert time_embed_dim is not None, "time_embed_dim required when use_adaln=True"
            self.norm1 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm3 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)
            self.norm3 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert coords is not None, "Coordinates are required for ReflexiveTransformerBlock"
        
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x, time_emb))
            # Layer norm + reflexive attention + residual
            x = x + self.refl_attn(self.norm2(x, time_emb), coords, mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x, time_emb))
        else:
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x))
            # Layer norm + reflexive attention + residual
            x = x + self.refl_attn(self.norm2(x), coords, mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x))
        return x

class ConsensusTransformerBlock(nn.Module):
    """Consensus transformer block: LayerNorm + SelfConsensus + LayerNorm + FeedForward."""

    def __init__(self, cfg, use_adaln: bool = False, time_embed_dim: int = None):
        super().__init__()
        self.use_adaln = use_adaln
        # Use SelfConsensus instead of SelfAttention
        # Get consensus parameters from first_block_cfg
        consensus_cfg = cfg.first_block_cfg
        self.consensus = SelfConsensus(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, num_iterations=consensus_cfg.consensus_num_iterations, connectivity_type=consensus_cfg.consensus_connectivity_type, 
                                       w=consensus_cfg.consensus_w, r=consensus_cfg.consensus_r, edge_hidden_dim=consensus_cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
        self.ff = FeedForward(cfg.d_model, hidden_dim=cfg.ff_hidden_dim, dropout=cfg.dropout)
        
        if use_adaln:
            assert time_embed_dim is not None, "time_embed_dim required when use_adaln=True"
            self.norm1 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            # Layer norm + consensus + residual
            x = x + self.consensus(self.norm1(x, time_emb), mask=mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm2(x, time_emb))
        else:
            # Layer norm + consensus + residual
            x = x + self.consensus(self.norm1(x), mask=mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm2(x))
        return x
