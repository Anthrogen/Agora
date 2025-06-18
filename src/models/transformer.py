from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from src.attention.GeometricAttention import GeometricAttention
from src.attention.SelfAttention import SelfAttention
from src.attention.ReflexiveAttention import ReflexiveAttention
from src.attention.Consensus import Consensus

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
                coord_mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Layer norm + self-attention + residual
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            x = x + self.self_attn(self.norm1(x, time_emb))
            # Layer norm + feed-forward + residual  
            x = x + self.ff(self.norm2(x, time_emb))
        else:
            x = x + self.self_attn(self.norm1(x))
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
                coord_mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert coords is not None, "Coordinates are required for GeometricTransformerBlock"
        
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x, time_emb))
            # Layer norm + geometric attention + residual
            x = x + self.geom_attn(self.norm2(x, time_emb), coords, coord_mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x, time_emb))
        else:
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x))
            # Layer norm + geometric attention + residual
            x = x + self.geom_attn(self.norm2(x), coords, coord_mask)
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
                coord_mask: Optional[torch.Tensor] = None, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert coords is not None, "Coordinates are required for ReflexiveTransformerBlock"
        
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x, time_emb))
            # Layer norm + reflexive attention + residual
            x = x + self.refl_attn(self.norm2(x, time_emb), coords, coord_mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x, time_emb))
        else:
            # Layer norm + self-attention + residual
            x = x + self.self_attn(self.norm1(x))
            # Layer norm + reflexive attention + residual
            x = x + self.refl_attn(self.norm2(x), coords, coord_mask)
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm3(x))
        return x

class ConsensusTransformerBlock(nn.Module):
    """Consensus transformer block: LayerNorm + Consensus + LayerNorm + FeedForward."""

    def __init__(self, cfg, use_adaln: bool = False, time_embed_dim: int = None):
        super().__init__()
        self.use_adaln = use_adaln
        # Use Consensus instead of SelfAttention
        self.consensus = Consensus(dim=cfg.d_model, dropout=cfg.dropout, num_iterations=cfg.consensus_num_iterations, connectivity_type=cfg.consensus_connectivity_type, 
                                   w=cfg.consensus_w, r=cfg.consensus_r, edge_hidden_dim=cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
        self.ff = FeedForward(cfg.d_model, hidden_dim=cfg.ff_hidden_dim, dropout=cfg.dropout)
        
        if use_adaln:
            assert time_embed_dim is not None, "time_embed_dim required when use_adaln=True"
            self.norm1 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_adaln:
            assert time_emb is not None, "time_emb required when use_adaln=True"
            # Layer norm + consensus + residual
            x = x + self.consensus(self.norm1(x, time_emb))
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm2(x, time_emb))
        else:
            # Layer norm + consensus + residual
            x = x + self.consensus(self.norm1(x))
            # Layer norm + feed-forward + residual
            x = x + self.ff(self.norm2(x))
        return x

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

# --------------------------------------------------------------------------- #
#  Top-level model                                                             #
# --------------------------------------------------------------------------- #
class TransformerTrunk(nn.Module):
    """Unified transformer model with configurable first layer."""

    def __init__(self, cfg, use_adaln: bool = False):
        super().__init__()
        # Store config
        self.cfg = cfg
        self.use_adaln = use_adaln

        # Sequence Embedding
        self.seq_embed = nn.Embedding(cfg.seq_vocab, cfg.d_model)
        # Structure Embedding, tokenized
        self.struct_embed = nn.Embedding(cfg.struct_vocab, cfg.d_model)
        
        # Time embedding for diffusion models following DiT
        if use_adaln:
            time_embed_dim = cfg.d_model * 4  # Following DiT paper
            # DiT-style time embeddings: sinusoidal embeddings + MLP
            self.time_embed = nn.Sequential(
                SinusoidalPositionEmbeddings(cfg.d_model),
                nn.Linear(cfg.d_model, time_embed_dim),
                nn.SiLU(),  # DiT uses SiLU activation
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        # Transformer stack
        self.layers = nn.ModuleList()
        
        # First block: either geometric transformer block or standard block
        if cfg.model_type == "GA":
            self.layers.append(GeometricTransformerBlock(cfg, use_adaln, time_embed_dim))
        elif cfg.model_type == "SA":
            self.layers.append(StandardTransformerBlock(cfg, use_adaln, time_embed_dim))
        elif cfg.model_type == "RA":
            self.layers.append(ReflexiveTransformerBlock(cfg, use_adaln, time_embed_dim))
        elif cfg.model_type == "C":
            self.layers.append(ConsensusTransformerBlock(cfg, use_adaln, time_embed_dim))
        else:
            raise ValueError(f"Invalid model_type type: {cfg.model_type}")
        
        # Remaining blocks
        if cfg.model_type == "C":
            # For Consensus, all blocks are ConsensusTransformerBlocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(ConsensusTransformerBlock(cfg, use_adaln, time_embed_dim))
        else:
            # For GA/SA/RA, remaining blocks are all standard transformer blocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(StandardTransformerBlock(cfg, use_adaln, time_embed_dim))

        if use_adaln:
            self.final_norm = AdaptiveLayerNorm(cfg.d_model, time_embed_dim)
        else:
            self.final_norm = nn.LayerNorm(cfg.d_model)
            
        # Prediction heads for sequence and structure tokens
        self.seq_logits = nn.Linear(cfg.d_model, cfg.seq_vocab)
        self.struct_logits = nn.Linear(cfg.d_model, cfg.struct_vocab)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],  # (seq_tokens, struct_tokens)
        coords: Optional[torch.Tensor] = None,  # [B, L, 4, 3] backbone coordinates
        coord_mask: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        timesteps: Optional[torch.Tensor] = None,  # [B, 1] timesteps for diffusion
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tuple of integer token tensors of shape ``[B, L]`` for sequence and structure
               tracks, respectively.
            coords: Optional backbone coordinates ``[B, L, 4, 3]`` (N, CA, C, CB) – only
               required when the first layer uses geometric attention (``GA``) or reflexive attention (``RA``).
            coord_mask: Optional boolean mask ``[B, L]`` indicating which positions
               have valid coordinates. True = valid, False = masked.
            timesteps: Optional timesteps for diffusion models, shape [B, 1] with values in [0, 1]

        Returns:
            Tuple of logits tensors of shape ``[B, L, V]`` where ``V`` is ``seq_vocab`` or
            ``struct_vocab`` for the respective track.  Raw (unnormalised)
            scores are returned so they can be passed directly to
            ``F.cross_entropy``.
        """
        seq_tokens, struct_tokens = x  # unpack tuple

        if seq_tokens.shape != struct_tokens.shape:
            raise ValueError("Sequence and structure token tensors must have identical shape")

        B, L = seq_tokens.shape
        if L > self.cfg.max_len:
            raise ValueError("Sequence length exceeds model max_len")

        # Embed each track (nn.Embedding expects integer indices, so no one-hot required)
        seq_emb = self.seq_embed(seq_tokens)          # [B, L, d]
        struct_emb = self.struct_embed(struct_tokens) # [B, L, d]

        h = seq_emb + struct_emb
        
        # Get time embeddings if using AdaLN
        time_emb = None
        if self.use_adaln:
            assert timesteps is not None, "timesteps required when use_adaln=True"
            time_emb = self.time_embed(timesteps)  # [B, time_embed_dim]

        # --- Transformer trunk ------------------------------------------------
        # Pass through all transformer blocks
        for block in self.layers:
            if self.cfg.model_type in ("GA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for geometric first layer"
                h = block(h, coords[:,:,:3,:], coord_mask, time_emb)
            elif self.cfg.model_type in ("RA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for reflexive first layer"
                h = block(h, coords, coord_mask, time_emb)
            else:
                # Standard blocks don't need coordinates
                h = block(h, time_emb=time_emb)

        if self.use_adaln:
            h = self.final_norm(h, time_emb)  # [B, L, d]
        else:
            h = self.final_norm(h)

        return self.seq_logits(h), self.struct_logits(h)
