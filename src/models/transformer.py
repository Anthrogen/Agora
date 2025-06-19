from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from src.models.blocks import SinusoidalPositionEmbeddings, AdaptiveLayerNorm, ConvBlock, StandardTransformerBlock, GeometricTransformerBlock, ReflexiveTransformerBlock, ConsensusTransformerBlock

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
