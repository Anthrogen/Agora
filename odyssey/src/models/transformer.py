from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from odyssey.src.models.blocks import SinusoidalPositionEmbeddings, AdaptiveLayerNorm, ConvBlock, StandardTransformerBlock, GeometricTransformerBlock, ReflexiveTransformerBlock, ConsensusTransformerBlock
from odyssey.src.attention.CrossAttention import CrossAttention  
from odyssey.src.attention.CrossConsensus import CrossConsensus

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
        # SS8 Embedding
        self.ss8_embed = nn.Embedding(cfg.ss8_vocab, cfg.d_model)
        # SASA Embedding
        self.sasa_embed = nn.Embedding(cfg.sasa_vocab, cfg.d_model)
        # pLDDT Embedding
        self.plddt_embed = nn.Embedding(cfg.plddt_vocab, cfg.d_model)
        # Per-residue annotation embedding
        self.per_residue_annotation_embed = nn.Embedding(cfg.per_residue_annotation_vocab, cfg.d_model)
        # Global annotation embedding
        self.global_annotation_embed = nn.Embedding(cfg.global_annotation_vocab, cfg.d_model)
        
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
        model_type = cfg.first_block_cfg.initials()
        if model_type == "GA":
            self.layers.append(GeometricTransformerBlock(cfg, use_adaln, time_embed_dim))
        elif model_type == "SA":
            self.layers.append(StandardTransformerBlock(cfg, use_adaln, time_embed_dim))
        elif model_type == "RA":
            self.layers.append(ReflexiveTransformerBlock(cfg, use_adaln, time_embed_dim))
        elif model_type == "SC":
            self.layers.append(ConsensusTransformerBlock(cfg, use_adaln, time_embed_dim))
        else:
            raise ValueError(f"Invalid model_type type: {model_type}")
        
        # Remaining blocks
        if model_type == "SC":
            # For SelfConsensus, all blocks are ConsensusTransformerBlocks
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
            
        # Context injection module for SS8/SASA/pLDDT/Per-residue annotations (if configured)
        context_type = cfg.context_cfg.initials()
        if context_type == "CA":  # CrossAttention
            self.context_ss8 = CrossAttention(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
            self.context_sasa = CrossAttention(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
            self.context_plddt = CrossAttention(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
            self.context_per_residue_annotation = CrossAttention(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
        elif context_type == "CC":  # CrossConsensus
            self.context_ss8 = CrossConsensus(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, num_iterations=cfg.context_cfg.consensus_num_iterations,
                connectivity_type=cfg.context_cfg.consensus_connectivity_type, w=cfg.context_cfg.consensus_w, r=cfg.context_cfg.consensus_r,
                edge_hidden_dim=cfg.context_cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
            self.context_sasa = CrossConsensus(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, num_iterations=cfg.context_cfg.consensus_num_iterations,
                connectivity_type=cfg.context_cfg.consensus_connectivity_type, w=cfg.context_cfg.consensus_w, r=cfg.context_cfg.consensus_r,
                edge_hidden_dim=cfg.context_cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
            self.context_plddt = CrossConsensus(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, num_iterations=cfg.context_cfg.consensus_num_iterations,
                connectivity_type=cfg.context_cfg.consensus_connectivity_type, w=cfg.context_cfg.consensus_w, r=cfg.context_cfg.consensus_r,
                edge_hidden_dim=cfg.context_cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
            self.context_per_residue_annotation = CrossConsensus(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, num_iterations=cfg.context_cfg.consensus_num_iterations,
                connectivity_type=cfg.context_cfg.consensus_connectivity_type, w=cfg.context_cfg.consensus_w, r=cfg.context_cfg.consensus_r,
                edge_hidden_dim=cfg.context_cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
        
        # Context injection module for global annotation
        # Use max of sequence length and global annotation length for positional embeddings
        max_pos_embeddings_global = max(cfg.max_len, cfg.max_len_global)
        self.context_global_annotation = CrossAttention(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=max_pos_embeddings_global)

        # Prediction heads for sequence and structure tokens
        self.seq_logits = nn.Linear(cfg.d_model, cfg.seq_vocab)
        self.struct_logits = nn.Linear(cfg.d_model, cfg.struct_vocab)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  # (seq_tokens, struct_tokens, ss8_tokens, sasa_tokens, global_annotation_tokens, per_residue_annotation_tokens, plddt_tokens)
        coords: Optional[torch.Tensor] = None,  # [B, L, 4, 3] backbone coordinates
        content_elements: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank_ss8: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank_sasa: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank_global_annotation: Optional[torch.Tensor] = None,  # [B, G] boolean mask
        nonbeospank_per_residue_annotation: Optional[torch.Tensor] = None,  # [B, L, K] boolean mask
        nonbeospank_plddt: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        timesteps: Optional[torch.Tensor] = None,  # [B, 1] timesteps for diffusion
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tuple of integer token tensors for sequence, structure, ss8, sasa, global_annotation, 
               per_residue_annotation, and plddt tracks.
            coords: Optional backbone coordinates ``[B, L, 4, 3]`` (N, CA, C, CB) - only
               required when the first layer uses geometric attention (``GA``) or reflexive attention (``RA``).
            content_elements: Optional boolean mask ``[B, L]`` indicating which positions
               have valid coordinates. True = valid, False = masked.
            nonbeospank: Optional boolean mask ``[B, L]`` indicating which positions
               have valid coordinates. True = valid, False = masked.
            nonbeospank_ss8: Optional boolean mask ``[B, L]`` indicating which positions
               have valid ss8 tokens. True = valid, False = masked.
            nonbeospank_sasa: Optional boolean mask ``[B, L]`` indicating which positions
               have valid sasa tokens. True = valid, False = masked.
            nonbeospank_global_annotation: Optional boolean mask ``[B, G]`` indicating which positions
               have valid global annotation tokens. True = valid, False = masked.
            nonbeospank_per_residue_annotation: Optional boolean mask ``[B, L, K]`` indicating which positions
               have valid per-residue annotation tokens. True = valid, False = masked.
            nonbeospank_plddt: Optional boolean mask ``[B, L]`` indicating which positions
               have valid plddt tokens. True = valid, False = masked.
            timesteps: Optional timesteps for diffusion models, shape [B, 1] with values in [0, 1]

        Returns:
            Tuple of logits tensors of shape ``[B, L, V]`` where ``V`` is ``seq_vocab`` or
            ``struct_vocab`` for the respective track.  Raw (unnormalised)
            scores are returned so they can be passed directly to
            ``F.cross_entropy``.
        """
        seq_tokens, struct_tokens, ss8_tokens, sasa_tokens, global_annotation_tokens, per_residue_annotation_tokens, plddt_tokens = x  # unpack tuple

        if seq_tokens.shape != struct_tokens.shape:
            raise ValueError(f"Sequence and structure token tensors must have identical shape: {seq_tokens.shape} != {struct_tokens.shape}")
        if ss8_tokens.shape != sasa_tokens.shape:
            raise ValueError(f"SS8 and sasa token tensors must have identical shape: {ss8_tokens.shape} != {sasa_tokens.shape}")

        B, L, K = per_residue_annotation_tokens.shape
        _, G = global_annotation_tokens.shape
        if L > self.cfg.max_len:
            raise ValueError("Sequence length exceeds model max_len")

        # Embed each track (nn.Embedding expects integer indices, so no one-hot required)
        seq_emb = self.seq_embed(seq_tokens)  # [B, L, d]
        struct_emb = self.struct_embed(struct_tokens)  # [B, L, d]
        ss8_emb = self.ss8_embed(ss8_tokens)  # [B, L, d]
        sasa_emb = self.sasa_embed(sasa_tokens)  # [B, L, d]
        global_annotation_emb = self.global_annotation_embed(global_annotation_tokens)  # [B, G, d]
        plddt_emb = self.plddt_embed(plddt_tokens)  # [B, L, d]
        per_residue_annotation_emb = self.per_residue_annotation_embed(per_residue_annotation_tokens)  # [B, L, K, d]
        
        # Masked mean pooling: only average over valid annotation tokens (non-BOS/EOS/PAD/UNK)
        # Zero out embeddings for special tokens and sum valid embeddings
        # Count valid tokens per position and compute masked mean (avoid division by zero)
        valid_per_residue_annotations = nonbeospank_per_residue_annotation.unsqueeze(-1)  # [B, L, K, 1] True = valid annotation, False = special token
        valid_per_residue_annotation_emb = per_residue_annotation_emb * valid_per_residue_annotations.float()  # [B, L, K, d]
        sum_embeddings = valid_per_residue_annotation_emb.sum(dim=2)  # [B, L, d]
        num_valid = torch.clamp(valid_per_residue_annotations.sum(dim=2).float(), min=1.0)  # [B, L, 1]
        per_residue_annotation_emb = sum_embeddings / num_valid  # [B, L, d] / [B, L, 1] -> [B, L, d]

        h = seq_emb + struct_emb
        
        # Inject SS8 context
        # Only apply context injection to rows that have at least one valid context position
        valid_ss8_indices = nonbeospank_ss8.any(dim=1).nonzero(as_tuple=True)[0]  # Get indices of valid rows
        if len(valid_ss8_indices) > 0:
            ss8_context = self.context_ss8(h[valid_ss8_indices], ss8_emb[valid_ss8_indices], nonbeospank[valid_ss8_indices], nonbeospank_ss8[valid_ss8_indices])
            h[valid_ss8_indices] = h[valid_ss8_indices] + ss8_context
        
        # Inject SASA context
        # Only apply context injection to rows that have at least one valid context position
        valid_sasa_indices = nonbeospank_sasa.any(dim=1).nonzero(as_tuple=True)[0]  # Get indices of valid rows
        if len(valid_sasa_indices) > 0:
            sasa_context = self.context_sasa(h[valid_sasa_indices], sasa_emb[valid_sasa_indices], nonbeospank[valid_sasa_indices], nonbeospank_sasa[valid_sasa_indices])
            h[valid_sasa_indices] = h[valid_sasa_indices] + sasa_context

        # Inject global annotation context
        # Only apply context injection to rows that have at least one valid context position
        valid_global_annotation_indices = nonbeospank_global_annotation.any(dim=1).nonzero(as_tuple=True)[0]  # Get indices of valid rows
        if len(valid_global_annotation_indices) > 0:
            global_annotation_context = self.context_global_annotation(h[valid_global_annotation_indices], global_annotation_emb[valid_global_annotation_indices], nonbeospank[valid_global_annotation_indices], nonbeospank_global_annotation[valid_global_annotation_indices])
            h[valid_global_annotation_indices] = h[valid_global_annotation_indices] + global_annotation_context

        # Per-residue annotation context
        # Simple [B, L] mask: True where at least one of K positions is valid (non-BOS/EOS/PAD/UNK)
        nonbeospank_per_residue_annotation_simplified = (~nonbeospank_per_residue_annotation).any(dim=2)  # [B, L]
        valid_per_residue_annotation_indices = nonbeospank_per_residue_annotation_simplified.any(dim=1).nonzero(as_tuple=True)[0]  # Get indices of valid rows
        if len(valid_per_residue_annotation_indices) > 0:
            per_residue_annotation_context = self.context_per_residue_annotation(h[valid_per_residue_annotation_indices], per_residue_annotation_emb[valid_per_residue_annotation_indices], nonbeospank[valid_per_residue_annotation_indices], nonbeospank_per_residue_annotation_simplified[valid_per_residue_annotation_indices])
            h[valid_per_residue_annotation_indices] = h[valid_per_residue_annotation_indices] + per_residue_annotation_context

        # Inject pLDDT context
        # Only apply context injection to rows that have at least one valid context position
        valid_plddt_indices = nonbeospank_plddt.any(dim=1).nonzero(as_tuple=True)[0]  # Get indices of valid rows
        if len(valid_plddt_indices) > 0:
            plddt_context = self.context_plddt(h[valid_plddt_indices], plddt_emb[valid_plddt_indices], nonbeospank[valid_plddt_indices], nonbeospank_plddt[valid_plddt_indices])
            h[valid_plddt_indices] = h[valid_plddt_indices] + plddt_context

        # Get time embeddings if using AdaLN
        time_emb = None
        if self.use_adaln:
            assert timesteps is not None, "timesteps required when use_adaln=True"
            time_emb = self.time_embed(timesteps)  # [B, time_embed_dim]

        # --- Transformer trunk ------------------------------------------------
        # Pass through all transformer blocks
        model_type = self.cfg.first_block_cfg.initials()
        for block in self.layers:
            if model_type in ("GA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for geometric first layer"
                h = block(h, coords[:,:,:3,:], content_elements, nonbeospank, time_emb)
            elif model_type in ("RA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for reflexive first layer"
                h = block(h, coords, content_elements, nonbeospank, time_emb)
            else:
                # Standard blocks don't need coordinates
                h = block(h, nonbeospank=nonbeospank, time_emb=time_emb)

        if self.use_adaln:
            h = self.final_norm(h, time_emb)  # [B, L, d]
        else:
            h = self.final_norm(h)

        return self.seq_logits(h), self.struct_logits(h)
