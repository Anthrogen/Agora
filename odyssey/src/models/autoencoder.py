from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import matplotlib.pyplot as plt
from odyssey.src.quantizer import Quantizer

from odyssey.src.models.blocks import ConvBlock, StandardTransformerBlock, GeometricTransformerBlock, ReflexiveTransformerBlock, ConsensusTransformerBlock
from odyssey.src.attention.CrossAttention import CrossAttention
from odyssey.src.attention.CrossConsensus import CrossConsensus

class FSQEncoder(nn.Module):
    """
    The encoder part of the FSQ autoencoder.
    Maps from backbone coordinates to discrete tokens.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # Store config for access in forward
        self.input_num_atoms = 3
        in_dim = self.input_num_atoms * 3  # N, CA, C, CB each have (x,y,z)
        
        # Encoder
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        
        # Convolutional blocks (operate on channel dimension)
        self.encoder_conv1 = ConvBlock(cfg.d_model)
        self.encoder_conv2 = ConvBlock(cfg.d_model)
        
        # Transformer stack
        self.layers = nn.ModuleList()
        
        # Transformer blocks
        model_type = cfg.first_block_cfg.initials()
        if model_type == "GA":
            self.layers.append(GeometricTransformerBlock(cfg))
        elif model_type == "SA":
            self.layers.append(StandardTransformerBlock(cfg))
        elif model_type == "RA":
            self.layers.append(ReflexiveTransformerBlock(cfg))
        elif model_type == "SC":
            self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            raise ValueError(f"Invalid model_type type: {model_type}")
        
        # Remaining blocks
        if model_type == "SC":
            # For SelfConsensus, all blocks are ConsensusTransformerBlocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            # For GA/SA/RA, remaining blocks are all standard transformer blocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(StandardTransformerBlock(cfg))
        
        # Project to FSQ dimension
        self.encoder_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.fsq_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,  # [B, L, 3, 3] or [B, L, 4, 3] for RA
        content_elements: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        ):
        """
        Args:
          x: [B, L, 3, 3]
        Returns:
          z_q: [B, L, fsq_dim], indices: [B, L] discrete codes
        """
        assert content_elements is None or content_elements.dtype == torch.bool
        assert nonbeospank is None or nonbeospank.dtype == torch.bool
        B, L, _, _ = x.shape
        
        # Flatten per residue to 9-dim and project to d_model
        x_flat = x.view(B, L, -1)           # [B, L, 9]
        h = self.input_proj(x_flat)         # [B, L, d_model]
        
        # Convolutional blocks (transpose for conv1d)
        h_conv = h.transpose(1, 2)          # [B, d_model, L]
        h_conv = self.encoder_conv1(h_conv) # [B, d_model, L]
        h_conv = self.encoder_conv2(h_conv) # [B, d_model, L]
        h = h_conv.transpose(1, 2)          # [B, L, d_model]
        
        # Pass through all transformer blocks
        model_type = self.cfg.first_block_cfg.initials()
        for block in self.layers:
            if model_type in ("GA") and block is self.layers[0]:
                assert coords is not None, "Coordinates required for geometric first layer"
                h = block(h, coords[:,:,:3,:], content_elements, nonbeospank)
            elif model_type in ("RA") and block is self.layers[0]:
                assert coords is not None, "Coordinates required for reflexive first layer"
                h = block(h, coords, content_elements, nonbeospank)
            else:
                # Standard blocks don't need coordinates
                h = block(h, nonbeospank=nonbeospank)
        
        # Project to FSQ dimension
        z = self.encoder_proj(h)            # [B, L, fsq_dim]
        
        return z


class FSQDecoder(nn.Module):
    """
    The decoder part of the FSQ autoencoder.
    Maps from quantized vectors back to backbone coordinates.
    """
    def __init__(self, cfg, out_dim, use_seq_context=False, seq_vocab=None):
        super().__init__()
        self.out_dim = out_dim
        self.cfg = cfg
        
        # Decoder projection - symmetric to encoder_proj
        # fsq_dim -> latent_dim -> d_model (reverse of encoder_proj)
        self.decoder_proj = nn.Sequential(
            nn.Linear(cfg.fsq_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.d_model)
        )
        
        # Context sequence for stage 2
        if use_seq_context:
            assert cfg.context_cfg is not None, "Context configuration is required when use_seq_context=True"
            assert seq_vocab is not None, "seq_vocab must be provided when use_seq_context=True"
            # Sequence embedding layer for context sequence
            self.seq_embed = nn.Embedding(seq_vocab, cfg.d_model)
            
            # Initialize context sequence based on configuration
            context_type = cfg.context_cfg.initials()
            if context_type == "CA":  # CrossAttention
                self.context_sequence = CrossAttention(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, max_position_embeddings=cfg.max_len)
            elif context_type == "CC":  # CrossConsensus
                self.context_sequence = CrossConsensus(dim=cfg.d_model, heads=cfg.n_heads, dropout=cfg.dropout, num_iterations=cfg.context_cfg.consensus_num_iterations,
                                                        connectivity_type=cfg.context_cfg.consensus_connectivity_type, w=cfg.context_cfg.consensus_w, 
                                                        r=cfg.context_cfg.consensus_r, edge_hidden_dim=cfg.context_cfg.consensus_edge_hidden_dim, max_len=cfg.max_len)
            else: raise ValueError(f"Unknown context type: {context_type}")
        else: self.context_sequence = None

        # Transformer stack
        self.layers = nn.ModuleList()
        
        # Transformer blocks
        model_type = cfg.first_block_cfg.initials()
        if model_type == "GA":
            self.layers.append(GeometricTransformerBlock(cfg))
        elif model_type == "SA":
            self.layers.append(StandardTransformerBlock(cfg))
        elif model_type == "RA":
            self.layers.append(ReflexiveTransformerBlock(cfg))
        elif model_type == "SC":
            self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            raise ValueError(f"Invalid model_type type: {model_type}")
        
        # Remaining blocks
        if model_type == "SC":
            # For SelfConsensus, all blocks are ConsensusTransformerBlocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            # For GA/SA/RA, remaining blocks are all standard transformer blocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(StandardTransformerBlock(cfg))

        # Convolutional blocks
        self.decoder_conv1 = ConvBlock(cfg.d_model)
        self.decoder_conv2 = ConvBlock(cfg.d_model)
        
        # Output projection - symmetric to input_proj
        # d_model -> coordinates (no latent_dim intermediate)
        self.output_proj = nn.Linear(cfg.d_model, out_dim)
        
    def forward(
        self,
        z_q: torch.Tensor,  # [B, L, fsq_dim] quantized vectors
        coords: Optional[torch.Tensor] = None,  # [B, L, 3, 3] or [B, L, 4, 3] for RA
        content_elements: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        seq_tokens: Optional[torch.Tensor] = None,  # [B, L] sequence tokens for stage 2 context sequence
    ) -> torch.Tensor:
        """
        Args:
          z_q: [B, L, fsq_dim] quantized vectors
          coords: Optional coordinates for GA/RA models
          content_elements: Optional mask for GA/RA models
          nonbeospank: Optional mask for GA/RA models
          seq_tokens: Optional sequence tokens for stage 2 context sequence
        Returns:
          x_rec: [B, L, num_atoms, 3] reconstructed coordinates
                 where num_atoms = in_dim // 3
        """
        B, L, _ = z_q.shape
        
        # Decoder projection: from quantized z to d_model (symmetric to encoder_proj)
        h = self.decoder_proj(z_q)         # [B, L, d_model]
        
        # Context sequence for stage 2
        if self.context_sequence is not None and seq_tokens is not None:
            # Embed sequence tokens
            seq_emb = self.seq_embed(seq_tokens)  # [B, L, d_model]
            # Apply context sequence: z_q features as target, sequence as context
            h = h + self.context_sequence(target=h, context=seq_emb, target_mask=nonbeospank, context_mask=nonbeospank)

        # Pass through all transformer blocks
        model_type = self.cfg.first_block_cfg.initials()
        for block in self.layers:
            if model_type in ("GA") and block is self.layers[0]:
                assert coords is not None, "Coordinates required for geometric first layer"
                h = block(h, coords[:,:,:3,:], content_elements, nonbeospank)
            elif model_type in ("RA") and block is self.layers[0]:
                assert coords is not None, "Coordinates required for reflexive first layer"
                h = block(h, coords, content_elements, nonbeospank)
            else:
                # Standard blocks don't need coordinates
                h = block(h, nonbeospank=nonbeospank)
        
        # Convolutional blocks
        h_conv = h.transpose(1, 2)          # [B, d_model, L]
        h_conv = self.decoder_conv1(h_conv) # [B, d_model, L]
        h_conv = self.decoder_conv2(h_conv) # [B, d_model, L]
        h = h_conv.transpose(1, 2)          # [B, L, d_model]
        
        # Output projection: d_model -> coordinates (symmetric to input_proj)
        out = self.output_proj(h)           # [B, L, out_dim]
        
        # Reshape based to coordinate format
        num_atoms = self.out_dim // 3
        x_rec = out.view(B, L, num_atoms, 3)  # [B, L, num_atoms, 3]
        
        return x_rec

# --------------------------------------------------------------------------- #
#  Top-level model                                                             #
# --------------------------------------------------------------------------- #
class Autoencoder(nn.Module):
    """
    An enhanced per-residue autoencoder with FSQ quantization bottleneck.
      - encoder: Conv1D → Conv1D → Transformer → Transformer → Linear to bottleneck
      - quantizer: FSQ on bottleneck dims
      - decoder: Linear → Transformer → Transformer → Conv1D → Conv1D → Output
    """

    def __init__(self, cfg):
        super().__init__()
        # Store config
        self.cfg = cfg
        
        # Create encoder and decoder using their sub-configs directly
        self.encoder = FSQEncoder(cfg.encoder_cfg)
        
        # Create decoder with appropriate in_dim based on stage
        if cfg.style == "stage_2": decoder_out_dim = 14 * 3 # Stage 2: decoder outputs 14 atoms
        else: decoder_out_dim = 3 * 3 # Stage 1: decoder outputs 3 atoms
            
        self.decoder = FSQDecoder(cfg.decoder_cfg, out_dim=decoder_out_dim, use_seq_context=(cfg.style == "stage_2"), seq_vocab=cfg.seq_vocab)

        # FSQ Quantizer: Using custom level structure from encoder config
        self.quantizer = Quantizer(levels=cfg.encoder_cfg.fsq_levels, dim=cfg.encoder_cfg.fsq_dim, scale=1)
        self.codebook_size = math.prod(cfg.encoder_cfg.fsq_levels)

    def encode_to_tokens(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None, content_elements: Optional[torch.Tensor] = None, nonbeospank: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method for tokenization that returns only the indices.
        
        Args:
            x: [B, L, 3, 3] backbone coordinates
            coords: Optional additional coordinates for GA/RA models
            content_elements: Optional mask for GA/RA models
            nonbeospank: Optional mask for GA/RA models
        Returns:
            indices: [B, L] discrete token indices
        """
        z = self.encoder(x, coords, content_elements, nonbeospank)
        _, indices = self.quantizer(z)
        return indices

    def forward(self, x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,  # [B, L, 3, 3] or [B, L, 4, 3] for GA/RA
        content_elements: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        nonbeospank: Optional[torch.Tensor] = None,  # [B, L] boolean mask
        seq_tokens: Optional[torch.Tensor] = None,  # [B, L] sequence tokens for stage 2 context sequence
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B, L, 3, 3] - main coordinate input
          coords: Optional additional coordinates for GA/RA models
          content_elements: Optional mask for GA/RA models
          nonbeospank: Optional mask for GA/RA/SA/SC models
          seq_tokens: Optional sequence tokens for stage 2 context sequence
        Returns:
          x_rec: [B, L, num_atoms, 3] - reconstructed coordinates (num_atoms depends on stage)
          indices: [B, L] discrete codes
        """
        # Encode
        z = self.encoder(x, coords, content_elements, nonbeospank)

        # Quantize
        z_q, indices = self.quantizer(z)    # [B, L, fsq_dim], [B, L]
        
        # Decode
        x_rec = self.decoder(z_q, coords, content_elements, nonbeospank, seq_tokens)
        
        return x_rec, indices 
