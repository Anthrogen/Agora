from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import matplotlib.pyplot as plt
from src.quantizer import Quantizer

from src.models.blocks import ConvBlock, StandardTransformerBlock, GeometricTransformerBlock, ReflexiveTransformerBlock, ConsensusTransformerBlock

class FSQEncoder(nn.Module):
    """
    The encoder part of the FSQ autoencoder.
    Maps from backbone coordinates to discrete tokens.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # Store config for access in forward
        in_dim = 3 * 3  # N, CA, C, CB each have (x,y,z)
        
        # Encoder
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        
        # Convolutional blocks (operate on channel dimension)
        self.encoder_conv1 = ConvBlock(cfg.d_model)
        self.encoder_conv2 = ConvBlock(cfg.d_model)
        
        # Transformer stack
        self.layers = nn.ModuleList()
        
        # Transformer blocks
        if cfg.model_type == "GA":
            self.layers.append(GeometricTransformerBlock(cfg))
        elif cfg.model_type == "SA":
            self.layers.append(StandardTransformerBlock(cfg))
        elif cfg.model_type == "RA":
            self.layers.append(ReflexiveTransformerBlock(cfg))
        elif cfg.model_type == "C":
            self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            raise ValueError(f"Invalid model_type type: {cfg.model_type}")
        
        # Remaining blocks
        if cfg.model_type == "C":
            # For Consensus, all blocks are ConsensusTransformerBlocks
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
        
        # FSQ Quantizer
        # Using custom level structure: [7, 5, 5, 5, 5]
        # Codebook size = 7 × 5 × 5 × 5 × 5 = 4,375 discrete codes
        self.quantizer = Quantizer(levels=cfg.fsq_levels, dim=cfg.fsq_dim, scale=1)
        
    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,  # [B, L, 3, 3] or [B, L, 4, 3] for RA
        coord_mask: Optional[torch.Tensor] = None  # [B, L] boolean mask
        ):
        """
        Args:
          x: [B, L, 3, 3]
        Returns:
          z_q: [B, L, fsq_dim], indices: [B, L] discrete codes
        """
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
        for block in self.layers:
            if self.cfg.model_type in ("GA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for geometric first layer"
                h = block(h, coords[:,:,:3,:], coord_mask)
            elif self.cfg.model_type in ("RA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for reflexive first layer"
                h = block(h, coords, coord_mask)
            else:
                # Standard blocks don't need coordinates
                h = block(h)
        
        # Project to FSQ dimension
        z = self.encoder_proj(h)            # [B, L, fsq_dim]
        
        # Quantize
        z_q, indices = self.quantizer(z)    # [B, L, fsq_dim], [B, L]
        
        return z_q, indices
    
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for tokenization that returns only the indices.
        
        Args:
            x: [B, L, 3, 3] backbone coordinates
        Returns:
            indices: [B, L] discrete token indices
        """
        _, indices = self.forward(x)
        return indices

class FSQDecoder(nn.Module):
    """
    The decoder part of the FSQ autoencoder.
    Maps from quantized vectors back to backbone coordinates.
    """
    def __init__(self, cfg, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.cfg = cfg
        
        # Determine input dimension for decoder_input layer
        # Stage 2: fsq_dim + 1 (concatenated z_q and seq_tokens)
        # Stage 1: fsq_dim (just z_q)
        if cfg.stage == "stage_2":
            decoder_input_dim = cfg.fsq_dim + 1
        else:
            decoder_input_dim = cfg.fsq_dim
        
        # Decoder
        self.decoder_input = nn.Linear(decoder_input_dim, cfg.d_model)

        # Transformer stack
        self.layers = nn.ModuleList()
        
        # Transformer blocks
        if cfg.model_type == "GA":
            self.layers.append(GeometricTransformerBlock(cfg))
        elif cfg.model_type == "SA":
            self.layers.append(StandardTransformerBlock(cfg))
        elif cfg.model_type == "RA":
            self.layers.append(ReflexiveTransformerBlock(cfg))
        elif cfg.model_type == "C":
            self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            raise ValueError(f"Invalid model_type type: {cfg.model_type}")
        
        # Remaining blocks
        if cfg.model_type == "C":
            # For Consensus, all blocks are ConsensusTransformerBlocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(ConsensusTransformerBlock(cfg))
        else:
            # For GA/SA/RA, remaining blocks are all standard transformer blocks
            for _ in range(cfg.n_layers - 1):
                self.layers.append(StandardTransformerBlock(cfg))

        # Convolutional blocks
        self.decoder_conv1 = ConvBlock(cfg.d_model)
        self.decoder_conv2 = ConvBlock(cfg.d_model)
        
        # Output projection - uses custom out_dim
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, out_dim)
        )
        
    def forward(
        self,
        z_q: torch.Tensor,  # [B, L, fsq_dim] quantized vectors
        coords: Optional[torch.Tensor] = None,  # [B, L, 3, 3] or [B, L, 4, 3] for RA
        coord_mask: Optional[torch.Tensor] = None,  # [B, L] boolean mask
    ) -> torch.Tensor:
        """
        Args:
          z_q: [B, L, fsq_dim] quantized vectors
          coords: Optional coordinates for GA/RA models
          coord_mask: Optional mask for GA/RA models
        Returns:
          x_rec: [B, L, num_atoms, 3] reconstructed coordinates
                 where num_atoms = in_dim // 3
        """
        B, L, _ = z_q.shape
        
        # Decoder: from quantized z to d_model
        h = self.decoder_input(z_q)         # [B, L, d_model]

        # Pass through all transformer blocks
        for block in self.layers:
            if self.cfg.model_type in ("GA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for geometric first layer"
                h = block(h, coords[:,:,:3,:], coord_mask)
            elif self.cfg.model_type in ("RA") and block is self.layers[0]:
                # First block may require coordinates
                assert coords is not None, "Coordinates required for reflexive first layer"
                h = block(h, coords, coord_mask)
            else:
                # Standard blocks don't need coordinates
                h = block(h)
        
        # Convolutional blocks
        h_conv = h.transpose(1, 2)          # [B, d_model, L]
        h_conv = self.decoder_conv1(h_conv) # [B, d_model, L]
        h_conv = self.decoder_conv2(h_conv) # [B, d_model, L]
        h = h_conv.transpose(1, 2)          # [B, L, d_model]
        
        # Project to output dimensions
        out = self.output_proj(h)           # [B, L, in_dim]
        
        # Reshape based on in_dim
        num_atoms = self.out_dim // 3
        x_rec = out.view(B, L, num_atoms, 3)  # [B, L, num_atoms, 3]
        
        return x_rec
        
    def decode_from_tokens(self, indices: torch.Tensor, quantizer: FSQ):
        """
        Convenience method to decode directly from token indices.
        
        Args:
          indices: [B, L] discrete token indices
          quantizer: FSQ quantizer instance to map indices to embeddings
        Returns:
          x_rec: [B, L, 3, 3] reconstructed coordinates
        """
        # Convert indices to embeddings
        z_q = quantizer.embed(indices)
        return self.forward(z_q)

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
        
        # Create encoder and decoder
        self.encoder = FSQEncoder(cfg)
        
        # Create decoder with appropriate in_dim based on stage
        if cfg.stage == "stage_2":
            # Stage 2: decoder outputs 14 atoms
            decoder_out_dim = 14 * 3
        else:
            # Stage 1: decoder outputs 3 atoms
            decoder_out_dim = 3 * 3
            
        self.decoder = FSQDecoder(cfg, out_dim=decoder_out_dim)
        
        # Expose quantizer for easier access
        self.quantizer = self.encoder.quantizer

    def forward(self, x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,  # [B, L, 3, 3] or [B, L, 4, 3] for GA/RA
        coord_mask: Optional[torch.Tensor] = None,  # [B, L] boolean mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B, L, 3, 3] - main coordinate input
          coords: Optional additional coordinates for GA/RA models
          coord_mask: Optional mask for GA/RA models
        Returns:
          x_rec: [B, L, num_atoms, 3] - reconstructed coordinates (num_atoms depends on stage)
          indices: [B, L] discrete codes
        """
        # Encode
        z_q, indices = self.encoder(x, coords, coord_mask)
        
        # Decode
        x_rec = self.decoder(z_q, coords, coord_mask)
        
        return x_rec, indices 
