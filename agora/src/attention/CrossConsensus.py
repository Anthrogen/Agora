import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding

class CrossConsensus(nn.Module):
    """
    Multi-head cross-consensus layer - consensus between target and context sequences
    
    Cross-consensus refines target representations by iteratively aligning them
    with context representations through learnable consensus updates.
    
    Key design:
    • Multi-head architecture: consensus operates independently in each head subspace
    • Bipartite connectivity: edges only between target and context sequences
    • Target-only updates: only target sequence is refined, context stays fixed
    • Configurable connectivity: scored_window or local_window (if sequences aligned)
    • Per-edge learnable matrices R_ij = alpha_ij I + Lambda_ij Lambda_ij^T
    • Learnable step sizes eta^(t) for each iteration (shared across heads)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,  # Number of attention heads
        dropout: float = 0.1,
        num_iterations: int = 2, # Number of consensus gradient iterations
        connectivity_type: str = "scored_window",  # "scored_window" or "local_window"
        w: int = 4,  # Window size for local_window, or w value for scored_window mode
        r: int = 8,  # Rank of Lambda_ij matrices
        edge_hidden_dim: int = 16,
        max_len: int = 2048,  # Maximum sequence length
    ):
        """
        Args:
            dim: Model dimension
            heads: Number of attention heads
            num_iterations: Number of consensus gradient iterations to unroll
            connectivity_type: Type of connectivity pattern ("scored_window" or "local_window")
            w: Window size for local_window mode, or w value for scored_window mode
            r: Rank of Lambda_ij in R_ij = alpha_ij I + Lambda_ij Lambda_ij^T
            edge_hidden_dim: Hidden dimension for edge parameter networks
            max_len: Maximum sequence length (for step size initialization)
        """
        super().__init__()
        
        if connectivity_type not in ["local_window", "scored_window"]:
            raise ValueError(f"connectivity_type must be 'local_window' or 'scored_window', got '{connectivity_type}'")
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "Dimension must be divisible by number of heads"
        
        self.num_iterations = num_iterations
        self.connectivity_type = connectivity_type
        self.w = w
        self.r = r
        
        # Target and context encoders
        self.target_encoder = nn.Linear(dim, dim)
        self.context_encoder = nn.Linear(dim, dim)
        
        # Rotary position embeddings
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_len)
        
        # For scored_window connectivity, we need similarity scores
        if connectivity_type == "scored_window":
            self.similarity_scorer = nn.Sequential(
                nn.Linear(2 * dim, edge_hidden_dim),
                nn.GELU(),
                nn.Linear(edge_hidden_dim, 1)
            )
        
        # Edge parameter networks
        # α_ij scalar per edge
        self.edge_to_alpha = nn.Sequential(
            nn.Linear(2 * dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, 1),
            nn.Softplus()
        )
        
        if self.r > 0:
            # Lambda_ij matrix per edge (rank × head_dim) for each head
            self.edge_to_Lambda = nn.Sequential(
                nn.Linear(2 * dim, edge_hidden_dim),
                nn.GELU(),
                nn.Linear(edge_hidden_dim, heads * r * self.head_dim)
            )
        
        # Learnable step sizes η(t,i) for each iteration and residue (shared across heads)
        # Initialize such that softplus(x) ≈ 0.1, which means x ≈ log(exp(0.1) - 1) ≈ -2.2
        self.step_sizes = nn.Parameter(torch.ones(num_iterations, max_len) * -2.2)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def get_scored_window_connectivity(self, target, context, B, L, K, device, target_mask=None, context_mask=None):
        """Get scored_window connectivity pattern based on similarity scores."""
        # Compute all pairwise features between target and context
        target_expanded = target.unsqueeze(2)  # [B, L, 1, D]
        context_expanded = context.unsqueeze(1)  # [B, 1, K, D]
        
        # Expand and concatenate
        target_tiled = target_expanded.expand(-1, -1, K, -1)  # [B, L, K, D]
        context_tiled = context_expanded.expand(-1, L, -1, -1)  # [B, L, K, D]
        pair_features = torch.cat([target_tiled, context_tiled], dim=-1)  # [B, L, K, 2*D]
        
        # Compute similarity scores for all pairs at once
        scores = self.similarity_scorer(pair_features).squeeze(-1)  # [B, L, K]
        
        # Apply validity masking if provided
        if target_mask is not None and context_mask is not None:
            # Mask out connections involving invalid positions
            valid_mask = target_mask.unsqueeze(2) & context_mask.unsqueeze(1)  # [B, L, K]
            scores.masked_fill_(~valid_mask, float('-inf'))
        
        # Get scored_window indices for each target position
        _, scored_window_indices = torch.topk(scores, min(self.w, K), dim=-1)  # [B, L, w]
        
        # Create source indices (each target position repeated w times)
        source_indices = torch.arange(L, device=device).view(1, L, 1).expand(B, -1, min(self.w, K))
        
        # Flatten to create edge lists
        edge_i = source_indices.reshape(B, -1)  # [B, L*w]
        edge_j = scored_window_indices.reshape(B, -1)   # [B, L*w]
        
        # Create edge validity mask: E_masked = {(i,j) ∈ E | target_i = 1 ∧ context_j = 1}
        if target_mask is not None and context_mask is not None: edge_valid = target_mask.gather(1, edge_i) & context_mask.gather(1, edge_j)  # [B, L*w]
        else: edge_valid = None
        
        return edge_i, edge_j, edge_valid
    
    def get_local_window_edges(self, B, L, K, device, target_mask=None, context_mask=None):
        """Get local window connectivity pattern (only valid if sequences are aligned)."""
        window = self.w
        
        # For each target position, connect to context positions within window
        target_positions = torch.arange(L, device=device)
        
        # Create offset pattern (excluding 0 for self-connections if L==K)
        offsets = torch.arange(-window, window + 1, device=device)
        if L == K:
            offsets = offsets[offsets != 0]  # Remove self-connections
        
        # Broadcast to get all (target_pos, offset) pairs
        pos_expanded = target_positions.unsqueeze(1)  # [L, 1]
        offsets_expanded = offsets.unsqueeze(0)  # [1, num_offsets]
        
        # Compute context positions
        context_positions = pos_expanded + offsets_expanded  # [L, num_offsets]
        
        # Create validity mask for in-bounds edges
        valid_mask = (context_positions >= 0) & (context_positions < K)  # [L, num_offsets]
        
        # Get valid edges using masking
        valid_indices = valid_mask.nonzero(as_tuple=False)  # [num_valid_edges, 2]
        
        # Extract source and target indices
        edge_i = valid_indices[:, 0]  # Target positions
        edge_j = context_positions[valid_mask]   # Context positions
        
        # Expand for batch dimension
        edge_i = edge_i.unsqueeze(0).expand(B, -1)  # [B, num_edges]
        edge_j = edge_j.unsqueeze(0).expand(B, -1)  # [B, num_edges]
        
        # Create edge validity mask: E_masked = {(i,j) ∈ E | target_i = 1 ∧ context_j = 1}
        if target_mask is not None and context_mask is not None: edge_valid = target_mask.gather(1, edge_i) & context_mask.gather(1, edge_j)  # [B, num_edges]
        else: edge_valid = None
        
        return edge_i, edge_j, edge_valid
    
    def get_connectivity(self, target, context, B, L, K, device, target_mask=None, context_mask=None):
        """Get connectivity pattern based on configured type."""
        if self.connectivity_type == "scored_window":
            return self.get_scored_window_connectivity(target, context, B, L, K, device, target_mask, context_mask)
        elif self.connectivity_type == "local_window":
            return self.get_local_window_edges(B, L, K, device, target_mask, context_mask)
        else:
            raise ValueError(f"Unknown connectivity type: {self.connectivity_type}")
    
    def compute_edge_params(self, target, context, edge_i, edge_j):
        """Compute edge-specific restriction parameters."""
        B = target.size(0)
        
        # Gather target and context features for edges
        target_features = target.gather(1, edge_i.unsqueeze(-1).expand(-1, -1, self.dim))  # [B, E, D]
        context_features = context.gather(1, edge_j.unsqueeze(-1).expand(-1, -1, self.dim))  # [B, E, D]
        
        # Compute edge encodings
        edge_features = torch.cat([target_features, context_features], dim=-1)  # [B, E, 2D]
        
        # Compute alpha and Lambda separately
        alphas = self.edge_to_alpha(edge_features).squeeze(-1)  # [B, E]

        if self.r > 0:
            Lambda_flat = self.edge_to_Lambda(edge_features)  # [B, E, heads*r*head_dim]
            Lambda = Lambda_flat.view(B, -1, self.heads, self.r, self.head_dim)  # [B, E, heads, r, head_dim]
            # Normalize Lambda columns
            Lambda = F.normalize(Lambda, dim=-1)
            return alphas, Lambda
        else:
            return alphas
    
    def consensus_step(self, u, v, edge_i, edge_j, alphas, step_sizes, Lambda=None, edge_valid=None):
        """One cross-consensus gradient update step."""
        B, H, L, D = u.shape  # H = heads
        _, _, K, _ = v.shape
        
        # Gather u and v values at edge endpoints
        u_i = u.gather(2, edge_i.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D))  # [B, H, E, D]
        v_j = v.gather(2, edge_j.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D))  # [B, H, E, D]
        
        # Apply RoPE rotation to source (target) nodes
        u_i = self.rope(u_i=u_i, model_type="SC", edge_i=edge_i, edge_j=edge_j)  # [B, H, E, D]
        
        # Compute difference (target - context)
        diff = u_i - v_j  # [B, H, E, D]
        
        # Apply restriction: R_ij @ diff
        # R_ij = alpha_ij * I + Lambda_ij @ Lambda_ij^T
        if self.r > 0:
            Lambda_diff = torch.einsum('behrd,bhed->bher', Lambda, diff)  # [B, H, E, r]
            residual = alphas.unsqueeze(1).unsqueeze(-1) * diff + torch.einsum('bher,behrd->bhed', Lambda_diff, Lambda)  # [B, H, E, D]
        else:
            residual = alphas.unsqueeze(1).unsqueeze(-1) * diff
        
        # Zero out residuals for invalid edges (implements E_masked filtering)
        if edge_valid is not None: residual = residual * edge_valid.unsqueeze(1).unsqueeze(-1)
        
        # Accumulate updates at target vertices only (cross-consensus)
        vertex_updates = torch.zeros_like(u)
        # Ensure residual matches vertex_updates dtype for scatter_add_ (fp16 compatibility)
        vertex_updates.scatter_add_(2, edge_i.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D), residual.to(vertex_updates.dtype))
        
        # Cross-consensus update with shared step sizes across heads
        u = u - step_sizes * vertex_updates
        
        return u
    
    def forward(self, target, context, target_mask=None, context_mask=None):
        """
        Args:
            target: [B, L, dim] target sequence features
            context: [B, K, dim] context sequence features
            target_mask: [B, L] boolean tensor where True = valid, False = invalid/padding
            context_mask: [B, K] boolean tensor where True = valid, False = invalid/padding

        Returns:
            y: [B, L, dim] refined target features
        """
        B, L, D = target.shape
        _, K, _ = context.shape
        device = target.device
        
        # Initial encoding
        u = self.target_encoder(target)  # [B, L, dim]
        v = self.context_encoder(context)  # [B, K, dim]
        
        # Reshape to multi-head format
        u = u.reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        v = v.reshape(B, K, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, K, head_dim]
        
        # Get connectivity pattern with edge filtering based on masks
        edge_i, edge_j, edge_valid = self.get_connectivity(target, context, B, L, K, device, target_mask, context_mask)
        
        # Compute edge parameters
        if self.r > 0:
            alphas, Lambda = self.compute_edge_params(target, context, edge_i, edge_j)
        
            # Unrolled cross-consensus gradient updates
            for t in range(self.num_iterations):
                # Get step sizes for this iteration (shared across heads)
                step_sizes_t = F.softplus(self.step_sizes[t].unsqueeze(0).unsqueeze(0).unsqueeze(-1))  # [1, 1, L, 1]
                step_sizes_t = step_sizes_t.expand(B, self.heads, -1, -1)  # [B, heads, L, 1]
                u = self.consensus_step(u, v, edge_i, edge_j, alphas, step_sizes_t, Lambda, edge_valid)
        
        else:
            alphas = self.compute_edge_params(target, context, edge_i, edge_j)
            for t in range(self.num_iterations):
                # Get step sizes for this iteration (shared across heads)
                step_sizes_t = F.softplus(self.step_sizes[t].unsqueeze(0).unsqueeze(0).unsqueeze(-1))  # [1, 1, L, 1]
                step_sizes_t = step_sizes_t.expand(B, self.heads, -1, -1)  # [B, heads, L, 1]
                u = self.consensus_step(u, v, edge_i, edge_j, alphas, step_sizes_t, None, edge_valid)
        
        # Reshape back to original format
        u = u.permute(0, 2, 1, 3).reshape(B, L, D)  # [B, L, dim]
            
        # Output projection
        y = self.out_proj(u)
        y = self.dropout(y)
        
        return y 