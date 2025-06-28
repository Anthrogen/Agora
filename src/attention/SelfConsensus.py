import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding

class SelfConsensus(nn.Module):
    """
    Multi-head self-consensus layer - unrolled learnable self-consensus with configurable connectivity
    
    Fully differentiable consensus solver using
    unrolled consensus gradient updates with learnable components.
    
    Key design:
    • Multi-head architecture: consensus operates independently in each head subspace
    • Configurable connectivity: top-w or local window
    • Per-edge learnable matrices R_ij = alpha_ij I + Lambda_ij Lambda_ij^T
    • Learnable step sizes eta^(t) for each iteration (shared across heads)
    • Complexity varies by connectivity type
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,  # Number of attention heads
        dropout: float = 0.1,
        num_iterations: int = 2, # Number of consensus gradient iterations
        connectivity_type: str = "local_window",  # "top_w" or "local_window"
        w: int = 4,  # Window size for local_window, or w value for top_w
        r: int = 8,  # Rank of Lambda_ij matrices
        edge_hidden_dim: int = 16,
        max_len: int = 2048,  # Maximum sequence length
    ):
        """
        Args:
            dim: Model dimension
            heads: Number of attention heads
            num_iterations: Number of consensus gradient iterations to unroll
            connectivity_type: Type of connectivity pattern ("top_w" or "local_window")
            w: Window size for local_window mode, or w value for top_w mode
            r: Rank of Lambda_ij in R_ij = alpha_ij I + Lambda_ij Lambda_ij^T
            edge_hidden_dim: Hidden dimension for edge parameter networks
            max_len: Maximum sequence length (for step size initialization)
        """
        super().__init__()
        
        if connectivity_type not in ["local_window", "top_w"]:
            raise ValueError(f"connectivity_type must be 'local_window' or 'top_w', got '{connectivity_type}'")
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "Dimension must be divisible by number of heads"
        
        self.num_iterations = num_iterations
        self.connectivity_type = connectivity_type
        self.w = w
        self.r = r
        
        # Initial token encoding
        self.token_encoder = nn.Linear(dim, dim)
        
        # Rotary position embeddings
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_len)
        
        # For top-w connectivity, we need similarity scores
        if connectivity_type == "top_w":
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
    
    def get_top_w_connectivity(self, x, B, L, device):
        """Get top-w connectivity pattern based on similarity scores."""
        # Compute all pairwise features at once
        x_i = x.unsqueeze(2)  # [B, L, 1, D]
        x_j = x.unsqueeze(1)  # [B, 1, L, D]
        
        # Expand and concatenate
        x_i_expanded = x_i.expand(-1, -1, L, -1)  # [B, L, L, D]
        x_j_expanded = x_j.expand(-1, L, -1, -1)  # [B, L, L, D]
        pair_features = torch.cat([x_i_expanded, x_j_expanded], dim=-1)  # [B, L, L, 2*D]
        
        # Compute similarity scores for all pairs at once
        scores = self.similarity_scorer(pair_features).squeeze(-1)  # [B, L, L]
        
        # Mask out self-connections
        mask = torch.eye(L, device=device, dtype=torch.bool)
        scores.masked_fill_(mask, float('-inf'))
        
        # Get top-w indices for each position
        _, top_w_indices = torch.topk(scores, self.w, dim=-1)  # [B, L, w]
        
        # Create source indices (each position repeated w times)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, L, self.w)
        source_indices = torch.arange(L, device=device).view(1, L, 1).expand(B, -1, self.w)
        
        # Flatten to create edge lists
        edge_i = source_indices.reshape(B, -1)  # [B, L*w]
        edge_j = top_w_indices.reshape(B, -1)   # [B, L*w]
        
        return edge_i, edge_j
    
    def get_local_window_edges(self, B, L, device):
        """Get local window connectivity pattern."""
        window = self.w
        
        # Create all possible position indices
        positions = torch.arange(L, device=device)
        
        # Create offset pattern (excluding 0 for self-connections)
        offsets = torch.arange(-window, window + 1, device=device)
        offsets = offsets[offsets != 0]  # Remove self-connections
        
        # Broadcast to get all (position, offset) pairs
        pos_expanded = positions.unsqueeze(1)  # [L, 1]
        offsets_expanded = offsets.unsqueeze(0)  # [1, num_offsets]
        
        # Compute target positions
        targets = pos_expanded + offsets_expanded  # [L, num_offsets]
        
        # Create validity mask for in-bounds edges
        valid_mask = (targets >= 0) & (targets < L)  # [L, num_offsets]
        
        # Get valid edges using masking
        valid_indices = valid_mask.nonzero(as_tuple=False)  # [num_valid_edges, 2]
        
        # Extract source and target indices
        edge_i = valid_indices[:, 0]  # Source positions
        edge_j = targets[valid_mask]   # Target positions
        
        # Expand for batch dimension
        edge_i = edge_i.unsqueeze(0).expand(B, -1)  # [B, num_edges]
        edge_j = edge_j.unsqueeze(0).expand(B, -1)  # [B, num_edges]
        
        return edge_i, edge_j
    
    def get_connectivity(self, x, B, L, device):
        """Get connectivity pattern based on configured type."""
        if self.connectivity_type == "top_w":
            return self.get_top_w_connectivity(x, B, L, device)
        elif self.connectivity_type == "local_window":
            return self.get_local_window_edges(B, L, device)
        else:
            raise ValueError(f"Unknown connectivity type: {self.connectivity_type}")
    
    def compute_edge_params(self, x, edge_i, edge_j):
        """Compute edge-specific restriction parameters."""
        B = x.size(0)
        
        # Gather node features for edges
        x_i = x.gather(1, edge_i.unsqueeze(-1).expand(-1, -1, self.dim))  # [B, E, D]
        x_j = x.gather(1, edge_j.unsqueeze(-1).expand(-1, -1, self.dim))  # [B, E, D]
        
        # Compute edge encodings
        edge_features = torch.cat([x_i, x_j], dim=-1)  # [B, E, 2D]
        
        # Compute alpha and U separately
        alphas = self.edge_to_alpha(edge_features).squeeze(-1)  # [B, E]

        if self.r > 0:
            Lambda_flat = self.edge_to_Lambda(edge_features)  # [B, E, heads*r*head_dim]
            Lambda = Lambda_flat.view(B, -1, self.heads, self.r, self.head_dim)  # [B, E, heads, r, head_dim]
            # Normalize Lambda columns
            Lambda = F.normalize(Lambda, dim=-1)
            return alphas, Lambda
        else:
            return alphas
    
    def consensus_step(self, u, edge_i, edge_j, alphas, step_sizes, Lambda=None):
        """One consensus gradient update step."""
        B, H, L, D = u.shape  # H = heads
        
        # Gather u values at edge endpoints
        u_i = u.gather(2, edge_i.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D))  # [B, H, E, D]
        u_j = u.gather(2, edge_j.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D))  # [B, H, E, D]
        
        # Apply RoPE rotation to source nodes
        u_i = self.rope(u_i=u_i, model_type="SC", edge_i=edge_i, edge_j=edge_j)  # [B, H, E, D]
        
        # Compute difference
        diff = u_i - u_j  # [B, H, E, D]
        
        # Apply restriction: R_ij @ diff
        # R_ij = alpha_ij * I + Lambda_ij @ Lambda_ij^T
        # R_ij @ diff = alpha_ij * diff + Lambda_ij @ (Lambda_ij^T @ diff)
        if self.r > 0:
            Lambda_diff = torch.einsum('behrd,bhed->bher', Lambda, diff)  # [B, H, E, r]
            residual = alphas.unsqueeze(1).unsqueeze(-1) * diff + torch.einsum('bher,behrd->bhed', Lambda_diff, Lambda)  # [B, H, E, D]
        else:
            residual = alphas.unsqueeze(1).unsqueeze(-1) * diff
        
        # Accumulate updates at vertices
        vertex_updates = torch.zeros_like(u)
        vertex_updates.scatter_add_(2, edge_i.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D), residual)
        vertex_updates.scatter_add_(2, edge_j.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D), -residual)
        
        # Consensus update with shared step sizes across heads
        # step_sizes is [B, H, L, 1] (expanded from shared values), vertex_updates is [B, H, L, D]
        u = u - step_sizes * vertex_updates
        
        return u
    
    def forward(self, x):
        """
        Args:
            x: [B, L, dim] input features
            position_ids: Not used (for compatibility)

        Returns:
            y: [B, L, dim] output features
        """
        B, L, D = x.shape
        device = x.device
        
        # Initial encoding
        u = self.token_encoder(x)  # [B, L, dim]
        
        # Reshape to multi-head format
        u = u.reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        
        # Get connectivity pattern (based on original features)
        edge_i, edge_j = self.get_connectivity(x, B, L, device)
        
        # Compute edge parameters
        if self.r > 0:
            alphas, Lambda = self.compute_edge_params(x, edge_i, edge_j)
        
            # Unrolled consensus gradient updates
            for t in range(self.num_iterations):
                # Get step sizes for this iteration (shared across heads)
                step_sizes_t = F.softplus(self.step_sizes[t].unsqueeze(0).unsqueeze(0).unsqueeze(-1))  # [1, 1, L, 1]
                step_sizes_t = step_sizes_t.expand(B, self.heads, -1, -1)  # [B, heads, L, 1]
                u = self.consensus_step(u, edge_i, edge_j, alphas, step_sizes_t, Lambda)
        
        else:
            alphas = self.compute_edge_params(x, edge_i, edge_j)
            for t in range(self.num_iterations):
                # Get step sizes for this iteration (shared across heads)
                step_sizes_t = F.softplus(self.step_sizes[t].unsqueeze(0).unsqueeze(0).unsqueeze(-1))  # [1, 1, L, 1]
                step_sizes_t = step_sizes_t.expand(B, self.heads, -1, -1)  # [B, heads, L, 1]
                u = self.consensus_step(u, edge_i, edge_j, alphas, step_sizes_t)
        
        # Reshape back to original format
        u = u.permute(0, 2, 1, 3).reshape(B, L, D)  # [B, L, dim]
            
        # Output projection
        y = self.out_proj(u)
        y = self.dropout(y)
        
        return y
