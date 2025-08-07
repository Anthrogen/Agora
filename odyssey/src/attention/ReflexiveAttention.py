import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# --------------------------------------------------------------------------- #
#  Algorithm 5 (Extended) – per-residue SE(3) frames with C-beta 
# --------------------------------------------------------------------------- #
_GS_EPS = 1e-8  # numerical safety for normalization


def _construct_se3_frames_with_cbeta(coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal per-residue frame using 4 atoms (N, CA, C, CB)
    Extended version of Algorithm 5 that includes C-beta information.
    
    This implementation uses the following construction to encode chirality:
    1. Vc = C - CA, Vn = N - CA, Vb = CB - CA
    2. x = Vn, x-bar = x / ||x||
    3. y = Vc - proj_{x-bar}(Vc) = Vc - (Vc · x-bar) * x-bar
    4. y-bar = y / ||y||
    5. w = x-bar x y-bar, w-bar = w / ||w||
    6. z = projection of Vb onto w-bar
    7. z-bar = z / ||z||
    
    The sign of (Vb · w-bar) encodes the chirality (L vs D configuration).
    
    **Row-major convention**: row vectors are multiplied on the *right*
    by rotation matrices (v_row x R).

    Parameters
    ----------
    coords : Tensor[B, L, 4, 3]
        Backbone coordinates ordered as (N, CA, C, CB).
        CB is always present (virtual CB for glycine).
        Invalid positions should be all zeros.
    mask : Optional[Tensor[B, L]]
        Boolean mask indicating which residues have valid (non-zero) coordinates.

    Returns
    -------
    R : Tensor[B, L, 3, 3]   row-major rotation matrices (global ← local)
    t : Tensor[B, L, 3]      frame origins (Ca positions)
    valid_frames : Tensor[B, L]  Boolean mask for valid frames
    """
    B, L = coords.shape[:2]
    device = coords.device
    dtype = coords.dtype
    
    # If no mask provided, detect zero coordinates
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    
    # Initialize outputs
    # Identity rotation for all positions (especially important for invalid ones)
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    # Zero translation for all positions initially
    t = torch.zeros(B, L, 3, device=device, dtype=dtype)
    
    # Only compute frames for valid positions
    if mask.any():
        n = coords[..., 0, :]           # N
        ca = coords[..., 1, :]          # CA (origin)
        c = coords[..., 2, :]           # C
        cb = coords[..., 3, :]          # CB (real or virtual)

        # Step 1: Define vectors relative to CA
        Vn = n - ca   # N - CA
        Vc = c - ca   # C - CA
        Vb = cb - ca  # CB - CA
        
        # Step 2: x-bar = normalized Vn
        x = Vn
        x_bar = F.normalize(x, dim=-1, eps=_GS_EPS)
        
        # Step 3: y = Vc - projection of Vc onto x-bar
        proj_Vc_on_x = (Vc * x_bar).sum(-1, keepdim=True) * x_bar
        y = Vc - proj_Vc_on_x
        y_bar = F.normalize(y, dim=-1, eps=_GS_EPS)
        
        # Step 4: w = x-bar × y-bar (normal to the N-CA-C plane)
        w = torch.cross(x_bar, y_bar, dim=-1)
        w_bar = F.normalize(w, dim=-1, eps=_GS_EPS)
        
        # Step 5: z = projection of Vb onto w-bar
        # This encodes chirality: (Vb · w-bar) is positive for L-amino acids, negative for D
        z = (Vb * w_bar).sum(-1, keepdim=True) * w_bar
        z_bar = F.normalize(z, dim=-1, eps=_GS_EPS)
        
        # Note: z-bar will be either +w-bar or -w-bar depending on chirality
        
        # Step 6: Construct rotation matrix using x-bar, y-bar, z-bar as rows
        R_computed = torch.stack((x_bar, y_bar, z_bar), dim=-2)  # [B,L,3,3]
        
        # Replace R with computed values only where mask is True
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        R = torch.where(mask_expanded, R_computed, R)
        
        # Use CA as translation only for valid positions
        t = torch.where(mask.unsqueeze(-1), ca, t)
    
    return R, t, mask

# --------------------------------------------------------------------------- #
#  Extended Geometric multi-head attention with C-beta information
# --------------------------------------------------------------------------- #
class ReflexiveAttention(nn.Module):
    """
    SE(3)-invariant multi-head attention with explicit 3-D orientation
    and distance channels, extended to use C-beta information.
    
    This is an extended version of GeometricAttention that uses 4 backbone atoms
    (N, CA, C, CB) instead of 3, providing richer structural information.

    Row-vector convention: a local row vector v is transformed to
    global coords by v_global = v_local x Rᵢ  (multiply on the right).
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads

        vdim = 3                  # per-head vector width (3D space)
        fused = heads * vdim      # projection size

        # Linear maps
        self.to_qr = nn.Linear(dim, fused)   # orientation queries
        self.to_kr = nn.Linear(dim, fused)   # orientation keys
        self.to_qd = nn.Linear(dim, fused)   # distance    queries
        self.to_kd = nn.Linear(dim, fused)   # distance    keys
        self.to_v  = nn.Linear(dim, fused)   # values

        # Learned positive scalars w_r, w_d (softplus-param.)
        self.w_r = nn.Parameter(torch.zeros(heads))
        self.w_d = nn.Parameter(torch.zeros(heads))

        self.proj = nn.Linear(fused, dim)
        self.dropout = nn.Dropout(dropout)

        # √3 constant as a registered buffer
        self.register_buffer("sqrt3", torch.sqrt(torch.tensor(3.0)))

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, L, H*3] → [B, H, L, 3]"""
        B, L, _ = t.shape
        return (t.view(B, L, self.heads, 3).permute(0, 2, 1, 3).contiguous()) # B H L 3

    def forward(self, x: torch.Tensor, coords: torch.Tensor, content_elements: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : Tensor[B, L, dim]      token embeddings
        coords : Tensor[B, L, 4, 3]     backbone coords (N, CA, C, CB)
        content_elements : Optional[Tensor[B, L]]  boolean mask for valid coordinates
        mask : Optional[Tensor[B, L]]  boolean mask for valid coordinates

        Returns
        -------
        Tensor[B, L, dim]  — updated embeddings
        """
        # (1) project & split heads
        qr = self._split_heads(self.to_qr(x))
        kr = self._split_heads(self.to_kr(x))
        qd = self._split_heads(self.to_qd(x))
        kd = self._split_heads(self.to_kd(x))
        v  = self._split_heads(self.to_v(x))

        # (2) construct per-residue frames with C-beta
        R, t, valid_frames = _construct_se3_frames_with_cbeta(coords, content_elements)  # [B,L,3,3], [B,L,3], [B,L]
        R_T = R.transpose(-1, -2)
        t = t.unsqueeze(1)                             # [B,1,L,3]

        # (3) rotate Q/K into global frame
        qr_g = torch.einsum("bhld,bldk->bhlk", qr, R)
        kr_g = torch.einsum("bhld,bldk->bhlk", kr, R)

        # (3b) rotate/translate distance channels
        qd_g = torch.einsum("bhld,bldk->bhlk", qd, R) + t
        kd_g = torch.einsum("bhld,bldk->bhlk", kd, R) + t

        # (4a) orientation similarity ⟨q_r, k_r⟩ / √3
        dir_score = (qr_g.unsqueeze(-2) * kr_g.unsqueeze(-3)).sum(-1) / self.sqrt3

        # (4b) distance term ‖q_d − k_d‖ / √3
        # Reshape for broadcasting: queries get a dummy key dimension, keys get a dummy query dimension
        qd_g_expanded = qd_g.unsqueeze(-2)  # [B, H, L, 1, 3]
        kd_g_expanded = kd_g.unsqueeze(-3)  # [B, H, 1, L, 3]
        
        # Compute pairwise distances using broadcasting
        dist_score = (qd_g_expanded - kd_g_expanded).norm(dim=-1) / self.sqrt3  # [B, H, L, L]

        # (4c) combine with learned weights
        w_r = F.softplus(self.w_r).view(1, self.heads, 1, 1)
        w_d = F.softplus(self.w_d).view(1, self.heads, 1, 1)
        scores = w_r * dir_score - w_d * dist_score
        
        # (4d) Apply coordinate mask to keys - mask out invalid key positions
        if mask is not None:
            # Create attention mask: [B, H, L_query, L_key]
            # Invalid keys (mask=False) should not be attended to
            key_mask = valid_frames.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_key]
            scores = scores.masked_fill(~key_mask, float('-inf'))

        # (5) attention softmax + dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # **Only now** rotate values into global frame (after softmax)
        v_g = torch.einsum("bhld,bldk->bhlk", v, R)

        # (6) aggregate values in global frame
        o_g = torch.matmul(attn, v_g)

        # (7) rotate back to local frames
        o_local = torch.einsum("bhlk,bldk->bhld", o_g, R_T)

        # (8) merge heads → project
        o = o_local.permute(0, 2, 1, 3).reshape(x.size(0), x.size(1), -1)
        out = self.dropout(self.proj(o))
        
        # (9) Zero output for invalid coordinate positions
        if mask is not None:
            out = out * valid_frames.unsqueeze(-1).to(out.dtype)
        
        return out 