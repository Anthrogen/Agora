import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# --------------------------------------------------------------------------- #
#  Algorithm 5 – per-residue SE(3) frames built with Gram–Schmidt (row-major)
# --------------------------------------------------------------------------- #
_GS_EPS = 1e-8  # numerical safety for normalization


def _construct_se3_frames(coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal per-residue frame (Jumper et al. Appendix A, Alg. 5)
    **Row-major convention**: row vectors are multiplied on the *right*
    by rotation matrices (v_row x R).

    Parameters
    ----------
    coords : Tensor[B, L, 3, 3]
        Backbone coordinates ordered as (N, CA, C).
        Invalid positions should be all zeros.
    mask : Optional[Tensor[B, L]]
        Boolean mask indicating which residues have valid (non-zero) coordinates.

    Returns
    -------
    R : Tensor[B, L, 3, 3]   row-major rotation matrices (global ← local)
    t : Tensor[B, L, 3]      frame origins (Ca positions)
    """
    B, L = coords.shape[:2]
    device = coords.device
    dtype = coords.dtype
    
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    
    # Initialize outputs
    # Identity rotation for all positions (especially important for invalid ones)
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
    # Zero translation for all positions initially
    t = torch.zeros(B, L, 3, device=device, dtype=dtype)
    
    # Only compute frames for valid positions
    if mask.any():
        n = coords[..., 0, :]   # N
        ca = coords[..., 1, :]  # CA (origin)
        c = coords[..., 2, :]   # C
        
        # Compute frames only for valid positions
        # x-axis: points from CA towards -C
        x_axis = F.normalize(ca - c, dim=-1, eps=_GS_EPS)
        
        # y-axis: N direction, orthogonalised against x̂
        y_raw = n - ca
        y_axis = F.normalize(
            y_raw - (x_axis * y_raw).sum(-1, keepdim=True) * x_axis,
            dim=-1,
            eps=_GS_EPS,
        )
        
        # right-handed z-axis
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_axis = F.normalize(z_axis, dim=-1, eps=_GS_EPS)
        
        # Stack to form rotation matrices
        R_computed = torch.stack((x_axis, y_axis, z_axis), dim=-2)
        
        # Replace R with computed values only where mask is True
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        R = torch.where(mask_expanded, R_computed, R)
        
        # Use CA as translation only for valid positions
        t = torch.where(mask.unsqueeze(-1), ca, t)
    
    return R, t

# --------------------------------------------------------------------------- #
#  Algorithm 6 – Geometric multi-head attention (post-LN variant)
# --------------------------------------------------------------------------- #
class GeometricAttention(nn.Module):
    """
    SE(3)-invariant multi-head attention with explicit 3-D orientation
    and distance channels. Supports masked coordinates where undefined
    coordinates are handled by masking keys and zeroing attention output.

    Row-vector convention: a local row vector v is transformed to
    global coords by v_global = v_local x Rᵢ  (multiply on the right).
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads

        vdim = 3                  # per-head vector width
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

        # √3 constant as a registered buffer (keeps correct dtype/device)
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
        coords : Tensor[B, L, 3, 3]     backbone coords (N, CA, C)
        content_elements : Optional[Tensor[B, L]]  boolean mask for valid coordinates
        mask : Optional[Tensor[B, L]]  Boolean mask for valid coordinates

        Returns
        -------
        Tensor[B, L, dim]  — updated embeddings
        """
        # (1) project & split heads; LN on orientation channels only
        qr = self._split_heads(self.to_qr(x))
        kr = self._split_heads(self.to_kr(x))
        qd = self._split_heads(self.to_qd(x))
        kd = self._split_heads(self.to_kd(x))
        v  = self._split_heads(self.to_v(x))

        # (2) construct per-residue frames
        R, t = _construct_se3_frames(coords, content_elements)  # [B,L,3,3], [B,L,3]
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
        
        # (4d) Apply coordinate validity masking
        if mask is not None:
            # Mask attention scores where keys (columns) are invalid
            # Invalid positions shouldn't be attended to
            score_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L_key]
            scores = scores.masked_fill(~score_mask, float('-inf'))

        # (5) attention softmax + dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # **Only now** rotate values into global frame (after softmax)
        v_g = torch.einsum("bhld,bldk->bhlk", v, R)

        # (6) aggregate values in global frame
        o_g = torch.matmul(attn, v_g)

        # (7) rotate back to local frames
        o_local = torch.einsum("bhlk,bldk->bhld", o_g, R_T)

        # (8) merge heads → project → add residual → LayerNorm
        o = o_local.permute(0, 2, 1, 3).reshape(x.size(0), x.size(1), -1)
        out = self.dropout(self.proj(o))
        
        # (9) Zero output for invalid coordinate positions before residual
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        
        return out
