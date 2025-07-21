from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, pack, unpack

def exists(val):
    return val is not None

def default(*vals):
    for v in vals:
        if exists(v):
            return v
    return None

def round_ste(z: Tensor) -> Tensor:
    """Round with straight‐through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def pack_one(t: Tensor, pattern: str):
    return pack([t], pattern)

def unpack_one(t: Tensor, ps, pattern: str):
    return unpack(t, ps, pattern)[0]

class Quantizer(nn.Module):
    """
    Finite Scalar Quantizer.
    levels: list of ints, number of quantization levels along each code dimension.
    dim: latent feature size (will be projected in/out if != prod(levels)).
    num_codebooks: number of separate codebooks (default 1).
    """
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        super().__init__()
        # store levels & basis for index calculation
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0)
        self.register_buffer("_basis", basis, persistent=False)

        max_output = 1
        for l in levels:
            max_output *= l
        self.max_output = max_output - 1

        self.scale = scale
        self.num_codebooks = num_codebooks
        self.codebook_dim = len(levels)
        self.effective_codebook_dim = self.codebook_dim * num_codebooks

        # whether to keep codebook dim in output
        keep = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep)
        self.keep_num_codebooks_dim = keep

        # feature dimension expected
        self.dim = default(dim, self.effective_codebook_dim)

        # projections if latent_dim != codebook_dim * num_codebooks
        self.has_projections = (self.dim != self.effective_codebook_dim)
        self.project_in  = nn.Linear(self.dim,  self.effective_codebook_dim) if self.has_projections else nn.Identity()
        self.project_out = nn.Linear(self.effective_codebook_dim, self.dim)  if self.has_projections else nn.Identity()

        # build implicit codebook for debugging/inspection if needed
        self.codebook_size = int(self._levels.prod())
        idxs = torch.arange(self.codebook_size, dtype=torch.int32)
        with torch.no_grad():
            self.register_buffer(
                "implicit_codebook",
                self.indices_to_codes(idxs, project_out=False),
                persistent=False
            )



    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Map to quantization range per dimension."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Straight‐through quantization of z, same shape output."""
        if self.scale is not None:
            z = z * self.scale
        q = round_ste(self.bound(z))
        half_width = self._levels // 2
        return q / half_width

    def _scale_and_shift(self, zhat_norm: Tensor) -> Tensor:
        half = self._levels // 2
        return (zhat_norm * half) + half

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half = self._levels // 2
        return (zhat - half) / half

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Convert codes to flat codebook index."""
        # zhat shape (..., codebook_dim)
        z = self._scale_and_shift(zhat)
        retval = (z * self._basis).sum(dim=-1).to(torch.int32)

        assert retval.max() <= self.max_output
        assert retval.min() >= 0
        return retval

    def indices_to_codes(self, indices: Tensor, project_out: bool = True) -> Tensor:
        """Inverse of `codes_to_indices`."""
        assert indices.max() <= self.max_output
        assert indices.min() >= 0

        is_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        idx = rearrange(indices, '... -> ... 1')
        codes_nc = (idx // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_nc)
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')
        if project_out:
            codes = self.project_out(codes)
        if is_video:
            codes = rearrange(codes, 'b ... d -> b d ...')
        return codes

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        z: [B, N, dim] or image/video
        Returns: (quantized_z, indices)
        """
        is_video = (z.ndim >= 4)
        if is_video:
            z, ps = pack_one(z, 'b * d')
        # project into code space
        z_flat = self.project_in(z)
        # reshape into (B, N, c, d) for multiple codebooks
        zcb = rearrange(z_flat, 'b n (c d) -> b n c d', c=self.num_codebooks)
        # quantize & get indices
        codes = self.quantize(zcb)
        indices = self.codes_to_indices(codes)
        # flatten codes back
        codes_flat = rearrange(codes, 'b n c d -> b n (c d)')
        out = self.project_out(codes_flat)
        if is_video:
            out = unpack_one(out, ps, 'b * d')
            indices = unpack_one(indices, ps, 'b * c')
            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, '... 1 -> ...')
        return out, indices 
    
###### 2. Transformer processes indices (discrete tokens)
# transformed_indices = transformer(indices)  # [B, L] -> [B, L]
# # 3. Convert indices back to continuous vectors
# z_q_from_indices = encoder.quantizer.indices_to_codes(transformed_indices)  # [B, L] -> [B, L, 5]
# # 4. Pass to decoder
# x_rec = decoder(z_q_from_indices)