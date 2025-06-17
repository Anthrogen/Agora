# loss_functions.py

import torch
import torch.nn.functional as F

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean-squared error between pred and target.
    Args:
        pred, target: (..., L, 3, 3) or (..., N, 3) tensors or list of [1, M, 3] tensors
    Returns:
        scalar MSE
    """
    if isinstance(pred, list):
        losses = []
        for p, t in zip(pred, target):
            mse = (p - t).pow(2).mean()
            losses.append(mse)
        return torch.stack(losses).mean()
    return torch.mean((pred - target) ** 2)


def simple_rmsd_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Batched simple RMSD:
      √( mean_{points} ‖pred - target‖² + eps ), averaged over the batch.
    Args:
      pred, target: [B, ..., 3] coordinate tensors or list of [1, M, 3] tensors
    Returns:
      scalar loss = mean over batch of per-sample RMSD
    """
    if isinstance(pred, list):
        losses = []
        for p, t in zip(pred, target):
            sq_dist = (p - t).pow(2).sum(dim=-1)  # [1, M]
            msd = sq_dist.mean()  # scalar
            rmsd = torch.sqrt(msd + eps)  # scalar
            losses.append(rmsd)
        return torch.stack(losses).mean()

    B = pred.size(0)
    pred_pts = pred.reshape(B, -1, 3)
    tgt_pts  = target.reshape(B, -1, 3)

    sq_dist = (pred_pts - tgt_pts).pow(2).sum(dim=-1)  # [B, M]
    msd     = sq_dist.mean(dim=-1)                    # [B]
    rmsd    = torch.sqrt(msd + eps)                   # [B]
    return rmsd.mean()


def _kabsch_align(pred_pts: torch.Tensor,
                  tgt_pts: torch.Tensor) -> torch.Tensor:
    """
    Internal helper: center pred_pts/tgt_pts (B×N×3), compute optimal
    rotation (row-vector convention), and return rotated pred.
    """
    # 1) center each sample
    pred_c = pred_pts - pred_pts.mean(dim=1, keepdim=True)
    tgt_c  = tgt_pts  - tgt_pts.mean(dim=1, keepdim=True)

    # 2) covariance matrix (swap order for row-vectors)
    H = tgt_c.transpose(1, 2) @ pred_c

    # 3) batched SVD
    U, S, Vh = torch.linalg.svd(H)

    # 4) reflection correction per‐batch
    dets = torch.linalg.det(torch.bmm(
        Vh.transpose(-2, -1),
        U.transpose(-2, -1),
    ))                                   # [B]
    D = torch.diag_embed(torch.stack([
        torch.ones_like(dets),
        torch.ones_like(dets),
        dets.sign()
    ], dim=1))                                              # [B,3,3]

    # 5) build optimal rotation for row-vectors
    R = Vh.transpose(-2, -1) @ D @ U.transpose(-2, -1)                     # now maps pred_c → tgt_c

    # 6) apply rotation to centered pred
    pred_rot = torch.bmm(pred_c, R)     # [B, N, 3]
    return pred_rot


def kabsch_rmsd_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """
    Batched RMSD after optimal rigid-body alignment (Kabsch).
    Args:
      pred, target: [B, N, 3] or [B, ..., 3] coordinate tensors or list of [1, M, 3] tensors
    Returns:
      scalar loss = mean over batch of per-sample aligned RMSD
    """
    if isinstance(pred, list):
        losses = []
        for p, t in zip(pred, target):
            pred_rot = _kabsch_align(p, t)
            tgt_c = t - t.mean(dim=1, keepdim=True)
            sq_dist = (pred_rot - tgt_c).pow(2).sum(dim=-1)  # [1, M]
            msd = sq_dist.mean()  # scalar
            rmsd = torch.sqrt(msd + eps)  # scalar
            losses.append(rmsd)
        return torch.stack(losses).mean()

    B = pred.size(0)
    pred_pts = pred.reshape(B, -1, 3)
    tgt_pts  = target.reshape(B, -1, 3)

    pred_rot = _kabsch_align(pred_pts, tgt_pts)           # [B,N,3]
    tgt_c    = tgt_pts - tgt_pts.mean(dim=1, keepdim=True)

    sq_dist = (pred_rot - tgt_c).pow(2).sum(dim=-1)        # [B,N]
    msd     = sq_dist.mean(dim=-1)                        # [B]
    rmsd    = torch.sqrt(msd + eps)                       # [B]
    return rmsd.mean()

def squared_kabsch_rmsd_loss(pred: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
    """
    Batched squared RMSD (i.e. MSE) after Kabsch alignment.
    No square-root ⇒ fully smooth gradients at zero.
    Args:
      pred, target: [B, N, 3] or [B, ..., 3] coordinate tensors or list of [1, M, 3] tensors
    Returns:
      scalar loss = mean over batch of per-sample aligned MSE
    """
    # Handle case where inputs are lists of tensors
    if isinstance(pred, list):
        losses = []
        for p, t in zip(pred, target):
            # Each element is [1, M, 3]
            pred_pts = p
            tgt_pts = t
            
            pred_rot = _kabsch_align(pred_pts, tgt_pts) 
            tgt_c = tgt_pts - tgt_pts.mean(dim=1, keepdim=True)
            
            mse = (pred_rot - tgt_c).pow(2).mean()
            losses.append(mse)
        return torch.stack(losses).mean()
    
    # Original tensor case
    B = pred.size(0)
    pred_pts = pred.reshape(B, -1, 3)
    tgt_pts = target.reshape(B, -1, 3)

    pred_rot = _kabsch_align(pred_pts, tgt_pts)
    tgt_c = tgt_pts - tgt_pts.mean(dim=1, keepdim=True)

    mse = (pred_rot - tgt_c).pow(2).sum(dim=-1).mean(dim=-1) 
    return mse.mean()


def huber_kabsch_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      delta: float = 1.0) -> torch.Tensor:
    """
    Batched Huber (smooth-L1) loss after Kabsch alignment:
      robust to occasional large errors.
    Args:
      pred, target: [B, N, 3] or [B, ..., 3] or list of [1, M, 3] tensors
      delta: Huber threshold
    Returns:
      scalar loss = mean over batch of per-sample Huber loss
    """
    if isinstance(pred, list):
        losses = []
        for p, t in zip(pred, target):
            pred_rot = _kabsch_align(p, t)
            tgt_c = t - t.mean(dim=1, keepdim=True)
            loss = F.smooth_l1_loss(pred_rot, tgt_c, reduction="mean")
            losses.append(loss)
        return torch.stack(losses).mean()

    B = pred.size(0)
    pred_pts = pred.reshape(B, -1, 3)
    tgt_pts  = target.reshape(B, -1, 3)

    pred_rot = _kabsch_align(pred_pts, tgt_pts)
    tgt_c    = tgt_pts - tgt_pts.mean(dim=1, keepdim=True)

    # smooth_l1_loss defaults to reduction='mean'
    return F.smooth_l1_loss(pred_rot, tgt_c, reduction="mean")


def soft_lddt_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   cutoff: float = 15.0,
                   thresholds: list[float] = (0.5, 1.0, 2.0, 4.0),
                   beta: float = 10.0) -> torch.Tensor:
    """
    Differentiable proxy for Local Distance Difference Test (lDDT):
      - Compute pairwise distances within pred and within target.
      - Mask to local pairs (d_tgt < cutoff, i != j).
      - For each Δ-threshold, score via sigmoid(beta*(t - |d_pred - d_tgt|)).
      - Average scores over thresholds & pairs, then loss = 1 - mean_lDDT.
    Args:
      pred, target: [B, N, 3] coordinate tensors or list of [1, M, 3] tensors
    Returns:
      scalar loss = mean over batch of (1 - soft-lDDT)
    """
    if isinstance(pred, list):
        losses = []
        for p, t in zip(pred, target):
            _, M, _ = p.shape
            d_pred = torch.cdist(p, p, p=2)  # [1,M,M]
            d_tgt = torch.cdist(t, t, p=2)  # [1,M,M]
            
            device = d_tgt.device
            eye = torch.eye(M, dtype=torch.bool, device=device).unsqueeze(0)
            mask = (d_tgt < cutoff) & ~eye
            
            diff = torch.abs(d_pred - d_tgt)
            scores = [torch.sigmoid(beta * (t - diff)) for t in thresholds]
            score = torch.stack(scores, dim=0).mean(dim=0)
            
            masked = torch.where(mask, score, torch.zeros_like(score))
            sum_score = masked.sum(dim=[1, 2])
            count = mask.sum(dim=[1, 2]).clamp(min=1)
            lddt = sum_score / count
            losses.append(1.0 - lddt)
        return torch.cat(losses).mean()

    B, N, _ = pred.shape

    # 1) pairwise distances
    d_pred = torch.cdist(pred,  pred,  p=2)  # [B,N,N]
    d_tgt  = torch.cdist(target, target, p=2)  # [B,N,N]

    # 2) mask local & not self
    device = d_tgt.device
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1,N,N]
    mask = (d_tgt < cutoff) & ~eye                                  # [B,N,N]

    # 3) abs difference
    diff = torch.abs(d_pred - d_tgt)                                # [B,N,N]

    # 4) soft-threshold scores
    scores = [torch.sigmoid(beta * (t - diff)) for t in thresholds]  # list of [B,N,N]
    score  = torch.stack(scores, dim=0).mean(dim=0)                 # [B,N,N]

    # 5) masked mean per-sample
    masked    = torch.where(mask, score, torch.zeros_like(score))
    sum_score = masked.sum(dim=[1, 2])                              # [B]
    count     = mask.sum(dim=[1, 2]).clamp(min=1)                   # [B]
    lddt      = sum_score / count                                   # [B]

    return (1.0 - lddt).mean()


def mask_and_flatten(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Given backbone coords and a boolean mask (residue-level), flatten to atom coords
    and filter out padded positions.
    Args:
      coords: [B, L, 3, 3] or [L, 3, 3]
      mask:   [B, L] or [L]
    Returns:
      valid_pts: List of [1, M, 3] of size Bor [1, M, 3] flattened and masked
    """
    # handle unbatched coords
    if coords.ndim == 3:
        dims = coords.shape
        coords = coords.unsqueeze(0)
        mask = mask.unsqueeze(0)
        added_batch = True
    else:
        added_batch = False
    B, L, _, _ = coords.shape
    pts = coords.view(B, L * 3, 3)
    # expand residue mask to atom level
    mask3 = mask.repeat_interleave(3, dim=1)  # [B, L*3]
    # collect valid per sample and return as list
    valid_list = []
    for b in range(B):
        valid_pts = pts[b][mask3[b]]  # [Mb, 3]
        valid_list.append(valid_pts.unsqueeze(0))  # [1, Mb, 3]
    return valid_list
