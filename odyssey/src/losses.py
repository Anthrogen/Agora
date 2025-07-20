# loss_functions.py
import torch
import torch.nn.functional as F
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
import pdb

# --------------------------------------------------------------------------- #
#  MLM Utilities                                                              #
# --------------------------------------------------------------------------- #
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, loss_elements: torch.Tensor) -> float:
    """Calculate accuracy for masked positions only."""
    B, L, V = logits.shape
    content_logits = logits[:,:,:V-len(SPECIAL_TOKENS)]
    predictions = torch.argmax(content_logits, dim=-1)
    correct = (predictions == labels) & loss_elements

    num_loss_elements = torch.sum(loss_elements.float())

    if num_loss_elements == 0.0:
        return None

    # Just return the unconditioned sample mean.
    # Not everything has to be complicated.

    retval = torch.sum(correct.float()) / num_loss_elements
    assert not retval.isnan()

    return retval

def cross_entropy_loss(logits, labels, loss_elements, reduction='sum'):
    """
    Logits is (B,L,V)
    Labels (B,L), each element of which is in [0,V)
    Loss elements is a boolean (B,L) tensor that denotes which elements contribute to the loss.
    """
    
    B, L, V = logits.shape
    assert labels.shape == (B, L)
    assert loss_elements.shape == (B, L)

    logits = logits.clone()
    num_content_tokens = V - len(SPECIAL_TOKENS)
    logits_content = logits[:,:,:num_content_tokens]
    nlls = -torch.log_softmax(logits_content, dim=-1)

    # DO NOT set special tokens to have -inf values.  After the gather operation, these will be zeroed out
    #  when multiplied by loss_elements.  Zero times infinity is NaN!!!
    special_nlls = torch.full((B, L, len(SPECIAL_TOKENS)), -1, device=logits.device)
    nlls = torch.cat([nlls,special_nlls], dim=2)

    # Calculate the pdf over V for each element of (B,L)
    # For Odyssey 1, we shall softmax over only the content tokens of V -- we do not include the special tokens.
    entropies = torch.gather(nlls, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    values = entropies * loss_elements.float()

    assert not (values < 0).any()

    per_row_sum = values.sum(dim=1)
    per_row_denom = loss_elements.float().sum(dim=1)
  
    # We need to normalize each row by the number of elements that contribute to the loss within that row.
    # For some rows, this will be zero.  We cannot normalize by zero, but the good news is 
    # the numerator will be zero for these rows too, so we can divide by anything you want
    # and the quotient will just be zero.
    # Clamp this denominator to be one at the minimum.
    per_row_average = per_row_sum / per_row_denom.clamp(min=1.0)

    retval = torch.mean(per_row_average)

    assert not torch.isnan(retval).any()
    return retval

# --------------------------------------------------------------------------- #
#  Discrete Diffusion Utilities                                               #
# --------------------------------------------------------------------------- #
def score_entropy_loss_absorb(output, x_0, x_t, cumulative_noise_levels, inst_noise_levels, mask_token, valid_mask):
    """
    Score entropy loss for absorb discrete diffusion.
    
    Args:
        output: Model predictions of shape (B, L, V)
        x_0: Original tokens of shape (B, L)
        x_t: Noisy tokens of shape (B, L)
        cumulative_noise_levels: Cumulative noise at time t, shape (B, 1)
        inst_noise_levels: Instantaneous noise at time t, shape (B, 1)
        mask_token: Index of the absorbing/mask token
        valid_mask: Mask for valid positions (excluding BOS/EOS/Padding), shape (B, L).
    
    Returns:
        Loss value
    """
    B, L, V = output.shape
    
    # Compute esigm1 = exp(sigma) - 1, with numerical stability for small sigma
    esigm1 = torch.where(
        cumulative_noise_levels < 0.5,
        torch.expm1(cumulative_noise_levels),
        torch.exp(cumulative_noise_levels) - 1
    )  # shape (B, 1)
    
    # Identify positions that are in the absorbing state (masked)
    rel_ind = (x_t == mask_token)  # [B, L]
    
    # Compute ratio = 1 / esigm1 for all positions
    ratio = 1 / esigm1  # [B, 1]
    
    # Negative term: ratio * score[x_0] for masked positions, 0 for others
    neg_term = ratio * torch.gather(output, -1, x_0[..., None]).squeeze(-1)  # [B, L]
    neg_term = neg_term * rel_ind.float()  # Zero out non-masked positions
    
    # Positive term: sum of exp(score) over all tokens except the mask token
    # Create a one-hot mask to exclude the mask token dimension while maintaining gradients
    mask_onehot = F.one_hot(torch.tensor(mask_token, device=output.device), num_classes=V).float()  # [V]
    exclude_mask = 1.0 - mask_onehot  # [V] - zeros out the mask token position
    
    score_exp = torch.exp(output)  # [B, L, V]
    pos_term = (score_exp * exclude_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, L]
    pos_term = pos_term * rel_ind.float()  # Only for masked positions
    
    # Constant term: ratio * (log(ratio) - 1) for masked positions, 0 for others
    const = ratio * (torch.log(torch.clamp(ratio, min=1e-10)) - 1)  # [B, 1]
    const = const.expand(-1, L) * rel_ind.float()  # [B, L], only for masked positions
    
    # Combine terms: entropy = pos_term - neg_term + const
    entropy = pos_term - neg_term + const  # [B, L]
    
    # Apply instantaneous noise levels and valid mask
    per_residue = entropy * inst_noise_levels  # [B, L] * [B, 1] -> [B, L]
    
    # Apply mask to per_residue losses and compute average over valid positions only
    per_residue_masked = per_residue * valid_mask.float()  # [B, L]
    valid_counts = valid_mask.sum(dim=1).float()  # [B]
    per_protein = per_residue_masked.sum(dim=1) / valid_counts  # (B,)

    return per_protein.mean(0)  # scalar

def score_entropy_loss_uniform(output, x_0, x_t, cumulative_noise_levels, inst_noise_levels, mask_token, valid_mask):
    """
    Score entropy loss for uniform discrete diffusion.
    
    Args:
        output: Model predictions of shape (B, L, V)
        x_0: Original tokens of shape (B, L)
        x_t: Noisy tokens of shape (B, L)
        cumulative_noise_levels: Cumulative noise at time t, shape (B, 1)
        inst_noise_levels: Instantaneous noise at time t, shape (B, 1)
        mask_token: Index of the absorbing/mask token
        valid_mask: Mask for valid positions (excluding BOS/EOS/Padding), shape (B, L).
    
    Returns:
        Loss value
    """
    B, L, V = output.shape
    
    # Compute esigm1 = exp(sigma) - 1, with numerical stability for small sigma
    esigm1 = torch.where(
        cumulative_noise_levels < 0.5,
        torch.expm1(cumulative_noise_levels),
        torch.exp(cumulative_noise_levels) - 1
    )  # shape (B, 1)
    
    # Ratio for uniform discrete diffusion
    ratio = 1 - V / (esigm1 + V)  # shape (B, 1)
    
    # Positive term: exp(score).mean(dim=-1) - exp(score)[x_t] / V
    sexp = torch.exp(output)  # [B, L, V]
    pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x_t[..., None]).squeeze(-1) / V  # [B, L]
    
    # Negative term: depends on whether x_t == x_0
    score_mean = output.mean(dim=-1)  # [B, L]
    score_xt = torch.gather(output, -1, x_t[..., None]).squeeze(-1)  # [B, L]
    
    neg_term_base = score_mean - score_xt / V  # [B, L]
    
    # Case 1: x_t == x_0 (unchanged)
    unchanged_positions = (x_t == x_0)  # [B, L]
    neg_term_unchanged = ratio * neg_term_base  # [B, 1] * [B, L] -> [B, L]
    
    # Case 2: x_t != x_0 (changed)
    score_x0 = torch.gather(output, -1, x_0[..., None]).squeeze(-1)  # [B, L]
    neg_term_changed = score_x0 / esigm1 + neg_term_base  # [B, L] (esigm1 broadcasts from [B, 1])
    
    # Combine negative terms based on position type
    neg_term = torch.where(unchanged_positions, neg_term_unchanged, neg_term_changed)  # [B, L]
    
    # Constant term: depends on whether x_t == x_0
    ratio_squeezed = ratio.squeeze(-1)  # [B]
    
    # Case 1: x_t == x_0
    const_unchanged = (V - 1) / V * ratio_squeezed * (torch.log(torch.clamp(ratio_squeezed, min=1e-10)) - 1)  # [B]
    
    # Case 2: x_t != x_0  
    const_changed = ((-torch.log(torch.clamp(ratio_squeezed, min=1e-10)) - 1) / torch.clamp(ratio_squeezed, min=1e-10) - (V - 2)) / V  # [B]
    
    # Combine constant terms and broadcast to [B, L]
    const = torch.where(unchanged_positions, const_unchanged.unsqueeze(-1), const_changed.unsqueeze(-1))  # [B, L]
    
    # Final entropy: pos_term - neg_term + const
    entropy = pos_term - neg_term + const  # [B, L]
    
    # Apply instantaneous noise levels and valid mask
    per_residue = entropy * inst_noise_levels  # [B, L] * [B, 1] -> [B, L]
    
    # Apply mask to per_residue losses and compute average over valid positions only
    per_residue_masked = per_residue * valid_mask.float()  # [B, L]
    valid_counts = valid_mask.sum(dim=1).float()  # [B]
    per_protein = per_residue_masked.sum(dim=1) / valid_counts  # (B,)

    return per_protein.mean(0)  # scalar

# --------------------------------------------------------------------------- #
#  FSQ Utilities                                                              #
# --------------------------------------------------------------------------- #
def _kabsch_align(pred_pts: torch.Tensor,
                  tgt_pts: torch.Tensor) -> torch.Tensor:
    """
    Internal helper: compute optimal rotation for already-centered pred_pts/tgt_pts (B×N×3)
    and return rotated pred. Assumes inputs are already mean-centered.
    """
    # 1) covariance matrix (swap order for row-vectors)
    H = tgt_pts.transpose(1, 2) @ pred_pts

    # 2) batched SVD
    U, S, Vh = torch.linalg.svd(H)

    # 3) reflection correction per‐batch
    dets = torch.linalg.det(torch.bmm(
        Vh.transpose(-2, -1),
        U.transpose(-2, -1),
    ))                                   # [B]
    D = torch.diag_embed(torch.stack([
        torch.ones_like(dets),
        torch.ones_like(dets),
        dets.sign()
    ], dim=1))                                              # [B,3,3]

    # 4) build optimal rotation for row-vectors
    R = Vh.transpose(-2, -1) @ D @ U.transpose(-2, -1)                     # now maps pred_pts → tgt_pts

    # 5) apply rotation to pred
    pred_rot = torch.bmm(pred_pts, R)     # [B, N, 3]
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
            sq_dist = (pred_rot - t).pow(2).sum(dim=-1)  # [1, M]
            msd = sq_dist.mean()  # scalar
            rmsd = torch.sqrt(msd + eps)  # scalar
            losses.append(rmsd)
        return torch.stack(losses).mean()

    B = pred.size(0)
    pred_pts = pred.reshape(B, -1, 3)
    tgt_pts  = target.reshape(B, -1, 3)

    pred_rot = _kabsch_align(pred_pts, tgt_pts)           # [B,N,3]

    sq_dist = (pred_rot - tgt_pts).pow(2).sum(dim=-1)     # [B,N]
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
            
            mse = (pred_rot - tgt_pts).pow(2).mean()
            losses.append(mse)
        return torch.stack(losses).mean()
    
    # Original tensor case
    B = pred.size(0)
    pred_pts = pred.reshape(B, -1, 3)
    tgt_pts = target.reshape(B, -1, 3)

    pred_rot = _kabsch_align(pred_pts, tgt_pts)

    mse = (pred_rot - tgt_pts).pow(2).sum(dim=-1).mean(dim=-1) 
    return mse.mean()


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
