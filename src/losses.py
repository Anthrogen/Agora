# loss_functions.py
import torch
import torch.nn.functional as F
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS

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

def cross_entropy_loss(logits, labels, loss_elements):
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
def score_entropy_loss(output, x_0, x_t, cumulative_noise_levels, inst_noise_levels, mask_token, valid_mask):
    """
    Score entropy loss from SEDD paper.
    
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
    
    # Output represents log score ratios, apply exp to get ratios p_t(y)/p_t(x^i)
    # This ensures the ratios are positive
    output = torch.exp(output)

    # Create one-hot encoding of x_t
    # x_t shape: [B, L] with values in [0, V-1]
    x_t_onecold = 1.0 - F.one_hot(x_t, num_classes=V).float()  # [B, L, V]
    
    # Calculate delta = output @ (1 - one_hot_{x_t})
    # This computes the dot product along the vocabulary dimension
    # (1 - x_t_onehot) zeros out the position corresponding to x_t
    #x_t_onecold = 1.0 - x_t_onehot  # [B, L, V]
    
    # Compute dot product: sum over vocabulary dimension
    # delta[b, l] = sum_v output[b, l, v] * x_t_onecold[b, l, v]
    delta = (output * x_t_onecold).sum(dim=-1)  # [B, L]
    
    # The delta tensor now has shape [B, L] where:
    # delta[b, l] = sum of all output[b, l, v] except where v == x_t[b, l]

    # More numerically stable computation of base
    # base = (1 - exp(-σ)) / exp(-σ) = exp(σ) - 1
    base = torch.exp(cumulative_noise_levels) - 1.0
    base = base.unsqueeze(1)
    masked_positions = (x_t == mask_token).bool()
    assert masked_positions.shape == (B,L)

    alpha = 1.0 - 2.0 * masked_positions.float()

    # Define "opposite": if x_t is masked, opposite is x_0; if x_t is not masked, opposite is mask_token
    opposite = torch.where(masked_positions, x_0, torch.full_like(x_t, mask_token))
    
    # epsilon_1[b, l] = output[b, l, opposite[b, l]]
    # Gather values from output at opposite positions
    # batch_indices = torch.arange(B, device=output.device).unsqueeze(1).expand(B, L)  # [B, L]
    # position_indices = torch.arange(L, device=output.device).unsqueeze(0).expand(B, L)  # [B, L]
    
    #epsilon_1 = output[batch_indices, position_indices, opposite]  # [B, L]
    epsilon_1 = torch.gather(output, dim=2, index=opposite.unsqueeze(-1)).squeeze(-1)
    
    # epsilon_2: base^alpha
    # base has shape [B, 1, 1] after unsqueeze, alpha has shape [B, L]
    # We need to squeeze base back to [B, 1] for proper broadcasting
    base = base.squeeze(-1)  # [B, 1, 1] -> [B, 1]
    epsilon_2 = torch.pow(base, alpha)  # [B, L]
    epsilon_2 = torch.clamp(epsilon_2, min=1e-10)

    # Clip epsilon_1 to prevent log of very small numbers
    epsilon_1 = torch.clamp(epsilon_1, min=1e-10)
    epsilon = epsilon_2 * torch.log(epsilon_1)

    gamma = (delta - epsilon) # Gamma is (B, L)
    K = epsilon_2 * (torch.log(epsilon_2) - 1) # (B, L)
    gamma += K # now >= 0 element-wise
    
    # inst_noise_levels has shape [B, 1], we want to broadcast with gamma [B, L]
    # We should NOT unsqueeze, as [B, 1] will broadcast correctly to [B, L]
    per_residue = gamma * inst_noise_levels  # [B, L] * [B, 1] -> [B, L]

    # Apply mask to per_residue losses and compute average over valid positions only
    per_residue_masked = per_residue * valid_mask.float()  # [B, L]
    valid_counts = valid_mask.sum(dim=1).float()  # [B]
    per_protein = per_residue_masked.sum(dim=1) / valid_counts  # (B,)

    return per_protein.mean(0) # scalar
    
# --------------------------------------------------------------------------- #
#  FSQ Utilities                                                              #
# --------------------------------------------------------------------------- #
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
