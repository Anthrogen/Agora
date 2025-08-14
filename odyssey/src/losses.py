# loss_functions.py
import torch
import torch.nn.functional as F
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
import pdb

# --------------------------------------------------------------------------- #
#  MLM Utilities                                                              #
# --------------------------------------------------------------------------- #
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, loss_elements: torch.Tensor, return_all=False) -> float:
    """Calculate accuracy for masked positions only."""
    B, L, V = logits.shape
    content_logits = logits[:,:,:V-len(SPECIAL_TOKENS)]
    predictions = torch.argmax(content_logits, dim=-1)
    correct = (predictions == labels) & loss_elements

    # Use consistent dtype (no unnecessary .float() conversions for fp16 compatibility)
    num_loss_elements = torch.sum(loss_elements.to(predictions.dtype))

    if num_loss_elements == 0.0:
        return None

    # Just return the unconditioned sample mean.
    # Not everything has to be complicated.

    if not return_all:
        retval = torch.sum(correct.to(predictions.dtype)) / num_loss_elements
        assert not retval.isnan()
        return retval
    else:
        return correct.to(predictions.dtype)

def cross_entropy_loss(logits, labels, loss_elements, reduction='sum', return_all=False):
    """
    Logits is (B,L,V)
    Labels (B,L), each element of which is in [0,V)
    Loss elements is a boolean (B,L) tensor that denotes which elements contribute to the loss.
    """
    B, L, V = logits.shape
    logits = logits.clone()
    num_content_tokens = V - len(SPECIAL_TOKENS)
    logits_content = logits[:,:,:num_content_tokens]
    nlls = -torch.log_softmax(logits_content, dim=-1)
    V = num_content_tokens

    assert labels.shape == (B, L)
    assert loss_elements.shape == (B, L)

    # DO NOT set special tokens to have -inf values.  After the gather operation, these will be zeroed out
    #  when multiplied by loss_elements.  Zero times infinity is NaN!!!
    special_nlls = torch.full((B, L, len(SPECIAL_TOKENS)), -1, device=logits.device)
    nlls = torch.cat([nlls,special_nlls], dim=2)

    # Calculate the pdf over V for each element of (B,L)
    # For Odyssey 1, we shall softmax over only the content tokens of V -- we do not include the special tokens.
    entropies = torch.gather(nlls, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    values = entropies * loss_elements.to(entropies.dtype)  # Match dtypes for fp16 compatibility

    assert not (values < 0).any()

    per_row_sum = values.sum(dim=1)
    per_row_denom = loss_elements.to(entropies.dtype).sum(dim=1)  # Match dtypes for fp16 compatibility
  
    # We need to normalize each row by the number of elements that contribute to the loss within that row.
    # For some rows, this will be zero.  We cannot normalize by zero, but the good news is 
    # the numerator will be zero for these rows too, so we can divide by anything you want
    # and the quotient will just be zero.
    # Clamp this denominator to be one at the minimum.
    per_row_average = per_row_sum / per_row_denom.clamp(min=1.0)

    if not return_all:
        retval = torch.mean(per_row_average)
        assert not torch.isnan(retval).any()
        return retval
    else:
        return per_row_average

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
    
    # Filter vocabulary to exclude special tokens but keep mask token
    num_content = V - len(SPECIAL_TOKENS)
    keep_pos = list(range(num_content)) + [mask_token]
    output = output[:, :, keep_pos]
    
    # Update indices: content tokens stay same, mask token goes to end
    V = len(keep_pos)
    x_t = torch.where(x_t == mask_token, V - 1, x_t); x_t = torch.where(~valid_mask, 0, x_t)
    x_0 = torch.where(x_0 == mask_token, V - 1, x_0); x_0 = torch.where(~valid_mask, 0, x_0)
    mask_token = V - 1
    
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

    # Clamp output to prevent exp overflow in float16 (max ~10.5) or float32 (max ~88)
    max_exp_input = 10.0 if output.dtype == torch.float16 else 88.0
    output_clamped = torch.clamp(output, max=max_exp_input)
    score_exp = torch.exp(output_clamped)  # [B, L, V]
    
    # Negative term: ratio * score[x_0] for masked positions, 0 for others
    neg_term = ratio * torch.gather(output, -1, x_0[..., None]).squeeze(-1)  # [B, L]
    neg_term = neg_term * rel_ind.to(neg_term.dtype)  # Zero out non-masked positions - match dtypes for fp16
    
    # Positive term: sum of exp(score) over all tokens except the mask token
    # Create a one-hot mask to exclude the mask token dimension while maintaining gradients
    mask_onehot = F.one_hot(torch.tensor(mask_token, device=output.device), num_classes=V).to(output.dtype)  # [V] - match dtypes for fp16
    exclude_mask = 1.0 - mask_onehot  # [V] - zeros out the mask token position

    pos_term = (score_exp * exclude_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, L]
    pos_term = pos_term * rel_ind.to(pos_term.dtype)  # Only for masked positions - match dtypes for fp16
    
    # Constant term: ratio * (log(ratio) - 1) for masked positions, 0 for others
    # Remove clamping - use ratio directly for log
    const = ratio * (torch.log(ratio) - 1) # [B, 1]
    const = const.expand(-1, L) * rel_ind.to(const.dtype)  # [B, L], only for masked positions - match dtypes for fp16
    
    # Combine terms: entropy = pos_term - neg_term + const
    entropy = pos_term - neg_term + const  # [B, L]
    
    # Apply instantaneous noise levels and valid mask
    per_residue = entropy * inst_noise_levels  # [B, L] * [B, 1] -> [B, L]
    
    # Apply mask to per_residue losses and compute average over valid positions only
    per_residue_masked = per_residue * valid_mask.to(per_residue.dtype)  # [B, L] - match dtypes for fp16
    valid_counts = valid_mask.sum(dim=1).to(per_residue.dtype)  # [B] - match dtypes for fp16
    per_protein = per_residue_masked.sum(dim=1) / valid_counts  # (B,)

    return per_protein.mean(0)  # scalar

def score_entropy_loss_uniform(output, x_0, x_t, cumulative_noise_levels, inst_noise_levels, mask_token, valid_mask):
    """
    Score entropy loss for uniform discrete diffusion.

    The repository at https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/ was referenced and used during the development of this loss function.
    Credit to the authors.
    
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
    num_content_tokens = V - len(SPECIAL_TOKENS)
    output = output[:,:,:num_content_tokens]
    V = num_content_tokens
    
    # Replace BEOSPANK (special token) indices with valid content token indices
    # Since these positions are zeroed out anyway, the actual values don't matter
    x_0 = torch.where(~valid_mask, 0, x_0)
    x_t = torch.where(~valid_mask, 0, x_t)
    
    # Compute esigm1 = exp(sigma) - 1, with numerical stability for small sigma
    esigm1 = torch.where(
        cumulative_noise_levels < 0.5,
        torch.expm1(cumulative_noise_levels),
        torch.exp(cumulative_noise_levels) - 1
    )  # shape (B, 1)
    
    # Ratio for uniform discrete diffusion
    ratio = 1 - V / (esigm1 + V)  # shape (B, 1)
    
    # Positive term: exp(score).mean(dim=-1) - exp(score)[x_t] / V
    # Clamp output to prevent exp overflow in float16 (max ~10.5) or float32 (max ~88)
    max_exp_input = 10.0 if output.dtype == torch.float16 else 88.0
    output_clamped = torch.clamp(output, max=max_exp_input)
    score_exp = torch.exp(output_clamped)  # [B, L, V]
    pos_term = score_exp.mean(dim=-1) - torch.gather(score_exp, -1, x_t[..., None]).squeeze(-1) / V  # [B, L]
    
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
    # Remove clamping - use ratio_squeezed directly for log
    const_unchanged = (V - 1) / V * ratio_squeezed * (torch.log(ratio_squeezed) - 1)  # [B]
    
    # Case 2: x_t != x_0  
    # Remove clamping - use ratio_squeezed directly for log and division
    const_changed = ((-torch.log(ratio_squeezed) - 1) / ratio_squeezed - (V - 2)) / V  # [B]
    
    # Combine constant terms and broadcast to [B, L]
    const = torch.where(unchanged_positions, const_unchanged.unsqueeze(-1), const_changed.unsqueeze(-1))  # [B, L]
    
    # Final entropy: pos_term - neg_term + const
    entropy = pos_term - neg_term + const  # [B, L]
    
    # Apply instantaneous noise levels and valid mask
    per_residue = entropy * inst_noise_levels  # [B, L] * [B, 1] -> [B, L]
    
    # Apply mask to per_residue losses and compute average over valid positions only
    per_residue_masked = per_residue * valid_mask.to(per_residue.dtype)  # [B, L] - match dtypes for fp16  
    valid_counts = valid_mask.sum(dim=1).to(per_residue.dtype)  # [B] - match dtypes for fp16
    per_protein = per_residue_masked.sum(dim=1) / valid_counts  # (B,)

    retval = per_protein.mean(0)  # scalar
    
    # NaN debugging - check if the return value contains NaN
    if retval.isnan().any() or retval.isinf().any():
        import os
        import sys
        from datetime import datetime
        
        # Simplified debugging focused on exp overflow
        debug_output = []
        debug_output.append("="*80)
        debug_output.append("EXP OVERFLOW DETECTED in score_entropy_loss_uniform!")
        debug_output.append(f"Time: {datetime.now()}")
        debug_output.append("="*80)
        
        # Basic info
        debug_output.append(f"\nBatch info:")
        debug_output.append(f"  output shape: {output.shape}, dtype: {output.dtype}")
        debug_output.append(f"  V (vocabulary size): {V}")
        
        # Focus on the problematic exp operation
        debug_output.append(f"\nEXP OVERFLOW ANALYSIS:")
        debug_output.append(f"  output min: {output.min():.6f}")
        debug_output.append(f"  output max: {output.max():.6f}")
        debug_output.append(f"  output mean: {output.mean():.6f}")
        
        # Check extreme values
        exp_threshold = 10.0 if output.dtype == torch.float16 else 88.0
        extreme_mask = output > exp_threshold
        num_extreme = extreme_mask.sum().item()
        
        debug_output.append(f"\n  Float16 safe exp threshold: {exp_threshold}")
        debug_output.append(f"  Values exceeding threshold: {num_extreme}")
        
        if num_extreme > 0:
            extreme_locations = torch.where(extreme_mask)
            debug_output.append(f"  Extreme value locations (first 10):")
            for i in range(min(10, len(extreme_locations[0]))):
                batch_idx = extreme_locations[0][i].item()
                seq_idx = extreme_locations[1][i].item()
                vocab_idx = extreme_locations[2][i].item()
                value = output[batch_idx, seq_idx, vocab_idx].item()
                debug_output.append(f"    Batch {batch_idx}, Seq {seq_idx}, Vocab {vocab_idx}: {value:.6f}")
        
        # Show exp results
        debug_output.append(f"\n  score_exp (exp(output)) stats:")
        debug_output.append(f"    min: {score_exp.min()}")
        debug_output.append(f"    max: {score_exp.max()}")
        debug_output.append(f"    has_inf: {score_exp.isinf().any()}")
        debug_output.append(f"    num_inf: {score_exp.isinf().sum().item()}")
        
        # Final loss value
        debug_output.append(f"\n  Final loss: {retval}")
        
        debug_output.append("\n" + "="*80)
        debug_output.append("SOLUTION: Use exp clamping as implemented in the code.")
        debug_output.append("="*80)
        
        # Print to console
        print("\n" + "="*80)
        print("EXP OVERFLOW DETECTED! Saving debug info and terminating...")
        print("="*80)
        
        # Save to text file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_filename = os.path.join(current_dir, f"CRASH_{timestamp}.txt")
        
        with open(debug_filename, 'w') as f:
            f.write('\n'.join(debug_output))
        
        print(f"Debug output saved to: {debug_filename}")
        print("Terminating training to prevent further overflow issues.")
        
        # Exit the program
        sys.exit(1)
    
    return retval

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
