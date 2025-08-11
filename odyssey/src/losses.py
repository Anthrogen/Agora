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
    neg_term = neg_term * rel_ind.to(neg_term.dtype)  # Zero out non-masked positions - match dtypes for fp16
    
    # Positive term: sum of exp(score) over all tokens except the mask token
    # Create a one-hot mask to exclude the mask token dimension while maintaining gradients
    mask_onehot = F.one_hot(torch.tensor(mask_token, device=output.device), num_classes=V).to(output.dtype)  # [V] - match dtypes for fp16
    exclude_mask = 1.0 - mask_onehot  # [V] - zeros out the mask token position
    
    score_exp = torch.exp(output)  # [B, L, V]
    pos_term = (score_exp * exclude_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, L]
    pos_term = pos_term * rel_ind.to(pos_term.dtype)  # Only for masked positions - match dtypes for fp16
    
    # Constant term: ratio * (log(ratio) - 1) for masked positions, 0 for others
    const = ratio * (torch.log(torch.clamp(ratio, min=1e-10)) - 1)  # [B, 1]
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
    per_residue_masked = per_residue * valid_mask.to(per_residue.dtype)  # [B, L] - match dtypes for fp16  
    valid_counts = valid_mask.sum(dim=1).to(per_residue.dtype)  # [B] - match dtypes for fp16
    per_protein = per_residue_masked.sum(dim=1) / valid_counts  # (B,)

    retval = per_protein.mean(0)  # scalar
    
    # NaN debugging - check if the return value contains NaN
    if retval.isnan().any() or retval.isinf().any():
        import os
        from datetime import datetime
        
        # Prepare debugging output
        debug_output = []
        debug_output.append("\n" + "="*80)
        debug_output.append("NaN/Inf DETECTED in score_entropy_loss_uniform!")
        debug_output.append("="*80)
        
        # Also print to console
        print("\n" + "="*80)
        print("NaN/Inf DETECTED in score_entropy_loss_uniform!")
        print("="*80)
        
        # Input shapes and types
        debug_output.append("\n--- INPUT INFORMATION ---")
        debug_output.append(f"output shape: {output.shape}, dtype: {output.dtype}")
        debug_output.append(f"x_0 shape: {x_0.shape}, dtype: {x_0.dtype}")
        debug_output.append(f"x_t shape: {x_t.shape}, dtype: {x_t.dtype}")
        debug_output.append(f"cumulative_noise_levels shape: {cumulative_noise_levels.shape}, dtype: {cumulative_noise_levels.dtype}")
        debug_output.append(f"inst_noise_levels shape: {inst_noise_levels.shape}, dtype: {inst_noise_levels.dtype}")
        debug_output.append(f"valid_mask shape: {valid_mask.shape}, dtype: {valid_mask.dtype}")
        debug_output.append(f"mask_token: {mask_token}")
        debug_output.append(f"V (vocabulary size): {V}")
        
        print("\n--- INPUT INFORMATION ---")
        print(f"output shape: {output.shape}, dtype: {output.dtype}")
        print(f"x_0 shape: {x_0.shape}, dtype: {x_0.dtype}")
        print(f"x_t shape: {x_t.shape}, dtype: {x_t.dtype}")
        print(f"cumulative_noise_levels shape: {cumulative_noise_levels.shape}, dtype: {cumulative_noise_levels.dtype}")
        print(f"inst_noise_levels shape: {inst_noise_levels.shape}, dtype: {inst_noise_levels.dtype}")
        print(f"valid_mask shape: {valid_mask.shape}, dtype: {valid_mask.dtype}")
        print(f"mask_token: {mask_token}")
        print(f"V (vocabulary size): {V}")
        
        # Full input tensors
        debug_output.append("\n--- FULL INPUT TENSORS ---")
        debug_output.append(f"x_0:\n{x_0}")
        debug_output.append(f"\nx_t:\n{x_t}")
        debug_output.append(f"\ncumulative_noise_levels:\n{cumulative_noise_levels}")
        debug_output.append(f"\ninst_noise_levels:\n{inst_noise_levels}")
        debug_output.append(f"\nvalid_mask:\n{valid_mask}")
        
        print("\n--- FULL INPUT TENSORS ---")
        print(f"x_0:\n{x_0}")
        print(f"\nx_t:\n{x_t}")
        print(f"\ncumulative_noise_levels:\n{cumulative_noise_levels}")
        print(f"\ninst_noise_levels:\n{inst_noise_levels}")
        print(f"\nvalid_mask:\n{valid_mask}")
        
        # Check for NaN/Inf in inputs
        debug_output.append("\n--- INPUT NaN/Inf CHECKS ---")
        debug_output.append(f"output has NaN: {output.isnan().any()}, has Inf: {output.isinf().any()}")
        debug_output.append(f"x_0 has NaN: {x_0.isnan().any()}, has Inf: {x_0.isinf().any()}")
        debug_output.append(f"x_t has NaN: {x_t.isnan().any()}, has Inf: {x_t.isinf().any()}")
        debug_output.append(f"cumulative_noise_levels has NaN: {cumulative_noise_levels.isnan().any()}, has Inf: {cumulative_noise_levels.isinf().any()}")
        debug_output.append(f"inst_noise_levels has NaN: {inst_noise_levels.isnan().any()}, has Inf: {inst_noise_levels.isinf().any()}")
        
        print("\n--- INPUT NaN/Inf CHECKS ---")
        print(f"output has NaN: {output.isnan().any()}, has Inf: {output.isinf().any()}")
        print(f"x_0 has NaN: {x_0.isnan().any()}, has Inf: {x_0.isinf().any()}")
        print(f"x_t has NaN: {x_t.isnan().any()}, has Inf: {x_t.isinf().any()}")
        print(f"cumulative_noise_levels has NaN: {cumulative_noise_levels.isnan().any()}, has Inf: {cumulative_noise_levels.isinf().any()}")
        print(f"inst_noise_levels has NaN: {inst_noise_levels.isnan().any()}, has Inf: {inst_noise_levels.isinf().any()}")
        
        # Output statistics
        debug_output.append("\n--- OUTPUT STATISTICS ---")
        debug_output.append(f"output min: {output.min()}, max: {output.max()}, mean: {output.mean()}")
        if output.isnan().any():
            debug_output.append(f"Number of NaN values in output: {output.isnan().sum()}")
            debug_output.append(f"NaN locations in output: {torch.where(output.isnan())}")
        
        print("\n--- OUTPUT STATISTICS ---")
        print(f"output min: {output.min()}, max: {output.max()}, mean: {output.mean()}")
        if output.isnan().any():
            print(f"Number of NaN values in output: {output.isnan().sum()}")
            print(f"NaN locations in output: {torch.where(output.isnan())}")
        
        # Intermediate values
        debug_output.append("\n--- INTERMEDIATE VALUES ---")
        debug_output.append(f"\nesigm1:\n{esigm1}")
        debug_output.append(f"esigm1 has NaN: {esigm1.isnan().any()}, has Inf: {esigm1.isinf().any()}")
        
        debug_output.append(f"\nratio:\n{ratio}")
        debug_output.append(f"ratio has NaN: {ratio.isnan().any()}, has Inf: {ratio.isinf().any()}")
        
        debug_output.append(f"\nsexp (exp(output)) min: {sexp.min()}, max: {sexp.max()}, mean: {sexp.mean()}")
        debug_output.append(f"sexp has NaN: {sexp.isnan().any()}, has Inf: {sexp.isinf().any()}")
        
        debug_output.append(f"\npos_term:\n{pos_term}")
        debug_output.append(f"pos_term has NaN: {pos_term.isnan().any()}, has Inf: {pos_term.isinf().any()}")
        
        debug_output.append(f"\nscore_mean:\n{score_mean}")
        debug_output.append(f"score_xt:\n{score_xt}")
        debug_output.append(f"score_x0:\n{score_x0}")
        
        debug_output.append(f"\nneg_term_base:\n{neg_term_base}")
        debug_output.append(f"neg_term_unchanged:\n{neg_term_unchanged}")
        debug_output.append(f"neg_term_changed:\n{neg_term_changed}")
        debug_output.append(f"neg_term:\n{neg_term}")
        debug_output.append(f"neg_term has NaN: {neg_term.isnan().any()}, has Inf: {neg_term.isinf().any()}")
        
        debug_output.append(f"\nunchanged_positions:\n{unchanged_positions}")
        debug_output.append(f"Number of unchanged positions: {unchanged_positions.sum()}")
        
        debug_output.append(f"\nratio_squeezed:\n{ratio_squeezed}")
        debug_output.append(f"const_unchanged:\n{const_unchanged}")
        debug_output.append(f"const_changed:\n{const_changed}")
        debug_output.append(f"const:\n{const}")
        debug_output.append(f"const has NaN: {const.isnan().any()}, has Inf: {const.isinf().any()}")
        
        debug_output.append(f"\nentropy:\n{entropy}")
        debug_output.append(f"entropy has NaN: {entropy.isnan().any()}, has Inf: {entropy.isinf().any()}")
        
        debug_output.append(f"\nper_residue:\n{per_residue}")
        debug_output.append(f"per_residue has NaN: {per_residue.isnan().any()}, has Inf: {per_residue.isinf().any()}")
        
        debug_output.append(f"\nper_residue_masked:\n{per_residue_masked}")
        debug_output.append(f"valid_counts:\n{valid_counts}")
        debug_output.append(f"per_protein:\n{per_protein}")
        debug_output.append(f"per_protein has NaN: {per_protein.isnan().any()}, has Inf: {per_protein.isinf().any()}")
        
        debug_output.append(f"\nretval: {retval}")
        
        print("\n--- INTERMEDIATE VALUES ---")
        print(f"\nesigm1:\n{esigm1}")
        print(f"esigm1 has NaN: {esigm1.isnan().any()}, has Inf: {esigm1.isinf().any()}")
        
        print(f"\nratio:\n{ratio}")
        print(f"ratio has NaN: {ratio.isnan().any()}, has Inf: {ratio.isinf().any()}")
        
        print(f"\nsexp (exp(output)) min: {sexp.min()}, max: {sexp.max()}, mean: {sexp.mean()}")
        print(f"sexp has NaN: {sexp.isnan().any()}, has Inf: {sexp.isinf().any()}")
        
        print(f"\npos_term:\n{pos_term}")
        print(f"pos_term has NaN: {pos_term.isnan().any()}, has Inf: {pos_term.isinf().any()}")
        
        print(f"\nscore_mean:\n{score_mean}")
        print(f"score_xt:\n{score_xt}")
        print(f"score_x0:\n{score_x0}")
        
        print(f"\nneg_term_base:\n{neg_term_base}")
        print(f"neg_term_unchanged:\n{neg_term_unchanged}")
        print(f"neg_term_changed:\n{neg_term_changed}")
        print(f"neg_term:\n{neg_term}")
        print(f"neg_term has NaN: {neg_term.isnan().any()}, has Inf: {neg_term.isinf().any()}")
        
        print(f"\nunchanged_positions:\n{unchanged_positions}")
        print(f"Number of unchanged positions: {unchanged_positions.sum()}")
        
        print(f"\nratio_squeezed:\n{ratio_squeezed}")
        print(f"const_unchanged:\n{const_unchanged}")
        print(f"const_changed:\n{const_changed}")
        print(f"const:\n{const}")
        print(f"const has NaN: {const.isnan().any()}, has Inf: {const.isinf().any()}")
        
        print(f"\nentropy:\n{entropy}")
        print(f"entropy has NaN: {entropy.isnan().any()}, has Inf: {entropy.isinf().any()}")
        
        print(f"\nper_residue:\n{per_residue}")
        print(f"per_residue has NaN: {per_residue.isnan().any()}, has Inf: {per_residue.isinf().any()}")
        
        print(f"\nper_residue_masked:\n{per_residue_masked}")
        print(f"valid_counts:\n{valid_counts}")
        print(f"per_protein:\n{per_protein}")
        print(f"per_protein has NaN: {per_protein.isnan().any()}, has Inf: {per_protein.isinf().any()}")
        
        print(f"\nretval: {retval}")
        
        # Additional diagnostics
        debug_output.append("\n--- ADDITIONAL DIAGNOSTICS ---")
        
        # Check for division by zero
        if (valid_counts == 0).any():
            debug_output.append(f"WARNING: valid_counts contains zeros at indices: {torch.where(valid_counts == 0)}")
        
        # Check extreme values in exp operation
        debug_output.append(f"\nExtreme values in output that might cause exp overflow:")
        debug_output.append(f"output values > 80: {(output > 80).sum().item()} locations")
        if (output > 80).any():
            debug_output.append(f"Max output value: {output.max()}")
            debug_output.append(f"Locations of extreme values: {torch.where(output > 80)}")
        
        # Check for numerical issues in log operations
        debug_output.append(f"\nratio_squeezed values close to 0 (might cause log issues):")
        debug_output.append(f"ratio_squeezed < 1e-10: {(ratio_squeezed < 1e-10).sum().item()} locations")
        
        # Check gather operations
        debug_output.append(f"\nx_0 min: {x_0.min()}, max: {x_0.max()}")
        debug_output.append(f"x_t min: {x_t.min()}, max: {x_t.max()}")
        debug_output.append(f"Are all indices valid for vocabulary size {V}? x_0: {(x_0 >= 0).all() and (x_0 < V).all()}, x_t: {(x_t >= 0).all() and (x_t < V).all()}")
        
        debug_output.append("\n" + "="*80)
        debug_output.append("END OF NaN DEBUGGING OUTPUT")
        debug_output.append("="*80 + "\n")
        
        print("\n--- ADDITIONAL DIAGNOSTICS ---")
        
        # Check for division by zero
        if (valid_counts == 0).any():
            print(f"WARNING: valid_counts contains zeros at indices: {torch.where(valid_counts == 0)}")
        
        # Check extreme values in exp operation
        print(f"\nExtreme values in output that might cause exp overflow:")
        print(f"output values > 80: {(output > 80).sum().item()} locations")
        if (output > 80).any():
            print(f"Max output value: {output.max()}")
            print(f"Locations of extreme values: {torch.where(output > 80)}")
        
        # Check for numerical issues in log operations
        print(f"\nratio_squeezed values close to 0 (might cause log issues):")
        print(f"ratio_squeezed < 1e-10: {(ratio_squeezed < 1e-10).sum().item()} locations")
        
        # Check gather operations
        print(f"\nx_0 min: {x_0.min()}, max: {x_0.max()}")
        print(f"x_t min: {x_t.min()}, max: {x_t.max()}")
        print(f"Are all indices valid for vocabulary size {V}? x_0: {(x_0 >= 0).all() and (x_0 < V).all()}, x_t: {(x_t >= 0).all() and (x_t < V).all()}")
        
        print("\n" + "="*80)
        print("END OF NaN DEBUGGING OUTPUT")
        print("="*80 + "\n")
        
        # Add detailed tensor values to debug output
        debug_output.append("\n" + "="*80)
        debug_output.append("DETAILED TENSOR VALUES")
        debug_output.append("="*80)
        
        debug_output.append(f"\n--- FULL OUTPUT TENSOR ---")
        debug_output.append(f"output:\n{output}")
        
        debug_output.append(f"\n--- ALL INTERMEDIATE TENSORS ---")
        debug_output.append(f"\nesigm1 (full tensor):\n{esigm1}")
        debug_output.append(f"\nratio (full tensor):\n{ratio}")
        debug_output.append(f"\nsexp (exp(output)) - showing min/max/mean due to size:")
        debug_output.append(f"sexp min: {sexp.min()}, max: {sexp.max()}, mean: {sexp.mean()}")
        debug_output.append(f"sexp shape: {sexp.shape}")
        debug_output.append(f"sexp (first 5x5 elements if large):\n{sexp.flatten()[:25] if sexp.numel() > 25 else sexp}")
        
        debug_output.append(f"\npos_term (full tensor):\n{pos_term}")
        debug_output.append(f"\nscore_mean (full tensor):\n{score_mean}")
        debug_output.append(f"\nscore_xt (full tensor):\n{score_xt}")
        debug_output.append(f"\nscore_x0 (full tensor):\n{score_x0}")
        debug_output.append(f"\nneg_term_base (full tensor):\n{neg_term_base}")
        debug_output.append(f"\nneg_term_unchanged (full tensor):\n{neg_term_unchanged}")
        debug_output.append(f"\nneg_term_changed (full tensor):\n{neg_term_changed}")
        debug_output.append(f"\nneg_term (full tensor):\n{neg_term}")
        debug_output.append(f"\nratio_squeezed (full tensor):\n{ratio_squeezed}")
        debug_output.append(f"\nconst_unchanged (full tensor):\n{const_unchanged}")
        debug_output.append(f"\nconst_changed (full tensor):\n{const_changed}")
        debug_output.append(f"\nconst (full tensor):\n{const}")
        debug_output.append(f"\nentropy (full tensor):\n{entropy}")
        debug_output.append(f"\nper_residue (full tensor):\n{per_residue}")
        debug_output.append(f"\nper_residue_masked (full tensor):\n{per_residue_masked}")
        debug_output.append(f"\nvalid_counts (full tensor):\n{valid_counts}")
        debug_output.append(f"\nper_protein (full tensor):\n{per_protein}")
        
        debug_output.append(f"\n--- TENSOR STATISTICS ---")
        debug_output.append(f"esigm1 - min: {esigm1.min()}, max: {esigm1.max()}, mean: {esigm1.mean()}")
        debug_output.append(f"ratio - min: {ratio.min()}, max: {ratio.max()}, mean: {ratio.mean()}")
        debug_output.append(f"pos_term - min: {pos_term.min()}, max: {pos_term.max()}, mean: {pos_term.mean()}")
        debug_output.append(f"neg_term - min: {neg_term.min()}, max: {neg_term.max()}, mean: {neg_term.mean()}")
        debug_output.append(f"const - min: {const.min()}, max: {const.max()}, mean: {const.mean()}")
        debug_output.append(f"entropy - min: {entropy.min()}, max: {entropy.max()}, mean: {entropy.mean()}")
        debug_output.append(f"per_residue - min: {per_residue.min()}, max: {per_residue.max()}, mean: {per_residue.mean()}")
        debug_output.append(f"per_protein - min: {per_protein.min()}, max: {per_protein.max()}, mean: {per_protein.mean()}")
        
        # Save to text file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_filename = os.path.join(current_dir, f"nan_debug_{timestamp}.txt")
        
        with open(debug_filename, 'w') as f:
            f.write('\n'.join(debug_output))
        
        print(f"Debug output saved to: {debug_filename}")
    
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
