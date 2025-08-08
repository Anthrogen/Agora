"""Utility functions for masking strategies used by dataloaders."""

import torch
import numpy as np
import math


def _get_noise_levels(s_min, s_max, T, schedule_type="linear"):
    """Generate instantaneous and cumulative noise levels for discrete diffusion.
    
    Args:
        s_min: Minimum noise level (sigma_min)
        s_max: Maximum noise level (sigma_max)
        T: Number of timesteps
        schedule_type: "linear", "inverted_u", or "uniform"
        
    Returns:
        inst_noise_levels: Tensor of shape (T,) with instantaneous noise at each timestep
        cumulative_noise_levels: Tensor of shape (T,) with cumulative noise up to each timestep
    """
    t = torch.arange(T, dtype=torch.float32)
    normalized_t = t / (T - 1) if T > 1 else torch.zeros_like(t)
    
    if schedule_type == "linear":
        # Linear schedule: σ(t) = σ_min + (σ_max - σ_min) * t/T
        inst_noise_levels = s_min + (s_max - s_min) * normalized_t
        
        # Cumulative noise: ∫_0^t σ(s) ds 
        # For linear schedule σ(s) = σ_min + (σ_max - σ_min) * s
        # ∫_0^t σ(s) ds = σ_min * t + 0.5 * (σ_max - σ_min) * t^2
        min_cumulative_noise = -torch.log(torch.tensor(1 - 0.05))  # ≈ 0.051293
        cumulative_noise_levels = min_cumulative_noise + s_min * normalized_t + 0.5 * (s_max - s_min) * normalized_t**2
        
    elif schedule_type == "inverted_u":
        # Inverted-U schedule: concentrated training time distribution
        mask_probs = torch.zeros(T)
        
        for i in range(T):
            t_norm = i / (T - 1)  # 0 to 1
            # Transform uniform t_norm to create concentrated density
            # Use inverse sine to concentrate values in the middle
            if t_norm <= 0.5:
                # First half: map [0, 0.5] to [0.05, 0.5] with more density in middle
                local_t = t_norm * 2  # Scale to [0, 1]
                # Use sqrt to concentrate more values toward the end (middle of overall range)
                transformed = math.sqrt(local_t)
                mask_probs[i] = 0.05 + 0.45 * transformed
            else:
                # Second half: map [0.5, 1] to [0.5, 0.95] with more density in middle  
                local_t = (t_norm - 0.5) * 2  # Scale to [0, 1]
                # Use (1 - sqrt(1 - t)) to concentrate more values toward the beginning (middle of overall range)
                transformed = 1 - math.sqrt(1 - local_t)
                mask_probs[i] = 0.5 + 0.45 * transformed
        
        # Convert to cumulative noise levels
        cumulative_noise_levels = -torch.log(1 - mask_probs + 1e-8)
        
        # Compute instantaneous noise levels as derivatives
        inst_noise_levels = torch.zeros_like(cumulative_noise_levels)
        inst_noise_levels[0] = cumulative_noise_levels[0]
        
        for i in range(1, T):
            dt = 1.0 / (T - 1)
            inst_noise_levels[i] = (cumulative_noise_levels[i] - cumulative_noise_levels[i-1]) / dt
        
        # Clamp instantaneous noise for return
        inst_noise_levels = torch.clamp(inst_noise_levels, s_min, s_max)
        

    elif schedule_type == "uniform":
        # Uniform schedule: equal time spent at all mask percentages
        # Linear progression from 5% to 95% mask probability
        mask_probs = 0.05 + 0.9 * normalized_t  # Maps [0,1] to [0.05, 0.95]
        
        # Convert to cumulative noise levels
        cumulative_noise_levels = -torch.log(1 - mask_probs + 1e-8)
        
        # Compute instantaneous noise as derivative of cumulative noise
        # d/dt[-log(1 - mask_probs)] = d/dt[-log(1 - (0.05 + 0.9*t))] = 0.9 / (1 - (0.05 + 0.9*t))
        inst_noise_levels = 0.9 / (1 - mask_probs + 1e-8)
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Must be 'linear', 'inverted_u', or 'uniform'")
    
    return inst_noise_levels, cumulative_noise_levels


def _sample_betalinear30(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from betalinear30 distribution. 80% Beta(3,9), 20% Uniform(0,1), avg ~30%"""
    mask_rates = torch.zeros(batch_size, device=device)
    use_beta = torch.rand(batch_size) < 0.8  # Choose distribution for each batch element
    
    beta_samples = torch.distributions.Beta(3.0, 9.0).sample((batch_size,)).to(device)  # Beta(3, 9) samples
    uniform_samples = torch.rand(batch_size, device=device)  # Uniform(0, 1) samples
    mask_rates = torch.where(use_beta.to(device), beta_samples, uniform_samples)  # Combine based on use_beta
    
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values


def _sample_cosine(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from cosine distribution using sin(u * π/2) for higher density near 1."""
    u = torch.rand(batch_size, device=device)  # Sample uniform values
    mask_rates = torch.sin(u * np.pi / 2)  # Apply sine transformation
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values


def _sample_sqrt(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample valid rates from sqrt distribution with linearly increasing PDF from 0 to 2 (favors high mask rates)."""
    u = torch.rand(batch_size, device=device)  # Sample uniform values
    mask_rates = torch.sqrt(u)  # Apply sqrt transformation for PDF f(x) = 2x
    return torch.clamp(mask_rates, min=0.05, max=0.95)  # Clamp to avoid extreme values