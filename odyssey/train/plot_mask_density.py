#!/usr/bin/env python3
"""
Plot mask percentage density distributions to visualize how much time is spent 
at different mask percentages for various noise schedules.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add both parent directory and src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from odyssey.src.dataloader import _get_noise_levels

def plot_mask_density():
    """Plot density distributions of mask percentages for different schedules."""
    
    # Parameters
    sigma_min = 0.31
    sigma_max = 5.68
    num_timesteps = 100
    
    # Get noise levels for different schedules
    schedules = {
        'Linear': 'linear',
        'Inverted-U': 'inverted_u', 
        'Uniform': 'uniform'
    }
    
    # Create subplots - one for each schedule
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['blue', 'red', 'green']
    
    for i, (name, schedule_type) in enumerate(schedules.items()):
        ax = axes[i]
        try:
            inst_noise, cumulative_noise = _get_noise_levels(
                sigma_min, sigma_max, num_timesteps, schedule_type
            )
            
            # Convert to mask probabilities
            mask_probs = 1 - torch.exp(-cumulative_noise)
            mask_percentages = mask_probs.numpy() * 100  # Convert to percentages
            
            # Create histogram/density plot with higher granularity
            # Each timestep represents equal probability (1/T), so each gets weight 1/T
            weights = np.ones(len(mask_percentages)) / len(mask_percentages)
            
            # Plot histogram with more bins for higher granularity
            n_bins = 100  # Increased from 50 to 100 for better granularity
            ax.hist(mask_percentages, bins=n_bins, alpha=0.7, 
                   color=colors[i], weights=weights, density=False, 
                   histtype='stepfilled', edgecolor='black', linewidth=0.5)
            
            # Configure individual subplot
            ax.set_xlabel('Mask Percentage (%)', fontsize=12)
            ax.set_ylabel('Probability Density\n(Time Spent)', fontsize=12)
            ax.set_title(f'{name} Schedule', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            
            # Add vertical lines for reference percentages
            for pct in [20, 40, 60, 80]:
                ax.axvline(x=pct, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(pct, ax.get_ylim()[1] * 0.95, f'{pct}%', 
                       horizontalalignment='center', fontsize=9, alpha=0.7)
            
            # Add statistics text box
            mean_mask = np.mean(mask_percentages)
            std_mask = np.std(mask_percentages)
            ax.text(0.02, 0.98, f'Mean: {mean_mask:.1f}%\nStd: {std_mask:.1f}%', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
        except ValueError as e:
            print(f"Warning: Could not plot {name}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig('mask_percentage_density_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create a comparison plot showing all three as lines
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, (name, schedule_type) in enumerate(schedules.items()):
        try:
            inst_noise, cumulative_noise = _get_noise_levels(
                sigma_min, sigma_max, num_timesteps, schedule_type
            )
            
            # Convert to mask probabilities
            mask_probs = 1 - torch.exp(-cumulative_noise)
            mask_percentages = mask_probs.numpy() * 100
            
            # Create histogram data
            weights = np.ones(len(mask_percentages)) / len(mask_percentages)
            hist, bin_edges = np.histogram(mask_percentages, bins=100, weights=weights)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot as line
            ax2.plot(bin_centers, hist, label=name, color=colors[i], linewidth=2)
            
        except ValueError as e:
            print(f"Warning: Could not plot {name}: {e}")
            continue
    
    # Configure comparison plot
    ax2.set_xlabel('Mask Percentage (%)', fontsize=14)
    ax2.set_ylabel('Probability Density (Time Spent)', fontsize=14)
    ax2.set_title('Comparison of Training Time Distribution Across Mask Percentages', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.set_xlim(0, 100)
    
    # Add vertical lines for reference percentages
    for pct in [20, 40, 60, 80]:
        ax2.axvline(x=pct, color='gray', linestyle='--', alpha=0.5)
        ax2.text(pct, ax2.get_ylim()[1] * 0.95, f'{pct}%', 
               horizontalalignment='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('mask_percentage_density_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis comparing schedules
    print("\n" + "="*80)
    print("MASK PERCENTAGE DENSITY ANALYSIS")
    print("="*80)
    
    # Analyze time spent in different ranges for each schedule
    ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    range_names = ["Low (0-25%)", "Medium-Low (25-50%)", "Medium-High (50-75%)", "High (75-100%)"]
    
    for name, schedule_type in schedules.items():
        try:
            inst_noise, cumulative_noise = _get_noise_levels(
                sigma_min, sigma_max, num_timesteps, schedule_type
            )
            mask_probs = 1 - torch.exp(-cumulative_noise)
            mask_percentages = mask_probs.numpy() * 100
            
            print(f"\n{name} Schedule:")
            print(f"  Mean: {np.mean(mask_percentages):.1f}%, Std: {np.std(mask_percentages):.1f}%")
            total_time = 0
            for (low, high), range_name in zip(ranges, range_names):
                count = np.sum((mask_percentages >= low) & (mask_percentages < high))
                percentage = (count / len(mask_percentages)) * 100
                total_time += percentage
                print(f"  {range_name:20s}: {percentage:5.1f}% of training time")
            print(f"  {'Total':20s}: {total_time:5.1f}%")
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    
    # Compare the new uniform schedule with validation timesteps
    print(f"\n" + "="*60)
    print("UNIFORM - VALIDATION TIMESTEPS ANALYSIS")
    print("="*60)
    
    try:
        inst_noise, cumulative_noise = _get_noise_levels(
            sigma_min, sigma_max, num_timesteps, 'uniform'
        )
        mask_probs = 1 - torch.exp(-cumulative_noise)
        
        validation_timesteps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        print("Mask percentages at validation timesteps:")
        for t in validation_timesteps:
            if t < num_timesteps:
                mask_pct = mask_probs[t].item() * 100
                print(f"  t={t:2d}: {mask_pct:5.1f}%")
        
        # Show characteristics of uniform schedule
        print(f"\nUniform schedule characteristics:")
        print(f"  - Equal time spent at all mask percentages")
        print(f"  - Linear progression from 5% to 95% mask probability") 
        print(f"  - Perfect flat density distribution")
        print(f"  - No bias towards lower or higher mask percentages")
                
    except Exception as e:
        print(f"Error analyzing uniform schedule: {e}")

if __name__ == "__main__":
    plot_mask_density() 