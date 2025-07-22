#!/usr/bin/env python3
"""
Plot the learning rate schedule for the warmup decay scheduler.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to import from train module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.train.train import create_warmup_decay

def plot_learning_rate_schedule(base_lr: float, min_lr: float, decay_epochs: int, warmup_epochs: int, total_steps: int = 20000):
    """
    Plot the learning rate schedule for the warmup decay scheduler.
    
    Args:
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        decay_epochs: Number of epochs for decay phase
        warmup_epochs: Number of epochs for warmup phase
        total_steps: Total number of steps to plot
    """
    # Create the lr_lambda function
    lr_lambda = create_warmup_decay(base_lr, min_lr, decay_epochs, warmup_epochs)
    
    # Generate learning rates for each step
    steps = np.arange(total_steps)
    learning_rates = []
    
    for step in steps:
        lr_multiplier = lr_lambda(step)
        actual_lr = base_lr * lr_multiplier
        learning_rates.append(actual_lr)
    
    learning_rates = np.array(learning_rates)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(steps, learning_rates, 'b-', linewidth=2, label='Learning Rate')
    
    # Add phase annotations
    plt.axvline(x=warmup_epochs, color='green', linestyle='--', alpha=0.7, 
                label=f'End of Warmup (step {warmup_epochs})')
    plt.axvline(x=warmup_epochs + decay_epochs, color='red', linestyle='--', alpha=0.7, 
                label=f'End of Decay (step {warmup_epochs + decay_epochs})')
    
    # Add horizontal lines for reference
    plt.axhline(y=base_lr, color='orange', linestyle=':', alpha=0.5, 
                label=f'Base LR ({base_lr})')
    plt.axhline(y=min_lr, color='purple', linestyle=':', alpha=0.5, 
                label=f'Min LR ({min_lr})')
    
    # Customize the plot
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Warmup Decay Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    y_min = min_lr * 0.8
    y_max = base_lr * 1.1
    plt.ylim(y_min, y_max)
    
    # Add text annotations for phases
    plt.annotate(f'Warmup Phase\n({warmup_epochs} steps)', 
                xy=(warmup_epochs/2, base_lr * 0.7), 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.annotate(f'Decay Phase\n({decay_epochs} steps)', 
                xy=(warmup_epochs + decay_epochs/2, base_lr * 0.7), 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.annotate('Constant Phase', 
                xy=(warmup_epochs + decay_epochs + (total_steps - warmup_epochs - decay_epochs)/2, min_lr * 1.5), 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'linear_decay_schedule.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    # Show statistics
    print(f"\nLearning Rate Schedule Statistics:")
    print(f"Base Learning Rate: {base_lr}")
    print(f"Min Learning Rate: {min_lr}")
    print(f"Warmup Steps: {warmup_epochs}")
    print(f"Decay Steps: {decay_epochs}")
    print(f"Total Steps Plotted: {total_steps}")
    print(f"LR at step 0: {learning_rates[0]:.6f}")
    print(f"LR at step {warmup_epochs}: {learning_rates[warmup_epochs]:.6f}")
    print(f"LR at step {warmup_epochs + decay_epochs}: {learning_rates[warmup_epochs + decay_epochs]:.6f}")
    print(f"LR at final step: {learning_rates[-1]:.6f}")
    
    # Show the plot
    plt.show()
    
    return steps, learning_rates

if __name__ == "__main__":
    # Parameters from fsq_stage_1_config.yaml
    base_learning_rate = 0.00025
    min_learning_rate = 0.00005
    num_epochs_decay = 10000
    num_epochs_warmup = 100
    total_steps = 20000
    
    print("Plotting Warmup Decay Learning Rate Schedule...")
    print(f"Configuration:")
    print(f"  Base LR: {base_learning_rate}")
    print(f"  Min LR: {min_learning_rate}")
    print(f"  Warmup Steps: {num_epochs_warmup}")
    print(f"  Decay Steps: {num_epochs_decay}")
    print(f"  Total Steps: {total_steps}")
    
    steps, learning_rates = plot_learning_rate_schedule(
        base_learning_rate, 
        min_learning_rate, 
        num_epochs_decay, 
        num_epochs_warmup, 
        total_steps
    ) 