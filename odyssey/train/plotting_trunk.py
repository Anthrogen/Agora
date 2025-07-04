"""
Plot validation losses for different first layer types.
Loads CSV files saved by train.py and creates comparison plots
with 95% confidence intervals on a log scale.

Supports different styles and masking strategies:
- mlm: MLM training with various masking strategies (simple, complex)
- discrete_diffusion: Discrete diffusion training

New format: Files named {model_type}_{style}_{mask_config}_epoch_metrics.csv
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats


def load_validation_data(checkpoint_dir: str, model_type: str, style: str, mask_config: str):
    """Load validation data for a given model type, style, and mask configuration."""
    # New filename pattern: {model_type}_{style}_{mask_config}_epoch_metrics.csv
    csv_path = Path(checkpoint_dir) / f"{model_type}_{style}_{mask_config}_epoch_metrics.csv"
    
    # Check if file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {csv_path}")
    
    # Load data with header
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)  # Skip header row
    
    # Read header to understand column structure
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
    
    # Create dictionary to access columns by name
    data_dict = {}
    for i, col_name in enumerate(header):
        if data.ndim == 1:  # Single epoch case
            data_dict[col_name] = data[i] if len(data) > i else 0
        else:  # Multiple epochs
            data_dict[col_name] = data[:, i] if data.shape[1] > i else np.zeros(data.shape[0])
    
    return data_dict


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval for validation losses."""
    if data.ndim == 1:
        return data, data, data  # Single run, no confidence interval
    
    # Calculate mean and standard error
    mean_val = np.mean(data, axis=0)
    std_error = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    
    # Calculate confidence interval using t-distribution
    df = data.shape[0] - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    ci_lower = mean_val - t_critical * std_error
    ci_upper = mean_val + t_critical * std_error
    
    return mean_val, ci_lower, ci_upper


def plot_validation_losses(checkpoint_dir: str = "checkpoints", 
                         model_types: list = ["SA", "GA", "RA", "SC"],
                         style: str = "mlm",
                         mask_config: str = "simple",
                         output_file: str = None):
    """Create validation loss plots with confidence intervals."""
    
    if output_file is None:
        output_file = f"validation_losses_{style}_{mask_config}.png"
    
    # Set up the figure with two subplots for sequence and structure losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different model types
    colors = {
        "SA": "#1f77b4",  # blue
        "GA": "#ff7f0e",  # orange
        "RA": "#2ca02c",  # green
        "SC": "#d62728"   # red for SelfConsensus
    }
    
    # Label mapping for display
    label_mapping = {
        "SA": "Self Attention",
        "GA": "Geometric Attention", 
        "RA": "Reflexive Attention",
        "SC": "Self Consensus"
    }
    
    data_found = False
    num_epochs = 0
    
    # Process each model type
    for model_type in model_types:
        try:
            # Load data
            data_dict = load_validation_data(checkpoint_dir, model_type, style, mask_config)
            
            # Check for validation sequence and structure losses
            val_loss_seq_key = 'val_loss_seq'
            val_loss_struct_key = 'val_loss_struct'
            
            if val_loss_seq_key not in data_dict or val_loss_struct_key not in data_dict:
                print(f"Warning: Missing validation loss columns for {model_type}")
                continue
            
            val_loss_seq = data_dict[val_loss_seq_key]
            val_loss_struct = data_dict[val_loss_struct_key]
            
            # Handle single values
            if isinstance(val_loss_seq, (int, float)):
                val_loss_seq = np.array([val_loss_seq])
            if isinstance(val_loss_struct, (int, float)):
                val_loss_struct = np.array([val_loss_struct])
            
            num_epochs = len(val_loss_seq)
            epochs = np.arange(1, num_epochs + 1)
            data_found = True
            
            # Calculate mean losses and confidence intervals
            if val_loss_seq.ndim == 1:
                seq_mean, seq_lower, seq_upper = val_loss_seq, val_loss_seq, val_loss_seq
                struct_mean, struct_lower, struct_upper = val_loss_struct, val_loss_struct, val_loss_struct
            else:
                seq_mean, seq_lower, seq_upper = calculate_confidence_interval(val_loss_seq)
                struct_mean, struct_lower, struct_upper = calculate_confidence_interval(val_loss_struct)
            
            # Get display label
            display_label = label_mapping.get(model_type, model_type)
            
            # Plot sequence losses
            ax1.plot(epochs, seq_mean, color=colors.get(model_type, "#333333"), 
                        label=display_label, linewidth=2)
            if seq_lower is not seq_upper:  # Only show CI if there's variance
                ax1.fill_between(epochs, seq_lower, seq_upper, 
                                color=colors.get(model_type, "#333333"), alpha=0.2)
            
            # Plot structure losses
            ax2.plot(epochs, struct_mean, color=colors.get(model_type, "#333333"), 
                        label=display_label, linewidth=2)
            if struct_lower is not struct_upper:  # Only show CI if there's variance
                ax2.fill_between(epochs, struct_lower, struct_upper, 
                                color=colors.get(model_type, "#333333"), alpha=0.2)
            
        except FileNotFoundError:
            print(f"Warning: Could not find metrics file for {model_type} {style} {mask_config}")
            continue
        except Exception as e:
            print(f"Error loading {model_type}: {e}")
            continue
    
    if not data_found:
        print(f"Error: No data files found for {style} with {mask_config} masking. Please check the checkpoint directory.")
        plt.close(fig)
        return
    
    # Customize sequence subplot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Sequence Prediction Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", alpha=0.3)
    if num_epochs > 0:
        ax1.set_xlim(1, num_epochs)
    ax1.set_yscale('log')
    
    # Customize structure subplot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Structure Prediction Loss', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", alpha=0.3)
    if num_epochs > 0:
        ax2.set_xlim(1, num_epochs)
    ax2.set_yscale('log')
    
    # Add overall title
    style_title = {
        "mlm": "Masked Language Modeling",
        "discrete_diffusion": "Discrete Diffusion"
    }.get(style, style.title())
    
    fig.suptitle(f'{style_title} - {mask_config.title()} Masking', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to {output_file}")


if __name__ == "__main__":
    # Default parameters - can be modified as needed
    checkpoint_dir = "../checkpoints/transformer_trunk"  # Correct path for trunk models
    model_types = ["SC", "SA"]  # Including SC for SelfConsensus
    style = "mlm"  # Can be "mlm" or "discrete_diffusion" 
    mask_config = "simple"  # Can be "simple", "complex", or "discrete_diffusion"
    output_file = f"validation_losses_{style}_{mask_config}.png"
    
    # Create the plots
    plot_validation_losses(checkpoint_dir, model_types, style, mask_config, output_file)