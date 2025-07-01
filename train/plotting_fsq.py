"""
Plot validation losses for different model types in FSQ training.
Loads CSV files saved by train.py and creates comparison plots
with 95% confidence intervals on a log scale.
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
                         style: str = "stage_1",
                         mask_config: str = "simple",
                         output_file: str = None):
    """Create validation loss plots with confidence intervals."""
    
    if output_file is None:
        output_file = f"validation_losses_{style}_{mask_config}.png"
    
    # Set up the figure with a single subplot for RMSD (FSQ models)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors for different model types
    colors = {
        "SA": "#1f77b4",  # blue
        "GA": "#ff7f0e",  # orange
        "RA": "#2ca02c",  # green
        "SC": "#d62728"   # red for SelfConsensus
    }
    
    # Full names for model types
    model_names = {
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
            
            # Get validation loss (assuming RMSD loss for FSQ models)
            val_loss_key = 'val_loss'
            if val_loss_key not in data_dict:
                print(f"Warning: No validation loss found for {model_type}")
                continue
                
            val_loss = data_dict[val_loss_key]
            if isinstance(val_loss, (int, float)):  # Single value
                val_loss = np.array([val_loss])
            
            num_epochs = len(val_loss)
            epochs = np.arange(1, num_epochs + 1)
            data_found = True
            
            # For single runs, no confidence interval calculation needed
            if val_loss.ndim == 1:
                rmsd_mean = val_loss
                rmsd_lower = val_loss
                rmsd_upper = val_loss
            else:
                rmsd_mean, rmsd_lower, rmsd_upper = calculate_confidence_interval(val_loss)
            
            # Plot RMSD values
            ax.plot(epochs, rmsd_mean, color=colors.get(model_type, "#333333"), 
                    label=f'{model_names.get(model_type, model_type)}', linewidth=2)
            
            if rmsd_lower is not rmsd_upper:  # Only show CI if there's variance
                ax.fill_between(epochs, rmsd_lower, rmsd_upper, 
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
    
    # Customize subplot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, num_epochs)
    ax.set_yscale('log')
    
    # Add overall title
    stage_title = "Stage 1 (Coordinate Reconstruction)" if style == "stage_1" else "Stage 2 (Full Structure Reconstruction)"
    fig.suptitle(f'{stage_title} - {mask_config.title()} Masking', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to {output_file}")


if __name__ == "__main__":
    # Default parameters - can be modified as needed
    checkpoint_dir = "../checkpoints/fsq"  # Relative to train folder
    model_types = ["SC", "SA"]  # Including SC for SelfConsensus
    style = "stage_1"  # Can be "stage_1" or "stage_2"
    mask_config = "complex"  # Can be "simple", "complex", "discrete_diffusion", or "no_mask"
    output_file = f"validation_losses_{style}_{mask_config}.png"
    
    # Create the plots
    plot_validation_losses(checkpoint_dir, model_types, style, mask_config, output_file)