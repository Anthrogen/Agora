"""
Plot validation losses for different model types in FSQ training.
Automatically discovers models in checkpoint directories by examining model.pt files,
extracts model types (GA/SA/RA/SC), and plots their loss histories.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import scipy.stats as stats

# Import required for loading checkpoints
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.model_librarian import load_model_from_checkpoint


def discover_models_in_directory(base_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    Discover models in checkpoint directories by examining model.pt files.
    
    Args:
        base_dir: Base directory containing subdirectories with model.pt files
        
    Returns:
        Dictionary mapping model_type -> (checkpoint_path, history_path)
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    
    models = {}
    
    # Scan all subdirectories for model.pt files
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
            
        model_path = subdir / "model.pt"
        history_path = subdir / "history.csv"
        
        if not model_path.exists():
            print(f"Warning: No model.pt found in {subdir}")
            continue
            
        if not history_path.exists():
            print(f"Warning: No history.csv found in {subdir}")
            continue
        
        try:
            # Load checkpoint to extract model type
            device = torch.device("cpu")  # Load on CPU for inspection only
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model_cfg = checkpoint['model_config']
            
            # Extract model type from configuration
            model_type = model_cfg.first_block_cfg.initials()
            
            # Store the mapping
            if model_type in models:
                print(f"Warning: Multiple models found for type {model_type}. Using {subdir}")
            
            models[model_type] = (str(model_path), str(history_path))
            print(f"Found {model_type} model in {subdir}")
            
        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            continue
    
    if not models:
        raise ValueError(f"No valid models found in {base_dir}")
    
    return models


def load_validation_data_from_history(history_path: str) -> Dict[str, np.ndarray]:
    """Load validation data from history.csv file."""
    if not Path(history_path).exists():
        raise FileNotFoundError(f"Could not find history file: {history_path}")
    
    # Load data with header
    data = np.loadtxt(history_path, delimiter=',', skiprows=1)  # Skip header row
    
    # Read header to understand column structure
    with open(history_path, 'r') as f:
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


def plot_validation_losses_auto(base_dir: str,
                               output_file: str = None,
                               loss_key: str = "train_loss"):
    """
    Automatically discover and plot validation losses for models in checkpoint directories.
    
    Args:
        base_dir: Base directory containing model checkpoint subdirectories
        output_file: Output filename for the plot (auto-generated if None)
        loss_key: Key to use for extracting loss values from history (default: "val_loss")
    """
    
    # Discover models in the directory
    print(f"Scanning directory: {base_dir}")
    models = discover_models_in_directory(base_dir)
    
    if not models:
        print(f"Error: No models found in {base_dir}")
        return
    
    print(f"Found models: {list(models.keys())}")
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = Path(base_dir).name
        output_file = f"validation_losses_{base_name}.png"
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors for different model types
    colors = {
        "SA": "#1f77b4",  # blue
        "GA": "#ff7f0e",  # orange
        "RA": "#2ca02c",  # green
        "SC": "#d62728"   # red
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
    
    # Process each discovered model
    for model_type in sorted(models.keys()):
        model_path, history_path = models[model_type]
        
        try:
            # Load history data
            data_dict = load_validation_data_from_history(history_path)
            
            # Get validation loss
            if loss_key not in data_dict:
                print(f"Warning: No {loss_key} found for {model_type}")
                continue
                
            val_loss = data_dict[loss_key]
            if isinstance(val_loss, (int, float)):  # Single value
                val_loss = np.array([val_loss])
            
            num_epochs = len(val_loss)
            epochs = np.arange(1, num_epochs + 1)
            data_found = True
            
            # For single runs, no confidence interval calculation needed
            if val_loss.ndim == 1:
                loss_mean = val_loss
                loss_lower = val_loss
                loss_upper = val_loss
            else:
                loss_mean, loss_lower, loss_upper = calculate_confidence_interval(val_loss)
            
            # Plot loss values
            color = colors.get(model_type, "#333333")
            label = model_names.get(model_type, model_type)
            
            ax.plot(epochs, loss_mean, color=color, label=label, linewidth=2)
            
            if not np.array_equal(loss_lower, loss_upper):  # Only show CI if there's variance
                ax.fill_between(epochs, loss_lower, loss_upper, color=color, alpha=0.2)
            
        except Exception as e:
            print(f"Error loading {model_type}: {e}")
            continue
    
    if not data_found:
        print(f"Error: No valid data loaded from {base_dir}")
        plt.close(fig)
        return
    
    # Customize plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, num_epochs)
    ax.set_yscale('log')
    
    # Add overall title based on directory name
    base_name = Path(base_dir).name
    fig.suptitle(f'{base_name} - Model Comparison', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to {output_file}")
    plt.show()

if __name__ == "__main__":
    # New auto-detection approach - just provide the base directory
    base_dir = "/workspace/demo/Odyssey/checkpoints/fsq/fsq_stage_1_config"  # Update this path as needed
    
    print("Using auto-detection approach...")
    plot_validation_losses_auto(base_dir)