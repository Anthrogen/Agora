"""
Plot validation losses for different first layer types.
Loads CSV files saved by train_transformer.py and creates comparison plots
with 95% confidence intervals on a log scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats


def load_validation_loss_data(checkpoint_dir: str, first_layer: str):
    """Load validation loss data for a given first layer type."""
    # Primary file naming pattern
    seq_csv_path = Path(checkpoint_dir) / f"{first_layer}_sequence_val_loss.csv"
    struct_csv_path = Path(checkpoint_dir) / f"{first_layer}_structure_val_loss.csv"
    
    # Check if files exist with primary pattern
    if seq_csv_path.exists() and struct_csv_path.exists():
        # Load data, skipping header rows
        seq_loss = np.loadtxt(seq_csv_path, delimiter=',', comments='#')
        struct_loss = np.loadtxt(struct_csv_path, delimiter=',', comments='#')
        return seq_loss, struct_loss
    
    # Try alternative naming pattern
    seq_csv_path = Path(checkpoint_dir) / f"{first_layer}_seq_val_loss.csv"
    struct_csv_path = Path(checkpoint_dir) / f"{first_layer}_struct_val_loss.csv"
    
    # Load data, skipping header rows
    seq_loss = np.loadtxt(seq_csv_path, delimiter=',', comments='#')
    struct_loss = np.loadtxt(struct_csv_path, delimiter=',', comments='#')
    
    return seq_loss, struct_loss


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval for validation losses."""
    # Calculate mean and standard error
    mean_loss = np.mean(data, axis=0)
    std_error = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    
    # Calculate confidence interval using t-distribution
    # degrees of freedom = n - 1
    df = data.shape[0] - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    ci_lower = mean_loss - t_critical * std_error
    ci_upper = mean_loss + t_critical * std_error
    
    return mean_loss, ci_lower, ci_upper


def plot_validation_losses(checkpoint_dir: str = "checkpoints", 
                         first_layers: list = ["SA", "GA", "RA", "C"],
                         output_file: str = "validation_losses.png"):
    """Create validation loss plots with confidence intervals."""
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different first layer types
    colors = {
        "SA": "#1f77b4",  # blue
        "GA": "#ff7f0e",  # orange
        "RA": "#2ca02c",  # green
        "C": "#d62728"    # red for Consensus
    }
    
    # Label mapping for display
    label_mapping = {
        "SA": "Self Attention",
        "GA": "Geometric Attention", 
        "RA": "Reflexive Attention",
        "C": "Self Consensus"
    }
    
    # Initialize data_found to track if any data was found
    data_found = False  # Track if any data was found
    num_epochs = None  # Will be set from first successful data load
    
    # Process each first layer type
    for first_layer in first_layers:
        try:
            # Load data
            seq_loss, struct_loss = load_validation_loss_data(checkpoint_dir, first_layer)
            
            # Get number of epochs
            num_epochs = seq_loss.shape[1]
            epochs = np.arange(1, num_epochs + 1)
            data_found = True
            
            # Calculate mean losses and confidence intervals
            seq_mean, seq_lower, seq_upper = calculate_confidence_interval(seq_loss)
            struct_mean, struct_lower, struct_upper = calculate_confidence_interval(struct_loss)
            
            # Get display label
            display_label = label_mapping.get(first_layer, first_layer)
            
            # Plot sequence losses
            ax1.plot(epochs, seq_mean, color=colors.get(first_layer, "#333333"), 
                        label=display_label, linewidth=2)
            ax1.fill_between(epochs, seq_lower, seq_upper, 
                            color=colors.get(first_layer, "#333333"), alpha=0.2)
            
            # Plot structure losses
            ax2.plot(epochs, struct_mean, color=colors.get(first_layer, "#333333"), 
                        label=display_label, linewidth=2)
            ax2.fill_between(epochs, struct_lower, struct_upper, 
                            color=colors.get(first_layer, "#333333"), alpha=0.2)
            
        except FileNotFoundError:
            print(f"Warning: Could not find loss data files for {first_layer}")
            continue
        except Exception as e:
            print(f"Error loading {first_layer}: {e}")
            continue
    
    if not data_found:
        print("Error: No data files found. Please check the checkpoint directory.")
        plt.close(fig)
        return
    
    # Customize sequence subplot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Sequence Prediction Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", alpha=0.3)
    if num_epochs is not None:
        ax1.set_xlim(1, num_epochs)
    ax1.set_yscale('log')
    
    # Customize structure subplot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Structure Prediction Loss', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", alpha=0.3)
    if num_epochs is not None:
        ax2.set_xlim(1, num_epochs)
    ax2.set_yscale('log')
    
    # Add overall title
    fig.suptitle('Validation Losses by First Layer Type (95% CI)', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to {output_file}")


if __name__ == "__main__":
    # Default parameters - can be modified as needed
    checkpoint_dir = "checkpoints"  # Relative to scripts folder
    first_layers = ["SA", "GA", "RA", "C"]  # Including C for Consensus
    loss_output_file = "validation_losses.png"
    
    # Create the plots
    plot_validation_losses(checkpoint_dir, first_layers, loss_output_file)