"""
Plot validation losses for different model types in FSQ training.
Loads CSV files saved by train_fsq.py and creates comparison plots
with 95% confidence intervals on a log scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats


def load_validation_loss_data(checkpoint_dir: str, model_type: str, stage: str = "stage_1", masking_strategy: str = "simple"):
    """Load validation loss data for a given model type and stage."""
    # File naming pattern for FSQ training
    loss_csv_path = Path(checkpoint_dir) / f"{model_type}_{stage}_val_loss_{masking_strategy}.csv"
    rmsd_csv_path = Path(checkpoint_dir) / f"{model_type}_{stage}_val_rmsd_{masking_strategy}.csv"
    
    # Check if files exist
    if not loss_csv_path.exists() or not rmsd_csv_path.exists():
        raise FileNotFoundError(f"Could not find validation data files for {model_type} {stage}")
    
    # Load data, skipping header rows
    val_loss = np.loadtxt(loss_csv_path, delimiter=',', comments='#')
    val_rmsd = np.loadtxt(rmsd_csv_path, delimiter=',', comments='#')
    
    return val_loss, val_rmsd


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
                         model_types: list = ["SA", "GA", "RA", "C"],
                         stage: str = "stage_1",
                         output_file: str = None,
                         masking_strategy: str = "simple"):
    """Create validation loss plots with confidence intervals."""
    
    if output_file is None:
        output_file = f"validation_losses_{stage}.png"
    
    # Set up the figure with a single subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Colors for different model types
    colors = {
        "SA": "#1f77b4",  # blue
        "GA": "#ff7f0e",  # orange
        "RA": "#2ca02c",  # green
        "C": "#d62728"    # red for Consensus
    }
    
    # Full names for model types
    model_names = {
        "SA": "Self Attention",
        "GA": "Geometric Attention",
        "RA": "Reflexive Attention",
        "C": "Self Consensus"
    }
    
    # Initialize num_epochs to avoid UnboundLocalError
    num_epochs = 100  # Default value
    data_found = False  # Track if any data was found
    
    # Process each model type
    for model_type in model_types:
        try:
            # Load data
            val_loss, val_rmsd = load_validation_loss_data(checkpoint_dir, model_type, stage, masking_strategy)
            
            # Get number of epochs
            num_epochs = val_loss.shape[1]
            epochs = np.arange(1, num_epochs + 1)
            data_found = True
            
            # Calculate mean RMSD and confidence intervals
            rmsd_mean, rmsd_lower, rmsd_upper = calculate_confidence_interval(val_rmsd)
            
            # Plot RMSD values
            ax.plot(epochs, rmsd_mean, color=colors.get(model_type, "#333333"), 
                    label=f'{model_names.get(model_type, model_type)}', linewidth=2)
            ax.fill_between(epochs, rmsd_lower, rmsd_upper, 
                           color=colors.get(model_type, "#333333"), alpha=0.2)
            
        except FileNotFoundError:
            print(f"Warning: Could not find loss data files for {model_type} {stage}")
            continue
        except Exception as e:
            print(f"Error loading {model_type}: {e}")
            continue
    
    if not data_found:
        print(f"Error: No data files found for {stage}. Please check the checkpoint directory.")
        plt.close(fig)
        return
    
    # Customize RMSD subplot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation RMSD (Å)', fontsize=12)
    ax.set_title('Kabsch RMSD', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, num_epochs)
    ax.set_yscale('log')
    
    # Add overall title
    stage_title = "Stage 1 (Masked Coordinate Reconstruction)" if stage == "stage_1" else "Stage 2 (Full Structure Reconstruction)"
    fig.suptitle(f'Validation RMSD by Model Type - {stage_title} (95% CI)', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to {output_file}")


if __name__ == "__main__":
    # Default parameters - can be modified as needed
    checkpoint_dir = "../checkpoints"  # Relative to scripts folder
    model_types = ["SA", "GA", "RA", "C"]  # Including C for Consensus
    stage = "stage_1"  # Can be "stage_1" or "stage_2"
    masking_strategy = "simple"  # Can be "simple", "complex", or "discrete_diffusion"
    loss_output_file = f"validation_losses_{stage}_{masking_strategy}.png"
    
    # Create the plots
    plot_validation_losses(checkpoint_dir, model_types, stage, loss_output_file, masking_strategy)