"""
FSQ Model Testing Script for SS8 Confusion Matrix
This script loads a trained FSQ model checkpoint and generates a confusion matrix
for 8-state secondary structure (SS8) prediction based on reconstructions.
Uses BioPython's DSSP implementation for accurate SS8 assignment.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import warnings
import tempfile
import subprocess
from Bio.PDB import PDBParser, PDBIO, DSSP
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
warnings.filterwarnings('ignore')

# Import the model and data loader from the src directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import ProteinBackboneDataset
from src.models.fsq_autoencoder_GA import FSQResidueAutoencoder

# Define SS8 classes
SS8_CLASSES = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'L']
SS8_NAMES = {
    'H': 'α-helix',
    'G': '3₁₀-helix', 
    'I': 'π-helix',
    'E': 'β-strand',
    'B': 'β-bridge',
    'T': 'Turn',
    'S': 'Bend',
    'L': 'Coil'
}

def load_model_checkpoint(checkpoint_path, device):
    """Load the trained FSQ model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default configuration matching FSQResidueAutoencoder parameters
        config = {
            'hidden_dim': 64,
            'latent_dim': 32,
            'fsq_dim': 5,
            'fsq_levels': [7, 5, 5, 5, 5],
            'attn_heads': 4
        }
    
    # Initialize model with correct parameters
    model = FSQResidueAutoencoder(
        hidden_dim=config.get('hidden_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        fsq_dim=config.get('fsq_dim', 5),
        fsq_levels=config.get('fsq_levels', [7, 5, 5, 5, 5]),
        attn_heads=config.get('attn_heads', 4)
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model

def coords_to_pdb_structure(coords, chain_id='A', structure_id='temp'):
    """Convert backbone coordinates to a BioPython Structure object."""
    # Create structure
    structure = Structure(structure_id)
    model = Model(0)
    chain = Chain(chain_id)
    
    # Add residues
    for i in range(len(coords)):
        res_id = (' ', i+1, ' ')
        residue = Residue(res_id, 'ALA', ' ')
        
        # Add backbone atoms
        n_coord = coords[i, 0]
        ca_coord = coords[i, 1]
        c_coord = coords[i, 2]
        
        # Calculate O position using standard geometry
        # O is approximately 1.23 Å from C in the C=O bond
        # The C-O vector is roughly perpendicular to the CA-C-N(i+1) plane
        if i < len(coords) - 1:
            next_n = coords[i+1, 0]
            # Calculate CA-C vector
            ca_c_vec = c_coord - ca_coord
            # Calculate C-N(i+1) vector  
            c_n_vec = next_n - c_coord
            # Cross product gives perpendicular direction
            perp_vec = np.cross(ca_c_vec, c_n_vec)
            if np.linalg.norm(perp_vec) > 0:
                perp_vec = perp_vec / np.linalg.norm(perp_vec)
            else:
                perp_vec = np.array([0, 0, 1])
            # O position
            o_coord = c_coord + 1.23 * perp_vec
        else:
            # For last residue, use a default direction
            ca_c_vec = c_coord - ca_coord
            ca_c_vec = ca_c_vec / np.linalg.norm(ca_c_vec)
            # Create perpendicular vector
            if abs(ca_c_vec[2]) < 0.9:
                perp_vec = np.cross(ca_c_vec, np.array([0, 0, 1]))
            else:
                perp_vec = np.cross(ca_c_vec, np.array([1, 0, 0]))
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            o_coord = c_coord + 1.23 * perp_vec
        
        n_atom = Atom('N', n_coord, 1.0, 1.0, ' ', 'N', i*4)
        ca_atom = Atom('CA', ca_coord, 1.0, 1.0, ' ', 'CA', i*4+1)
        c_atom = Atom('C', c_coord, 1.0, 1.0, ' ', 'C', i*4+2)
        o_atom = Atom('O', o_coord, 1.0, 1.0, ' ', 'O', i*4+3)
        
        residue.add(n_atom)
        residue.add(ca_atom)
        residue.add(c_atom)
        residue.add(o_atom)
        
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    
    return structure

def run_dssp_on_coords(coords):
    """
    Run DSSP on backbone coordinates to get SS8 assignments.
    
    Args:
        coords: Array of shape (L, 3, 3) with backbone coordinates
        
    Returns:
        List of SS8 assignments
    """
    # Convert to numpy if needed
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    
    # Create temporary PDB file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
        tmp_pdb_path = tmp_pdb.name
        
        # Convert coordinates to PDB structure
        structure = coords_to_pdb_structure(coords)
        
        # Write structure to PDB file
        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_pdb_path)
    
    try:
        # Create output DSSP file
        dssp_out = tmp_pdb_path.replace('.pdb', '.dssp')
        
        # Run mkdssp directly
        cmd = ['mkdssp', '--output-format', 'dssp', tmp_pdb_path, dssp_out]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: mkdssp failed: {result.stderr}")
            return ['L'] * len(coords)
        
        # Parse DSSP output
        ss8_assignments = []
        with open(dssp_out, 'r') as f:
            lines = f.readlines()
            in_data = False
            for line in lines:
                if line.startswith('  #  RESIDUE'):
                    in_data = True
                    continue
                if in_data and len(line) > 16:
                    ss_code = line[16]
                    # Convert DSSP codes to SS8
                    # DSSP uses: H (alpha helix), B (beta bridge), E (extended/beta strand),
                    # G (3-10 helix), I (pi helix), T (turn), S (bend), - (loop/coil)
                    # P (polyproline) - we'll map to coil
                    if ss_code == ' ' or ss_code == '-':
                        ss_code = 'L'
                    elif ss_code == 'P':  # Polyproline helix -> coil
                        ss_code = 'L'
                    elif ss_code not in SS8_CLASSES:
                        ss_code = 'L'
                    ss8_assignments.append(ss_code)
        
        # Clean up
        os.remove(dssp_out)
        
        return ss8_assignments
        
    except Exception as e:
        print(f"Warning: DSSP failed: {e}")
        # Return all coil as fallback
        return ['L'] * len(coords)
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_pdb_path):
            os.remove(tmp_pdb_path)

def get_true_ss8_from_json(json_path):
    """
    Extract true SS8 assignments from JSON file using DSSP.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get backbone coordinates
        coords_N = np.array(data["backbone_coordinates"]["N"], dtype=np.float32)
        coords_CA = np.array(data["backbone_coordinates"]["CA"], dtype=np.float32)
        coords_C = np.array(data["backbone_coordinates"]["C"], dtype=np.float32)
        
        # Verify shapes are correct
        if coords_N.ndim != 2 or coords_N.shape[1] != 3:
            raise ValueError(f"Invalid N coordinates shape: {coords_N.shape}")
        if coords_CA.ndim != 2 or coords_CA.shape[1] != 3:
            raise ValueError(f"Invalid CA coordinates shape: {coords_CA.shape}")
        if coords_C.ndim != 2 or coords_C.shape[1] != 3:
            raise ValueError(f"Invalid C coordinates shape: {coords_C.shape}")
        
        # Stack into (L, 3, 3) array
        coords = np.stack([coords_N, coords_CA, coords_C], axis=1)
        
        # Get SS8 assignments using DSSP
        ss8_assignments = run_dssp_on_coords(coords)
        
        return ss8_assignments
    except Exception as e:
        print(f"Warning: Error processing {json_path}: {e}")
        return None

def get_valid_json_files(data_dir):
    """Get list of valid JSON files that can be loaded by the DataLoader."""
    valid_files = []
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    for fn in all_files:
        path = os.path.join(data_dir, fn)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Verify backbone coordinates
            coords_N = np.array(data["backbone_coordinates"]["N"], dtype=np.float32)
            coords_CA = np.array(data["backbone_coordinates"]["CA"], dtype=np.float32)
            coords_C = np.array(data["backbone_coordinates"]["C"], dtype=np.float32)
            
            # Check shapes
            if (coords_N.ndim == 2 and coords_N.shape[1] == 3 and
                coords_CA.ndim == 2 and coords_CA.shape[1] == 3 and
                coords_C.ndim == 2 and coords_C.shape[1] == 3 and
                coords_N.shape[0] == coords_CA.shape[0] == coords_C.shape[0]):
                valid_files.append(fn)
            else:
                print(f"Warning: Skipping {fn} due to invalid coordinate shapes")
        except Exception as e:
            print(f"Warning: Skipping {fn}: {e}")
    
    return valid_files

def process_dataset(model, data_loader, device, data_dir):
    """Process entire dataset and collect true and predicted SS8 labels."""
    all_true_ss8 = []
    all_pred_ss8 = []
    
    print("Processing dataset...")
    
    # Get list of valid JSON files (same ones that DataLoader can process)
    json_files = get_valid_json_files(data_dir)
    print(f"Found {len(json_files)} valid JSON files")
    
    # Process batches
    file_idx = 0
    processed_count = 0
    with torch.no_grad():
        for batch_idx, (coords_batch, mask_batch) in enumerate(tqdm(data_loader)):
            coords_batch = coords_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Get batch size
            batch_size = coords_batch.shape[0]
            
            # Forward pass through model
            # FSQResidueAutoencoder returns (reconstructed_coords, indices)
            reconstructed, indices = model(coords_batch)
            
            # reconstructed already has shape (B, L, 3, 3)
            
            # Process each sample in batch
            for i in range(batch_size):
                if file_idx >= len(json_files):
                    break
                    
                # Get true SS8 from original structure
                json_path = os.path.join(data_dir, json_files[file_idx])
                true_ss8 = get_true_ss8_from_json(json_path)
                
                if true_ss8 is None:
                    file_idx += 1
                    continue
                
                # Get predicted SS8 from reconstruction using DSSP
                # Apply mask to get valid length
                valid_length = int(mask_batch[i].sum().item())
                reconstructed_coords = reconstructed[i][:valid_length]
                
                pred_ss8 = run_dssp_on_coords(reconstructed_coords)
                
                # Ensure same length
                min_len = min(len(true_ss8), len(pred_ss8))
                if min_len > 0:
                    true_ss8 = true_ss8[:min_len]
                    pred_ss8 = pred_ss8[:min_len]
                    
                    all_true_ss8.extend(true_ss8)
                    all_pred_ss8.extend(pred_ss8)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"\nProcessed {processed_count} proteins, {len(all_true_ss8)} residues total")
                
                file_idx += 1
            
            if file_idx >= len(json_files):
                break
    
    print(f"\nFinished processing {processed_count} proteins")
    return all_true_ss8, all_pred_ss8

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix plot."""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=SS8_CLASSES)
    
    # Normalize by row (true labels) - handle zero division
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=[f"{c}\n{SS8_NAMES[c]}" for c in SS8_CLASSES],
                yticklabels=[f"{c}\n{SS8_NAMES[c]}" for c in SS8_CLASSES],
                cbar_kws={'label': 'Proportion'})
    
    plt.title('SS8 Confusion Matrix\n(Rows: True SS8, Columns: Predicted SS8)', fontsize=14)
    plt.xlabel('Predicted SS8 Type', fontsize=12)
    plt.ylabel('True SS8 Type', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    # Also save the raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=[f"{c}\n{SS8_NAMES[c]}" for c in SS8_CLASSES],
                yticklabels=[f"{c}\n{SS8_NAMES[c]}" for c in SS8_CLASSES],
                cbar_kws={'label': 'Count'})
    
    plt.title('SS8 Confusion Matrix (Raw Counts)\n(Rows: True SS8, Columns: Predicted SS8)', fontsize=14)
    plt.xlabel('Predicted SS8 Type', fontsize=12)
    plt.ylabel('True SS8 Type', fontsize=12)
    plt.tight_layout()
    
    save_path_counts = save_path.replace('.png', '_counts.png')
    plt.savefig(save_path_counts, dpi=300, bbox_inches='tight')
    print(f"Raw count confusion matrix saved to {save_path_counts}")
    plt.close()
    
    # Print statistics
    total_correct = np.diag(cm).sum()
    total_samples = cm.sum()
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\nOverall SS8 Accuracy: {accuracy:.3f}")
    print("\nPer-class accuracy:")
    for i, ss8_class in enumerate(SS8_CLASSES):
        class_total = cm[i].sum()
        class_acc = cm[i, i] / class_total if class_total > 0 else 0
        print(f"  {ss8_class} ({SS8_NAMES[ss8_class]}): {class_acc:.3f} ({cm[i, i]}/{class_total})")
    
    # Save detailed report
    report_path = save_path.replace('.png', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("SS8 Confusion Matrix Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.3f}\n")
        f.write(f"Total samples: {total_samples}\n\n")
        f.write("Per-class Statistics:\n")
        f.write("-"*30 + "\n")
        
        for i, ss8_class in enumerate(SS8_CLASSES):
            total = cm[i].sum()
            correct = cm[i, i]
            acc = correct / total if total > 0 else 0
            f.write(f"{ss8_class} ({SS8_NAMES[ss8_class]}):\n")
            f.write(f"  Total: {total}\n")
            f.write(f"  Correct: {correct}\n")
            f.write(f"  Accuracy: {acc:.3f}\n\n")
        
        f.write("\nConfusion Matrix (normalized by row):\n")
        df = pd.DataFrame(cm_normalized, 
                         index=[f"{c} ({SS8_NAMES[c]})" for c in SS8_CLASSES],
                         columns=[f"{c} ({SS8_NAMES[c]})" for c in SS8_CLASSES])
        f.write(df.to_string())
        
        f.write("\n\nConfusion Matrix (raw counts):\n")
        df_raw = pd.DataFrame(cm, 
                             index=[f"{c} ({SS8_NAMES[c]})" for c in SS8_CLASSES],
                             columns=[f"{c} ({SS8_NAMES[c]})" for c in SS8_CLASSES])
        f.write(df_raw.to_string())
    
    print(f"Detailed report saved to {report_path}")

def main():
    # Configuration
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "checkpoint_epoch_60.pt")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_training_data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    # Load model
    model = load_model_checkpoint(checkpoint_path, device)
    
    # Create dataset and dataloader
    dataset = ProteinBackboneDataset(root_dir=data_dir, center=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)} proteins")
    
    # Process dataset
    true_ss8, pred_ss8 = process_dataset(model, data_loader, device, data_dir)
    
    print(f"\nTotal residues processed: {len(true_ss8)}")
    
    # Generate confusion matrix
    save_path = os.path.join(output_dir, "ss8_confusion_matrix.png")
    plot_confusion_matrix(true_ss8, pred_ss8, save_path)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
