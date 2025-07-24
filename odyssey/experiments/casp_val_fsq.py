#!/usr/bin/env python3
"""
CASP15 FSQ Autoencoder Evaluation Script

This script evaluates a trained FSQ autoencoder on CASP15 data by computing
Kabsch RMSD on reconstructed structures. It handles both stage 1 (backbone only)
and stage 2 (all heavy atoms) autoencoders automatically based on the model configuration.

Usage:
    python casp_val_fsq.py --checkpoint /path/to/autoencoder_checkpoint.pt
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import glob
import csv
from odyssey.src.configurations import NoMaskConfig

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from odyssey.src.models.autoencoder import Autoencoder
from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.dataset import ProteinDataset
from odyssey.src.dataloader import _get_training_dataloader, worker_init_fn
from odyssey.src.configurations import AutoencoderConfig
from odyssey.train.fsq_step import stage_1_step, stage_2_step

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeOptimizer:
    """Fake optimizer for validation (step functions expect an optimizer)"""
    def zero_grad(self): pass
    def step(self): pass

class FakeScheduler:
    """Fake scheduler for validation (step functions expect a scheduler)"""
    def step(self): pass

def evaluate_autoencoder(model, data_loader, model_cfg, train_cfg):
    """Evaluate autoencoder using the same step functions as training."""
    model.eval()
    
    # Create fake optimizer and scheduler (required by step functions but not used in eval mode)
    fake_optimizer = FakeOptimizer()
    fake_scheduler = FakeScheduler()
    
    # Determine which step function to use
    if model_cfg.style == "stage_1":
        step_fn = stage_1_step
        print("Evaluating Stage 1 autoencoder (backbone atoms only)...")
    elif model_cfg.style == "stage_2":
        step_fn = stage_2_step
        print("Evaluating Stage 2 autoencoder (all heavy atoms)...")
    else:
        raise ValueError(f"Unknown model style: {model_cfg.style}")
    
    # Accumulate metrics across all batches
    metrics_sum = {}
    metrics_count = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            if batch is None:
                continue
            
            # Call the step function in evaluation mode
            batch_metrics = step_fn(model, fake_optimizer, fake_scheduler, batch, model_cfg, train_cfg, train_mode=False)
            
            # Accumulate metrics (following train.py pattern)
            for k, (value, count) in batch_metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0.0
                    metrics_count[k] = 0
                metrics_sum[k] += value * count
                metrics_count[k] += count
    
    # Calculate final averages
    final_metrics = {k: metrics_sum[k] / metrics_count[k] for k in metrics_sum.keys()}
    
    return final_metrics, metrics_count

def create_casp_csv_file(casp_dir, output_file):
    """Create a CSV file listing all CASP JSON files for ProteinDataset."""
    json_files = []
    
    # Collect from both subdirectories
    for subdir in ["tsdomains"]: # ["oligos", "tsdomains"]:
        subdir_path = os.path.join(casp_dir, subdir)
        if os.path.exists(subdir_path):
            files = glob.glob(os.path.join(subdir_path, "*.json"))
            json_files.extend(files)
    
    # Write CSV file with 4 columns as expected by ProteinDataset
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header: sequence_id, rep_id, rep_json_path, member_json_path
        writer.writerow(['sequence_id', 'rep_id', 'rep_json_path', 'member_json_path'])
        
        for file_path in json_files:
            # Extract protein ID from filename (remove .json extension)
            protein_id = os.path.splitext(os.path.basename(file_path))[0]
            # sequence_id (empty), rep_id (protein_id), rep_json_path (file_path), member_json_path (empty)
            writer.writerow(['', protein_id, file_path, ''])
    
    print(f"Created CSV file with {len(json_files)} CASP structures: {output_file}")
    return len(json_files)

def main():
    parser = argparse.ArgumentParser(description='Evaluate FSQ autoencoder on CASP15 data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to FSQ autoencoder checkpoint (.pt file)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return
    
    casp_dir = "/workspace/casp_data/casp15/jsons"
    if not os.path.exists(casp_dir):
        print(f"Error: CASP data directory {casp_dir} does not exist")
        return
    
    # Use experiments directory for output
    output_dir = os.path.dirname(__file__)
    
    print("="*60)
    print("CASP15 FSQ Autoencoder Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    
    # Load model
    print("\nLoading model...")
    # try:
    model, model_cfg, train_cfg = load_model_from_checkpoint(args.checkpoint, DEVICE)
    
    # Call post_init after loading configs from checkpoint to set global and local annotation information
    # When deserializing from checkpoint, __post_init__ is not called, so vocab sizes aren't computed
    model_cfg.autoencoder_path = '../../checkpoints/AWS/STAGE_2/checkpoint_step_40000.pt'
    model_cfg.vocab_per_residue_path = "/workspace/demo/Odyssey/odyssey/train/vocab_per_residue_annotations.txt"
    model_cfg.vocab_global_path = "/workspace/demo/Odyssey/odyssey/train/vocab_global_annotations.txt"
    train_cfg.data_dir = "/workspace/casp_data/casp15/jsons"
    train_cfg.checkpoint_dir = "/workspace/demo/Odyssey/checkpoints/fsq"
    train_cfg.mask_cfg = NoMaskConfig()
    model_cfg.__post_init__()
    train_cfg.__post_init__()
    
    model.eval()
    print(f"Model loaded successfully")
    print(f"Model style: {model_cfg.style}")
    print(f"Model type: {model_cfg.encoder_cfg.first_block_cfg.initials()}")
    
    # Load autoencoder if needed (for Stage 2 and other models that require struct tokens)
    autoencoder = None
    if model_cfg.style in {"stage_2", "mlm", "discrete_diffusion"}:
        print(f"Loading autoencoder from: {model_cfg.autoencoder_path}")
        autoencoder, autoencoder_model_cfg, _ = load_model_from_checkpoint(model_cfg.autoencoder_path, DEVICE)
        autoencoder.eval()
        autoencoder.requires_grad_(False)
        print(f"Autoencoder loaded successfully")
    
    # Create temporary CSV file for CASP data
    casp_csv = os.path.join(output_dir, 'casp15_files.csv')
    num_files = create_casp_csv_file(casp_dir, casp_csv)
    
    if num_files == 0:
        print("No CASP files found. Check data directory.")
        return
    
    # Create dataset and dataloader
    print(f"\nCreating dataset from {num_files} CASP structures...")
    dataset_mode = "side_chain" if model_cfg.style == "stage_2" else "backbone"
    dataset = ProteinDataset(
        casp_csv, 
        mode=dataset_mode, 
        max_length=model_cfg.max_len - 2,
        max_length_global=getattr(model_cfg, 'max_len_global', 512) - 2
    )
    
    print(f"Dataset created with {len(dataset)} entries")
    
    # Configure tracks based on model stage
    if model_cfg.style == "stage_1":
        tracks = {'seq': False, 'struct': False, 'coords': True, 'ss8': False, 'sasa': False, 'global_annotation': False, 'per_residue_annotation': False, 'plddt': False}
        min_unmasked = {'seq': 0, 'coords': 1}
    elif model_cfg.style == "stage_2":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': False, 'sasa': False, 'global_annotation': False, 'per_residue_annotation': False, 'plddt': False}
        min_unmasked = {'seq': 0, 'coords': 0}
    else:
        raise ValueError(f"Unknown model style: {model_cfg.style}")
    
    # Create data loader
    print("Creating data loader...")
    g_eval = torch.Generator()
    g_eval.manual_seed(42)
    
    # Use no masking for evaluation - note: for autoencoder evaluation, we don't pass the autoencoder
    data_loader = _get_training_dataloader(
        dataset, model_cfg, train_cfg, tracks, DEVICE,
        batch_size=1, shuffle=False, generator=g_eval,
        worker_init_fn=worker_init_fn, min_unmasked=min_unmasked,
        autoencoder=autoencoder  # Pass the loaded autoencoder
    )
    
    print(f"Data loader created")
    
    # Evaluate the model
    final_metrics, total_counts = evaluate_autoencoder(model, data_loader, model_cfg, train_cfg)
    
    # Determine evaluation type
    evaluation_type = f"Stage {model_cfg.style.split('_')[1]} ({'Backbone' if model_cfg.style == 'stage_1' else 'All Heavy Atoms'})"
    
    # Display results
    if final_metrics:
        print(f"\n{evaluation_type} Evaluation Results:")
        print("="*50)
        
        # Display metrics
        for metric_name, value in final_metrics.items():
            total_structures = total_counts.get(metric_name, 0)
            if metric_name == 'rmsd':
                print(f"Mean Kabsch RMSD: {value:.4f} Å (from {total_structures} structures)")
            elif metric_name == 'loss':
                print(f"Mean Loss: {value:.4e} (from {total_structures} structures)")
            else:
                print(f"Mean {metric_name}: {value:.4f} (from {total_structures} structures)")
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'detailed_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"CASP15 FSQ Autoencoder Evaluation Results\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Evaluation Type: {evaluation_type}\n")
            f.write(f"Model Type: {model_cfg.encoder_cfg.first_block_cfg.initials()}\n")
            f.write("="*60 + "\n\n")
            
            f.write("Summary Statistics:\n")
            for metric_name, value in final_metrics.items():
                total_structures = total_counts.get(metric_name, 0)
                if metric_name == 'rmsd':
                    f.write(f"Mean Kabsch RMSD: {value:.4f} Å (from {total_structures} structures)\n")
                elif metric_name == 'loss':
                    f.write(f"Mean Loss: {value:.4e} (from {total_structures} structures)\n")
                else:
                    f.write(f"Mean {metric_name}: {value:.4f} (from {total_structures} structures)\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Save summary
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Evaluation Type: {evaluation_type}\n")
            for metric_name, value in final_metrics.items():
                total_structures = total_counts.get(metric_name, 0)
                f.write(f"{metric_name}: {value:.4f} ({total_structures} structures)\n")
        
        print(f"Summary saved to: {summary_file}")
        
    else:
        print("No results generated. Check model compatibility and data.")
    
    # Clean up temporary CSV file
    if os.path.exists(casp_csv):
        os.remove(casp_csv)

if __name__ == "__main__":
    main() 