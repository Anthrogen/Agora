import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple, Callable, Dict
import random
from types import SimpleNamespace
import argparse

# Import the model and data loader from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder, StandardTransformerBlock
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.models.autoencoder import FSQEncoder
from odyssey.src.dataloader import SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader, worker_init_fn
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from odyssey.src.configurations import TransformerConfig, TrainingConfig
from odyssey.src.configurations import *
from odyssey.src.config_loader import load_config, load_multi_configs
from odyssey.src.model_librarian import ensure_identical_parameters_transformers, ensure_identical_parameters_autoencoders, load_model_from_empty, load_model_from_checkpoint, save_model_checkpoint, save_summary_history
from odyssey.src.tokenizer import CorruptionMode

from odyssey.train.yaml_expander import expand_yaml_to_directory
from odyssey.train.generate_experiment_map import generate_experiment_map
from odyssey.generation.mlm_gen import generate_mlm
from odyssey.generation.discrete_diffusion_gen import generate_discrete_diffusion
from odyssey.src.losses import _kabsch_align
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb

def tokens_to_sequence(seq_tokens, vocab_mapping):
    """Convert sequence tokens back to amino acid sequence."""
    sequence = []
    for token in seq_tokens.squeeze():
        if token.item() < len(SEQUENCE_TOKENS):
            # Convert token index to amino acid
            aa = list(SEQUENCE_TOKENS)[token.item()].name
            sequence.append(aa)
        elif token.item() == SPECIAL_TOKENS.BOS.value + len(SEQUENCE_TOKENS):
            sequence.append('BOS')
        elif token.item() == SPECIAL_TOKENS.EOS.value + len(SEQUENCE_TOKENS):
            sequence.append('EOS')
        elif token.item() == SPECIAL_TOKENS.PAD.value + len(SEQUENCE_TOKENS):
            sequence.append('PAD')
        else:
            sequence.append('UNK')
    return ''.join([s for s in sequence if s not in ['BOS', 'EOS', 'PAD', 'UNK']])

def visualize_structure_comparison(original_coords, reconstructed_coords, generated_coords, 
                                  original_seq, generated_seq):
    """Create clean protein structure visualizations using matplotlib."""
    
    print("\n" + "="*80)
    print("PROTEIN STRUCTURE VISUALIZATION")
    print("="*80)
    print(f"Original sequence:  {original_seq}")
    print(f"Generated sequence: {generated_seq}")
    print(f"Sequence length: {len(original_seq)} residues")
    
    # Calculate sequence similarity
    matches = sum(1 for a, b in zip(original_seq, generated_seq) if a == b)
    similarity = matches / len(original_seq) * 100
    print(f"Sequence identity: {similarity:.1f}% ({matches}/{len(original_seq)})")
    
    # Trim coordinates to actual sequence length
    seq_len = len(original_seq)
    original_trimmed = original_coords[:seq_len, :3, :]  # [seq_len, 3_backbone, 3_coords]
    reconstructed_trimmed = reconstructed_coords[:seq_len, :3, :]
    generated_trimmed = generated_coords[:seq_len, :3, :]
    
    # Extract backbone atoms and flatten for alignment
    original_backbone = original_trimmed.reshape(-1, 3)
    reconstructed_backbone = reconstructed_trimmed.reshape(-1, 3)
    generated_backbone = generated_trimmed.reshape(-1, 3)
    
    # Center coordinates
    original_centered = original_backbone - original_backbone.mean(axis=0)
    reconstructed_centered = reconstructed_backbone - reconstructed_backbone.mean(axis=0)
    generated_centered = generated_backbone - generated_backbone.mean(axis=0)
    
    # Align structures using Kabsch algorithm
    original_tensor = torch.from_numpy(original_centered).float().unsqueeze(0)
    reconstructed_tensor = torch.from_numpy(reconstructed_centered).float().unsqueeze(0)
    generated_tensor = torch.from_numpy(generated_centered).float().unsqueeze(0)
    
    reconstructed_aligned = _kabsch_align(reconstructed_tensor, original_tensor)
    generated_aligned = _kabsch_align(generated_tensor, original_tensor)
    
    # Calculate RMSD
    recon_rmsd = torch.sqrt(((original_tensor - reconstructed_aligned) ** 2).sum(dim=-1).mean()).item()
    gen_rmsd = torch.sqrt(((original_tensor - generated_aligned) ** 2).sum(dim=-1).mean()).item()
    
    print(f"Reconstruction RMSD: {recon_rmsd:.2f} Å")
    print(f"Generation RMSD:     {gen_rmsd:.2f} Å")
    
    # Create visualization
    print("\nCreating protein structure visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    
    # Prepare aligned coordinate sets
    structures = [
        (original_centered, original_seq, 'Original', '#1f77b4'),
        (reconstructed_aligned.squeeze(0).numpy(), original_seq, 'Reconstructed', '#ff7f0e'),
        (generated_aligned.squeeze(0).numpy(), generated_seq, 'Generated', '#2ca02c')
    ]
    
    for ax, (coords, seq, title, color) in zip(axes, structures):
        # Reshape back to per-residue format for proper visualization
        coords_reshaped = coords.reshape(seq_len, 3, 3)  # [residues, atoms, xyz]
        
        # Extract CA atoms for backbone trace
        ca_coords = coords_reshaped[:, 1, :]  # CA is the second atom
        
        # Plot main backbone trace (CA atoms)
        ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                color=color, linewidth=3, alpha=0.8, label='Backbone')
        
        # Add CA atoms as spheres
        ax.scatter(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                   c=color, s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add bonds within each residue (N-CA-C)
        for i in range(seq_len):
            residue_coords = coords_reshaped[i]  # [3_atoms, 3_coords]
            # Draw N-CA bond
            ax.plot([residue_coords[0, 0], residue_coords[1, 0]],
                   [residue_coords[0, 1], residue_coords[1, 1]], 
                   [residue_coords[0, 2], residue_coords[1, 2]], 
                   color=color, linewidth=1.5, alpha=0.6)
            # Draw CA-C bond
            ax.plot([residue_coords[1, 0], residue_coords[2, 0]],
                   [residue_coords[1, 1], residue_coords[2, 1]], 
                   [residue_coords[1, 2], residue_coords[2, 2]], 
                   color=color, linewidth=1.5, alpha=0.6)
        
        # Add residue labels for key positions
        label_step = max(1, seq_len // 8)  # Show ~8 labels
        for i in range(0, seq_len, label_step):
            ax.text(ca_coords[i, 0], ca_coords[i, 1], ca_coords[i, 2], 
                   f'{seq[i]}{i+1}', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Styling
        ax.set_title(f'{title}\nSeq: {seq[:15]}{"..." if len(seq) > 15 else ""}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (Å)', fontsize=10)
        ax.set_ylabel('Y (Å)', fontsize=10)
        ax.set_zlabel('Z (Å)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Set consistent limits for all subplots
    all_coords = np.vstack([original_centered, reconstructed_aligned.squeeze(0).numpy(), generated_aligned.squeeze(0).numpy()])
    margin = 3
    xlim = [all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin]
    ylim = [all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin]
    zlim = [all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin]
    
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
    # Add overall title and metrics
    fig.suptitle(f'Protein Structure Comparison\n'
                f'Sequence Identity: {similarity:.1f}% | '
                f'Reconstruction RMSD: {recon_rmsd:.2f}Å | '
                f'Generation RMSD: {gen_rmsd:.2f}Å', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)  # Make more room for suptitle and subplot titles
    
    # Save the figure
    plt.savefig('protein_structure_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization as protein_structure_comparison.png")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

def generate(model_checkpoint, protein_json_path, callback=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###########################################################################
    # Load model
    ###########################################################################

    transformer, transformer_model_cfg, transformer_train_cfg = load_model_from_checkpoint(model_checkpoint, device)
    autoencoder, autoencoder_model_cfg, autoencoder_train_cfg = load_model_from_checkpoint(transformer_model_cfg.autoencoder_path, device)

    # Call post_init after loading configs from checkpoint to set global and local annotation information
    # When deserializing from checkpoint, __post_init__ is not called, so vocab sizes aren't computed
    transformer_model_cfg.__post_init__()
    transformer_train_cfg.__post_init__()
    autoencoder_model_cfg.__post_init__()
    autoencoder_train_cfg.__post_init__()

    transformer.eval()
    transformer.requires_grad_(False)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    
    assert isinstance(transformer, TransformerTrunk), "Model must be a TransformerTrunk"
    assert isinstance(autoencoder, Autoencoder), "Model must be an Autoencoder"

    # Use different masking strategies for stage 1 vs stage 2
    if transformer_model_cfg.style == "mlm":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}
    elif transformer_model_cfg.style == "discrete_diffusion":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}

    dataset_mode = "side_chain" if transformer_model_cfg.style == "stage_2" else "backbone"

    print(f"Generating for model:")
    print(transformer_model_cfg)
    print("with train config:")
    print(transformer_train_cfg)

    ###########################################################################
    #  Data Loading 
    ###########################################################################
    # Set seed for consistent generation
    data_seed = transformer_model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create DataLoaders with fixed seed for consistent masking
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000 + 1)

    print(f"Creating protein dataset from JSON: {protein_json_path}")
    dataset = ProteinDataset(protein_json_path, mode=dataset_mode, max_length=transformer_model_cfg.max_len - 2, max_length_orthologous_groups=transformer_model_cfg.max_len_orthologous_groups - 2, max_length_semantic_description=transformer_model_cfg.max_len_semantic_description - 2)
    print(f"Dataset size: {len(dataset)}")

    # Use the training dataloader for generation
    GENERATION_BATCH_SIZE = 1
    val_loader = _get_training_dataloader(dataset, transformer_model_cfg, transformer_train_cfg, tracks, device, 
                                         batch_size=GENERATION_BATCH_SIZE, shuffle=False, generator=g_val, 
                                         worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, autoencoder=autoencoder)

    transformer.eval()
    with torch.no_grad():
        
        for batch in val_loader:
            if batch is None: continue
            
            # Move batch to correct device (from CPU to GPU)
            batch = batch.to(device)
            
            if transformer_model_cfg.style == "mlm":
                batch = generate_mlm(transformer, transformer_model_cfg, transformer_train_cfg, batch) #, NUM_TO_UNMASK=2048)

            elif transformer_model_cfg.style == "discrete_diffusion" and transformer_train_cfg.mask_config.corruption_mode == "uniform":
                batch = generate_discrete_diffusion(CorruptionMode.UNIFORM, transformer, transformer_model_cfg, transformer_train_cfg, batch)

            elif transformer_model_cfg.style == "discrete_diffusion" and transformer_train_cfg.mask_config.corruption_mode == "absorb":
                raise NotImplementedError("Absorb corruption mode not implemented for generation at this time.")

            else: raise NotImplementedError(f"Style {transformer_model_cfg.style} not implemented for generation at this time.")
            
            # Visualize the generated protein
            print("Visualizing generated protein...")
            
            # Convert sequence tokens to amino acid sequence
            generated_sequence = tokens_to_sequence(batch.masked_data['seq'], SEQUENCE_TOKENS)
            original_sequence = tokens_to_sequence(batch.unmasked_data['seq'], SEQUENCE_TOKENS)
            print(f"Original sequence:  {original_sequence}")
            print(f"Generated sequence: {generated_sequence}")
            
            # Get original coordinates for comparison
            original_coords = batch.unmasked_data['coords'].squeeze(0).cpu().numpy()  # [L, num_atoms, 3]
            
            # Decode structure tokens to coordinates using autoencoder decoder
            with torch.no_grad():
                # Decode to coordinates
                content_elements = ~batch.beospank['coords'] & ~batch.masks['coords']
                nonbeospank = ~batch.beospank['coords']

                # Get structure tokens and feed through autoencoder decoder
                struct_tokens = batch.masked_data['struct']  # [B, L]
                nonbeospank_struct_tokens = torch.where(struct_tokens < autoencoder.codebook_size, struct_tokens, torch.zeros_like(struct_tokens))
                z_q = autoencoder.quantizer.indices_to_codes(nonbeospank_struct_tokens)  # [B, L, fsq_dim]

                # Zero out BOS/EOS/PAD/UNK positions in z_q
                z_q[batch.beospank['coords']] = 0.0
                
                # Different arguments based on autoencoder architecture
                model_type = autoencoder_model_cfg.first_block_cfg.initials()
                if model_type in ("GA", "RA"): 
                    four_atom = batch.masked_data['coords'][:, :, :4, :]  # [B, L, 4, 3] for GA/RA
                    coords = autoencoder.decoder(z_q, coords=four_atom, content_elements=content_elements, nonbeospank=nonbeospank, seq_tokens=batch.masked_data['seq'])
                else: coords = autoencoder.decoder(z_q, nonbeospank=nonbeospank, seq_tokens=batch.masked_data['seq'])
                
                # Get reconstructed coordinates by passing original unmasked data through full autoencoder
                with torch.no_grad():
                    three_atom = batch.unmasked_data['coords'][:, :, :3, :]
                    four_atom = batch.unmasked_data['coords'][:, :, :4, :]
                    
                    # Pass through full autoencoder with original sequence tokens
                    if model_type in ("GA", "RA"): 
                        reconstructed_coords, _ = autoencoder(three_atom, coords=four_atom, content_elements=content_elements, nonbeospank=nonbeospank, seq_tokens=batch.unmasked_data['seq'])
                    else: 
                        reconstructed_coords, _ = autoencoder(three_atom, content_elements=content_elements, nonbeospank=nonbeospank, seq_tokens=batch.unmasked_data['seq'])
                
                # Visualize the structure comparison
                generated_coords_np = coords.squeeze(0).cpu().numpy()
                reconstructed_coords_np = reconstructed_coords.squeeze(0).cpu().numpy()
                visualize_structure_comparison(original_coords, reconstructed_coords_np, generated_coords_np, original_sequence, generated_sequence)
            
            break  # Only generate one protein for visualization

    return batch

if __name__ == "__main__":
    """
    Easy way to test:
    python generation.py --checkpoint ../../checkpoints/transformer_trunk/mlm_simple_config/mlm_simple_config_000/checkpoint_step_89184.pt --protein ../../sample_data/27k/AF-A0A009G5B5-F1.json
    python generation.py --checkpoint ../../checkpoints/transformer_trunk/mlm_complex_config/mlm_complex_config_000/checkpoint_step_89184.pt --protein ../../sample_data/27k/AF-A0A009G5B5-F1.json
    python generation.py --checkpoint ../../checkpoints/transformer_trunk/discrete_diffusion_absorb_config/discrete_diffusion_absorb_config_000/checkpoint_step_89184.pt --protein ../../sample_data/27k/AF-A0A009G5B5-F1.json
    python generation.py --checkpoint ../../checkpoints/transformer_trunk/discrete_diffusion_uniform_config/discrete_diffusion_uniform_config_000/checkpoint_step_89184.pt --protein ../../sample_data/27k/AF-A0A009G5B5-F1.json
    """
    parser = argparse.ArgumentParser(description='Generate Odyssey models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fully trained model checkpoint')
    parser.add_argument('--protein', type=str, required=True, help='Path to protein JSON file')
    args = parser.parse_args()
    
    assert os.path.exists(args.checkpoint), f"Checkpoint {args.checkpoint} does not exist."
    assert os.path.exists(args.protein), f"Protein file {args.protein} does not exist."
    batch = generate(args.checkpoint, args.protein)
