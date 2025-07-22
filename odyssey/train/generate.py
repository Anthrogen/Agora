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
from odyssey.src.dataloader import MaskedBatch, SimpleDataLoader, ComplexDataLoader, DiffusionDataLoader, NoMaskDataLoader, _get_training_dataloader, worker_init_fn
from odyssey.src.dataset import ProteinDataset
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.losses import kabsch_rmsd_loss, squared_kabsch_rmsd_loss
from odyssey.src.configurations import TransformerConfig, TrainingConfig
from odyssey.train.fsq_step import stage_1_step, stage_2_step
from odyssey.train.mlm_step import mlm_step
from odyssey.train.discrete_diffusion_step import discrete_diffusion_step
from odyssey.src.configurations import *
from odyssey.src.config_loader import load_config, load_multi_configs
from odyssey.src.model_librarian import ensure_identical_parameters_transformers, ensure_identical_parameters_autoencoders, load_model_from_empty, load_model_from_checkpoint, save_model_checkpoint, save_summary_history

from odyssey.train.yaml_expander import expand_yaml_to_directory
from odyssey.train.generate_experiment_map import generate_experiment_map
from odyssey.train.mlm_step import generate_mlm
from odyssey.src.losses import _kabsch_align
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def tokens_to_sequence(seq_tokens, vocab_mapping):
    """Convert sequence tokens back to amino acid sequence."""
    sequence = []
    for token in seq_tokens.squeeze():
        if token.item() < len(SEQUENCE_TOKENS):
            # Convert token index to amino acid
            aa = list(SEQUENCE_TOKENS)[token.item()].name
            sequence.append(aa)
        elif token.item() == SPECIAL_TOKENS.BOS.value + len(SEQUENCE_TOKENS):
            sequence.append('[BOS]')
        elif token.item() == SPECIAL_TOKENS.EOS.value + len(SEQUENCE_TOKENS):
            sequence.append('[EOS]')
        elif token.item() == SPECIAL_TOKENS.PAD.value + len(SEQUENCE_TOKENS):
            sequence.append('[PAD]')
        else:
            sequence.append('[UNK]')
    return ''.join([s for s in sequence if s not in ['[BOS]', '[EOS]', '[PAD]', '[UNK]']])

def visualize_structure_comparison(original_coords, reconstructed_coords, generated_coords, 
                                  original_seq, generated_seq):
    """Visualize original, reconstructed, and generated protein structures after Kabsch alignment."""
    # Extract backbone atoms and flatten
    original_backbone = torch.from_numpy(original_coords[:, :3, :].reshape(-1, 3)).float().unsqueeze(0)
    reconstructed_backbone = torch.from_numpy(reconstructed_coords[:, :3, :].reshape(-1, 3)).float().unsqueeze(0)
    generated_backbone = torch.from_numpy(generated_coords[:, :3, :].reshape(-1, 3)).float().unsqueeze(0)
    
    # Mean center all structures
    original_centered = original_backbone - original_backbone.mean(dim=1, keepdim=True)
    reconstructed_centered = reconstructed_backbone - reconstructed_backbone.mean(dim=1, keepdim=True)
    generated_centered = generated_backbone - generated_backbone.mean(dim=1, keepdim=True)
    
    # Align reconstructed and generated to original
    reconstructed_aligned = _kabsch_align(reconstructed_centered, original_centered)
    generated_aligned = _kabsch_align(generated_centered, original_centered)
    
    # Convert to numpy
    original_np = original_centered.squeeze(0).numpy()
    reconstructed_np = reconstructed_aligned.squeeze(0).numpy()
    generated_np = generated_aligned.squeeze(0).numpy()
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Plot original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(original_np[:, 0], original_np[:, 1], original_np[:, 2], 'b-', alpha=0.7, linewidth=2)
    ax1.set_title(f'Original: {original_seq}')
    
    # Plot reconstructed
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(reconstructed_np[:, 0], reconstructed_np[:, 1], reconstructed_np[:, 2], 'orange', alpha=0.7, linewidth=2)
    ax2.set_title('Reconstructed')
    
    # Plot generated
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(generated_np[:, 0], generated_np[:, 1], generated_np[:, 2], 'g-', alpha=0.7, linewidth=2)
    ax3.set_title(f'Generated: {generated_seq}')
    
    # Set consistent limits and viewing angle for all three
    all_coords = np.vstack([original_np, reconstructed_np, generated_np])
    margin = 5
    xlim = [all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin]
    ylim = [all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin]
    zlim = [all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin]
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('protein_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization as protein_comparison.png")

def generate(model_checkpoint, callback=None):

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
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'global_annotation': True, 'per_residue_annotation': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}
    elif transformer_model_cfg.style == "discrete_diffusion":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'global_annotation': True, 'per_residue_annotation': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}

    dataset_mode = "side_chain" if transformer_model_cfg.style == "stage_2" else "backbone"

    print(f"Generating for model:")
    print(transformer_model_cfg)
    print("with train config:")
    print(transformer_train_cfg)

    ###########################################################################
    #  Data Loading 
    ###########################################################################
    # Set seed for dataset split
    data_seed = transformer_model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create DataLoaders with fixed seed for consistent masking
    g_train = torch.Generator()
    g_train.manual_seed(data_seed)
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000)

    print("Loading dataset...")
    dataset = ProteinDataset(transformer_train_cfg.data_dir, mode=dataset_mode, max_length=transformer_model_cfg.max_len - 2, max_length_global=transformer_model_cfg.max_len_global - 2)
    print("...done.")
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    # We use g_val as the generator of the split
    _, data = random_split(dataset, [train_size, val_size], generator=g_val)

    print(f"Constructed Validation Dataset of size {val_size}")
    print(data)

    # Do not mask anything.
    GENERATION_BATCH_SIZE = 1
    # For actual generation we need to use Nonmasking dataloader.
    # val_loader = NoMaskDataLoader(data, transformer_model_cfg, transformer_train_cfg, tracks, device, batch_size=GENERATION_BATCH_SIZE, shuffle=False, generator=g_val, 
    #                                     worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, 
    #                                     autoencoder=autoencoder)
    val_loader = _get_training_dataloader(data, transformer_model_cfg, transformer_train_cfg, tracks, device, batch_size=GENERATION_BATCH_SIZE, shuffle=False, generator=g_val, 
                                        worker_init_fn=worker_init_fn, min_unmasked=min_unmasked, 
                                        autoencoder=autoencoder)

    transformer.eval()
    with torch.no_grad():
        
        for batch in val_loader:
            if batch is None: continue
            
            if transformer_model_cfg.style == "mlm":
                batch = generate_mlm(transformer, transformer_model_cfg, transformer_train_cfg, batch)
            elif transformer_model_cfg.style == "discrete_diffusion" and transformer_train_cfg.mask_config.corruption_mode == "uniform":
                raise NotImplementedError("Uniform corruption mode not implemented for generation at this time.")
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
                
                # Get reconstructed coordinates by passing original through autoencoder
                with torch.no_grad():
                    three_atom = batch.unmasked_data['coords'][:, :, :3, :]
                    four_atom = batch.unmasked_data['coords'][:, :, :4, :]
                    if model_type in ("GA", "RA"): 
                        reconstructed_coords, _ = autoencoder(three_atom, four_atom, ~batch.beospank['coords'], ~batch.beospank['coords'])
                    else: reconstructed_coords, _ = autoencoder(three_atom, nonbeospank=~batch.beospank['coords'])
                
                # Visualize the structure comparison
                generated_coords_np = coords.squeeze(0).cpu().numpy()
                reconstructed_coords_np = reconstructed_coords.squeeze(0).cpu().numpy()
                visualize_structure_comparison(original_coords, reconstructed_coords_np, generated_coords_np, original_sequence, generated_sequence)
            
            break  # Only generate one protein for visualization

    return batch

if __name__ == "__main__":
    """
    Easy way to test:
    python generate.py --checkpoint ../../checkpoints/transformer_trunk/mlm_config/mlm_config_000/model.pt
    python generate.py --checkpoint ../../checkpoints/transformer_trunk/discrete_diffusion_config/discrete_diffusion_config_000/model.pt
    """
    parser = argparse.ArgumentParser(description='Train Odyssey models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fully trained model checkpoint')
    args = parser.parse_args()
    
    assert os.path.exists(args.checkpoint), f"Checkpoint {args.checkpoint} does not exist."
    batch = generate(args.checkpoint)
    
