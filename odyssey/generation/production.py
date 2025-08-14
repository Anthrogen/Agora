import os, sys
import torch
import json
import argparse
import numpy as np
import random
from dataclasses import replace
import matplotlib.pyplot as plt

# Import from existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.dataloader import ContentBasedDataLoader, worker_init_fn
from odyssey.src.dataset import ProteinDataset, Protein
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS
from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.tokenizer import CorruptionMode
from odyssey.generation.mlm_gen import generate_mlm
from odyssey.generation.discrete_diffusion_gen import generate_discrete_diffusion
from odyssey.src.configurations import NoMaskConfig
from odyssey.src.losses import _kabsch_align

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


def visualize_structures(reference_coords, generated_coords, reference_seq, generated_seq, output_path):
    """Visualize original vs generated protein structures with Kabsch alignment."""
    
    # Convert sequences to the same length for comparison
    seq_len = min(len(reference_seq), len(generated_seq))
    reference_seq = reference_seq[:seq_len]
    generated_seq = generated_seq[:seq_len]
    
    # Convert to numpy if tensors
    if isinstance(reference_coords, torch.Tensor):
        reference_coords = reference_coords.cpu().numpy()
    if isinstance(generated_coords, torch.Tensor):
        generated_coords = generated_coords.cpu().numpy()
    
    # Trim coordinates to sequence length and extract backbone atoms
    reference_trimmed = reference_coords[:seq_len, :3, :]  # [seq_len, 3_backbone, 3_coords]
    generated_trimmed = generated_coords[:seq_len, :3, :]
    
    # Extract backbone atoms and flatten for alignment
    reference_backbone = reference_trimmed.reshape(-1, 3)
    generated_backbone = generated_trimmed.reshape(-1, 3)
    
    # Center coordinates
    reference_centered = reference_backbone - reference_backbone.mean(axis=0)
    generated_centered = generated_backbone - generated_backbone.mean(axis=0)
    
    # Align structures using Kabsch algorithm
    reference_tensor = torch.from_numpy(reference_centered).float().unsqueeze(0)
    generated_tensor = torch.from_numpy(generated_centered).float().unsqueeze(0)
    
    generated_aligned = _kabsch_align(generated_tensor, reference_tensor)
    
    # Calculate RMSD
    rmsd = torch.sqrt(((reference_tensor - generated_aligned) ** 2).sum(dim=-1).mean()).item()
    
    print(f"Structure RMSD after alignment: {rmsd:.2f} Å")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': '3d'})
    
    # Prepare coordinate sets
    structures = [
        (reference_centered, reference_seq, 'Reference', '#1f77b4'),
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
    all_coords = np.vstack([reference_centered, generated_aligned.squeeze(0).numpy()])
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
    fig.suptitle(f'Protein Structure Comparison (RMSD: {rmsd:.2f} Å)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.json', '_structure_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Structure comparison plot saved to: {plot_path}")
    
    plt.show()


def generate_single_protein(protein_json_path: str, model_checkpoint: str, reference_json_path: str = None):
    """Generate sequences and structures for masked positions in a single protein JSON file."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading protein from {protein_json_path}")
    with open(protein_json_path, 'r') as f:
        original_data = json.load(f)
    
    original_sequence = original_data.get('sequence', '')
    print(f"Original sequence: {original_sequence}")
    print(f"Masked positions (*): {original_sequence.count('*')}")
    
    ###########################################################################
    # Load models (same as generate.py)
    ###########################################################################
    
    transformer, transformer_model_cfg, transformer_train_cfg = load_model_from_checkpoint(model_checkpoint, device)
    autoencoder, autoencoder_model_cfg, autoencoder_train_cfg = load_model_from_checkpoint(transformer_model_cfg.autoencoder_path, device)

    # Call post_init after loading configs from checkpoint
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

    # Set tracks and min_unmasked (same as generate.py)
    if transformer_model_cfg.style == "mlm":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}
    elif transformer_model_cfg.style == "discrete_diffusion":
        tracks = {'seq': True, 'struct': True, 'coords': True, 'ss8': True, 'sasa': True, 'orthologous_groups': True, 'semantic_description': True, 'domains': True, 'plddt': True}
        min_unmasked = {'seq': 0, 'coords': 1}
    else:
        raise NotImplementedError(f"Style {transformer_model_cfg.style} not implemented")

    dataset_mode = "side_chain" if transformer_model_cfg.style == "stage_2" else "backbone"

    ###########################################################################
    # Create dataset using JSON support (no temporary files!)
    ###########################################################################
    
    # Set seed (same as generate.py)
    data_seed = transformer_model_cfg.reference_model_seed
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    random.seed(data_seed)

    # Create generators (same as generate.py)
    g_val = torch.Generator()
    g_val.manual_seed(data_seed + 5000 + 1)

    print("Creating protein dataset from JSON...")
    # Use ProteinDataset with JSON support - it will automatically detect .json extension
    dataset = ProteinDataset(protein_json_path, mode=dataset_mode, 
                           max_length=transformer_model_cfg.max_len - 2, 
                           max_length_orthologous_groups=transformer_model_cfg.max_len_orthologous_groups - 2, 
                           max_length_semantic_description=transformer_model_cfg.max_len_semantic_description - 2)
    
    print(f"Dataset size: {len(dataset)}")

    # Use ContentBasedDataLoader to properly handle existing masks in the protein data
    GENERATION_BATCH_SIZE = 1
    print("Creating ContentBasedDataLoader (masks based on data content: *, [-1,-1,-1], -1, ...")
    
    val_loader = ContentBasedDataLoader(dataset, transformer_model_cfg, transformer_train_cfg, tracks, device, 
                                       batch_size=GENERATION_BATCH_SIZE, shuffle=False, generator=g_val, 
                                       autoencoder=autoencoder, min_unmasked=min_unmasked, worker_init_fn=worker_init_fn)

    ###########################################################################
    # Generation (same as generate.py)
    ###########################################################################
    
    transformer.eval()
    with torch.no_grad():
        
        for batch in val_loader:
            if batch is None: continue
            
            # Move batch to correct device (from CPU to GPU)
            batch = batch.to(device)
            
            # Generate based on model style (same as generate.py)
            if transformer_model_cfg.style == "mlm":
                batch = generate_mlm(transformer, transformer_model_cfg, transformer_train_cfg, batch)

            elif transformer_model_cfg.style == "discrete_diffusion" and transformer_train_cfg.mask_config.corruption_mode == "uniform":
                batch = generate_discrete_diffusion(CorruptionMode.UNIFORM, transformer, transformer_model_cfg, transformer_train_cfg, batch)

            elif transformer_model_cfg.style == "discrete_diffusion" and transformer_train_cfg.mask_config.corruption_mode == "absorb":
                raise NotImplementedError("Absorb corruption mode not implemented for generation at this time.")

            else: 
                raise NotImplementedError(f"Style {transformer_model_cfg.style} not implemented for generation at this time.")
            
            # Extract results (same as generate.py)
            generated_sequence = tokens_to_sequence(batch.masked_data['seq'], SEQUENCE_TOKENS)
            print(f"Generated sequence: {generated_sequence}")
            
            # Decode structure tokens to coordinates (same as generate.py)
            with torch.no_grad():
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
                else: 
                    coords = autoencoder.decoder(z_q, nonbeospank=nonbeospank, seq_tokens=batch.masked_data['seq'])
            
            # Update the protein data and save
            generated_coords = coords.squeeze(0).cpu()
            
            # Create a Protein object with the generated data
            protein = Protein(protein_json_path, mode=dataset_mode)
            
            # Update sequence and coordinates
            protein.seq = list(generated_sequence)
            protein.coords = generated_coords
            protein.len = len(protein.seq)  # Update length to match new sequence
            
            # Auto-create output path in gen folder
            protein_dir = os.path.dirname(protein_json_path)
            sample_data_dir = os.path.dirname(protein_dir)  # Go up to sample_data
            gen_dir = os.path.join(sample_data_dir, "gen")
            
            # Create gen directory if it doesn't exist
            os.makedirs(gen_dir, exist_ok=True)
            
            # Create output filename with _gen.json suffix
            protein_filename = os.path.basename(protein_json_path)
            output_filename = protein_filename.replace('.json', '_gen.json')
            output_path = os.path.join(gen_dir, output_filename)
            
            print(f"Saving generated protein to {output_path}")
            protein.dump_to_json(output_path, original_data)
            
            # Visualization if reference JSON provided
            if reference_json_path:
                print(f"\nLoading reference structure from {reference_json_path}")
                reference_protein = Protein(reference_json_path, mode=dataset_mode)
                reference_coords = reference_protein.coords
                reference_sequence = ''.join(reference_protein.seq)
                
                print("Creating structure comparison visualization...")
                visualize_structures(reference_coords, generated_coords, reference_sequence, generated_sequence, output_path)
            
            print("Generation complete!")
            break  # Only process the first (and only) protein
    
    return output_path


if __name__ == "__main__":
    """
    Easy way to test:
    python production.py --protein ../../sample_data/1k/AF-A0A009LZF3-F1.json --checkpoint ../../checkpoints/transformer_trunk/mlm_simple_config/mlm_simple_config_000/checkpoint_step_89184.pt --reference ../../sample_data/27k/AF-A0A009LZF3-F1.json    
    """
    parser = argparse.ArgumentParser(description='Generate sequences and structures for masked protein positions')
    parser.add_argument('--protein', type=str, required=True, 
                       help='Path to protein JSON file with masked positions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--reference', type=str, default=None,
                       help='Path to reference protein JSON for structure comparison visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.protein):
        raise FileNotFoundError(f"Protein file not found: {args.protein}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    generate_single_protein(args.protein, args.checkpoint, reference_json_path=args.reference) 