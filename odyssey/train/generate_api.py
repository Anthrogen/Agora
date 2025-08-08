import os, sys
import torch
import json
import argparse
import numpy as np
import random

# Import from existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from odyssey.src.models.autoencoder import Autoencoder
from odyssey.src.models.transformer import TransformerTrunk
from odyssey.src.dataloader import NoMaskDataLoader, worker_init_fn
from odyssey.src.dataset import ProteinDataset, Protein
from odyssey.src.vocabulary import SEQUENCE_TOKENS
from odyssey.src.model_librarian import load_model_from_checkpoint
from odyssey.src.tokenizer import CorruptionMode
from odyssey.train.mlm_step import generate_mlm
from odyssey.train.discrete_diffusion_step import generate_discrete_diffusion


def tokens_to_sequence(seq_tokens, vocab_mapping):
    """Convert sequence tokens back to amino acid sequence."""
    sequence = []
    for token in seq_tokens.squeeze():
        if token.item() < len(SEQUENCE_TOKENS):
            # Convert token index to amino acid
            aa = list(SEQUENCE_TOKENS)[token.item()].name
            sequence.append(aa)
    return ''.join(sequence)


def generate_single_protein(protein_json_path: str, model_checkpoint: str, output_path: str = None):
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

    # Use NoMaskDataLoader since asterisks are automatically converted to MASK tokens
    GENERATION_BATCH_SIZE = 1
    print("Creating NoMaskDataLoader (asterisks auto-converted to MASK tokens)...")
    val_loader = NoMaskDataLoader(dataset, transformer_model_cfg, transformer_train_cfg, tracks, device, 
                                 batch_size=GENERATION_BATCH_SIZE, shuffle=False, generator=g_val, 
                                 worker_init_fn=worker_init_fn, autoencoder=autoencoder)

    ###########################################################################
    # Generation (same as generate.py)
    ###########################################################################
    
    transformer.eval()
    with torch.no_grad():
        
        for batch in val_loader:
            if batch is None: continue
            
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
            original_sequence = tokens_to_sequence(batch.unmasked_data['seq'], SEQUENCE_TOKENS)
            print(f"Original sequence:  {original_sequence}")
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
            
            # Save output
            if output_path is None:
                output_path = protein_json_path.replace('.json', '_generated.json')
            
            print(f"Saving generated protein to {output_path}")
            protein.dump_to_json(output_path, original_data)
            
            print("Generation complete!")
            break  # Only process the first (and only) protein
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sequences and structures for masked protein positions')
    parser.add_argument('--protein', type=str, required=True, 
                       help='Path to protein JSON file with masked positions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for generated protein JSON (default: input_generated.json)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.protein):
        raise FileNotFoundError(f"Protein file not found: {args.protein}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    generate_single_protein(args.protein, args.checkpoint, args.output) 