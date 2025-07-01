'''
This is for the seperately saved encoder and decoder stack but same 
behavior as casp_val.py otherwise.
'''

#!/usr/bin/env python3
import os, sys, random
import json
import glob
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ensure project root is on PYTHONPATH so we can import src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fsq_autoencoder_var3 import FSQEncoder, FSQDecoder
from src.losses import (
    kabsch_rmsd_loss,
    soft_lddt_loss,
    mask_and_flatten,
    _kabsch_align,
)

# ============= Config =============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "checkpoints"
CASP_DIR = "/workspace/casp_data/casp15/jsons/tsdomains"
OUTPUT_DIR = "checkpoint_visuals"

# Model parameters
HIDDEN_DIM = 64
LATENT_DIM = 32
FSQ_DIM = 5
FSQ_LEVELS = [7, 5, 5, 5, 5]
ATTN_HEADS = 4

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Load CASP Structures ===========
def load_casp_structure(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract backbone coordinates
    coords_N = np.asarray(data["backbone_coordinates"]["N"], dtype=np.float32)
    coords_CA = np.asarray(data["backbone_coordinates"]["CA"], dtype=np.float32)
    coords_C = np.asarray(data["backbone_coordinates"]["C"], dtype=np.float32)
    
    # Stack into (L, 3 atom-types, 3 coords)
    coords = np.stack([coords_N, coords_CA, coords_C], axis=1)  # -> (L,3,3)
    
    # Center the coordinates
    centroid = coords.reshape(-1, 3).mean(axis=0)
    coords = coords - centroid  # broadcast over (L,3,3)
    
    # Convert to torch tensor
    coords_t = torch.from_numpy(coords)
    mask_t = torch.ones(coords.shape[0], dtype=torch.bool)
    
    return coords_t, mask_t

# Load all CASP structures
casp_files = glob.glob(os.path.join(CASP_DIR, "*.json"))
casp_structures = []
for json_file in casp_files:
    try:
        coords, mask = load_casp_structure(json_file)
        casp_structures.append((coords, mask, os.path.basename(json_file)))
    except Exception as e:
        print(f"Error loading {json_file}: {e}")

print(f"Loaded {len(casp_structures)} CASP structures")

# ========== Load Model Checkpoints ===========
def load_separate_encoder_decoder(epoch_id):
    """
    Load encoder and decoder from separate checkpoint files.
    
    Args:
        epoch_id: Either 'final' or an epoch number
    
    Returns:
        encoder, decoder, epoch_num
    """
    # Initialize encoder and decoder
    encoder = FSQEncoder(
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        fsq_dim=FSQ_DIM,
        fsq_levels=FSQ_LEVELS,
        attn_heads=ATTN_HEADS
    ).to(DEVICE)
    
    decoder = FSQDecoder(
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        fsq_dim=FSQ_DIM,
        attn_heads=ATTN_HEADS
    ).to(DEVICE)
    
    # Define file paths
    encoder_path = os.path.join(CKPT_DIR, f"encoder_{epoch_id}.pt")
    decoder_path = os.path.join(CKPT_DIR, f"decoder_{epoch_id}.pt")
    
    # Check if files exist
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Could not find encoder or decoder files for epoch {epoch_id}")
    
    # Load encoder and decoder weights
    encoder_state = torch.load(encoder_path, map_location=DEVICE)
    decoder_state = torch.load(decoder_path, map_location=DEVICE)
    
    # Remove 'module.' prefix if present (from DDP)
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    encoder.load_state_dict(clean_state_dict(encoder_state))
    decoder.load_state_dict(clean_state_dict(decoder_state))
    
    # Set to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Determine epoch number
    if epoch_id == 'final':
        # Look for metadata in the checkpoint directory
        metadata_files = glob.glob(os.path.join(CKPT_DIR, "checkpoint_*.pt"))
        if metadata_files:
            try:
                checkpoint = torch.load(metadata_files[0], map_location=DEVICE)
                epoch_num = checkpoint.get('epoch', 0)
            except:
                epoch_num = 0
        else:
            epoch_num = 0
    else:
        epoch_num = int(epoch_id)
    
    return encoder, decoder, epoch_num

# Load all checkpoint pairs
checkpoint_pairs = []
# First look for final checkpoints
if os.path.exists(os.path.join(CKPT_DIR, "encoder_final.pt")) and \
   os.path.exists(os.path.join(CKPT_DIR, "decoder_final.pt")):
    try:
        encoder, decoder, epoch = load_separate_encoder_decoder('final')
        checkpoint_pairs.append((encoder, decoder, epoch, 'final'))
        print(f"Loaded final encoder and decoder (epoch {epoch})")
    except Exception as e:
        print(f"Error loading final encoder and decoder: {e}")

# Then look for numbered epoch checkpoints
encoder_files = glob.glob(os.path.join(CKPT_DIR, "encoder_epoch_*.pt"))
for encoder_file in encoder_files:
    # Extract epoch number
    file_name = os.path.basename(encoder_file)
    epoch_id = file_name.replace("encoder_epoch_", "").replace(".pt", "")
    
    # Check if corresponding decoder exists
    decoder_file = os.path.join(CKPT_DIR, f"decoder_epoch_{epoch_id}.pt")
    if os.path.exists(decoder_file):
        try:
            encoder, decoder, epoch = load_separate_encoder_decoder(f"epoch_{epoch_id}")
            checkpoint_pairs.append((encoder, decoder, epoch, f"epoch_{epoch_id}"))
            print(f"Loaded encoder and decoder for epoch {epoch_id}")
        except Exception as e:
            print(f"Error loading encoder and decoder for epoch {epoch_id}: {e}")

print(f"Loaded {len(checkpoint_pairs)} encoder-decoder pairs")

# If no encoder-decoder pairs were found, fall back to full checkpoint files
if not checkpoint_pairs:
    print("No separate encoder-decoder pairs found, falling back to full checkpoint files...")
    from src.models.fsq_autoencoder_var3 import FSQResidueAutoencoder
    
    def load_checkpoint(checkpoint_path):
        # Initialize model with same architecture as training
        model = FSQResidueAutoencoder(
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            fsq_dim=FSQ_DIM,
            fsq_levels=FSQ_LEVELS,
            attn_heads=ATTN_HEADS
        ).to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        # Remove 'module.' prefix if present (from DDP)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        return model, checkpoint['epoch']

    # Load all checkpoints
    checkpoint_files = glob.glob(os.path.join(CKPT_DIR, "checkpoint_*.pt"))
    for ckpt_file in checkpoint_files:
        try:
            model, epoch = load_checkpoint(ckpt_file)
            checkpoint_pairs.append((model.encoder, model.decoder, epoch, os.path.basename(ckpt_file)))
        except Exception as e:
            print(f"Error loading {ckpt_file}: {e}")

    print(f"Loaded {len(checkpoint_pairs)} checkpoints")

# ========== Validation Loop ===========
def validate_model(encoder, decoder, structures):
    results = []
    with torch.no_grad():
        for coords, mask, name in structures:
            coords = coords.unsqueeze(0).to(DEVICE)  # Add batch dimension
            mask = mask.unsqueeze(0).to(DEVICE)
            
            # Forward pass through encoder and decoder
            z_q, _ = encoder(coords)
            x_rec = decoder(z_q)
            
            # Get flattened and masked points
            pts_pred = mask_and_flatten(x_rec, mask)
            pts_true = mask_and_flatten(coords, mask)
            
            # Calculate losses
            kabsch_loss = kabsch_rmsd_loss(pts_pred, pts_true).item()
            lddt_loss = soft_lddt_loss(pts_pred, pts_true).item()
            
            results.append({
                'name': name,
                'kabsch_rmsd': kabsch_loss,
                'lddt_loss': lddt_loss
            })
    return results

# Validate all checkpoints
all_results = {}
for encoder, decoder, epoch, ckpt_name in checkpoint_pairs:
    print(f"\nValidating encoder-decoder pair {ckpt_name} (epoch {epoch})...")
    results = validate_model(encoder, decoder, casp_structures)
    all_results[ckpt_name] = results

# ========== Generate Summary and Save to File ===========
with open(os.path.join(OUTPUT_DIR, 'casp_validation_results.txt'), 'w') as f:
    f.write("CASP Validation Results\n")
    f.write("=====================\n\n")
    
    # Write averages first
    f.write("Averages Summary:\n")
    f.write("----------------\n")
    for ckpt_name, results in all_results.items():
        avg_kabsch = np.mean([r['kabsch_rmsd'] for r in results])
        avg_lddt = np.mean([r['lddt_loss'] for r in results])
        f.write(f"\nCheckpoint: {ckpt_name}\n")
        f.write(f"  Mean Kabsch RMSD: {avg_kabsch:.4e}\n")
        f.write(f"  Mean LDDT Loss: {avg_lddt:.4e}\n")
    
    # Write detailed results
    f.write("\n\nDetailed Results:\n")
    f.write("----------------\n")
    for ckpt_name, results in all_results.items():
        f.write(f"\nCheckpoint: {ckpt_name}\n")
        f.write("Individual structure results:\n")
        for r in results:
            f.write(f"  {r['name']}:\n")
            f.write(f"    Kabsch RMSD: {r['kabsch_rmsd']:.4e}\n")
            f.write(f"    LDDT Loss: {r['lddt_loss']:.4e}\n")

# ========== Generate Bar Chart ===========
plt.figure(figsize=(12, 6))
x = np.arange(len(casp_structures))
width = 0.35

# Plot for each checkpoint
for i, (ckpt_name, results) in enumerate(all_results.items()):
    kabsch_values = [r['kabsch_rmsd'] for r in results]
    lddt_values = [r['lddt_loss'] for r in results]
    
    plt.bar(x + i*width, kabsch_values, width, label=f'{ckpt_name} - Kabsch')
    plt.bar(x + i*width + width/2, lddt_values, width, label=f'{ckpt_name} - LDDT')

plt.xlabel('CASP Structures')
plt.ylabel('Loss Value')
plt.title('CASP Validation Results')
plt.xticks(x + width/2, [r['name'] for r in results], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'casp_validation.png'), dpi=300)
plt.close()

# ========== Generate Reconstruction Visualizations ===========
# Select 10 random structures
sample_structures = random.sample(casp_structures, min(10, len(casp_structures)))

# Create visualization for each checkpoint
for encoder, decoder, epoch, ckpt_name in checkpoint_pairs:
    fig = plt.figure(figsize=(5*5, 2*5))
    for i, (coords, mask, name) in enumerate(sample_structures):
        ax = fig.add_subplot(2, 5, i+1, projection='3d')
        
        # Get reconstruction
        coords_i = coords.unsqueeze(0).to(DEVICE)
        mask_i = mask.unsqueeze(0).to(DEVICE)
        
        # Forward pass through encoder and decoder
        z_q, _ = encoder(coords_i)
        x_rec_i = decoder(z_q)
        
        # Get flattened and masked points
        pts_true = mask_and_flatten(coords_i, mask_i)[0]
        pts_pred = mask_and_flatten(x_rec_i, mask_i)[0]
        
        # Align reconstruction
        rec_align = _kabsch_align(pts_pred, pts_true)
        
        # Plot
        op = pts_true.squeeze(0).cpu().numpy()
        rp = rec_align.squeeze(0).cpu().detach().numpy()
        ax.scatter(op[:,0], op[:,1], op[:,2], c='blue', s=2, label='Original')
        ax.scatter(rp[:,0], rp[:,1], rp[:,2], c='red', s=2, label='Reconstructed')
        
        # Calculate losses
        kabsch = kabsch_rmsd_loss(pts_pred, pts_true).item()
        lddt = soft_lddt_loss(pts_pred, pts_true).item()
        
        ax.set_title(f"{name}\nKabsch: {kabsch:.2e}\nLDDT: {lddt:.2e}", fontsize=6)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'casp_reconstruction_{ckpt_name}.png'), dpi=300)
    plt.close()

print(f"\nAll outputs have been saved to the {OUTPUT_DIR} directory:")
print(f"- Validation results: {os.path.join(OUTPUT_DIR, 'casp_validation_results.txt')}")
print(f"- Bar chart: {os.path.join(OUTPUT_DIR, 'casp_validation.png')}")
print(f"- Reconstruction visualizations: {os.path.join(OUTPUT_DIR, 'casp_reconstruction_*.png')}")
