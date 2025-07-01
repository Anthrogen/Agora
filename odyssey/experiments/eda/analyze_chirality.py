import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import JSONWrapper
from tqdm import tqdm

_GS_EPS = 1e-8  # numerical safety for normalization

def analyze_residue_chirality(coords):
    """
    Analyze chirality for each residue in a protein using the same logic as ReflexiveAttention.
    
    Parameters:
    coords: np.array of shape (L, 4, 3) with atoms ordered as (N, CA, C, CB)
    
    Returns:
    chirality_signs: np.array of shape (L,) with values:
        > 0 for L-amino acids
        < 0 for D-amino acids
    """
    coords = torch.from_numpy(coords).float()
    
    n = coords[..., 0, :]   # N
    ca = coords[..., 1, :]  # CA 
    c = coords[..., 2, :]   # C
    cb = coords[..., 3, :]  # CB
    
    # Vectors relative to CA
    Vn = n - ca   # N - CA
    Vc = c - ca   # C - CA  
    Vb = cb - ca  # CB - CA
    
    # x-bar = normalized Vn
    x_bar = F.normalize(Vn, dim=-1, eps=_GS_EPS)
    
    # y = Vc - projection of Vc onto x-bar
    proj_Vc_on_x = (Vc * x_bar).sum(-1, keepdim=True) * x_bar
    y = Vc - proj_Vc_on_x
    y_bar = F.normalize(y, dim=-1, eps=_GS_EPS)
    
    # w = x-bar × y-bar (normal to the N-CA-C plane)
    w = torch.cross(x_bar, y_bar, dim=-1)
    w_bar = F.normalize(w, dim=-1, eps=_GS_EPS)
    
    # Chirality is determined by the sign of (Vb · w_bar)
    # Positive for L-amino acids, negative for D-amino acids
    chirality_sign = (Vb * w_bar).sum(-1)
    
    return chirality_sign.numpy()

def main():
    data_dir = "data/sample_training_data"
    
    # Get all JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    total_residues = 0
    l_amino_acids = 0
    d_amino_acids = 0
    proteins_with_d = []
    
    print(f"Analyzing chirality in {len(json_files)} protein files...")
    
    for json_file in tqdm(json_files):
        filepath = os.path.join(data_dir, json_file)
        
        try:
            # Load protein with CB coordinates
            protein = JSONWrapper(filepath, use_cbeta=True)
            coords = protein.coords.numpy()  # (L, 4, 3)
            
            # Analyze chirality for each residue
            chirality_signs = analyze_residue_chirality(coords)
            
            # Count L and D amino acids
            n_l = np.sum(chirality_signs > 0)
            n_d = np.sum(chirality_signs < 0)
            n_zero = np.sum(chirality_signs == 0)
            
            total_residues += len(chirality_signs)
            l_amino_acids += n_l
            d_amino_acids += n_d
            
            # Track proteins with D-amino acids
            if n_d > 0:
                proteins_with_d.append({
                    'file': json_file,
                    'n_residues': len(chirality_signs),
                    'n_l': n_l,
                    'n_d': n_d,
                    'n_zero': n_zero,
                    'd_positions': np.where(chirality_signs < 0)[0].tolist()
                })
                
        except Exception as e:
            print(f"\nError processing {json_file}: {e}")
            continue
    
    # Print results
    print("\n" + "="*60)
    print("CHIRALITY ANALYSIS RESULTS")
    print("="*60)
    print(f"Total proteins analyzed: {len(json_files)}")
    print(f"Total residues: {total_residues}")
    print(f"L-amino acids: {l_amino_acids} ({100*l_amino_acids/total_residues:.2f}%)")
    print(f"D-amino acids: {d_amino_acids} ({100*d_amino_acids/total_residues:.2f}%)")
    print(f"Proteins containing D-amino acids: {len(proteins_with_d)}")
    
    if proteins_with_d:
        print("\n" + "-"*60)
        print("Proteins with D-amino acids:")
        print("-"*60)
        for protein in proteins_with_d[:10]:  # Show first 10
            print(f"\n{protein['file']}:")
            print(f"  Total residues: {protein['n_residues']}")
            print(f"  L-amino acids: {protein['n_l']}")
            print(f"  D-amino acids: {protein['n_d']}")
            print(f"  D-amino acid positions: {protein['d_positions'][:10]}{'...' if len(protein['d_positions']) > 10 else ''}")
        
        if len(proteins_with_d) > 10:
            print(f"\n... and {len(proteins_with_d) - 10} more proteins with D-amino acids")
    
    # Additional statistics
    print("\n" + "-"*60)
    print("Additional Statistics:")
    print("-"*60)
    if proteins_with_d:
        d_percentages = [(p['n_d'] / p['n_residues'] * 100) for p in proteins_with_d]
        print(f"Average D-amino acid percentage in affected proteins: {np.mean(d_percentages):.2f}%")
        print(f"Max D-amino acid percentage: {np.max(d_percentages):.2f}%")

if __name__ == "__main__":
    main() 