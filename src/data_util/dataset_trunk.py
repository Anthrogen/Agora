import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
from enum import Enum

class Chirality(Enum):
    D = 1   # Dextrorotatory amino acid
    A = 0   # Achiral amino acid
    L = -1  # Levorotatory amino acid

ATOM_N_ENC = 0
ATOM_CA_ENC = 1
ATOM_C_ENC = 2
ATOM_CB_ENC = 3

BACKBONE = ["N", "CA", "C", "CB"]

class JSONWrapper():
    """
    Loads protein structure from JSON file with backbone atoms only.
    Always includes N, CA, C, and CB (virtual for glycine).
    """
    def __init__(self, file_path):
        assert os.path.exists(file_path)
        data = json.load(open(file_path))

        # This dataloader assumes that each amino acid begins with an "N".
        atom_names = data["all_atoms"]["atom_names"]
        sequence = data["sequence"]
        atom_coords = torch.Tensor(data["all_atoms"]["atom_coordinates"])

        # Validate that backbone coordinates (N, CA, C) are not null
        backbone_coords = data["backbone_coordinates"]
        for i, (n_coord, ca_coord, c_coord) in enumerate(zip(backbone_coords["N"], backbone_coords["CA"], backbone_coords["C"])):
            if n_coord is None or ca_coord is None or c_coord is None:
                raise ValueError(f"Null backbone coordinates found at residue {i}")

        # We always use 4 cells for backbone atoms (N, CA, C, CB)
        all_coords = torch.zeros(len(sequence), len(BACKBONE), 3)
        
        # Get indices of all "N" positions
        n_indices = [i for i, name in enumerate(atom_names) if name == "N"] + [len(atom_names)]

        # Iterate over all residues and record their atomic coordinates
        for residue_idx in range(len(n_indices) - 1):
            start_idx, end_idx = n_indices[residue_idx], n_indices[residue_idx + 1]

            filled = torch.zeros(len(BACKBONE))
            if sequence[residue_idx] == "G":
                filled[ATOM_CB_ENC] = 1  # We'll fill CB for glycine later

            for atom_idx in range(start_idx, end_idx):
                atom_name = atom_names[atom_idx]
                
                # Skip OXT and hydrogen atoms
                if atom_name == "OXT" or atom_name.startswith("H"):
                    continue
                
                if atom_name in BACKBONE:
                    position = BACKBONE.index(atom_name)
                    all_coords[residue_idx, position, :] = atom_coords[atom_idx, :]
                    filled[position] += 1

            # Check to make sure we have one (and only one) of each atom
            assert torch.allclose(filled, torch.ones_like(filled)), f"Residue {residue_idx} has missing or extra atoms."

        # Compute virtual CB for glycine residues
        gly_indices = [i for i, aa in enumerate(sequence) if aa == "G"]
        for gly_idx in gly_indices:
            gly_n = all_coords[gly_idx, ATOM_N_ENC]
            gly_ca = all_coords[gly_idx, ATOM_CA_ENC]
            gly_c = all_coords[gly_idx, ATOM_C_ENC]

            vec_b = gly_ca - gly_n
            vec_c = gly_c - gly_ca
            vec_a = torch.linalg.cross(vec_b, vec_c)
            virtual_cb = -0.58273431 * vec_a + 0.56802827 * vec_b - 0.54067466 * vec_c + gly_ca
            all_coords[gly_idx, ATOM_CB_ENC, :] = virtual_cb

        # Final checks
        assert not torch.isnan(all_coords).any(), "NaNs found in backbone coordinates"

        self.coords = all_coords  # Shape: [L, 4, 3]
        self.seq = sequence
        self.len = len(self.seq)

class ProteinBackboneDataset(Dataset):
    """Dataset for protein backbone structures with N, CA, C, CB atoms."""
    
    def __init__(self, root_dir: str, center: bool = True, max_length: int = 2048):
        self.root_dir = root_dir
        self.max_length = max_length
        
        # Gather and validate JSONs
        all_files = sorted(fn for fn in os.listdir(root_dir) if fn.lower().endswith(".json"))
        valid_paths = []
        
        for fn in all_files:
            path = os.path.join(root_dir, fn)
            try:
                protein = JSONWrapper(path)
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: skipping malformed file {fn}: {e}")
                      
        assert len(valid_paths) > 0, f"No valid JSON files found in {root_dir}!"
            
        self.file_paths = valid_paths
        self.center = center

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        protein = JSONWrapper(path)
        
        coords = protein.coords[:self.max_length]  # [L, 4, 3]
        seq = protein.seq[:self.max_length]
        length = torch.tensor(min(protein.len, self.max_length))

        # Optionally center the structure
        if self.center:
            centroid = coords.reshape(-1, 3).mean(dim=0)
            coords = coords - centroid

        return seq, coords, length
