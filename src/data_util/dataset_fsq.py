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
ATOM_O_ENC = 4

ENCODING_LEN = 14
BACKBONE = ["N", "CA", "C", "CB"]
CORE_BACKBONE = BACKBONE[:-1]

ATOMS = {'V': BACKBONE + ['O', 'CG1', 'CG2'], 
         'R': BACKBONE + ['O', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'], 
         'C': BACKBONE + ['O', 'SG'], 
         'S': BACKBONE + ['O', 'OG'], 
         'L': BACKBONE + ['O', 'CG', 'CD1', 'CD2'], 
         'Q': BACKBONE + ['O', 'CG', 'CD', 'OE1', 'NE2'], 
         'E': BACKBONE + ['O', 'CG', 'CD', 'OE1', 'OE2'], 
         'T': BACKBONE + ['O', 'OG1', 'CG2'], 
         'A': BACKBONE + ['O'], 
         'D': BACKBONE + ['O', 'CG', 'OD1', 'OD2'], 
         'G': BACKBONE + ['O'], # CB is the phantom side-chain 
         'H': BACKBONE + ['O', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'], 
         'I': BACKBONE + ['O', 'CG1', 'CG2', 'CD1'], 
         'P': BACKBONE + ['O', 'CG', 'CD'], 
         'F': BACKBONE + ['O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 
         'K': BACKBONE + ['O', 'CG', 'CD', 'CE', 'NZ'], 
         'N': BACKBONE + ['O', 'CG', 'OD1', 'ND2'], 
         'M': BACKBONE + ['O', 'CG', 'SD', 'CE'], 
         'Y': BACKBONE + ['O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'], 
         'W': BACKBONE + ['O', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']}



for k in ATOMS:
    assert ATOMS[k][ATOM_N_ENC] == "N"
    assert ATOMS[k][ATOM_CA_ENC] == "CA"
    assert ATOMS[k][ATOM_C_ENC] == "C"
    assert ATOMS[k][ATOM_O_ENC] == "O"
    assert len(ATOMS[k]) <= ENCODING_LEN

class JSONWrapper():
    """
    To test this code, try it with:

    JSONWrapper("/workspace/demo/transformer_stack/data/sample_training_data/7x99_B.json", True)

    Modes: side_chain, backbone
    """
    def __init__(self, file_path, mode="side_chain"):

        assert mode in ["side_chain", "backbone"]
        self.mode = mode

        assert os.path.exists(file_path)
        data = json.load(open(file_path))

        # This dataloader assumes that each amino acid begins with an "N".
        atom_names = data["all_atoms"]["atom_names"]
        sequence = data["sequence"]
        atom_coords = torch.Tensor(data["all_atoms"]["atom_coordinates"])

        # Validate that backbone coordinates (N, CA, C) are not null - if any are, this is a malformed file
        backbone_coords = data["backbone_coordinates"]
        for i, (n_coord, ca_coord, c_coord) in enumerate(zip(backbone_coords["N"], backbone_coords["CA"], backbone_coords["C"])):
            if n_coord is None or ca_coord is None or c_coord is None:
                raise ValueError(f"Null backbone coordinates found at residue {i}")

        # If self.side_chain is True, then we reserve 14 cells for atoms.
        # Otherwise, we reserve 4 cells for the atoms.
        # We store this number in enc_len.
        enc_len = ENCODING_LEN if self.mode == "side_chain" else len(BACKBONE)
        all_coords = torch.zeros(len(sequence), enc_len, 3)
        
        # We require that the first atom in the JSON file for each residue is an "N".
        # Get indices of all "N" positions:
        n_indices = [i for i, name in enumerate(atom_names) if name == "N"] + [len(atom_names)] # a "phantom" N at the end for ease of processing

        ##########################################################
        # Iterate over all residues and record their atomic coordinates:
        for residue_idx in range(len(n_indices) - 1): # -1 because of the phantom N
            start_idx, end_idx = n_indices[residue_idx], n_indices[residue_idx + 1]

            filled = torch.zeros(len(BACKBONE) if not self.mode == "side_chain" else len(ATOMS[sequence[residue_idx]]))
            if sequence[residue_idx] == "G":
                filled[ATOM_CB_ENC] = 1 # We'll fill it in later.

            for atom_idx in range(start_idx, end_idx):
                atom_name = atom_names[atom_idx]
                
                # Skip OXT and hydrogen atoms - these are terminal/additional atoms not part of the residue
                if atom_name == "OXT" or atom_name.startswith("H"): continue
                
                if self.mode == "side_chain" or ((not self.mode == "side_chain") and (atom_name in BACKBONE)):
                    position = ATOMS[sequence[residue_idx]].index(atom_name)
                    all_coords[residue_idx, position, :] = atom_coords[atom_idx, :]
                    filled[position] += 1

            # Check to make sure we have one (and only one) of each atom.
            assert torch.allclose(filled, torch.ones_like(filled)), f"Residue {residue_idx} has missing or extra atoms."
        

        ##########################################################
        # Glycine Phantom Coords:
        gly_indices = [i for i, name in enumerate(sequence) if name == "G"]
        for gly_idx in gly_indices:
            gly_n = all_coords[gly_idx, ATOMS["G"].index("N")]
            gly_ca = all_coords[gly_idx, ATOMS["G"].index("CA")]
            gly_c = all_coords[gly_idx, ATOMS["G"].index("C")]

            vec_b = gly_ca - gly_n
            vec_c = gly_c - gly_ca
            vec_a = torch.linalg.cross(vec_b, vec_c)
            virtual_cb = -0.58273431 * vec_a + 0.56802827 * vec_b - 0.54067466 * vec_c + gly_ca
            all_coords[gly_idx, ATOMS["G"].index("CB"), :] = virtual_cb

        ##########################################################
        # Final Checks and Outputs:
        assert not torch.isnan(all_coords).any(), "NaNs found in Protein Coordinates"

        self.coords = all_coords
        self.seq = sequence
        self.len = len(self.seq)

    def bond_angles(self):
        """
        Calculate all bond angles and return a tuple, all pivoting about the alpha carbon, excluding hydrogens.

        Returns: (Cb - N, Cb - C, N - C) for each residue.
        """

        assert self.mode != "backbone", "Bond angles are not available for backbone mode."
        # Extract coordinates for N, CA, C, and CB atoms
        coords_N = self.coords[:, ATOM_N_ENC, :]   # (L, 3)
        coords_CA = self.coords[:, ATOM_CA_ENC, :]  # (L, 3)
        coords_C = self.coords[:, ATOM_C_ENC, :]   # (L, 3)
        coords_CB = self.coords[:, ATOM_CB_ENC, :]  # (L, 3)

        # Calculate vectors from CA to each atom
        vec_CA_to_N = coords_N - coords_CA   # (L, 3)
        vec_CA_to_C = coords_C - coords_CA   # (L, 3)
        vec_CA_to_CB = coords_CB - coords_CA # (L, 3)
        
        # Calculate angles using dot product formula: cos(theta) = (u·v) / (|u||v|)
        # Angle between CB and N (pivoting at CA)
        dot_CB_N = torch.sum(vec_CA_to_CB * vec_CA_to_N, dim=-1)  # (L,)
        norm_CB = torch.norm(vec_CA_to_CB, dim=-1)  # (L,)
        norm_N = torch.norm(vec_CA_to_N, dim=-1)    # (L,)
        cos_CB_N = dot_CB_N / (norm_CB * norm_N + 1e-8)  # Add small epsilon to avoid division by zero
        angle_CB_N = torch.acos(torch.clamp(cos_CB_N, -1.0, 1.0))  # Clamp to avoid numerical errors
        
        # Angle between CB and C (pivoting at CA)
        dot_CB_C = torch.sum(vec_CA_to_CB * vec_CA_to_C, dim=-1)  # (L,)
        norm_C = torch.norm(vec_CA_to_C, dim=-1)    # (L,)
        cos_CB_C = dot_CB_C / (norm_CB * norm_C + 1e-8)
        angle_CB_C = torch.acos(torch.clamp(cos_CB_C, -1.0, 1.0))
        
        # Angle between N and C (pivoting at CA)
        dot_N_C = torch.sum(vec_CA_to_N * vec_CA_to_C, dim=-1)  # (L,)
        cos_N_C = dot_N_C / (norm_N * norm_C + 1e-8)
        angle_N_C = torch.acos(torch.clamp(cos_N_C, -1.0, 1.0))
        
        # Convert from radians to degrees
        angle_CB_N_deg = torch.rad2deg(angle_CB_N)
        angle_CB_C_deg = torch.rad2deg(angle_CB_C)
        angle_N_C_deg = torch.rad2deg(angle_N_C)
        
        return torch.stack([angle_CB_N_deg, angle_CB_C_deg, angle_N_C_deg], dim=1) # New shape is (L, 3)

    def chirality(self):
        """
        Returns and L-dimensional vector of chirality for each residue.
        """

        assert self.mode != "backbone", "Chirality is not available for backbone mode."
        
        # Extract coordinates for N, CA, C, and CB atoms
        coords_N = self.coords[:, ATOM_N_ENC, :]   # (L, 3)
        coords_CA = self.coords[:, ATOM_CA_ENC, :]  # (L, 3)
        coords_C = self.coords[:, ATOM_C_ENC, :]   # (L, 3)
        coords_CB = self.coords[:, ATOM_CB_ENC, :]  # (L, 3)
        
        # Calculate vectors from CA to other atoms
        vec_CA_to_N = coords_N - coords_CA   # (L, 3)
        vec_CA_to_C = coords_C - coords_CA   # (L, 3)
        vec_CA_to_CB = coords_CB - coords_CA # (L, 3)
        
        # Calculate the cross product of N-CA and C-CA vectors
        # This gives us a vector perpendicular to the N-CA-C plane
        cross_product = torch.cross(vec_CA_to_N, vec_CA_to_C, dim=-1)  # (L, 3)
        
        # Calculate the dot product of this perpendicular vector with the CB-CA vector
        # The sign of this dot product determines the chirality
        dot_product = torch.sum(cross_product * vec_CA_to_CB, dim=-1)  # (L,)
        
        # Initialize chirality vector
        chirality_vector = torch.zeros(self.len, dtype=torch.int)
        
        # Determine chirality based on the sign of the dot product
        # Positive dot product indicates L-chirality (most natural amino acids)
        # Negative dot product indicates D-chirality
        
        # Assign chirality values
        chirality_vector[dot_product > 0] = Chirality.L.value  # -1
        chirality_vector[dot_product < -0] = Chirality.D.value  # 1
        chirality_vector[self.seq == "G"] = Chirality.A.value # Glycine is achiral -- overwrite the dot product chirality assignment.
        
        return chirality_vector

class ProteinDataset(Dataset):
    def __init__(self, root_dir: str, center: bool = True, mode: str = "backbone", max_length: int = 2048):
        self.root_dir = root_dir

        # Maximum sequence length for padding/truncation
        # Proteins longer than this will be truncated
        self.max_length = max_length
        self.mode = mode
        
        # Gather and validate JSONs, discarding any that fail sanity checks
        # This ensures all data loaded during training will be valid
        all_files = sorted(fn for fn in os.listdir(root_dir) if fn.lower().endswith(".json"))
        valid_paths = []
        
        for fn in all_files:
            path = os.path.join(root_dir, fn)
            try:
                protein = JSONWrapper(path, mode=self.mode)
                valid_paths.append(path)
   
            except Exception as e:
                print(f"Warning: skipping malformed file {fn}: {e}")
                      
        assert len(valid_paths) > 0, f"No valid JSON Paths found in {root_dir}!"
            
        self.file_paths = valid_paths
        self.center = center

    def __len__(self):
        """Return the total number of protein structures in the dataset"""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:

        # 1) Load protein structure from JSON file
        path = self.file_paths[idx]

        protein = JSONWrapper(path, mode = self.mode)
        coords = protein.coords[:self.max_length]
        seq = protein.seq[:self.max_length]
        l = torch.tensor(min(protein.len, self.max_length))

        # 7) Optionally center the structure at origin
        # This is useful for translation-invariant applications
        if self.center:
            centroid = coords.reshape(-1, 3).mean(dim=0)  # Average position of all atoms
            coords = coords - centroid  # Subtract mean from all coordinates

        return seq, coords, l
