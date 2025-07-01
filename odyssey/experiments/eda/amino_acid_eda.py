import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import ProteinBackboneDataset, JSONWrapper, Chirality
from src.vocabulary import SEQUENCE_TOKENS
from matplotlib import pyplot as plt
import torch
import numpy as np
import json

def calculate_dihedral_angle(p1, p2, p3, p4):
    """
    Calculate dihedral angle between four points.
    Returns angle in degrees.
    """
    # Convert to numpy if needed
    if isinstance(p1, torch.Tensor):
        p1, p2, p3, p4 = p1.numpy(), p2.numpy(), p3.numpy(), p4.numpy()
    
    # Calculate vectors
    v1 = p2 - p1  # N -> CA
    v2 = p3 - p2  # CA -> CB  
    v3 = p4 - p3  # CB -> CG
    
    # Calculate normal vectors to the two planes
    n1 = np.cross(v1, v2)  # Normal to plane 1 (N-CA-CB)
    n2 = np.cross(v2, v3)  # Normal to plane 2 (CA-CB-CG)
    
    # Normalize normal vectors
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return np.nan
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Calculate angle between normal vectors
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Determine sign of angle using triple product
    if np.dot(np.cross(n1, n2), v2) < 0:
        angle = -angle
    
    return np.degrees(angle)

def extract_cg_coordinates(data):
    """
    Extract gamma atom coordinates from all_atoms data for residues that have them.
    Different amino acids use different gamma atom names:
    - Most: CG
    - Ile, Val: CG1 (primary), CG2 (secondary) 
    - Thr: OG1 (primary), CG2 (secondary)
    - Ser: OG
    - Cys: SG
    Returns array of gamma coordinates and mask indicating which residues have gamma atoms.
    """
    try:
        atom_names = data["all_atoms"]["atom_names"] 
        atom_coords = np.asarray(data["all_atoms"]["atom_coordinates"], dtype=np.float32)
        
        # Find residue boundaries - each residue starts with 'N'
        residue_starts = []
        for i, atom_name in enumerate(atom_names):
            if atom_name == 'N':
                residue_starts.append(i)
        
        # Add sentinel for the end
        residue_starts.append(len(atom_names))
        
        # Extract gamma atoms for each residue
        coords_gamma = []
        has_gamma_mask = []
        
        # Priority order for gamma atoms (prefer primary gamma atoms)
        gamma_atom_priority = ['CG', 'CG1', 'OG1', 'OG', 'SG', 'CG2']
        
        for res_idx in range(len(residue_starts) - 1):
            start_idx = residue_starts[res_idx]
            end_idx = residue_starts[res_idx + 1]
            
            # Look for gamma atoms in priority order
            gamma_found = False
            for gamma_name in gamma_atom_priority:
                for i in range(start_idx, end_idx):
                    if atom_names[i] == gamma_name:
                        coords_gamma.append(atom_coords[i])
                        gamma_found = True
                        break
                if gamma_found:
                    break
            
            if not gamma_found:
                coords_gamma.append([np.nan, np.nan, np.nan])  # Placeholder for no gamma
            
            has_gamma_mask.append(gamma_found)
        
        return np.array(coords_gamma, dtype=np.float32), np.array(has_gamma_mask)
    
    except Exception as e:
        print(f"Warning: Could not extract gamma coordinates: {e}")
        return None, None

def extract_carbonyl_oxygen_coordinates(data):
    """
    Extract carbonyl oxygen (O) coordinates from all_atoms data.
    Returns array of O coordinates matching the sequence length.
    """
    try:
        atom_names = data["all_atoms"]["atom_names"] 
        atom_coords = np.asarray(data["all_atoms"]["atom_coordinates"], dtype=np.float32)
        
        # Find residue boundaries - each residue starts with 'N'
        residue_starts = []
        for i, atom_name in enumerate(atom_names):
            if atom_name == 'N':
                residue_starts.append(i)
        
        # Add sentinel for the end
        residue_starts.append(len(atom_names))
        
        # Extract O for each residue
        coords_O = []
        has_o_mask = []
        
        for res_idx in range(len(residue_starts) - 1):
            start_idx = residue_starts[res_idx]
            end_idx = residue_starts[res_idx + 1]
            
            # Look for O in this residue (carbonyl oxygen)
            o_found = False
            for i in range(start_idx, end_idx):
                if atom_names[i] == 'O':
                    coords_O.append(atom_coords[i])
                    o_found = True
                    break
            
            if not o_found:
                coords_O.append([np.nan, np.nan, np.nan])  # Placeholder for no O
            
            has_o_mask.append(o_found)
        
        return np.array(coords_O, dtype=np.float32), np.array(has_o_mask)
    
    except Exception as e:
        print(f"Warning: Could not extract O coordinates: {e}")
        return None, None

def extract_cd_coordinates(data):
    """
    Extract delta atom coordinates from all_atoms data for residues that have them.
    Different amino acids use different delta atom names:
    - Most: CD
    - Ile: CD1 (primary)
    - Leu: CD1, CD2 (prefer CD1)
    - Phe, Tyr, Trp: CD1, CD2 (prefer CD1)
    - His: CD2, ND1 (prefer CD2)
    - Met: SD (sulfur delta)
    - Pro: CD
    Returns array of delta coordinates and mask indicating which residues have delta atoms.
    """
    try:
        atom_names = data["all_atoms"]["atom_names"] 
        atom_coords = np.asarray(data["all_atoms"]["atom_coordinates"], dtype=np.float32)
        
        # Find residue boundaries - each residue starts with 'N'
        residue_starts = []
        for i, atom_name in enumerate(atom_names):
            if atom_name == 'N':
                residue_starts.append(i)
        
        # Add sentinel for the end
        residue_starts.append(len(atom_names))
        
        # Extract delta atoms for each residue
        coords_delta = []
        has_delta_mask = []
        
        # Priority order for delta atoms
        delta_atom_priority = ['CD', 'CD1', 'CD2', 'SD', 'ND1']
        
        for res_idx in range(len(residue_starts) - 1):
            start_idx = residue_starts[res_idx]
            end_idx = residue_starts[res_idx + 1]
            
            # Look for delta atoms in priority order
            delta_found = False
            for delta_name in delta_atom_priority:
                for i in range(start_idx, end_idx):
                    if atom_names[i] == delta_name:
                        coords_delta.append(atom_coords[i])
                        delta_found = True
                        break
                if delta_found:
                    break
            
            if not delta_found:
                coords_delta.append([np.nan, np.nan, np.nan])  # Placeholder for no delta
            
            has_delta_mask.append(delta_found)
        
        return np.array(coords_delta, dtype=np.float32), np.array(has_delta_mask)
    
    except Exception as e:
        print(f"Warning: Could not extract delta coordinates: {e}")
        return None, None

def extract_ce_coordinates(data):
    """
    Extract epsilon atom coordinates from all_atoms data for residues that have them.
    Different amino acids use different epsilon atom names:
    - Arg: NE (epsilon nitrogen)
    - Lys: CE
    - Met: CE
    - Glu: OE1, OE2 (prefer OE1)
    - Gln: OE1, NE2 (prefer OE1)
    Returns array of epsilon coordinates and mask indicating which residues have epsilon atoms.
    """
    try:
        atom_names = data["all_atoms"]["atom_names"] 
        atom_coords = np.asarray(data["all_atoms"]["atom_coordinates"], dtype=np.float32)
        
        # Find residue boundaries - each residue starts with 'N'
        residue_starts = []
        for i, atom_name in enumerate(atom_names):
            if atom_name == 'N':
                residue_starts.append(i)
        
        # Add sentinel for the end
        residue_starts.append(len(atom_names))
        
        # Extract epsilon atoms for each residue
        coords_epsilon = []
        has_epsilon_mask = []
        
        # Priority order for epsilon atoms
        epsilon_atom_priority = ['CE', 'NE', 'OE1', 'OE2', 'NE2']
        
        for res_idx in range(len(residue_starts) - 1):
            start_idx = residue_starts[res_idx]
            end_idx = residue_starts[res_idx + 1]
            
            # Look for epsilon atoms in priority order
            epsilon_found = False
            for epsilon_name in epsilon_atom_priority:
                for i in range(start_idx, end_idx):
                    if atom_names[i] == epsilon_name:
                        coords_epsilon.append(atom_coords[i])
                        epsilon_found = True
                        break
                if epsilon_found:
                    break
            
            if not epsilon_found:
                coords_epsilon.append([np.nan, np.nan, np.nan])  # Placeholder for no epsilon
            
            has_epsilon_mask.append(epsilon_found)
        
        return np.array(coords_epsilon, dtype=np.float32), np.array(has_epsilon_mask)
    
    except Exception as e:
        print(f"Warning: Could not extract epsilon coordinates: {e}")
        return None, None

def extract_terminal_coordinates(data):
    """
    Extract terminal atom coordinates for Chi4 analysis.
    Different amino acids use different terminal atom names:
    - Arg: CZ (zeta carbon in guanidinium)
    - Lys: NZ (zeta nitrogen)
    - Met: NZ would not exist, use CE as terminal
    Returns array of terminal coordinates and mask indicating which residues have terminal atoms.
    """
    try:
        atom_names = data["all_atoms"]["atom_names"] 
        atom_coords = np.asarray(data["all_atoms"]["atom_coordinates"], dtype=np.float32)
        
        # Find residue boundaries - each residue starts with 'N'
        residue_starts = []
        for i, atom_name in enumerate(atom_names):
            if atom_name == 'N':
                residue_starts.append(i)
        
        # Add sentinel for the end
        residue_starts.append(len(atom_names))
        
        # Extract terminal atoms for each residue
        coords_terminal = []
        has_terminal_mask = []
        
        # Priority order for terminal atoms (Chi4 needs these)
        terminal_atom_priority = ['NZ', 'CZ', 'NH1', 'NH2']
        
        for res_idx in range(len(residue_starts) - 1):
            start_idx = residue_starts[res_idx]
            end_idx = residue_starts[res_idx + 1]
            
            # Look for terminal atoms in priority order
            terminal_found = False
            for terminal_name in terminal_atom_priority:
                for i in range(start_idx, end_idx):
                    if atom_names[i] == terminal_name:
                        coords_terminal.append(atom_coords[i])
                        terminal_found = True
                        break
                if terminal_found:
                    break
            
            if not terminal_found:
                coords_terminal.append([np.nan, np.nan, np.nan])  # Placeholder for no terminal
            
            has_terminal_mask.append(terminal_found)
        
        return np.array(coords_terminal, dtype=np.float32), np.array(has_terminal_mask)
    
    except Exception as e:
        print(f"Warning: Could not extract terminal coordinates: {e}")
        return None, None

class ExtendedJSONWrapper(JSONWrapper):
    """Extended version that extracts atom coordinates for Chi1, Chi2, Chi3, and Chi4 analysis."""
    
    def __init__(self, fqfp, use_cbeta: bool = False):
        super().__init__(fqfp, use_cbeta)
        
        # Load all sidechain atom coordinates
        data = json.load(open(fqfp))
        self.coords_gamma, self.has_gamma_mask = extract_cg_coordinates(data)
        self.coords_delta, self.has_delta_mask = extract_cd_coordinates(data)
        self.coords_epsilon, self.has_epsilon_mask = extract_ce_coordinates(data)
        self.coords_terminal, self.has_terminal_mask = extract_terminal_coordinates(data)
        self.coords_O, self.has_o_mask = extract_carbonyl_oxygen_coordinates(data)
    
    def chi1_angles(self):
        """
        Calculate Chi1 dihedral angles (N-CA-CB-gamma) for residues that have gamma atoms.
        Gamma atoms can be: CG, CG1, OG1, OG, SG, or CG2 depending on amino acid type.
        Returns Chi1 angles in degrees and mask indicating which residues have valid Chi1.
        """
        if self.coords_gamma is None or not self.use_cbeta:
            return None, None
        
        # Extract coordinates
        coords_N = self.coords[:, 0, :].numpy()   # (L, 3)
        coords_CA = self.coords[:, 1, :].numpy()  # (L, 3) 
        coords_CB = self.coords[:, 3, :].numpy()  # (L, 3)
        coords_gamma = self.coords_gamma          # (L, 3)
        
        chi1_angles = []
        valid_mask = []
        
        for i in range(len(self.seq)):
            if self.has_gamma_mask[i] and not np.any(np.isnan(coords_gamma[i])):
                # Calculate Chi1 angle for this residue
                angle = calculate_dihedral_angle(
                    coords_N[i], coords_CA[i], coords_CB[i], coords_gamma[i]
                )
                chi1_angles.append(angle)
                valid_mask.append(not np.isnan(angle))
            else:
                chi1_angles.append(np.nan)
                valid_mask.append(False)
        
        return np.array(chi1_angles), np.array(valid_mask)
    
    def chi2_angles(self):
        """
        Calculate Chi2 dihedral angles (CA-CB-CG-CD) for residues that have delta atoms.
        Returns Chi2 angles in degrees and mask indicating which residues have valid Chi2.
        """
        if self.coords_gamma is None or self.coords_delta is None or not self.use_cbeta:
            return None, None
        
        # Extract coordinates
        coords_CA = self.coords[:, 1, :].numpy()  # (L, 3) 
        coords_CB = self.coords[:, 3, :].numpy()  # (L, 3)
        coords_gamma = self.coords_gamma          # (L, 3)
        coords_delta = self.coords_delta          # (L, 3)
        
        chi2_angles = []
        valid_mask = []
        
        for i in range(len(self.seq)):
            if (self.has_gamma_mask[i] and self.has_delta_mask[i] and 
                not np.any(np.isnan(coords_gamma[i])) and not np.any(np.isnan(coords_delta[i]))):
                # Calculate Chi2 angle for this residue
                angle = calculate_dihedral_angle(
                    coords_CA[i], coords_CB[i], coords_gamma[i], coords_delta[i]
                )
                chi2_angles.append(angle)
                valid_mask.append(not np.isnan(angle))
            else:
                chi2_angles.append(np.nan)
                valid_mask.append(False)
        
        return np.array(chi2_angles), np.array(valid_mask)
    
    def chi3_angles(self):
        """
        Calculate Chi3 dihedral angles (CB-CG-CD-CE) for residues that have epsilon atoms.
        Returns Chi3 angles in degrees and mask indicating which residues have valid Chi3.
        """
        if (self.coords_gamma is None or self.coords_delta is None or 
            self.coords_epsilon is None or not self.use_cbeta):
            return None, None
        
        # Extract coordinates
        coords_CB = self.coords[:, 3, :].numpy()  # (L, 3)
        coords_gamma = self.coords_gamma          # (L, 3)
        coords_delta = self.coords_delta          # (L, 3)
        coords_epsilon = self.coords_epsilon      # (L, 3)
        
        chi3_angles = []
        valid_mask = []
        
        for i in range(len(self.seq)):
            if (self.has_gamma_mask[i] and self.has_delta_mask[i] and self.has_epsilon_mask[i] and 
                not np.any(np.isnan(coords_gamma[i])) and not np.any(np.isnan(coords_delta[i])) and
                not np.any(np.isnan(coords_epsilon[i]))):
                # Calculate Chi3 angle for this residue
                angle = calculate_dihedral_angle(
                    coords_CB[i], coords_gamma[i], coords_delta[i], coords_epsilon[i]
                )
                chi3_angles.append(angle)
                valid_mask.append(not np.isnan(angle))
            else:
                chi3_angles.append(np.nan)
                valid_mask.append(False)
        
        return np.array(chi3_angles), np.array(valid_mask)
    
    def chi4_angles(self):
        """
        Calculate Chi4 dihedral angles (CG-CD-CE-terminal) for residues that have terminal atoms.
        Returns Chi4 angles in degrees and mask indicating which residues have valid Chi4.
        """
        if (self.coords_gamma is None or self.coords_delta is None or 
            self.coords_epsilon is None or self.coords_terminal is None or not self.use_cbeta):
            return None, None
        
        # Extract coordinates
        coords_gamma = self.coords_gamma          # (L, 3)
        coords_delta = self.coords_delta          # (L, 3)
        coords_epsilon = self.coords_epsilon      # (L, 3)
        coords_terminal = self.coords_terminal    # (L, 3)
        
        chi4_angles = []
        valid_mask = []
        
        for i in range(len(self.seq)):
            if (self.has_gamma_mask[i] and self.has_delta_mask[i] and 
                self.has_epsilon_mask[i] and self.has_terminal_mask[i] and
                not np.any(np.isnan(coords_gamma[i])) and not np.any(np.isnan(coords_delta[i])) and
                not np.any(np.isnan(coords_epsilon[i])) and not np.any(np.isnan(coords_terminal[i]))):
                # Calculate Chi4 angle for this residue
                angle = calculate_dihedral_angle(
                    coords_gamma[i], coords_delta[i], coords_epsilon[i], coords_terminal[i]
                )
                chi4_angles.append(angle)
                valid_mask.append(not np.isnan(angle))
            else:
                chi4_angles.append(np.nan)
                valid_mask.append(False)
        
        return np.array(chi4_angles), np.array(valid_mask)
    
    def carbonyl_bond_angles(self):
        """
        Calculate Cα-C'-O bond angles for carbonyl groups.
        Returns bond angles in degrees and mask indicating which residues have valid angles.
        """
        if self.coords_O is None:
            return None, None
        
        # Extract coordinates
        coords_CA = self.coords[:, 1, :].numpy()  # (L, 3) - alpha carbon
        coords_C = self.coords[:, 2, :].numpy()   # (L, 3) - carbonyl carbon
        coords_O = self.coords_O                  # (L, 3) - carbonyl oxygen
        
        bond_angles = []
        valid_mask = []
        
        for i in range(len(self.seq)):
            if self.has_o_mask[i] and not np.any(np.isnan(coords_O[i])) and \
               not np.any(np.isnan(coords_C[i])) and not np.any(np.isnan(coords_CA[i])):
                
                # Calculate Cα-C'-O bond angle
                # Vectors from C' to CA and from C' to O
                vec_C_to_CA = coords_CA[i] - coords_C[i]  # C' -> CA
                vec_C_to_O = coords_O[i] - coords_C[i]    # C' -> O
                
                # Calculate angle using dot product formula: cos(θ) = (u·v) / (|u||v|)
                dot_product = np.dot(vec_C_to_CA, vec_C_to_O)
                norm_CA = np.linalg.norm(vec_C_to_CA)
                norm_O = np.linalg.norm(vec_C_to_O)
                
                if norm_CA > 1e-8 and norm_O > 1e-8:
                    cos_angle = dot_product / (norm_CA * norm_O)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    bond_angles.append(angle_deg)
                    valid_mask.append(True)
                else:
                    bond_angles.append(np.nan)
                    valid_mask.append(False)
            else:
                bond_angles.append(np.nan)
                valid_mask.append(False)
        
        return np.array(bond_angles), np.array(valid_mask)

def create_chi_angle_analysis(chi_angles_all, chi_by_residue, proteins_with_chi, chi_name, 
                             residue_counts, expected_amino_acids):
    """
    Create a comprehensive chi angle analysis plot similar to chi1_angle_analysis.png
    
    Parameters:
    - chi_angles_all: list of all chi angles
    - chi_by_residue: dict of chi angles grouped by amino acid
    - proteins_with_chi: number of proteins with this chi angle
    - chi_name: string like "Chi1", "Chi2", etc.
    - residue_counts: dict of amino acid counts in dataset
    - expected_amino_acids: list of amino acids that SHOULD have this chi angle
    """
    if not chi_angles_all:
        print(f"No {chi_name} angles found in the dataset!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Overall Chi distribution
    plt.subplot(2, 2, 1)
    plt.hist(chi_angles_all, bins=60, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    plt.title(f"Overall {chi_name} Angle Distribution\n({len(chi_angles_all)} angles from {proteins_with_chi} proteins)")
    plt.xlabel(f"{chi_name} Angle (degrees)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    
    # Chi by residue type (show ALL residue types with this chi angle)
    plt.subplot(2, 2, 2)
    residue_chi_counts = {aa: len(angles) for aa, angles in chi_by_residue.items() if len(angles) > 0}
    all_residues_with_chi = sorted(residue_chi_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Use a colormap to distinguish amino acids
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_residues_with_chi)))
    
    for i, (aa, count) in enumerate(all_residues_with_chi):
        if count > 5:  # Only plot if we have reasonable data
            plt.hist(chi_by_residue[aa], bins=25, alpha=0.6, label=f"{aa} (n={count})", 
                    density=True, color=colors[i], histtype='step', linewidth=2)
    
    plt.title(f"{chi_name} Angles by Residue Type (All Amino Acids)")
    plt.xlabel(f"{chi_name} Angle (degrees)")
    plt.ylabel("Density") 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Residue-specific Chi counts (show ALL amino acids in dataset)
    plt.subplot(2, 2, 3)
    aa_names = []
    aa_counts = []
    
    # Get all amino acids that appear in the dataset (from residue_counts)
    all_dataset_aas = [aa for aa, count in residue_counts.items() if count > 0]
    
    for aa in sorted(all_dataset_aas):
        aa_names.append(aa)
        # Get chi count for this amino acid (0 if no chi angles)
        chi_count = len(chi_by_residue.get(aa, []))
        aa_counts.append(chi_count)
    
    # Color bars: blue for amino acids that SHOULD have this chi, gray for those that shouldn't
    colors = []
    for aa in aa_names:
        if aa in expected_amino_acids:
            colors.append('steelblue' if aa_counts[aa_names.index(aa)] > 0 else 'lightcoral')  # Red if expected but missing
        else:
            colors.append('lightgray')  # Gray if not expected
    
    plt.bar(aa_names, aa_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.title(f"{chi_name} Angle Counts by Amino Acid\n(Gray = No {chi_name} expected, Red = Expected but missing)")
    plt.xlabel("Amino Acid")
    plt.ylabel(f"Number of {chi_name} Angles")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add text annotation for amino acids without chi
    no_chi_aas = [aa for aa, count in zip(aa_names, aa_counts) if count == 0 and aa in expected_amino_acids]
    expected_no_chi_aas = [aa for aa in aa_names if aa not in expected_amino_acids]
    
    annotation_text = ""
    if no_chi_aas:
        annotation_text += f"Missing {chi_name}: {', '.join(no_chi_aas)}\n"
    if expected_no_chi_aas:
        annotation_text += f"No {chi_name} expected: {', '.join(expected_no_chi_aas)}"
    
    if annotation_text:
        plt.text(0.02, 0.98, annotation_text.strip(), 
                transform=plt.gca().transAxes, fontsize=7, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    # Chi angle circular plot (Ramachandran-style)
    plt.subplot(2, 2, 4, projection='polar')
    chi_radians = np.radians(chi_angles_all)
    plt.hist(chi_radians, bins=36, alpha=0.7, color='green')
    plt.title(f"{chi_name} Angles (Circular)")
    plt.grid(True)
    
    plt.tight_layout()
    filename = f"{chi_name.lower()}_angle_analysis.png"
    plt.savefig(os.path.join("..", "results", filename), dpi=300, bbox_inches='tight')
    
    print(f"\n{chi_name} analysis complete! Saved {filename}")
    if all_residues_with_chi:
        print(f"Top 5 residues by {chi_name} count: {dict(all_residues_with_chi[:5])}")

# Expected amino acids for each chi angle (based on sidechain length and structure)
# These are amino acids that SHOULD have the chi angle if the structure is complete
EXPECTED_CHI1 = ['C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']  # All except G, A
EXPECTED_CHI2 = ['E', 'F', 'H', 'I', 'K', 'L', 'M', 'P', 'Q', 'R', 'W', 'Y']  # Need at least 4 heavy atoms in sidechain
EXPECTED_CHI3 = ['E', 'K', 'M', 'Q', 'R']  # Need at least 5 heavy atoms in sidechain  
EXPECTED_CHI4 = ['K', 'R']  # Only Lys and Arg have long enough sidechains

# Load dataset
pbd = ProteinBackboneDataset(root_dir="/workspace/demo/transformer_stack/data/sample_training_data", center=True,use_cbeta=True, max_length=10**10)
paths = pbd.file_paths

# Initialize counts and lists
chirality_counts = {chirality: 0 for chirality in Chirality}
residue_counts = {token.name: 0 for token in SEQUENCE_TOKENS}

num_dextro = 0
num_levo = 0
num_achiral = 0

# Bond angle lists
CB_N_pro = []
CB_C_pro = []
N_C_pro = []
CB_N_nonpro = []
CB_C_nonpro = []
N_C_nonpro = []

# Chi1 angle lists
chi1_angles_all = []
chi1_by_residue = {aa: [] for aa in "ACDEFGHIKLMNPQRSTVWY"}  # Standard amino acids
proteins_with_chi1 = 0
total_chi1_angles = 0

# Chi2 angle lists
chi2_angles_all = []
chi2_by_residue = {aa: [] for aa in "ACDEFGHIKLMNPQRSTVWY"}
proteins_with_chi2 = 0
total_chi2_angles = 0

# Chi3 angle lists
chi3_angles_all = []
chi3_by_residue = {aa: [] for aa in "ACDEFGHIKLMNPQRSTVWY"}
proteins_with_chi3 = 0
total_chi3_angles = 0

# Chi4 angle lists
chi4_angles_all = []
chi4_by_residue = {aa: [] for aa in "ACDEFGHIKLMNPQRSTVWY"}
proteins_with_chi4 = 0
total_chi4_angles = 0

# C'-O bond angle lists
co_bond_angles_all = []
proteins_with_co_angles = 0
total_co_angles = 0

# Correlation analysis: collect Chi1 with corresponding backbone angles
chi1_with_backbone_angles = []  # Will store tuples of (chi1, cb_n_angle, cb_c_angle, n_c_angle, residue_type)

print("Analyzing proteins for Chi1, Chi2, Chi3, and Chi4 angles...")
print("Now extracting atoms: CG/CG1/OG/SG, CD/CD1/SD, CE/NE/OE1, NZ/CZ...")
print("Also extracting carbonyl oxygen atoms for Cα-C'-O bond angle analysis...")

for i, pth in enumerate(paths):
    if i % 10 == 0:
        print(f"Processing protein {i+1}/{len(paths)}")
    
    try:
        item = ExtendedJSONWrapper(pth, use_cbeta=True)
        
        ch = item.chirality()
        ba = item.bond_angles()
        chi1_angles, chi1_valid = item.chi1_angles()
        chi2_angles, chi2_valid = item.chi2_angles()
        chi3_angles, chi3_valid = item.chi3_angles()
        chi4_angles, chi4_valid = item.chi4_angles()
        co_angles, co_valid = item.carbonyl_bond_angles()
        
        # Track proteins that have at least one Chi1 angle
        if chi1_valid is not None and np.any(chi1_valid):
            proteins_with_chi1 += 1
            total_chi1_angles += np.sum(chi1_valid)
            
            # Store valid Chi1 angles
            valid_chi1 = chi1_angles[chi1_valid]
            chi1_angles_all.extend(valid_chi1[~np.isnan(valid_chi1)])
            
            # Group Chi1 angles by residue type
            for j, (aa, has_chi1, chi1_val) in enumerate(zip(item.seq, chi1_valid, chi1_angles)):
                if has_chi1 and not np.isnan(chi1_val):
                    if aa in chi1_by_residue:
                        chi1_by_residue[aa].append(chi1_val)
            
            # Collect correlation data: Chi1 vs backbone angles
            for j, (aa, has_chi1, chi1_val) in enumerate(zip(item.seq, chi1_valid, chi1_angles)):
                if has_chi1 and not np.isnan(chi1_val) and j < len(ba):
                    # Get corresponding backbone angles for this residue
                    cb_n_angle = ba[j, 0].item()  # CB-CA-N 
                    cb_c_angle = ba[j, 1].item()  # CB-CA-C
                    n_c_angle = ba[j, 2].item()   # N-CA-C
                    chi1_with_backbone_angles.append((chi1_val, cb_n_angle, cb_c_angle, n_c_angle, aa))
        
        # Track proteins that have at least one Chi2 angle
        if chi2_valid is not None and np.any(chi2_valid):
            proteins_with_chi2 += 1
            total_chi2_angles += np.sum(chi2_valid)
            
            # Store valid Chi2 angles
            valid_chi2 = chi2_angles[chi2_valid]
            chi2_angles_all.extend(valid_chi2[~np.isnan(valid_chi2)])
            
            # Group Chi2 angles by residue type
            for j, (aa, has_chi2, chi2_val) in enumerate(zip(item.seq, chi2_valid, chi2_angles)):
                if has_chi2 and not np.isnan(chi2_val):
                    if aa in chi2_by_residue:
                        chi2_by_residue[aa].append(chi2_val)
        
        # Track proteins that have at least one Chi3 angle
        if chi3_valid is not None and np.any(chi3_valid):
            proteins_with_chi3 += 1
            total_chi3_angles += np.sum(chi3_valid)
            
            # Store valid Chi3 angles
            valid_chi3 = chi3_angles[chi3_valid]
            chi3_angles_all.extend(valid_chi3[~np.isnan(valid_chi3)])
            
            # Group Chi3 angles by residue type
            for j, (aa, has_chi3, chi3_val) in enumerate(zip(item.seq, chi3_valid, chi3_angles)):
                if has_chi3 and not np.isnan(chi3_val):
                    if aa in chi3_by_residue:
                        chi3_by_residue[aa].append(chi3_val)
        
        # Track proteins that have at least one Chi4 angle
        if chi4_valid is not None and np.any(chi4_valid):
            proteins_with_chi4 += 1
            total_chi4_angles += np.sum(chi4_valid)
            
            # Store valid Chi4 angles
            valid_chi4 = chi4_angles[chi4_valid]
            chi4_angles_all.extend(valid_chi4[~np.isnan(valid_chi4)])
            
            # Group Chi4 angles by residue type
            for j, (aa, has_chi4, chi4_val) in enumerate(zip(item.seq, chi4_valid, chi4_angles)):
                if has_chi4 and not np.isnan(chi4_val):
                    if aa in chi4_by_residue:
                        chi4_by_residue[aa].append(chi4_val)
        
        # Track C'-O bond angles
        if co_valid is not None and np.any(co_valid):
            proteins_with_co_angles += 1
            total_co_angles += np.sum(co_valid)
            
            # Store valid C'-O bond angles
            valid_co = co_angles[co_valid]
            co_bond_angles_all.extend(valid_co[~np.isnan(valid_co)])
        
        # Split angles based on whether residue is proline or not
        is_pro = torch.tensor([aa == "P" for aa in item.seq])

        for s in item.seq:
            residue_counts[s] += 1
        
        # Update chirality counts
        for c in ch:
            chirality_counts[Chirality(c.item())] += 1
        
        # Proline angles
        CB_N_pro.append(ba[is_pro, 0])
        CB_C_pro.append(ba[is_pro, 1])
        N_C_pro.append(ba[is_pro, 2])
        
        # Non-proline angles
        CB_N_nonpro.append(ba[~is_pro, 0])
        CB_C_nonpro.append(ba[~is_pro, 1])
        N_C_nonpro.append(ba[~is_pro, 2])

        num_dextro += (ch == Chirality.D.value).sum()
        num_levo += (ch == Chirality.L.value).sum()
        num_achiral += (ch == Chirality.A.value).sum()
        
    except Exception as e:
        print(f"Warning: Error processing {pth}: {e}")
        continue

# Concatenate all angles
CB_N_pro = torch.cat(CB_N_pro, dim=0) if CB_N_pro else torch.tensor([])
CB_C_pro = torch.cat(CB_C_pro, dim=0) if CB_C_pro else torch.tensor([])
N_C_pro = torch.cat(N_C_pro, dim=0) if N_C_pro else torch.tensor([])
CB_N_nonpro = torch.cat(CB_N_nonpro, dim=0) if CB_N_nonpro else torch.tensor([])
CB_C_nonpro = torch.cat(CB_C_nonpro, dim=0) if CB_C_nonpro else torch.tensor([])
N_C_nonpro = torch.cat(N_C_nonpro, dim=0) if N_C_nonpro else torch.tensor([])

# Print statistics
print(f"\nChirality: {num_dextro} dextro, {num_levo} levo, {num_achiral} achiral")
print(f"\nChi1 Angle Statistics:")
print(f"Proteins with Chi1 angles: {proteins_with_chi1}/{len(paths)} ({100*proteins_with_chi1/len(paths):.1f}%)")
print(f"Total Chi1 angles found: {total_chi1_angles}")

if chi1_angles_all:
    chi1_array = np.array(chi1_angles_all)
    print(f"Chi1 angle range: {chi1_array.min():.1f}° to {chi1_array.max():.1f}°")
    print(f"Chi1 angle mean: {chi1_array.mean():.1f}° ± {chi1_array.std():.1f}°")
    
    # Show which amino acids have Chi1 angles
    aas_with_chi1 = [aa for aa, angles in chi1_by_residue.items() if len(angles) > 0]
    print(f"Amino acids with Chi1 angles: {sorted(aas_with_chi1)}")

print(f"\nChi2 Angle Statistics:")
print(f"Proteins with Chi2 angles: {proteins_with_chi2}/{len(paths)} ({100*proteins_with_chi2/len(paths):.1f}%)")
print(f"Total Chi2 angles found: {total_chi2_angles}")

if chi2_angles_all:
    chi2_array = np.array(chi2_angles_all)
    print(f"Chi2 angle range: {chi2_array.min():.1f}° to {chi2_array.max():.1f}°")
    print(f"Chi2 angle mean: {chi2_array.mean():.1f}° ± {chi2_array.std():.1f}°")
    
    # Show which amino acids have Chi2 angles
    aas_with_chi2 = [aa for aa, angles in chi2_by_residue.items() if len(angles) > 0]
    print(f"Amino acids with Chi2 angles: {sorted(aas_with_chi2)}")

print(f"\nChi3 Angle Statistics:")
print(f"Proteins with Chi3 angles: {proteins_with_chi3}/{len(paths)} ({100*proteins_with_chi3/len(paths):.1f}%)")
print(f"Total Chi3 angles found: {total_chi3_angles}")

if chi3_angles_all:
    chi3_array = np.array(chi3_angles_all)
    print(f"Chi3 angle range: {chi3_array.min():.1f}° to {chi3_array.max():.1f}°")
    print(f"Chi3 angle mean: {chi3_array.mean():.1f}° ± {chi3_array.std():.1f}°")
    
    # Show which amino acids have Chi3 angles
    aas_with_chi3 = [aa for aa, angles in chi3_by_residue.items() if len(angles) > 0]
    print(f"Amino acids with Chi3 angles: {sorted(aas_with_chi3)}")

print(f"\nChi4 Angle Statistics:")
print(f"Proteins with Chi4 angles: {proteins_with_chi4}/{len(paths)} ({100*proteins_with_chi4/len(paths):.1f}%)")
print(f"Total Chi4 angles found: {total_chi4_angles}")

if chi4_angles_all:
    chi4_array = np.array(chi4_angles_all)
    print(f"Chi4 angle range: {chi4_array.min():.1f}° to {chi4_array.max():.1f}°")
    print(f"Chi4 angle mean: {chi4_array.mean():.1f}° ± {chi4_array.std():.1f}°")
    
    # Show which amino acids have Chi4 angles
    aas_with_chi4 = [aa for aa, angles in chi4_by_residue.items() if len(angles) > 0]
    print(f"Amino acids with Chi4 angles: {sorted(aas_with_chi4)}")

print(f"\nC'-O Bond Angle Statistics:")
print(f"Proteins with C'-O angles: {proteins_with_co_angles}/{len(paths)} ({100*proteins_with_co_angles/len(paths):.1f}%)")
print(f"Total C'-O angles found: {total_co_angles}")

if co_bond_angles_all:
    co_array = np.array(co_bond_angles_all)
    print(f"C'-O bond angle range: {co_array.min():.3f} to {co_array.max():.3f} °")
    print(f"C'-O bond angle mean: {co_array.mean():.3f} ± {co_array.std():.3f} °")

print(f"\nCorrelation Data:")
print(f"Chi1-backbone angle pairs collected: {len(chi1_with_backbone_angles)}")

# Create plots
plt.figure(figsize=(10, 12))

# Bond angles plot (existing)
plt.subplot(3, 1, 1)
plt.hist(CB_N_nonpro.numpy(), bins=50, alpha=0.5, label='Non-Proline', density=True)
plt.hist(CB_N_pro.numpy(), bins=50, alpha=0.5, label='Proline', density=True)
plt.title("Cβ-Cα-N")
plt.legend()

plt.subplot(3, 1, 2)
plt.hist(CB_C_nonpro.numpy(), bins=50, alpha=0.5, label='Non-Proline', density=True)
plt.hist(CB_C_pro.numpy(), bins=50, alpha=0.5, label='Proline', density=True)
plt.title("Cβ-Cα-C")
plt.legend()

plt.subplot(3, 1, 3)
plt.hist(N_C_nonpro.numpy(), bins=50, alpha=0.5, label='Non-Proline', density=True)
plt.hist(N_C_pro.numpy(), bins=50, alpha=0.5, label='Proline', density=True)
plt.title("N-Cα-C")
plt.legend()

plt.suptitle("Bond Angles Distribution: Proline vs Non-Proline (degrees)")
plt.tight_layout()
plt.savefig(os.path.join("..", "results", "bond_angles_pro_vs_nonpro.png"))

# Residue counts plot (existing)
plt.figure(figsize=(15, 5))
sorted_items = sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_items)

# Calculate total residues for percentage calculation
total_residues = sum(values)

bars = plt.bar(labels, values)
plt.title("Residue Counts")

# Add percentage labels on top of each bar
for i, (bar, value) in enumerate(zip(bars, values)):
    percentage = (value / total_residues) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
             f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

plt.xticks()
plt.tight_layout()
plt.savefig(os.path.join("..", "results", "residue_counts.png"))

# Chirality counts plot (existing)
plt.figure(figsize=(8, 5))
chirality_labels = [c.name for c in chirality_counts.keys()]
chirality_values = list(chirality_counts.values())
plt.bar(chirality_labels, chirality_values)
plt.title("Chirality Counts")
plt.tight_layout()
plt.savefig(os.path.join("..", "results", "chirality_counts.png"))

# NEW: Chi1 angle distribution plot
if chi1_angles_all:
    create_chi_angle_analysis(chi1_angles_all, chi1_by_residue, proteins_with_chi1, "Chi1", 
                             residue_counts, EXPECTED_CHI1)

# NEW: Chi2 angle distribution plot
if chi2_angles_all:
    create_chi_angle_analysis(chi2_angles_all, chi2_by_residue, proteins_with_chi2, "Chi2", 
                             residue_counts, EXPECTED_CHI2)
else:
    print("No Chi2 angles found in the dataset!")

# NEW: Chi3 angle distribution plot
if chi3_angles_all:
    create_chi_angle_analysis(chi3_angles_all, chi3_by_residue, proteins_with_chi3, "Chi3", 
                             residue_counts, EXPECTED_CHI3)
else:
    print("No Chi3 angles found in the dataset!")

# NEW: Chi4 angle distribution plot
if chi4_angles_all:
    create_chi_angle_analysis(chi4_angles_all, chi4_by_residue, proteins_with_chi4, "Chi4", 
                             residue_counts, EXPECTED_CHI4)
else:
    print("No Chi4 angles found in the dataset!")

# NEW: Chi1 vs Backbone Angles Correlation Analysis
if chi1_with_backbone_angles:
    print(f"\nCreating Chi1-backbone correlation analysis...")
    
    # Convert to arrays for plotting
    correlation_data = np.array(chi1_with_backbone_angles)
    chi1_vals = correlation_data[:, 0].astype(float)
    cb_n_vals = correlation_data[:, 1].astype(float)
    cb_c_vals = correlation_data[:, 2].astype(float)
    n_c_vals = correlation_data[:, 3].astype(float)
    aa_types = correlation_data[:, 4]
    
    plt.figure(figsize=(15, 10))
    
    # Scatter plots of Chi1 vs each backbone angle
    plt.subplot(2, 3, 1)
    plt.scatter(chi1_vals, cb_n_vals, alpha=0.5, s=10, c='blue')
    plt.xlabel("Chi1 Angle (degrees)")
    plt.ylabel("Cβ-Cα-N Angle (degrees)")
    plt.title("Chi1 vs Cβ-Cα-N")
    plt.grid(True, alpha=0.3)
    
    # Calculate and display correlation coefficient
    corr_chi1_cb_n = np.corrcoef(chi1_vals, cb_n_vals)[0, 1]
    plt.text(0.05, 0.95, f"r = {corr_chi1_cb_n:.3f}", transform=plt.gca().transAxes, 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.subplot(2, 3, 2)
    plt.scatter(chi1_vals, cb_c_vals, alpha=0.5, s=10, c='red')
    plt.xlabel("Chi1 Angle (degrees)")
    plt.ylabel("Cβ-Cα-C Angle (degrees)")
    plt.title("Chi1 vs Cβ-Cα-C")
    plt.grid(True, alpha=0.3)
    
    corr_chi1_cb_c = np.corrcoef(chi1_vals, cb_c_vals)[0, 1]
    plt.text(0.05, 0.95, f"r = {corr_chi1_cb_c:.3f}", transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.subplot(2, 3, 3)
    plt.scatter(chi1_vals, n_c_vals, alpha=0.5, s=10, c='green')
    plt.xlabel("Chi1 Angle (degrees)")
    plt.ylabel("N-Cα-C Angle (degrees)")
    plt.title("Chi1 vs N-Cα-C")
    plt.grid(True, alpha=0.3)
    
    corr_chi1_n_c = np.corrcoef(chi1_vals, n_c_vals)[0, 1]
    plt.text(0.05, 0.95, f"r = {corr_chi1_n_c:.3f}", transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Histogram of correlation coefficients
    plt.subplot(2, 3, 4)
    correlations = [corr_chi1_cb_n, corr_chi1_cb_c, corr_chi1_n_c]
    angle_names = ['Chi1-CβN', 'Chi1-CβC', 'Chi1-NC']
    colors = ['blue', 'red', 'green']
    plt.bar(angle_names, correlations, color=colors, alpha=0.7)
    plt.ylabel("Correlation Coefficient")
    plt.title("Chi1 vs Backbone Angle Correlations")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Chi1 distribution by amino acid type (subset for clarity)
    plt.subplot(2, 3, 5)
    top_aa_types = ['P', 'L', 'V', 'I', 'F', 'Y']  # Common amino acids with distinct chi1 patterns
    for aa in top_aa_types:
        aa_mask = aa_types == aa
        if np.sum(aa_mask) > 20:  # Only if we have enough data
            plt.hist(chi1_vals[aa_mask], bins=30, alpha=0.6, label=f"{aa} (n={np.sum(aa_mask)})", density=True)
    plt.xlabel("Chi1 Angle (degrees)")
    plt.ylabel("Density")
    plt.title("Chi1 Distribution by Amino Acid")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
    Chi1-Backbone Correlation Summary
    
    Sample Size: {len(chi1_vals):,} residues
    
    Correlation Coefficients:
    • Chi1 vs Cβ-Cα-N:  {corr_chi1_cb_n:+.3f}
    • Chi1 vs Cβ-Cα-C:  {corr_chi1_cb_c:+.3f}  
    • Chi1 vs N-Cα-C:   {corr_chi1_n_c:+.3f}
    
    Independence Test:
    |r| < 0.1 suggests independence
    Chi1 appears {"independent" if all(abs(r) < 0.1 for r in correlations) else "weakly correlated"}
    from backbone geometry
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    plt.suptitle("Chi1 vs Backbone Angles Independence Analysis", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "results", "chi1_backbone_correlation.png"), dpi=300, bbox_inches='tight')
    print(f"Saved chi1_backbone_correlation.png")

# NEW: C'-O Bond Angle Distribution Analysis  
if co_bond_angles_all:
    print(f"\nCreating C'-O bond angle analysis...")
    
    plt.figure(figsize=(12, 8))
    
    co_array = np.array(co_bond_angles_all)
    
    # Main distribution plot
    plt.subplot(2, 2, 1)
    plt.hist(co_array, bins=50, alpha=0.7, density=True, color='darkred', edgecolor='black')
    plt.xlabel("Cα-C'-O Bond Angle (°)")
    plt.ylabel("Density")
    plt.title(f"Cα-C'-O Bond Angle Distribution\n({len(co_array):,} angles from {proteins_with_co_angles} proteins)")
    plt.grid(True, alpha=0.3)
    
    # Add theoretical C=O bond angle reference
    theoretical_co = 120.0  # Typical Cα-C'-O bond angle for sp2 hybridized carbonyl carbon
    plt.axvline(theoretical_co, color='red', linestyle='--', linewidth=2, 
               label=f'Theoretical Cα-C\'-O: {theoretical_co:.0f}°')
    plt.legend()
    
    # Cumulative distribution
    plt.subplot(2, 2, 2)
    sorted_co = np.sort(co_array)
    cumulative = np.arange(1, len(sorted_co) + 1) / len(sorted_co)
    plt.plot(sorted_co, cumulative, linewidth=2, color='darkred')
    plt.xlabel("Cα-C'-O Bond Angle (°)")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution")
    plt.grid(True, alpha=0.3)
    plt.axvline(theoretical_co, color='red', linestyle='--', alpha=0.7)
    
    # Violin plot
    plt.subplot(2, 2, 3)
    violin_parts = plt.violinplot([co_array], positions=[0], widths=[0.5], showmeans=True, showmedians=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('darkred')
        pc.set_alpha(0.7)
    plt.ylabel("Cα-C'-O Bond Angle (°)")
    plt.title("Distribution Shape")
    plt.xticks([0], ['Cα-C\'-O Angles'])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Statistics summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calculate percentiles
    p5 = np.percentile(co_array, 5)
    p25 = np.percentile(co_array, 25)
    p50 = np.percentile(co_array, 50)  # median
    p75 = np.percentile(co_array, 75)
    p95 = np.percentile(co_array, 95)
    
    stats_text = f"""
    C'-O Bond Angle Statistics
    
    Sample Size: {len(co_array):,} angles
    Proteins: {proteins_with_co_angles:,}
    
    Central Tendency:
    • Mean:    {co_array.mean():.3f} ± {co_array.std():.3f} °
    • Median:  {p50:.3f} °
    
    Percentiles:
    •  5th:    {p5:.3f} °
    • 25th:    {p25:.3f} °  
    • 75th:    {p75:.3f} °
    • 95th:    {p95:.3f} °
    
    Range: {co_array.min():.3f} - {co_array.max():.3f} °
    
    Expected Cα-C'-O: ~120° (sp2 trigonal planar)
    Deviation: {abs(co_array.mean() - theoretical_co):.3f} °
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
    
    plt.suptitle("Carbonyl C'-O Bond Angle Analysis", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "results", "co_bond_angle_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"Saved co_bond_angle_analysis.png")
else:
    print("No C'-O bond angle data found!")

plt.show()
