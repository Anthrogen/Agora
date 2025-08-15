import os
import orjson # TODO: swap for faster json library
import mmap
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
from enum import Enum
import array
import re
import boto3
from botocore.exceptions import NoCredentialsError
import smart_open

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
         'W': BACKBONE + ['O', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
         'X': BACKBONE + ['O'],  # Unknown amino acid - treat like Glycine (minimal sidechain)
         'U': BACKBONE + ['O', 'SE', 'O', 'H', 'HA', 'HB2', 'HB3'], # Selenocysteine
         'Z': BACKBONE + ['O', 'CG', 'CD', 'OE1', 'OE2'],  # Glutamic acid or Glutamine (ambiguous) - treat as Glutamic acid
         'B': BACKBONE + ['O', 'CG', 'OD1', 'ND2'],  # Asparagine or Aspartic acid (ambiguous) - treat as Asparagine
         'O': BACKBONE + ['O', 'CG', 'CD', 'CE', 'NZ'],  # Ornithine - similar to Lysine
         'J': BACKBONE + ['O', 'CG', 'CD1', 'CD2']  # Leucine or Isoleucine (ambiguous) - treat as Leucine
}

for k in ATOMS:
    assert ATOMS[k][ATOM_N_ENC] == "N"
    assert ATOMS[k][ATOM_CA_ENC] == "CA"
    assert ATOMS[k][ATOM_C_ENC] == "C"
    assert ATOMS[k][ATOM_O_ENC] == "O"
    assert len(ATOMS[k]) <= ENCODING_LEN


# Note: this does not include "struct", which we do not have at this point.
ALL_TRACKS = ('seq', 'coords', 'ss8', 'sasa', 'orthologous_groups', 'semantic_description', 'domains', 'plddt')

class Protein():
    """
    To test this code, try it with:

    Protein("/workspace/demo/Odyssey/sample_data/3k/1a04_A.json", mode="side_chain")

    Modes: side_chain, backbone

    TODO: this class should have two methods of construction: one from PDB/JSON, one from sequence and coordinate tensors.
    TODO: furthermore, we should be able to "dump" to PDB/JSON from this class.
    """
    def __init__(self, file_path, mode="side_chain"):

        assert mode in ["side_chain", "backbone"]
        self.mode = mode

        # Check if this is an S3/R2 URL
        if file_path.startswith(('s3://', 'r2://', 'https://')):
            # Handle R2 URLs by converting to S3 format for boto3
            if file_path.startswith('r2://'):
                file_path = file_path.replace('r2://', 's3://')
            
            # Use smart_open for S3/R2 streaming
            endpoint_url = os.environ.get('AWS_ENDPOINT_URL', None)
            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_DEFAULT_REGION', 'auto')
            ) if endpoint_url else None
            
            with smart_open.open(file_path, 'rb', transport_params={'client': s3_client} if s3_client else {}) as f:
                data = orjson.loads(f.read())
        else:
            # Local file path
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as f:
                data = orjson.loads(f.read())

        # This dataloader assumes that each amino acid begins with an "N".
        atom_names = data.get("all_atoms", {}).get('atom_names', None)
        sequence = data.get('sequence', None)
        atom_coords = data.get('all_atoms', {}).get('atom_coordinates', None)
        ss8 = data.get('dssp', None)
        sasa = data.get('sasa', None)
        orthologous_groups = data.get('orthologous_groups', None)
        semantic_description = data.get('semantic_description', None)
        domains = data.get('domains', None)
        plddt = data.get('plddt', None)
        structure_source = data.get('structure_source', None)

        if atom_coords is not None:
            atom_coords = torch.Tensor(atom_coords)

        # Handle secondary structure
        if ss8 is None or len(ss8) == 0: ss8 = [None] * len(sequence) # If entire field is None, create array of None with sequence length
        elif isinstance(ss8, list): # assert isinstance(ss8, list) # Replace individual NaN/None elements with None
            ss8[:] = [None if (x is None or str(x).lower() == 'nan' or (isinstance(x, (int, float)) and np.isnan(x))) else x for x in ss8]

        # Handle sasa
        if sasa is None or len(sasa) == 0: sasa = [None] * len(sequence) # If entire field is None, create array of None with sequence length
        elif isinstance(sasa, list): # Replace individual NaN/None elements with None
            sasa[:] = [None if (x is None or str(x).lower() == 'nan' or (isinstance(x, (int, float)) and np.isnan(x))) else x for x in sasa]

        # Handle orthologous groups
        if orthologous_groups is None or orthologous_groups == "": orthologous_groups = []
        elif isinstance(orthologous_groups, str): # Split by commas and clean up terms
            terms = orthologous_groups.split(',')
            orthologous_groups = [term.strip() for term in terms if term.strip()]

        # Handle semantic description
        if semantic_description is None or semantic_description == "": semantic_description = []
        elif isinstance(semantic_description, str): # Split by whitespace, punctuation and clean up terms
            # Split on punctuation and whitespace, keeping letters and numbers together
            terms = re.split(r'[^a-zA-Z0-9]+', semantic_description)
            semantic_description = [term.strip() for term in terms if term.strip()]

        # Handle domains
        if not domains: domains = [[] for _ in range(len(sequence))] # Handles both None and empty list
        elif isinstance(domains, list):
            assert len(domains) == len(sequence)
            for i, domain_list in enumerate(domains):
                if not isinstance(domain_list, list): raise ValueError(f"Domain at position {i} is not a list: {domain_list}")
        
        # Handle plddt with structure source-specific scaling
        if plddt is None: plddt = [None] * len(sequence) # If entire field is None, create array of None with sequence length
        elif isinstance(plddt, list): 
            # Scale AFDB values from 0-100 to 0-1 (PDB and ESM values are already in 0-1 range, no scaling needed)
            scaling = (1/100.0) if structure_source == "AFDB" else 1.0
            plddt[:] = [None if (x is None or str(x).lower() == 'nan' or (isinstance(x, (int, float)) and np.isnan(x))) else x * scaling for x in plddt]

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

            if sequence[residue_idx] == '*':
                # For masked sequence positions, read all coordinates sequentially
                coord_idx = 0
                for atom_idx in range(start_idx, end_idx):
                    atom_name = atom_names[atom_idx]
                    
                    # Skip OXT - these are terminal/additional atoms not part of the residue
                    # Also skip all hydrogen atoms.
                    if atom_name == "OXT" or atom_name.startswith("H"): continue
                    
                    # If we exceed the tensor size, just skip the remaining atoms
                    if coord_idx < all_coords.shape[1]:
                        all_coords[residue_idx, coord_idx, :] = atom_coords[atom_idx, :]
                        coord_idx += 1
            else:
                # For unmasked sequence positions, use proper amino acid lookup
                filled = torch.zeros(len(BACKBONE) if not self.mode == "side_chain" else len(ATOMS[sequence[residue_idx]]))
                if sequence[residue_idx] == "G":
                    filled[ATOM_CB_ENC] = 1 # We'll fill it in later with a phantom CB.

                for atom_idx in range(start_idx, end_idx):
                    atom_name = atom_names[atom_idx]
                    
                    # Skip OXT - these are terminal/additional atoms not part of the residue
                    # Also skip all hydrogen atoms.
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

            # For construction of the virtual/"phantom" CB atom, we use the formula developed by:
            # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011330
            virtual_cb = -0.58273431 * vec_a + 0.56802827 * vec_b - 0.54067466 * vec_c + gly_ca
            all_coords[gly_idx, ATOMS["G"].index("CB"), :] = virtual_cb

        ##########################################################
        # Final Checks and Outputs:
        assert not torch.isnan(all_coords).any(), "NaNs found in Protein Coordinates"

        self.coords = all_coords
        self.seq = sequence
        self.len = len(self.seq)
        self.ss8 = ss8
        self.sasa = sasa
        self.orthologous_groups = orthologous_groups
        self.semantic_description = semantic_description
        self.domains = domains
        self.plddt = plddt

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

    def dump_to_json(self, file_path, original_json_data=None):
        """
        Save the protein data to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
            original_json_data: Optional original JSON data to preserve fields not in the Protein object
        """
        # Start with original data if provided, otherwise create new dict
        if original_json_data is not None:
            data = original_json_data.copy()
        else:
            data = {}
        
        # Update sequence
        data['sequence'] = ''.join(self.seq)
        
        # Update all_atoms and backbone_coordinates if coordinates are available
        if hasattr(self, 'coords') and self.coords is not None:
            if 'all_atoms' not in data:
                data['all_atoms'] = {'atom_names': [], 'atom_coordinates': []}
            if 'backbone_coordinates' not in data:
                data['backbone_coordinates'] = {'N': [], 'CA': [], 'C': []}
            
            atom_coords_list = []
            atom_names_list = []
            backbone_n_coords = []
            backbone_ca_coords = []
            backbone_c_coords = []
            
            # Reconstruct atom coordinates for each residue
            for residue_idx in range(self.len):
                residue_char = self.seq[residue_idx]
                    
                # Get atoms for this residue type
                residue_atoms = ATOMS[residue_char]
                
                # Add each atom (only up to the number that this residue type should have)
                for atom_idx, atom_name in enumerate(residue_atoms):
                    coord = self.coords[residue_idx, atom_idx, :].tolist()
                    # Add coordinates (including zeros) - we need all atoms for proper structure
                    atom_names_list.append(atom_name)
                    atom_coords_list.append(coord)
                    
                    # Also update backbone coordinates
                    if atom_name == 'N':
                        backbone_n_coords.append(coord)
                    elif atom_name == 'CA':
                        backbone_ca_coords.append(coord)
                    elif atom_name == 'C':
                        backbone_c_coords.append(coord)
            
            data['all_atoms']['atom_names'] = atom_names_list
            data['all_atoms']['atom_coordinates'] = atom_coords_list
            data['backbone_coordinates']['N'] = backbone_n_coords
            data['backbone_coordinates']['CA'] = backbone_ca_coords
            data['backbone_coordinates']['C'] = backbone_c_coords
        
        # Write to file
        with open(file_path, 'wb') as f:
            # json.dump(data, f, indent=2)
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    
# JSON at /workspace/cmu_vqvae_data/single_chain_clusters_full.csv
# See development at tmp_csv_parser.py
class ProteinDataset(Dataset):
    """
    Test me in shell:
    ProteinDataset("/workspace/demo/Odyssey/sample_data/tiny_set.csv")
    """
    PROTEIN_ID_COL = 1
    JSON_PATH_COL = 2 # Within the index.csv file, this is the colun (0-indexed) that points ot member Json Path 
    TOTAL_COLS = 4
    def __init__(self, index_csv_path: str, center: bool = True, mode: str = "backbone", max_length: int = 2048, critical_tracks=None, max_length_orthologous_groups: int = 512, max_length_semantic_description: int = 128, eager: bool = False, verbose: bool = False):
        """
        ProteinDataset constructor that handles both CSV index files and single JSON files.
        
        Args:
            index_csv_path: Path to either a CSV index file or a single JSON protein file
            
        If the path ends with .json, creates a single-protein dataset.
        If the path ends with .csv, creates a multi-protein dataset from the CSV index.
        
        Eager: if true, check for malformed files up front. Otherwise, do so on the fly and potentially return None from __getitem__.
        """
        if critical_tracks is None:
            # Default value: everything.
            critical_tracks = {t: True for t in ALL_TRACKS}

        self.critical_tracks = critical_tracks
        
        # Maximum sequence length for padding/truncation
        # Proteins longer than this will be truncated
        self.max_length = max_length
        self.max_length_orthologous_groups = max_length_orthologous_groups
        self.max_length_semantic_description = max_length_semantic_description
        self.mode = mode
        self.center = center
        self.verbose = verbose

        # Check file extension to determine dataset type
        if index_csv_path.endswith('.json'):
            # Single JSON protein file - create a simple list with one path
            self.protein_paths = [index_csv_path]
            self.offsets = array.array("Q", [0])  # Single offset for one protein
            self.index_csv_path = None
            self.mm = None  # No memory mapping needed
            
            if self.verbose: print(f"Creating single-protein dataset from {index_csv_path}")
                
        elif index_csv_path.endswith('.csv'):
            # CSV index file (original behavior)
            self.protein_paths = None  # Will be read from CSV via memory mapping
            self.index_csv_path = index_csv_path
            self.index_csv_dir = os.path.dirname(index_csv_path) # all paths expressed in the .csv file are relative to this position.

            # Iterate through the CSV file.
            # For each row, record the offset (number of bytes from the beginning of the file) of the row on the disk.
            self.offsets = array.array("Q", [])
            
            with open(self.index_csv_path, "rb") as f:

                last_id = None # filter out non-unique entries.
                for line in f:
                    pos = f.tell() - len(line)
                    line = line.decode("utf-8").rstrip("\r\n").split(",")

                    # print(f"Line: {line}")

                    if len(line) != self.TOTAL_COLS:
                        if self.verbose: print(f"Skipping bad csv line: {line}")
                        continue

                    new_id = line[self.PROTEIN_ID_COL].strip()
                    if self.verbose: print(f"new_id = {new_id}")

                    if last_id is None or last_id != new_id: 
                        self.offsets.append(pos)
                    elif self.verbose: 
                        print(f"Skipping duplicate entry for {new_id}")

                    last_id = new_id

            self.offsets.pop(0) # Get rid of header row.
            assert len(self.offsets) > 0, f"No valid JSON Paths found in {self.index_csv_path}!"

            # print(f"Full offsets: {self.offsets}")
            
            # Create a memory map into the index.csv file.
            # This virtually "maps" the disk space (large memory storage) file into the memory of the python process.
            # This allows us to read the contents of the file 
            #  without re-loading it into memory at each invocation of __getitem__.
            fd = os.open(self.index_csv_path, os.O_RDONLY)
            self.mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)

            if eager: self.offsets = array.array("Q", [off for idx, off in enumerate(self.offsets) if self.get_protein(idx) is not None])
        else: raise ValueError(f"Unsupported file type: {index_csv_path}. Expected .csv or .json file.")
            
    def __getstate__(self):
        """Handle pickling by excluding the mmap object."""
        state = self.__dict__.copy()
        # Remove the mmap object as it cannot be pickled
        if 'mm' in state:
            del state['mm']
        return state
    
    def __setstate__(self, state):
        """Handle unpickling by recreating the mmap object."""
        self.__dict__.update(state)
        # Recreate the mmap object in the new process (only for CSV case)
        if self.index_csv_path is not None:
            try:
                fd = os.open(self.index_csv_path, os.O_RDONLY)
                self.mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                os.close(fd)
            except (OSError, IOError) as e:
                raise RuntimeError(f"Failed to recreate mmap for {self.index_csv_path} in worker process: {e}")

    def __len__(self):
        """Return the total number of protein structures in the dataset"""
        return len(self.offsets)

    def get_protein(self, idx) -> Protein:
        """
        It is important to separate out this functionality from __getitem__
        so that we can speed up eager mode.
        """
        assert idx >= 0 and idx < len(self.offsets), f"Index {idx} out of bounds for dataset of length {len(self.offsets)}"

        if self.protein_paths is not None:
            # Single JSON protein case - path is directly available
            json_path = self.protein_paths[idx]
        else:
            # CSV index case - read path from memory-mapped CSV
            start = self.offsets[idx]
            end = self.mm.find(b"\n", start)
            if end == -1:
                end = len(self.mm)

            rel_path_to_json = (
                self.mm[start:end]
                .decode()
                .split(",")[self.JSON_PATH_COL]
                .strip()  # remove any whitespace/newline characters
            )

            # Check if it's an S3/R2 URL or local path
            if rel_path_to_json.startswith(('s3://', 'r2://', 'https://')):
                json_path = rel_path_to_json  # Use URL directly
            else:
                json_path = os.path.join(self.index_csv_dir, rel_path_to_json)

        # Load protein (same for both cases)
        try:
            return Protein(json_path, mode=self.mode)
        except (AssertionError, ValueError, orjson.JSONDecodeError, PermissionError, OSError, FileNotFoundError, NoCredentialsError) as e:
            if self.verbose:
                print(f"Warning: encountered malformed file {json_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> torch.Tensor:
        protein = self.get_protein(idx)
        if protein is None:
            return None

        coords = protein.coords[:self.max_length]
        seq = protein.seq[:self.max_length]
        ss8 = protein.ss8[:self.max_length]
        sasa = protein.sasa[:self.max_length]
        orthologous_groups = protein.orthologous_groups[:self.max_length_orthologous_groups]
        semantic_description = protein.semantic_description[:self.max_length_semantic_description]
        domains = protein.domains[:self.max_length]
        plddt = protein.plddt[:self.max_length]
        l = torch.tensor(min(protein.len, self.max_length))

        # Optionally center the structure at origin
        # This is useful for translation-invariant applications
        if self.center:
            # Compute centroid using only backbone atoms (N, CA, C) which are always present
            backbone_coords = coords[:, :3, :]  # [L, 3, 3] - only N, CA, C
            centroid = backbone_coords.reshape(-1, 3).mean(dim=0)  # Average position of backbone atoms
            
            # Only center positive atoms (real atoms), leave missing atoms (zeros and [-1,-1,-1]) unchanged
            positive_mask = (coords > 0.0).any(dim=-1)  # [L, H] - True for real atoms with positive coordinates
            coords = torch.where(positive_mask.unsqueeze(-1), coords - centroid, coords)

        return seq, coords, ss8, sasa, orthologous_groups, semantic_description, domains, plddt, l
