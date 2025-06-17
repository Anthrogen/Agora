"""Test script to verify DSSP is working correctly."""

import os
import json
import numpy as np
import tempfile
import subprocess
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

def coords_to_pdb_structure(coords, chain_id='A', structure_id='temp'):
    """Convert backbone coordinates to a BioPython Structure object."""
    structure = Structure(structure_id)
    model = Model(0)
    chain = Chain(chain_id)
    
    for i in range(len(coords)):
        res_id = (' ', i+1, ' ')
        residue = Residue(res_id, 'ALA', ' ')
        
        n_coord = coords[i, 0]
        ca_coord = coords[i, 1]
        c_coord = coords[i, 2]
        
        # Calculate O position using standard geometry
        if i < len(coords) - 1:
            next_n = coords[i+1, 0]
            ca_c_vec = c_coord - ca_coord
            c_n_vec = next_n - c_coord
            perp_vec = np.cross(ca_c_vec, c_n_vec)
            if np.linalg.norm(perp_vec) > 0:
                perp_vec = perp_vec / np.linalg.norm(perp_vec)
            else:
                perp_vec = np.array([0, 0, 1])
            o_coord = c_coord + 1.23 * perp_vec
        else:
            ca_c_vec = c_coord - ca_coord
            ca_c_vec = ca_c_vec / np.linalg.norm(ca_c_vec)
            if abs(ca_c_vec[2]) < 0.9:
                perp_vec = np.cross(ca_c_vec, np.array([0, 0, 1]))
            else:
                perp_vec = np.cross(ca_c_vec, np.array([1, 0, 0]))
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            o_coord = c_coord + 1.23 * perp_vec
        
        n_atom = Atom('N', n_coord, 1.0, 1.0, ' ', 'N', i*4)
        ca_atom = Atom('CA', ca_coord, 1.0, 1.0, ' ', 'CA', i*4+1)
        c_atom = Atom('C', c_coord, 1.0, 1.0, ' ', 'C', i*4+2)
        o_atom = Atom('O', o_coord, 1.0, 1.0, ' ', 'O', i*4+3)
        
        residue.add(n_atom)
        residue.add(ca_atom)
        residue.add(c_atom)
        residue.add(o_atom)
        
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    
    return structure

def test_dssp():
    """Test DSSP with a sample protein."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_training_data")
    
    # Get first valid JSON file
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON files found!")
        return
    
    json_path = os.path.join(data_dir, json_files[0])
    print(f"Testing with: {json_files[0]}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        coords_N = np.array(data["backbone_coordinates"]["N"], dtype=np.float32)
        coords_CA = np.array(data["backbone_coordinates"]["CA"], dtype=np.float32)
        coords_C = np.array(data["backbone_coordinates"]["C"], dtype=np.float32)
        
        print(f"Protein length: {len(coords_N)} residues")
        
        # Stack into (L, 3, 3) array
        coords = np.stack([coords_N, coords_CA, coords_C], axis=1)
        
        # Create PDB structure
        structure = coords_to_pdb_structure(coords)
        
        # Save to temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
            tmp_pdb_path = tmp_pdb.name
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp_pdb_path)
        
        print(f"Saved PDB to: {tmp_pdb_path}")
        
        # Create output DSSP file
        dssp_out = tmp_pdb_path.replace('.pdb', '.dssp')
        
        # Run mkdssp directly
        cmd = ['mkdssp', '--output-format', 'dssp', tmp_pdb_path, dssp_out]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"mkdssp failed: {result.stderr}")
            return
        
        print(f"mkdssp succeeded, output saved to: {dssp_out}")
        
        # Read and display first few lines of DSSP output
        with open(dssp_out, 'r') as f:
            lines = f.readlines()
            print(f"\nFirst 20 lines of DSSP output:")
            for i, line in enumerate(lines[:20]):
                print(f"{i}: {line.rstrip()}")
        
        # Parse DSSP output
        ss8_assignments = []
        with open(dssp_out, 'r') as f:
            lines = f.readlines()
            in_data = False
            for line in lines:
                if line.startswith('  #  RESIDUE'):
                    in_data = True
                    continue
                if in_data and len(line) > 16:
                    ss_code = line[16]
                    if ss_code == ' ' or ss_code == '-':
                        ss_code = 'L'
                    ss8_assignments.append(ss_code)
        
        print(f"DSSP assignments: {len(ss8_assignments)} residues")
        print(f"First 20 SS8: {''.join(ss8_assignments[:20])}")
        
        # Count SS8 types
        ss8_counts = {}
        for ss in ss8_assignments:
            ss8_counts[ss] = ss8_counts.get(ss, 0) + 1
        print(f"SS8 counts: {ss8_counts}")
        
        # Clean up
        os.remove(tmp_pdb_path)
        os.remove(dssp_out)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dssp() 