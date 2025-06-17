"""
Test script to verify SS8 computation setup using BioPython's DSSP
"""

import os
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the DSSP function from the main script
from GA_testing import run_dssp_on_coords

def test_single_protein():
    """Test SS8 computation on a single protein using DSSP."""
    
    # Load a sample protein
    data_dir = "../data/sample_training_data"
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in data directory")
        return
    
    # Use the first file
    test_file = os.path.join(data_dir, json_files[0])
    print(f"Testing with file: {json_files[0]}")
    
    # Load data
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Get backbone coordinates
    try:
        coords_N = np.array(data["backbone_coordinates"]["N"], dtype=np.float32)
        coords_CA = np.array(data["backbone_coordinates"]["CA"], dtype=np.float32)
        coords_C = np.array(data["backbone_coordinates"]["C"], dtype=np.float32)
        
        # Stack into (L, 3, 3) array
        coords = np.stack([coords_N, coords_CA, coords_C], axis=1)
        
        print(f"Protein length: {len(coords)} residues")
        print(f"Coordinate shape: {coords.shape}")
        
        # Compute SS8 assignments using DSSP
        print("\nRunning DSSP...")
        ss8_assignments = run_dssp_on_coords(coords)
        
        print(f"\nSS8 assignments (first 50): {''.join(ss8_assignments[:50])}")
        
        # Count SS8 types
        ss8_counts = {}
        for ss in ss8_assignments:
            ss8_counts[ss] = ss8_counts.get(ss, 0) + 1
        
        print(f"\nSS8 distribution:")
        for ss_type in ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'L']:
            count = ss8_counts.get(ss_type, 0)
            percent = 100 * count / len(ss8_assignments)
            print(f"  {ss_type}: {count:4d} ({percent:5.1f}%)")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    test_single_protein() 