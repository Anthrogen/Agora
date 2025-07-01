# SS8 Confusion Matrix Generation for FSQ Model

This directory contains scripts for evaluating a trained FSQ (Finite Scalar Quantization) model by generating confusion matrices for 8-state secondary structure (SS8) prediction.

## Overview

The SS8 prediction task involves classifying each residue in a protein into one of 8 secondary structure types:
- **H**: α-helix
- **G**: 3₁₀-helix  
- **I**: π-helix (rare)
- **E**: β-strand
- **B**: β-bridge (isolated β)
- **T**: Turn
- **S**: Bend
- **L**: Coil/loop

## Scripts

### 1. `GA_testing.py`
Main script that:
- Loads a trained FSQ model from checkpoint
- Processes the entire dataset
- Uses BioPython's DSSP implementation to compute SS8 assignments
- Generates confusion matrices and detailed reports

### 2. `test_ss8_setup.py`
Test script to verify DSSP is working correctly on a single protein

## Usage

### Prerequisites
```bash
# Required packages
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm biopython

# Install DSSP (required for BioPython's DSSP module)
# On Ubuntu/Debian:
sudo apt-get install dssp

# On macOS with Homebrew:
brew install brewsci/bio/dssp
```

### Running the Script
```bash
cd scripts
python GA_testing.py
```

### Running the Test
```bash
cd scripts
python test_ss8_setup.py
```

### Configuration
Edit the following parameters in `GA_testing.py`:
```python
checkpoint_path = "../checkpoints/checkpoint_epoch_60.pt"  # Path to model checkpoint
data_dir = "../data/sample_training_data"                  # Path to dataset
output_dir = "../results"                                  # Where to save results
batch_size = 4                                            # Batch size for processing
```

## Output

The script generates the following files in the `results/` directory:

1. **`ss8_confusion_matrix.png`**: Normalized confusion matrix (proportions)
2. **`ss8_confusion_matrix_counts.png`**: Raw count confusion matrix
3. **`ss8_confusion_matrix_report.txt`**: Detailed text report containing:
   - Overall accuracy
   - Per-class statistics
   - Full confusion matrices (normalized and raw)

## Interpreting the Confusion Matrix

- **Rows**: True SS8 types (ground truth)
- **Columns**: Predicted SS8 types (model output)
- **Diagonal elements**: Correct predictions
- **Off-diagonal elements**: Misclassifications

Common patterns to look for:
- H↔G confusion (both are helical structures)
- E↔B confusion (both involve β-sheet hydrogen bonding)
- T,S→L confusion (turns/bends often misclassified as coil)

## Technical Details

### SS8 Assignment Algorithm
The script uses BioPython's DSSP implementation ([Bio.PDB.DSSP](https://biopython.org/docs/1.76/api/Bio.PDB.DSSP.html)):
1. Converts backbone coordinates to PDB format
2. Runs DSSP algorithm to assign secondary structures
3. Maps DSSP codes to our SS8 classification (converts '-' to 'L')

### Model Architecture
The FSQ autoencoder:
- Encodes protein backbone coordinates to discrete codes
- Uses finite scalar quantization with levels [7, 5, 5, 5, 5]
- Reconstructs backbone coordinates from quantized representation

### DSSP Integration
- Creates temporary PDB files for DSSP processing
- Uses BioPython's DSSP wrapper with default settings
- Falls back to all-coil assignment if DSSP fails

## Troubleshooting

1. **"DSSP failed"**: Ensure DSSP is installed and accessible in PATH
2. **"Checkpoint file not found"**: Ensure the checkpoint path is correct
3. **Out of memory**: Reduce batch_size
4. **No SS8 type 'I' in results**: π-helices are very rare in proteins

## References

- DSSP: Kabsch W, Sander C. (1983) Dictionary of protein secondary structure
- BioPython DSSP module: https://biopython.org/docs/1.76/api/Bio.PDB.DSSP.html

## Future Improvements

1. Integrate actual DSSP for more accurate SS8 assignment
2. Add hydrogen bond analysis for better β-structure detection
3. Include confidence scores for predictions
4. Add visualization of specific protein examples 