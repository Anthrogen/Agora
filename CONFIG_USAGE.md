# Configuration System Usage Guide

## Overview

The Odyssey project now uses a YAML-based configuration system that replaces the hardcoded configurations in `train.py`. This allows for more flexible experimentation and easier configuration management.

## Basic Usage

### 1. Using Default Configuration

```bash
python train_from_config.py
```

This will use `configs/default_config.yaml` by default.

### 2. Using a Specific Configuration File

```bash
python train_from_config.py --config configs/fsq_stage1_config.yaml
```

### 3. Overriding Configuration Parameters

You can override specific parameters from the command line:

```bash
python train_from_config.py --config configs/fsq_stage1_config.yaml --batch-size 8 --lr 5e-5 --epochs 100
```

### 4. Available Configuration Files

- `configs/default_config.yaml` - Default FSQ Stage 1 configuration
- `configs/fsq_stage1_config.yaml` - FSQ Stage 1 training configuration
- `configs/fsq_stage2_config.yaml` - FSQ Stage 2 training configuration
- `configs/mlm_config.yaml` - Masked Language Modeling configuration
- `configs/discrete_diffusion_config.yaml` - Discrete Diffusion configuration

## Configuration Structure

Each YAML configuration file has the following sections:

### Model Configuration
```yaml
model:
  style: "stage_1"              # Training style: stage_1, stage_2, mlm, discrete_diffusion
  d_model: 768                  # Model dimension
  n_heads: 12                   # Number of attention heads
  n_layers: 12                  # Number of transformer layers
  max_len: 2048                 # Maximum sequence length
  dropout: 0.1                  # Dropout rate
  ff_mult: 4                    # Feedforward multiplier
  reference_model_seed: 42      # Seed for model initialization
  
  # Block configuration
  block_type: "self_consensus"  # Options: self_attention, geometric_attention, reflexive_attention, self_consensus
  block_params:
    # Parameters specific to the block type
    num_iterations: 1
    connectivity_type: "local_window"
    w: 2
    r: 8
    edge_hidden_dim: 24
```

### Training Configuration
```yaml
training:
  batch_size: 32
  max_epochs: 100
  learning_rate: 1e-4
  data_dir: "sample_data/1k"
  checkpoint_dir: "checkpoints"
```

### Loss Configuration
```yaml
loss:
  type: "cross_entropy"         # Options: cross_entropy, kabsch_rmsd, score_entropy
  weights:
    sequence: 1.0
    structure: 1.0
  loss_elements: "masked"       # Options: masked, non_beospank, non_special
```

### Masking Configuration
```yaml
masking:
  strategy: "simple"            # Options: simple, complex, discrete_diffusion, none
  simple:
    mask_prob_seq: 0.15
    mask_prob_struct: 0.15
```

## Command-Line Options

- `--config`: Path to YAML configuration file
- `--style`: Override model style
- `--batch-size`: Override batch size
- `--learning-rate`, `--lr`: Override learning rate
- `--epochs`: Override number of epochs
- `--device`: Override device (cuda/cpu)
- `--print-config`: Print configuration and exit
- `--save-config`: Save effective configuration to file

## Examples

### FSQ Stage 1 Training
```bash
python train_from_config.py --config configs/fsq_stage1_config.yaml
```

### FSQ Stage 2 Training
```bash
python train_from_config.py --config configs/fsq_stage2_config.yaml
```

### MLM Training with Custom Parameters
```bash
python train_from_config.py --config configs/mlm_config.yaml --batch-size 8 --epochs 100
```

### Print Configuration Without Training
```bash
python train_from_config.py --config configs/fsq_stage1_config.yaml --print-config
```

### Save Modified Configuration
```bash
python train_from_config.py --config configs/fsq_stage1_config.yaml --batch-size 16 --save-config configs/my_custom_config.yaml
```

## Creating Custom Configurations

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Save with a descriptive name
4. Use it for training:

```bash
python train_from_config.py --config configs/my_custom_config.yaml
```

## Integration with Existing Code

The configuration system creates the same `model_cfg` and `train_cfg` objects that the original `train.py` expects, so it's fully compatible with the existing training infrastructure.