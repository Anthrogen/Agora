# Odyssey Configuration Constructor Guide

This guide provides a comprehensive reference for all configuration options available in the Odyssey YAML configuration files. Use this guide to understand what parameters you can set and their valid values.

## Table of Contents
- [Model Configuration](#model-configuration)
  - [Core Parameters](#core-parameters)
  - [Block Configuration](#block-configuration)
  - [FSQ Parameters](#fsq-parameters)
- [Training Configuration](#training-configuration)
- [Loss Configuration](#loss-configuration)
- [Masking Configuration](#masking-configuration)
- [Configuration Examples](#configuration-examples)

---

## Model Configuration

The model configuration determines the architecture and parameters of your transformer model.

### Core Parameters

```yaml
model:
  # Model style - determines the type of model and training approach
  # Required field that must be one of the following:
  style: "stage_1"  # Options: "stage_1", "stage_2", "mlm", "discrete_diffusion"
  
  # Core transformer parameters
  d_model: 768      # Model dimension (hidden size) - must be positive integer
  n_heads: 12       # Number of attention heads - must be positive integer
  n_layers: 12      # Number of transformer layers - must be positive integer
  max_len: 2048     # Maximum sequence length - must be positive integer
  dropout: 0.1      # Dropout rate - float between 0 and 1
  ff_mult: 4        # Feedforward multiplier - positive integer
                    # (ff_dim = d_model * ff_mult)
```

#### Model Styles Explained:
- **`stage_1`**: First stage FSQ training - trains the FSQ encoder from scratch
- **`stage_2`**: Second stage FSQ training - uses pre-trained FSQ encoder
- **`mlm`**: Masked Language Modeling on trunk model
- **`discrete_diffusion`**: Discrete diffusion training on trunk model

### Block Configuration

The first transformer block can use different attention mechanisms:

```yaml
model:
  # Block type for the first transformer block
  block_type: "self_consensus"  # Options: "self_attention", "geometric_attention", 
                                #          "reflexive_attention", "self_consensus"
  
  # Block-specific parameters
  block_params:
    # Parameters vary based on block_type
```

#### Block Types:

**1. Self-Attention Block** (standard transformer attention)
```yaml
block_type: "self_attention"
block_params: {}  # No additional parameters needed
```

**2. Geometric Attention Block** (incorporates geometric information)
```yaml
block_type: "geometric_attention"
block_params: {}  # No additional parameters needed
```

**3. Reflexive Attention Block** (reflexive attention mechanism)
```yaml
block_type: "reflexive_attention"
block_params: {}  # No additional parameters needed
```

**4. Self-Consensus Block** (consensus-based attention)
```yaml
block_type: "self_consensus"
block_params:
  num_iterations: 1              # Number of consensus gradient iterations (positive integer)
  connectivity_type: "local_window"  # Options: "local_window", "top_w"
  w: 2                          # Window size (local_window) or number of connections (top_w)
  r: 24                         # Rank of Lambda_ij matrices (positive integer)
  edge_hidden_dim: 12           # Hidden dimension for edge networks (positive integer)
```

### FSQ Parameters

FSQ (Finite Scalar Quantization) parameters are used for structure encoding:

```yaml
model:
  # FSQ parameters (required for stage_1 and stage_2)
  fsq:
    latent_dim: 32              # Pre-quantized continuous latent dimension (positive integer)
    levels: [7, 5, 5, 5, 5]     # Quantization levels for each dimension (list of positive integers)
    encoder_path: null          # Path to pre-trained encoder (required for stage_2, string)
  
  # FSQ encoder path for trunk models (required for mlm and discrete_diffusion)
  fsq_encoder_path: "path/to/encoder.pt"  # Path to pre-trained FSQ encoder
```

---

## Training Configuration

Basic training parameters:

```yaml
training:
  batch_size: 32        # Training batch size (positive integer)
  max_epochs: 100       # Maximum number of epochs (positive integer)
  learning_rate: 1e-4   # Learning rate (positive float)
  
  # Data paths (required)
  data_dir: "sample_data/1k"         # Directory containing protein data files
  checkpoint_dir: "odyssey/checkpoints"  # Directory for saving checkpoints
```

---

## Loss Configuration

Configure the loss function used during training:

```yaml
loss:
  # Loss type
  type: "cross_entropy"  # Options: "cross_entropy", "kabsch_rmsd", "diffusion"
  
  # Cross-entropy specific parameters
  weights:
    sequence: 1.0       # Weight for sequence reconstruction loss (float)
    structure: 1.0      # Weight for structure reconstruction loss (float)
  
  # Loss elements - which positions contribute to loss
  loss_elements: "masked"  # Options: "masked", "non_beospank", "non_special"
  
  # RMSD specific parameters (for kabsch_rmsd loss)
  rmsd_elements: "non_masked"  # Options: "masked", "non_beospank", 
                               #          "non_special", "non_masked"
```

### Loss Types Explained:

**1. Cross-Entropy Loss** (`cross_entropy`)
- Standard classification loss for sequence and structure tokens
- Uses `weights` and `loss_elements` parameters

**2. Kabsch RMSD Loss** (`kabsch_rmsd`)
- Root Mean Square Deviation loss for structural alignment
- Uses `rmsd_elements` parameter

**3. Diffusion Loss** (`diffusion`)
- Used with discrete diffusion training
- Parameters are defined in the masking configuration

### Loss Elements Options:
- **`masked`**: Only masked positions contribute to loss
- **`non_beospank`**: All positions except BOS, EOS, PAD, UNK tokens
- **`non_special`**: All positions except special tokens (BOS, EOS, PAD, UNK, MASK)
- **`non_masked`**: All positions except MASK tokens (RMSD only)

---

## Masking Configuration

Configure how inputs are masked during training:

```yaml
masking:
  # Masking strategy
  strategy: "simple"  # Options: "simple", "complex", "discrete_diffusion", "none"
```

### Masking Strategies:

**1. Simple Masking** (random masking with fixed probability)
```yaml
masking:
  strategy: "simple"
  simple:
    mask_prob_seq: 0.2      # Probability of masking sequence tokens (0-1)
    mask_prob_struct: 0.2   # Probability of masking structure tokens (0-1)
```

**2. Complex Masking** (advanced masking - parameters TBD)
```yaml
masking:
  strategy: "complex"
  complex:
    # Parameters to be defined when ComplexMaskConfig is implemented
```

**3. Discrete Diffusion** (diffusion-based masking)
```yaml
masking:
  strategy: "discrete_diffusion"
  discrete_diffusion:
    noise_schedule: "linear"   # Options: "linear", "inverted_u", "uniform"
    sigma_min: 0.31           # Minimum noise level (positive float)
    sigma_max: 5.68           # Maximum noise level (positive float)
    num_timesteps: 100        # Number of discrete timesteps (positive integer)
```

**4. No Masking** (for evaluation)
```yaml
masking:
  strategy: "none"
  # No additional parameters needed
```

---

## Configuration Examples

### Example 1: Stage 1 FSQ Training
```yaml
model:
  style: "stage_1"
  d_model: 128
  n_heads: 8
  n_layers: 6
  max_len: 1024
  dropout: 0.1
  ff_mult: 4
  
  block_type: "self_attention"
  block_params: {}
  
  fsq:
    latent_dim: 32
    levels: [7, 5, 5, 5, 5]

training:
  batch_size: 64
  max_epochs: 70
  learning_rate: 1e-4
  data_dir: "data/proteins"
  checkpoint_dir: "checkpoints/fsq"

loss:
  type: "cross_entropy"
  weights:
    sequence: 1.0
    structure: 1.0
  loss_elements: "masked"

masking:
  strategy: "simple"
  simple:
    mask_prob_seq: 0.15
    mask_prob_struct: 0.15
```

### Example 2: MLM Trunk Training with Self-Consensus
```yaml
model:
  style: "mlm"
  d_model: 768
  n_heads: 12
  n_layers: 12
  max_len: 2048
  dropout: 0.1
  ff_mult: 4
  
  block_type: "self_consensus"
  block_params:
    num_iterations: 2
    connectivity_type: "local_window"
    w: 3
    r: 32
    edge_hidden_dim: 16
  
  fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"

training:
  batch_size: 32
  max_epochs: 100
  learning_rate: 5e-5
  data_dir: "data/proteins"
  checkpoint_dir: "checkpoints/mlm"

loss:
  type: "cross_entropy"
  weights:
    sequence: 1.0
    structure: 1.0
  loss_elements: "non_beospank"

masking:
  strategy: "simple"
  simple:
    mask_prob_seq: 0.2
    mask_prob_struct: 0.2
```

### Example 3: Discrete Diffusion Training
```yaml
model:
  style: "discrete_diffusion"
  d_model: 768
  n_heads: 12
  n_layers: 12
  max_len: 2048
  dropout: 0.1
  ff_mult: 4
  
  block_type: "geometric_attention"
  block_params: {}
  
  fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"

training:
  batch_size: 16
  max_epochs: 200
  learning_rate: 1e-4
  data_dir: "data/proteins"
  checkpoint_dir: "checkpoints/diffusion"

loss:
  type: "diffusion"  # Loss is handled by diffusion config

masking:
  strategy: "discrete_diffusion"
  discrete_diffusion:
    noise_schedule: "inverted_u"
    sigma_min: 0.1
    sigma_max: 10.0
    num_timesteps: 200
```

### Example 4: Stage 2 FSQ Training with Pre-trained Encoder
```yaml
model:
  style: "stage_2"
  d_model: 256
  n_heads: 8
  n_layers: 10
  max_len: 1024
  dropout: 0.1
  ff_mult: 4
  
  block_type: "reflexive_attention"
  block_params: {}
  
  fsq:
    latent_dim: 32
    levels: [7, 5, 5, 5, 5]
    encoder_path: "checkpoints/stage1/best_encoder.pt"

training:
  batch_size: 48
  max_epochs: 30
  learning_rate: 5e-5
  data_dir: "data/proteins"
  checkpoint_dir: "checkpoints/stage2"

loss:
  type: "cross_entropy"
  weights:
    sequence: 1.0
    structure: 1.2  # Slightly higher weight for structure
  loss_elements: "non_special"

masking:
  strategy: "simple"
  simple:
    mask_prob_seq: 0.25
    mask_prob_struct: 0.25
```

---

## Validation Rules

When creating configurations, keep in mind these validation rules:

1. **Model Style**: Must be one of `["stage_1", "stage_2", "mlm", "discrete_diffusion"]`
2. **Positive Integers**: `d_model`, `n_heads`, `n_layers`, `max_len`, `batch_size`, `max_epochs` must be > 0
3. **Positive Floats**: `learning_rate`, `sigma_min`, `sigma_max` must be > 0
4. **Probabilities**: `dropout`, `mask_prob_seq`, `mask_prob_struct` must be between 0 and 1
5. **Block Type**: Must be one of `["self_attention", "geometric_attention", "reflexive_attention", "self_consensus"]`
6. **Connectivity Type**: For self_consensus, must be one of `["local_window", "top_w"]`
7. **Loss Type**: Must be one of `["cross_entropy", "kabsch_rmsd", "diffusion"]`
8. **Loss Elements**: Must be one of `["masked", "non_beospank", "non_special"]` (plus `"non_masked"` for RMSD)
9. **Noise Schedule**: Must be one of `["linear", "inverted_u", "uniform"]`
10. **Required Paths**:
    - `data_dir` must exist
    - `checkpoint_dir` must exist
    - `fsq_encoder_path` required for `mlm` and `discrete_diffusion` styles
    - `fsq.encoder_path` required for `stage_2` style

---

## Tips for Configuration

1. **Start Simple**: Begin with default values and adjust based on your needs
2. **Match Style and Parameters**: Ensure your parameters match the model style (e.g., FSQ parameters for stage_1/stage_2)
3. **Consistent Paths**: Use absolute paths or paths relative to the project root
4. **Validate Early**: The configuration loader will validate your settings and provide clear error messages
5. **Save Configurations**: Use `--save-config` to save effective configurations for reproducibility

For more examples and use cases, refer to the example configuration files in the `configs/` directory.