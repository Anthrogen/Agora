# Odyssey Configuration Constructor Guide

This guide provides a comprehensive reference for all configuration options available in the Odyssey YAML configuration files. Use this guide to understand what parameters you can set and their valid values.

## Configuration Safety Features

All configuration objects now include built-in safety features:

1. **Automatic Dictionary Backup**: Each configuration stores an immutable copy of its values at creation time
2. **JSON Persistence**: Configurations can be saved to and loaded from JSON files
3. **Recovery Capability**: Original configuration values are preserved even if the object is modified

These features ensure configuration reproducibility and provide recovery options if needed.

## Table of Contents
- [Configuration Safety Features](#configuration-safety-features)
- [Model Configuration](#model-configuration)
  - [Core Parameters](#core-parameters)
  - [Block Configuration](#block-configuration)
  - [FSQ Parameters](#fsq-parameters)
  - [Trunk Model Parameters](#trunk-model-parameters)
- [Training Configuration](#training-configuration)
- [Loss Configuration](#loss-configuration)
- [Masking Configuration](#masking-configuration)
- [Configuration Examples](#configuration-examples)
- [Configuration Safety and Recovery](#configuration-safety-and-recovery)

---

## Model Configuration

The model configuration determines the architecture and parameters of your transformer model.

**Configuration Class Hierarchy:**
- `TransformerConfig` (base class) - contains core transformer parameters
  - `FSQConfig` - for stage_1 and stage_2 training (adds FSQ-specific parameters)
  - `TrunkConfig` - for mlm and discrete_diffusion training (adds trunk-specific parameters)

### Core Parameters

All model configurations share these core parameters:

```yaml
model:
  # Model style - determines the type of model and training approach
  # Required field that must be one of the following:
  style: "stage_1"  # Options: "stage_1", "stage_2", "mlm", "discrete_diffusion"
  
  # Core transformer parameters (required for all styles)
  d_model: 768      # Model dimension (hidden size) - must be positive integer
  n_heads: 12       # Number of attention heads - must be positive integer
  n_layers: 12      # Number of transformer layers - must be positive integer
  max_len: 2048     # Maximum sequence length - must be positive integer
  dropout: 0.1      # Dropout rate - float between 0 and 1
  ff_mult: 4        # Feedforward multiplier - positive integer
                    # (ff_dim = d_model * ff_mult)
  reference_model_seed: 42  # Seed for model initialization - integer
```

### Model Configuration Types

Based on the `style` parameter, you'll build one of these configurations:

#### 1. FSQ Stage 1 Configuration (FSQConfig)
For training FSQ encoder from scratch:

```yaml
model:
  style: "stage_1"
  
  # Core parameters (as above)
  d_model: 128
  n_heads: 8
  n_layers: 6
  max_len: 1024
  dropout: 0.1
  ff_mult: 4
  reference_model_seed: 42
  
  # Block configuration
  block_type: "self_attention"
  block_params: {}
  
  # FSQ-specific parameters
  fsq:
    latent_dim: 32              # Pre-quantized latent dimension
    levels: [7, 5, 5, 5, 5]     # Quantization levels per dimension
    # encoder_path not needed for stage_1
```

#### 2. FSQ Stage 2 Configuration (FSQConfig)
For fine-tuning with pre-trained FSQ encoder:

```yaml
model:
  style: "stage_2"
  
  # Core parameters (as above)
  d_model: 128
  n_heads: 8
  n_layers: 6
  max_len: 1024
  dropout: 0.1
  ff_mult: 4
  reference_model_seed: 42
  
  # Block configuration
  block_type: "self_consensus"
  block_params:
    num_iterations: 1
    connectivity_type: "local_window"
    w: 2
    r: 24
    edge_hidden_dim: 12
  
  # FSQ-specific parameters
  fsq:
    latent_dim: 32              # Must match the pre-trained encoder
    levels: [7, 5, 5, 5, 5]     # Must match the pre-trained encoder
    encoder_path: "checkpoints/fsq/stage1_model.pt"  # Required for stage_2
```

#### 3. MLM Trunk Configuration (TrunkConfig)
For masked language modeling:

```yaml
model:
  style: "mlm"
  
  # Core parameters (as above)
  d_model: 768
  n_heads: 12
  n_layers: 12
  max_len: 2048
  dropout: 0.1
  ff_mult: 4
  reference_model_seed: 42
  
  # Block configuration
  block_type: "geometric_attention"
  block_params: {}
  
  # Trunk-specific parameters
  fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"  # Required
```

#### 4. Discrete Diffusion Trunk Configuration (TrunkConfig)
For discrete diffusion training:

```yaml
model:
  style: "discrete_diffusion"
  
  # Core parameters (as above)
  d_model: 768
  n_heads: 12
  n_layers: 12
  max_len: 2048
  dropout: 0.1
  ff_mult: 4
  reference_model_seed: 42
  
  # Block configuration
  block_type: "reflexive_attention"
  block_params: {}
  
  # Trunk-specific parameters
  fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"  # Required
```

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


### Configuration Summary

#### Parameters by Model Style

| Parameter | stage_1 | stage_2 | mlm | discrete_diffusion |
|-----------|---------|---------|-----|-------------------|
| **Core Parameters** | ✓ | ✓ | ✓ | ✓ |
| **Block Configuration** | ✓ | ✓ | ✓ | ✓ |
| **fsq.latent_dim** | ✓ | ✓ | ✗ | ✗ |
| **fsq.levels** | ✓ | ✓ | ✗ | ✗ |
| **fsq.encoder_path** | ✗ | ✓ | ✗ | ✗ |
| **fsq_encoder_path** | ✗ | ✗ | ✓ | ✓ |

✓ = Required, ✗ = Not used

#### Configuration Classes Used

- **stage_1** → `FSQConfig` (trains FSQ encoder from scratch)
- **stage_2** → `FSQConfig` (fine-tunes with pre-trained encoder)
- **mlm** → `TrunkConfig` (masked language modeling)
- **discrete_diffusion** → `TrunkConfig` (diffusion modeling)

#### Common Configuration Patterns

**FSQ Training Pattern (stage_1/stage_2):**
```yaml
model:
  style: "stage_1"  # or "stage_2"
  # Core parameters...
  # Block configuration...
  fsq:
    latent_dim: 32
    levels: [7, 5, 5, 5, 5]
    encoder_path: null  # or path for stage_2
```

**Trunk Training Pattern (mlm/discrete_diffusion):**
```yaml
model:
  style: "mlm"  # or "discrete_diffusion"
  # Core parameters...
  # Block configuration...
  fsq_encoder_path: "checkpoints/fsq/encoder.pt"
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
  checkpoint_dir: "checkpoints"      # Directory for saving checkpoints
```

---

## Loss Configuration

Configure the loss function used during training:

```yaml
loss:
  # Loss type
  type: "cross_entropy"  # Options: "cross_entropy", "kabsch_rmsd", "score_entropy"
  
  # Cross-entropy and score_entropy specific parameters
  weights:
    sequence: 1.0       # Weight for sequence reconstruction loss (float)
    structure: 1.0      # Weight for structure reconstruction loss (float)
  
  # Loss elements - which positions contribute to loss (cross_entropy only)
  loss_elements: "masked"  # Options: "masked", "non_beospank", "non_special"
  
  # RMSD specific parameters (for kabsch_rmsd loss)
  # Note: Current implementation doesn't use rmsd_elements
  # rmsd_elements: "non_masked"  # Options: "masked", "non_beospank", 
                                 #          "non_special", "non_masked"
```

### Loss Types Explained:

**1. Cross-Entropy Loss** (`cross_entropy`)
- Standard classification loss for sequence and structure tokens
- Uses `weights` and `loss_elements` parameters
- Typically used with MLM training

**2. Kabsch RMSD Loss** (`kabsch_rmsd`)
- Root Mean Square Deviation loss for structural alignment
- Used for FSQ stage 1 and stage 2 training
- Current implementation doesn't use rmsd_elements parameter

**3. Score Entropy Loss** (`score_entropy`)
- Entropy-based loss for discrete diffusion training
- Uses `weights` parameters for sequence and structure
- Typically paired with discrete_diffusion masking strategy

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
  type: "score_entropy"
  weights:
    sequence: 1.0
    structure: 1.0

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
7. **Loss Type**: Must be one of `["cross_entropy", "kabsch_rmsd", "score_entropy"]`
8. **Loss Elements**: Must be one of `["masked", "non_beospank", "non_special"]` (for cross_entropy only)
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

---

## Configuration Safety and Recovery

### Built-in Safety Features

All configuration objects (ModelConfig, TrainingConfig, etc.) automatically include safety mechanisms:

#### 1. Automatic Dictionary Backup

When a configuration is created, it automatically stores a copy of all values:

```python
# When you create a configuration
model_cfg = FSQConfig(style='stage_1', d_model=768, ...)

# A backup is automatically stored in model_cfg._config_dict
# This backup is immutable and preserves the original values
```

#### 2. Accessing Configuration Dictionaries

```python
# Get the stored configuration as a dictionary
config_dict = model_cfg.get_config_dict()

# Convert current configuration to dictionary (excludes computed fields)
current_dict = model_cfg.to_dict()
```

#### 3. JSON Persistence

Save configurations for backup or sharing:

```python
# Save configuration to JSON
model_cfg.save_to_json('backup_model_config.json')
train_cfg.save_to_json('backup_train_config.json')

# Load configuration from JSON
recovered_model = Config.load_from_json('backup_model_config.json')
recovered_train = Config.load_from_json('backup_train_config.json')
```

#### 4. Safety Example

```python
# Original configuration
model_cfg = FSQConfig(d_model=768, ...)
original_d_model = model_cfg.get_config_dict()['d_model']  # 768

# Even if modified later...
model_cfg.d_model = 999

# The original value is still preserved
stored_d_model = model_cfg.get_config_dict()['d_model']  # Still 768!
```

### When to Use Safety Features

1. **Before Training**: Save configurations to JSON for exact reproducibility
2. **During Debugging**: Access original values if configuration was modified
3. **For Checkpointing**: Store configuration alongside model checkpoints
4. **For Sharing**: Export configurations as JSON for collaboration

### Integration with Training

The training scripts automatically benefit from these safety features:

```python
# In train.py or similar
model_cfg, train_cfg = load_config_from_args()

# Configurations are automatically backed up
# Save them with checkpoints for full reproducibility
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_cfg.get_config_dict(),  # Original config preserved
    'training_config': train_cfg.get_config_dict()
}, checkpoint_path)
```

### Recovery from Checkpoints

```python
# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# Reconstruct exact configuration
model_cfg = Config.load_from_json('model_config.json')
# Or from checkpoint dict if saved there
```

These safety features ensure that:
- Experiments are fully reproducible
- Configuration values are never lost
- Recovery is always possible
- Sharing and collaboration are simplified