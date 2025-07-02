# Odyssey Configuration Constructor Guide

This guide provides a comprehensive reference for all configuration options available in the Odyssey YAML configuration files. Use this guide to understand the new hierarchical configuration system with automatic type resolution.

## New Configuration System Overview

The configuration system now uses a **registry-based approach** with hierarchical type/params structure:

1. **Type-based Resolution**: Each configuration section specifies a `type` field that maps to a registered configuration class
2. **Automatic Building**: The config loader recursively builds configurations by looking for `type` and `params` fields
3. **Registry Pattern**: All configuration classes are registered with decorators for automatic discovery

## Configuration Structure

All configurations now follow this hierarchical pattern:

```yaml
model_cfg:
  type: "config_type"  # Maps to registered configuration class
  params:
    # All parameters for this configuration type
    param1: value1
    param2: value2
    
    # Nested configurations also use type/params
    nested_cfg:
      type: "nested_type"
      params:
        nested_param1: value1

train_cfg:
  type: "training_cfg"
  params:
    # Training parameters...
```

## Table of Contents
- [Configuration Registry](#configuration-registry)
- [Model Configuration](#model-configuration)
  - [FSQ Configuration](#fsq-configuration)
  - [Trunk Configuration](#trunk-configuration)
  - [Block Configurations](#block-configurations)
- [Training Configuration](#training-configuration)
- [Loss Configuration](#loss-configuration)
- [Masking Configuration](#masking-configuration)
- [Configuration Examples](#configuration-examples)
- [Configuration Safety Features](#configuration-safety-features)

---

## Configuration Registry

All configuration types are registered and can be used via their type identifier:

### Model Configuration Types
- `fsq_cfg` - FSQ model configuration (stage_1 and stage_2)
- `trunk_cfg` - Trunk model configuration (mlm and discrete_diffusion)

### Block Configuration Types
- `self_attention_cfg` - Standard transformer self-attention
- `geometric_attention_cfg` - Geometric-aware attention
- `reflexive_attention_cfg` - Reflexive attention mechanism
- `self_consensus_cfg` - Consensus-based attention with parameters

### Training Configuration Types
- `training_cfg` - Standard training configuration

### Loss Configuration Types
- `cross_entropy_loss_cfg` - Cross-entropy loss for sequence/structure
- `kabsch_rmsd_loss_cfg` - RMSD loss for structural alignment
- `score_entropy_loss_cfg` - Score entropy loss for diffusion

### Masking Configuration Types
- `simple_mask_cfg` - Simple random masking
- `complex_mask_cfg` - Complex masking patterns
- `diffusion_mask_cfg` - Diffusion-based masking
- `no_mask_cfg` - No masking (for evaluation)

---

## Model Configuration

### FSQ Configuration

For FSQ stage 1 and stage 2 training:

```yaml
model_cfg:
  type: "fsq_cfg"
  params:
    style: "stage_1"  # or "stage_2"
    
    # Core transformer parameters
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    # First block configuration
    first_block_cfg:
      type: "self_attention_cfg"
      params: {}
    
    # FSQ-specific parameters
    latent_dim: 32
    fsq_levels: [7, 5, 5, 5, 5]
    fsq_encoder_path: null  # Required for stage_2, null for stage_1
```

### Trunk Configuration

For MLM and discrete diffusion training:

```yaml
model_cfg:
  type: "trunk_cfg"
  params:
    style: "mlm"  # or "discrete_diffusion"
    
    # Core transformer parameters
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    # First block configuration
    first_block_cfg:
      type: "geometric_attention_cfg"
      params: {}
    
    # Trunk-specific parameter
    fsq_encoder_path: "checkpoints/fsq/encoder.pt"  # Required
```

### Block Configurations

#### Self-Attention Block
```yaml
first_block_cfg:
  type: "self_attention_cfg"
  params: {}  # No additional parameters
```

#### Geometric Attention Block
```yaml
first_block_cfg:
  type: "geometric_attention_cfg"
  params: {}  # No additional parameters
```

#### Reflexive Attention Block
```yaml
first_block_cfg:
  type: "reflexive_attention_cfg"
  params: {}  # No additional parameters
```

#### Self-Consensus Block
```yaml
first_block_cfg:
  type: "self_consensus_cfg"
  params:
    num_iterations: 1
    connectivity_type: "local_window"  # or "top_w"
    w: 2
    r: 24
    edge_hidden_dim: 12
```

---

## Training Configuration

```yaml
train_cfg:
  type: "training_cfg"
  params:
    batch_size: 32
    max_epochs: 100
    learning_rate: 1e-4
    data_dir: "sample_data/1k"
    checkpoint_dir: "checkpoints"
    
    # Nested loss configuration
    loss_config:
      type: "cross_entropy_loss_cfg"
      params:
        seq_loss_weight: 1.0
        struct_loss_weight: 1.0
        loss_elements: "masked"
    
    # Nested mask configuration
    mask_config:
      type: "simple_mask_cfg"
      params:
        mask_prob_seq: 0.15
        mask_prob_struct: 0.15
```

---

## Loss Configuration

### Cross Entropy Loss
```yaml
loss_config:
  type: "cross_entropy_loss_cfg"
  params:
    seq_loss_weight: 1.0
    struct_loss_weight: 1.0
    loss_elements: "masked"  # Options: "masked", "non_beospank", "non_special"
```

### Kabsch RMSD Loss
```yaml
loss_config:
  type: "kabsch_rmsd_loss_cfg"
  params: {}  # No additional parameters
```

### Score Entropy Loss
```yaml
loss_config:
  type: "score_entropy_loss_cfg"
  params:
    seq_loss_weight: 1.0
    struct_loss_weight: 1.0
```

---

## Masking Configuration

### Simple Masking
```yaml
mask_config:
  type: "simple_mask_cfg"
  params:
    mask_prob_seq: 0.15
    mask_prob_struct: 0.15
```

### Complex Masking
```yaml
mask_config:
  type: "complex_mask_cfg"
  params: {}  # Parameters TBD
```

### Diffusion Masking
```yaml
mask_config:
  type: "diffusion_mask_cfg"
  params:
    noise_schedule: "linear"  # Options: "linear", "inverted_u", "uniform"
    sigma_min: 0.31
    sigma_max: 5.68
    num_timesteps: 100
```

### No Masking
```yaml
mask_config:
  type: "no_mask_cfg"
  params: {}
```

---

## Configuration Examples

### Example 1: FSQ Stage 1 Training
```yaml
model_cfg:
  type: "fsq_cfg"
  params:
    style: "stage_1"
    d_model: 128
    n_heads: 8
    n_layers: 6
    max_len: 1024
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      type: "self_attention_cfg"
      params: {}
    
    latent_dim: 32
    fsq_levels: [7, 5, 5, 5, 5]
    fsq_encoder_path: null

train_cfg:
  type: "training_cfg"
  params:
    batch_size: 64
    max_epochs: 70
    learning_rate: 1e-4
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/fsq"
    
    loss_config:
      type: "kabsch_rmsd_loss_cfg"
      params: {}
    
    mask_config:
      type: "simple_mask_cfg"
      params:
        mask_prob_seq: 0.15
        mask_prob_struct: 0.15
```

### Example 2: MLM with Self-Consensus
```yaml
model_cfg:
  type: "trunk_cfg"
  params:
    style: "mlm"
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      type: "self_consensus_cfg"
      params:
        num_iterations: 2
        connectivity_type: "local_window"
        w: 3
        r: 32
        edge_hidden_dim: 16
    
    fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"

train_cfg:
  type: "training_cfg"
  params:
    batch_size: 32
    max_epochs: 100
    learning_rate: 5e-5
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/mlm"
    
    loss_config:
      type: "cross_entropy_loss_cfg"
      params:
        seq_loss_weight: 1.0
        struct_loss_weight: 1.0
        loss_elements: "non_beospank"
    
    mask_config:
      type: "simple_mask_cfg"
      params:
        mask_prob_seq: 0.2
        mask_prob_struct: 0.2
```

### Example 3: Discrete Diffusion
```yaml
model_cfg:
  type: "trunk_cfg"
  params:
    style: "discrete_diffusion"
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      type: "geometric_attention_cfg"
      params: {}
    
    fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"

train_cfg:
  type: "training_cfg"
  params:
    batch_size: 16
    max_epochs: 200
    learning_rate: 1e-4
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/diffusion"
    
    loss_config:
      type: "score_entropy_loss_cfg"
      params:
        seq_loss_weight: 1.0
        struct_loss_weight: 1.0
    
    mask_config:
      type: "diffusion_mask_cfg"
      params:
        noise_schedule: "inverted_u"
        sigma_min: 0.1
        sigma_max: 10.0
        num_timesteps: 200
```

### Example 4: FSQ Stage 2 with Reflexive Attention
```yaml
model_cfg:
  type: "fsq_cfg"
  params:
    style: "stage_2"
    d_model: 256
    n_heads: 8
    n_layers: 10
    max_len: 1024
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      type: "reflexive_attention_cfg"
      params: {}
    
    latent_dim: 32
    fsq_levels: [7, 5, 5, 5, 5]
    fsq_encoder_path: "checkpoints/stage1/best_encoder.pt"

train_cfg:
  type: "training_cfg"
  params:
    batch_size: 48
    max_epochs: 30
    learning_rate: 5e-5
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/stage2"
    
    loss_config:
      type: "kabsch_rmsd_loss_cfg"
      params: {}
    
    mask_config:
      type: "no_mask_cfg"
      params: {}
```

---

## Configuration Safety Features

All configuration objects include built-in safety features:

### 1. Automatic Dictionary Backup
When a configuration is created, it automatically stores an immutable copy of all values:

```python
# Configuration automatically backs up values
model_cfg = FSQConfig(style='stage_1', d_model=768, ...)
# Backup stored in model_cfg._config_dict
```

### 2. JSON Persistence
Save and load configurations:

```python
# Save configuration
model_cfg.save_to_json('model_config.json')

# Load configuration
recovered_model = Config.load_from_json('model_config.json')
```

### 3. Dictionary Access
```python
# Get stored configuration dictionary
config_dict = model_cfg.get_config_dict()

# Convert to dictionary (excludes computed fields)
current_dict = model_cfg.to_dict()
```

### 4. Integration with Training
The training scripts automatically preserve configuration:

```python
# Configurations are backed up in checkpoints
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_cfg.get_config_dict(),
    'training_config': train_cfg.get_config_dict()
}, checkpoint_path)
```

---

## Migration from Old Format

To convert from the old flat configuration format to the new hierarchical format:

### Old Format:
```yaml
model:
  style: "stage_1"
  d_model: 128
  block_type: "self_attention"
  block_params: {}
  fsq:
    latent_dim: 32

training:
  batch_size: 64
  learning_rate: 1e-4

loss:
  type: "kabsch_rmsd"

masking:
  strategy: "simple"
  simple:
    mask_prob_seq: 0.15
```

### New Format:
```yaml
model_cfg:
  type: "fsq_cfg"
  params:
    style: "stage_1"
    d_model: 128
    first_block_cfg:
      type: "self_attention_cfg"
      params: {}
    latent_dim: 32
    fsq_levels: [7, 5, 5, 5, 5]
    fsq_encoder_path: null

train_cfg:
  type: "training_cfg"
  params:
    batch_size: 64
    learning_rate: 1e-4
    loss_config:
      type: "kabsch_rmsd_loss_cfg"
      params: {}
    mask_config:
      type: "simple_mask_cfg"
      params:
        mask_prob_seq: 0.15
```

Key changes:
1. Add `type` field to each configuration section
2. Wrap parameters in `params` field
3. Use registered type names (e.g., `fsq_cfg`, `training_cfg`)
4. Flatten nested configurations (e.g., `fsq.latent_dim` → `latent_dim`)
5. Move loss and masking into training configuration

---

## Tips for Using the New System

1. **Always specify type**: Every configuration section must have a `type` field
2. **Use registered names**: Type names must match registered configuration classes
3. **Hierarchical nesting**: Child configurations also use type/params structure
4. **Validation**: The config loader validates types and parameters automatically
5. **Auto-completion**: IDEs can provide better support with explicit types

For more examples, see the configuration files in the `configs/` directory.