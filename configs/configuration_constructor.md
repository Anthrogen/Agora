# Odyssey Configuration Constructor Guide

This guide provides a comprehensive reference for all configuration options available in the Odyssey YAML configuration files. Use this guide to understand the new hierarchical configuration system with automatic type resolution.

## New Configuration System Overview

The configuration system now uses a **simplified hierarchical approach** with list support:

1. **Direct Type Headers**: Configuration types are used directly as headers (e.g., `trunk_cfg:` instead of `type: "trunk_cfg"`)
2. **List Support**: Any parameter can be specified as a list using YAML dash notation (`-`)
3. **Multi-level Lists**: Lists can span across hierarchy levels, allowing multiple configuration options at any level
4. **Registry Pattern**: All configuration classes are registered with decorators for automatic discovery

## Configuration Structure

All configurations now follow this simplified hierarchical pattern with list support:

```yaml
model_cfg:
  config_type:  # Configuration type used directly as header
    # All parameters for this configuration type
    param1: value1
    param2: value2
    
    # Lists can be used for parameters
    list_param:
      - item1
      - item2
    
    # Nested configurations can include multiple options as lists
    nested_cfg:
      - option1_type:
      - option2_type:
      - option3_type:
        param1: value1
        list_param:
          - item1
          - item2

train_cfg:
  training_cfg:
    # Training parameters...
    
    # Configuration sections can also use lists
    loss_config:
      loss_type:
        param1: value1
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

All configuration types are registered and used directly as headers:

### Model Configuration Types
- `fsq_cfg:` - FSQ model configuration (stage_1 and stage_2)
- `trunk_cfg:` - Trunk model configuration (mlm and discrete_diffusion)

### Block Configuration Types
- `self_attention_cfg:` - Standard transformer self-attention
- `geometric_attention_cfg:` - Geometric-aware attention
- `reflexive_attention_cfg:` - Reflexive attention mechanism
- `self_consensus_cfg:` - Consensus-based attention with parameters

### Training Configuration Types
- `training_cfg:` - Standard training configuration

### Loss Configuration Types
- `cross_entropy_loss_cfg:` - Cross-entropy loss for sequence/structure
- `kabsch_rmsd_loss_cfg:` - RMSD loss for structural alignment
- `score_entropy_loss_cfg:` - Score entropy loss for diffusion

### Masking Configuration Types
- `simple_mask_cfg:` - Simple random masking
- `complex_mask_cfg:` - Complex masking patterns
- `diffusion_cfg:` - Diffusion-based masking
- `no_mask_cfg:` - No masking (for evaluation)

---

## Model Configuration

### FSQ Configuration

For FSQ stage 1 and stage 2 training:

```yaml
model_cfg:
  fsq_cfg:
    style: "stage_1"  # or "stage_2"
    
    # Core transformer parameters
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    # First block configuration - multiple options as list
    first_block_cfg:
      - self_attention_cfg:
      - geometric_attention_cfg:
      - reflexive_attention_cfg:
      - self_consensus_cfg:
        consensus_num_iterations: 1
        consensus_connectivity_type: "local_window"
        consensus_w:
          - 2
          - 3
        consensus_r: 24
        consensus_edge_hidden_dim: 12
    
    # FSQ-specific parameters
    latent_dim: 32
    fsq_levels: "7x5x5x5x5"
    fsq_encoder_path: null  # Required for stage_2, null for stage_1
```

### Trunk Configuration

For MLM and discrete diffusion training:

```yaml
model_cfg:
  trunk_cfg:
    style: "mlm"  # or "discrete_diffusion"
    
    # Core transformer parameters
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    # First block configuration - multiple options as list
    first_block_cfg:
      - self_attention_cfg:
      - geometric_attention_cfg:
      - reflexive_attention_cfg:
      - self_consensus_cfg:
        consensus_num_iterations: 1
        consensus_connectivity_type: "local_window"
        consensus_w:
          - 2
          - 3
        consensus_r: 24
        consensus_edge_hidden_dim: 12
    
    # Loss configuration options
    loss_config:
      - cross_entropy_loss_cfg:
      - kabsch_rmsd_loss_cfg:
      - score_entropy_loss_cfg:
    
    # Trunk-specific parameter
    fsq_encoder_path: "checkpoints/fsq/encoder.pt"  # Required
```

### Block Configurations

#### Self-Attention Block
```yaml
first_block_cfg:
  - self_attention_cfg:  # No additional parameters
```

#### Geometric Attention Block
```yaml
first_block_cfg:
  - geometric_attention_cfg:  # No additional parameters
```

#### Reflexive Attention Block
```yaml
first_block_cfg:
  - reflexive_attention_cfg:  # No additional parameters
```

#### Self-Consensus Block
```yaml
first_block_cfg:
  - self_consensus_cfg:
    consensus_num_iterations: 1
    consensus_connectivity_type: "local_window"  # or "scored_window"
    consensus_w:
      - 2
      - 3
    consensus_r: 24
    consensus_edge_hidden_dim: 12
```

#### Multiple Block Options
```yaml
first_block_cfg:
  - self_attention_cfg:
  - geometric_attention_cfg:
  - reflexive_attention_cfg:
  - self_consensus_cfg:
    consensus_num_iterations: 1
    consensus_connectivity_type: "local_window"
    consensus_w:
      - 2
      - 3
    consensus_r: 24
    consensus_edge_hidden_dim: 12
```

---

## Training Configuration

```yaml
train_cfg:
  training_cfg:
    batch_size: 32
    max_epochs: 100
    learning_rate: 1e-4
    data_dir: "sample_data/1k"
    checkpoint_dir: "checkpoints"
    
    # Loss configuration
    loss_config:
      cross_entropy_loss_cfg:
        seq_loss_weight: 1.0
        struct_loss_weight: 1.0
        loss_elements: "masked"
    
    # Mask configuration
    mask_config:
      simple_mask_cfg:
        mask_prob_seq: 0.15
        mask_prob_struct: 0.15
```

---

## Loss Configuration

### Cross Entropy Loss
```yaml
loss_config:
  cross_entropy_loss_cfg:
    seq_loss_weight: 1.0
    struct_loss_weight: 1.0
    loss_elements: "masked"  # Options: "masked", "non_beospank", "non_special"
```

### Kabsch RMSD Loss
```yaml
loss_config:
  kabsch_rmsd_loss_cfg:  # No additional parameters
```

### Score Entropy Loss
```yaml
loss_config:
  score_entropy_loss_cfg:
    seq_loss_weight: 1.0
    struct_loss_weight: 1.0
```

### Multiple Loss Options
```yaml
loss_config:
  - cross_entropy_loss_cfg:
  - kabsch_rmsd_loss_cfg:
  - score_entropy_loss_cfg:
    seq_loss_weight: 1.0
    struct_loss_weight: 1.0
```

---

## Masking Configuration

### Simple Masking
```yaml
mask_config:
  simple_mask_cfg:
    mask_prob_seq: 0.15
    mask_prob_struct: 0.15
```

### Complex Masking
```yaml
mask_config:
  complex_mask_cfg:  # Parameters TBD
```

### Diffusion Masking
```yaml
mask_config:
  diffusion_cfg:
    noise_schedule: "linear"  # Options: "linear", "inverted_u", "uniform"
    sigma_min: 0.31
    sigma_max: 5.68
    num_timesteps: 100
```

### No Masking
```yaml
mask_config:
  no_mask_cfg:  # No additional parameters
```

### Multiple Masking Options
```yaml
mask_config:
  - simple_mask_cfg:
    mask_prob_seq: 0.15
    mask_prob_struct: 0.15
  - complex_mask_cfg:
  - diffusion_cfg:
    noise_schedule: "uniform"
    sigma_min: 0.31
    sigma_max: 5.68
    num_timesteps: 100
  - no_mask_cfg:
```

---

## Configuration Examples

### Example 1: FSQ Stage 1 Training
```yaml
model_cfg:
  fsq_cfg:
    style: "stage_1"
    d_model: 128
    n_heads: 8
    n_layers: 6
    max_len: 1024
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      - self_attention_cfg:
    
    latent_dim: 32
    fsq_levels: "7x5x5x5x5"
    fsq_encoder_path: null

train_cfg:
  training_cfg:
    batch_size: 64
    max_epochs: 70
    learning_rate: 1e-4
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/fsq"
    
    loss_config:
      kabsch_rmsd_loss_cfg:
    
    mask_config:
      simple_mask_cfg:
        mask_prob_seq: 0.15
        mask_prob_struct: 0.15
```

### Example 2: MLM with Self-Consensus
```yaml
model_cfg:
  trunk_cfg:
    style: "mlm"
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      - self_consensus_cfg:
        consensus_num_iterations: 2
        consensus_connectivity_type: "local_window"
        consensus_w:
          - 3
        consensus_r: 32
        consensus_edge_hidden_dim: 16
    
    fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"

train_cfg:
  training_cfg:
    batch_size: 32
    max_epochs: 100
    learning_rate: 5e-5
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/mlm"
    
    loss_config:
      cross_entropy_loss_cfg:
        seq_loss_weight: 1.0
        struct_loss_weight: 1.0
        loss_elements: "non_beospank"
    
    mask_config:
      simple_mask_cfg:
        mask_prob_seq: 0.2
        mask_prob_struct: 0.2
```

### Example 3: Discrete Diffusion
```yaml
model_cfg:
  trunk_cfg:
    style: "discrete_diffusion"
    d_model: 768
    n_heads: 12
    n_layers: 12
    max_len: 2048
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      - geometric_attention_cfg:
    
    fsq_encoder_path: "checkpoints/fsq/best_encoder.pt"

train_cfg:
  training_cfg:
    batch_size: 16
    max_epochs: 200
    learning_rate: 1e-4
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/diffusion"
    
    loss_config:
      score_entropy_loss_cfg:
        seq_loss_weight: 1.0
        struct_loss_weight: 1.0
    
    mask_config:
      diffusion_cfg:
        noise_schedule: "inverted_u"
        sigma_min: 0.1
        sigma_max: 10.0
        num_timesteps: 200
```

### Example 4: FSQ Stage 2 with Reflexive Attention
```yaml
model_cfg:
  fsq_cfg:
    style: "stage_2"
    d_model: 256
    n_heads: 8
    n_layers: 10
    max_len: 1024
    dropout: 0.1
    ff_mult: 4
    reference_model_seed: 42
    
    first_block_cfg:
      - reflexive_attention_cfg:
    
    latent_dim: 32
    fsq_levels: "7x5x5x5x5"
    fsq_encoder_path: "checkpoints/stage1/best_encoder.pt"

train_cfg:
  training_cfg:
    batch_size: 48
    max_epochs: 30
    learning_rate: 5e-5
    data_dir: "data/proteins"
    checkpoint_dir: "checkpoints/stage2"
    
    loss_config:
      kabsch_rmsd_loss_cfg:
    
    mask_config:
      no_mask_cfg:
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

### Old Format (type/params structure):
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

### New Format (simplified with lists):
```yaml
model_cfg:
  fsq_cfg:
    style: "stage_1"
    d_model: 128
    first_block_cfg:
      - self_attention_cfg:
      - geometric_attention_cfg:
      - self_consensus_cfg:
        consensus_num_iterations: 1
        consensus_w:
          - 2
          - 3
    latent_dim: 32
    fsq_levels: "7x5x5x5x5"
    fsq_encoder_path: null

train_cfg:
  training_cfg:
    batch_size: 64
    learning_rate: 1e-4
    loss_config:
      kabsch_rmsd_loss_cfg:
    mask_config:
      simple_mask_cfg:
        mask_prob_seq: 0.15
```

Key changes:
1. Remove `type:` and `params:` fields - use configuration type directly as header
2. Support lists using YAML dash notation (`-`) for multiple options
3. Lists can contain both empty configurations and configurations with parameters
4. Parameters can themselves be lists (e.g., `consensus_w: [2, 3]` or dash notation)
5. Cleaner, more readable structure with less nesting

---

## Tips for Using the New System

1. **Use configuration types as headers**: Replace `type:` and `params:` with the configuration type directly as the header
2. **Leverage lists for options**: Use YAML dash notation (`-`) to specify multiple configuration options at any level
3. **Mix empty and parameterized configs**: Lists can contain both configurations without parameters and those with parameters
4. **Use lists for parameter values**: Any parameter can be a list using either `[item1, item2]` or dash notation
5. **Hierarchical nesting**: Child configurations follow the same pattern - type as header with optional parameters
6. **Validation**: The config loader validates types and parameters automatically
7. **Clean structure**: The new format reduces nesting and improves readability

## List Usage Examples

### Single vs List Parameters
```yaml
# Single value
consensus_w: 2

# List with bracket notation  
consensus_w: [2, 3]

# List with dash notation
consensus_w:
  - 2
  - 3
```

### Multiple Configuration Options
```yaml
# Multiple options in a list
first_block_cfg:
  - self_attention_cfg:
  - geometric_attention_cfg:
  - self_consensus_cfg:
    consensus_num_iterations: 1
```

### Nested Lists
```yaml
# Lists can be used at any level
model_cfg:
  - fsq_cfg:
    first_block_cfg:
      - self_attention_cfg:
      - self_consensus_cfg:
        consensus_w:
          - 2
          - 3
  - trunk_cfg:
    first_block_cfg:
      - geometric_attention_cfg:
```

For more examples, see the configuration files in the `configs/` directory.