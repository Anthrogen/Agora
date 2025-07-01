# Odyssey Configuration Files

This directory contains YAML configuration files for different training scenarios in the Odyssey project.

## Configuration Files

### Core Configurations

1. **`default_config.yaml`** - Default FSQ Stage 1 configuration
   - Model: FSQConfig with Self-Consensus attention
   - Loss: Kabsch RMSD
   - Masking: Simple (20% mask probability)

2. **`fsq_stage1_config.yaml`** - FSQ Stage 1 training
   - Trains FSQ encoder from scratch
   - Uses Self Attention block
   - Small model size (d_model=128)

3. **`fsq_stage2_config.yaml`** - FSQ Stage 2 training
   - Fine-tunes with pre-trained encoder
   - Uses Self-Consensus attention
   - No masking strategy

4. **`mlm_config.yaml`** - Masked Language Modeling
   - Uses TrunkConfig with pre-trained FSQ encoder
   - Cross-entropy loss on masked positions
   - 15% masking probability

5. **`discrete_diffusion_config.yaml`** - Discrete Diffusion training
   - Uses TrunkConfig with diffusion masking
   - Score entropy loss
   - Uniform noise schedule

### Example Configurations

6. **`fsq_stage1_geometric.yaml`** - FSQ with Geometric Attention
   - Example of using different attention mechanisms
   - Larger model (d_model=256)

7. **`mlm_reflexive.yaml`** - MLM with Reflexive Attention
   - Different attention block type
   - Higher masking probability (25%)
   - Uses "non_beospank" loss elements

8. **`discrete_diffusion_large.yaml`** - Large-scale Diffusion Model
   - Scaled up architecture (d_model=1024, 24 layers)
   - Self-Consensus with top_w connectivity
   - Inverted-U noise schedule

## Configuration Structure

Each configuration file has four main sections:

```yaml
model:          # Model architecture and style
training:       # Training hyperparameters and paths
loss:           # Loss function configuration
masking:        # Masking strategy configuration
```

## Usage

```bash
# Use default configuration
python train_from_config.py

# Use specific configuration
python train_from_config.py --config configs/mlm_config.yaml

# Override parameters
python train_from_config.py --config configs/fsq_stage1_config.yaml --batch-size 16 --lr 1e-4
```

## Configuration Guidelines

1. **Model Style** determines which configuration class is used:
   - `stage_1`, `stage_2` → FSQConfig
   - `mlm`, `discrete_diffusion` → TrunkConfig

2. **Loss Type** should match the model style:
   - FSQ models typically use `kabsch_rmsd`
   - MLM models use `cross_entropy`
   - Discrete diffusion uses `score_entropy`

3. **Masking Strategy** options:
   - `simple` - Random masking with fixed probability
   - `none` - No masking (typically for stage_2)
   - `discrete_diffusion` - Time-dependent diffusion masking

4. **Block Types** can be mixed and matched:
   - `self_attention` - Standard transformer attention
   - `geometric_attention` - Incorporates geometric information
   - `reflexive_attention` - Reflexive attention mechanism
   - `self_consensus` - Consensus-based attention with additional parameters

For detailed configuration options, see `configuration_constructor.md`.

## Configuration Safety Features

The Odyssey configuration system includes several safety features to protect your configurations:

### 1. Automatic Backup

When configurations are loaded, the original values are automatically stored internally. This is done through the `_config_dict` attribute that preserves the initial configuration state.

```python
# Example: Configurations automatically store their initial state
config = FSQConfig(...)
original_config = config.get_config_dict()  # Returns the preserved original configuration
```

### 2. Saving Configurations to JSON

You can save any configuration to a JSON file for backup or sharing:

```python
# Save configuration to JSON
model_config.save_to_json('backup_configs/my_model_config.json')
training_config.save_to_json('backup_configs/my_training_config.json')
```

The saved JSON includes:
- All configuration parameters
- The configuration class type (for proper deserialization)
- Human-readable formatting

### 3. Loading Configurations from JSON

You can restore configurations from previously saved JSON files:

```python
# Load configuration from JSON
from odyssey.src.configurations import Config

restored_config = Config.load_from_json('backup_configs/my_model_config.json')
```

The loader automatically:
- Detects the correct configuration class
- Handles nested configurations (block configs, mask configs, loss configs)
- Validates all parameters

### 4. Command-Line Configuration Management

The `train_from_config.py` script supports saving effective configurations:

```bash
# Save the effective configuration (after merging YAML + command-line args)
python train_from_config.py --config configs/fsq_stage1_config.yaml --save-config configs/backup/fsq_stage1_effective.yaml

# You can also save configurations in JSON format programmatically
python save_config_example.py --config configs/fsq_stage1_config.yaml --output configs/backup/fsq_stage1.json
```

### 5. Best Practices for Using Safety Features

1. **Before Training**: Save your configuration
   ```bash
   # Create a backup directory
   mkdir -p configs/backup/$(date +%Y%m%d)
   
   # Save configuration before training
   python train_from_config.py --config configs/my_config.yaml \
     --save-config configs/backup/$(date +%Y%m%d)/my_config_backup.yaml
   ```

2. **Version Control**: Keep configuration backups in version control
   ```bash
   git add configs/backup/
   git commit -m "Backup configuration for experiment X"
   ```

3. **Experiment Tracking**: Use descriptive names for saved configurations
   ```
   configs/backup/
   ├── 20240115_fsq_stage1_baseline.json
   ├── 20240116_fsq_stage1_larger_model.json
   └── 20240117_mlm_with_geometric_attention.json
   ```

4. **Configuration Validation**: Always validate loaded configurations
   ```python
   # The configuration loader automatically validates consistency
   config_loader.validate_config_consistency()
   ```

5. **Safe Modification**: When modifying configurations programmatically
   ```python
   # Get a copy of the original configuration
   original_dict = config.get_config_dict()
   
   # Make modifications
   config.learning_rate = 0.001
   
   # Save both original and modified
   Config.from_dict(original_dict).save_to_json('configs/backup/original.json')
   config.save_to_json('configs/backup/modified.json')
   ```

## Example: Complete Configuration Safety Workflow

See `configs/safety_example.py` for a complete example demonstrating all safety features.