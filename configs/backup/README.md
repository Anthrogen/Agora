# Configuration Backup Directory

This directory contains backed-up configurations for experiments and training runs.

## Directory Structure

```
backup/
├── daily/           # Daily configuration snapshots
├── experiments/     # Experiment-specific configurations
├── checkpoints/     # Configurations saved with model checkpoints
└── README.md        # This file
```

## Usage

### Daily Backups
Save daily snapshots of configurations you're actively working on:
```bash
python configs/save_config_to_json.py --config ../my_config.yaml \
  --output daily/$(date +%Y%m%d)_my_config.json
```

### Experiment Backups
Save configurations for specific experiments:
```bash
python configs/save_config_to_json.py --config ../experiment_config.yaml \
  --output experiments/experiment_name_v1.json
```

### Checkpoint Configurations
When saving model checkpoints, also save the configuration:
```python
# In your training script
model_config.save_to_json(f'configs/backup/checkpoints/epoch_{epoch}_model.json')
training_config.save_to_json(f'configs/backup/checkpoints/epoch_{epoch}_training.json')
```

## Best Practices

1. **Naming Convention**: Use descriptive names with dates
   - `20240115_fsq_baseline.json`
   - `20240116_mlm_large_model.json`
   - `experiment_geometric_attention_v2.json`

2. **Version Control**: Add important configurations to git
   ```bash
   git add experiments/successful_experiment.json
   git commit -m "Configuration for successful experiment X"
   ```

3. **Regular Cleanup**: Remove old daily backups periodically, keep experiment configs

4. **Documentation**: Add notes about what changed in each configuration:
   ```bash
   echo "Increased learning rate to 1e-3, added geometric attention" > \
     experiments/experiment_v2_notes.txt
   ```

## Recovery

To recover a configuration:
```python
from odyssey.src.configurations import Config

# Load from JSON backup
config = Config.load_from_json('backup/experiments/my_experiment.json')
```

Or use with training scripts:
```bash
# First convert JSON back to YAML if needed
python -c "
import json, yaml
with open('backup/experiments/config.json') as f:
    config = json.load(f)
with open('recovered_config.yaml', 'w') as f:
    yaml.dump(config, f)
"

# Then use with training
python train_from_config.py --config recovered_config.yaml
```