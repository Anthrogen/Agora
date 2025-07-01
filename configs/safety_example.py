#!/usr/bin/env python
"""
Configuration Safety Example

This script demonstrates all the safety features available in the Odyssey
configuration system, including:
1. Automatic backup of original configurations
2. Saving configurations to JSON
3. Loading configurations from JSON
4. Safe modification practices
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from datetime import datetime
import json

from odyssey.src.config_loader import ConfigLoader
from odyssey.src.configurations import Config, FSQConfig, TrunkConfig


def demonstrate_automatic_backup():
    """Demonstrate how configurations automatically backup their initial state."""
    print("\n" + "="*60)
    print("1. AUTOMATIC BACKUP DEMONSTRATION")
    print("="*60)
    
    # Load a configuration
    config_loader = ConfigLoader('configs/fsq_stage1_config.yaml')
    config_loader._resolve_paths()
    model_config = config_loader.get_model_config()
    
    # Show that original configuration is preserved
    print(f"Original learning rate: {config_loader.config['training']['learning_rate']}")
    print(f"Original d_model: {model_config.d_model}")
    
    # Get the preserved original configuration
    original_dict = model_config.get_config_dict()
    print(f"\nPreserved configuration has {len(original_dict)} fields")
    print(f"Configuration class: {original_dict.get('_config_class', 'FSQConfig')}")
    
    # Modify the configuration object
    model_config.d_model = 256  # Change from original value
    
    # Show that we can still access the original
    print(f"\nModified d_model: {model_config.d_model}")
    print(f"Original d_model (from backup): {original_dict['d_model']}")
    

def demonstrate_save_to_json():
    """Demonstrate saving configurations to JSON files."""
    print("\n" + "="*60)
    print("2. SAVING TO JSON DEMONSTRATION")
    print("="*60)
    
    # Create backup directory
    backup_dir = Path('configs/backup')
    backup_dir.mkdir(exist_ok=True)
    
    # Load configurations
    config_loader = ConfigLoader('configs/fsq_stage1_config.yaml')
    config_loader._resolve_paths()
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = backup_dir / f'model_config_{timestamp}.json'
    training_path = backup_dir / f'training_config_{timestamp}.json'
    
    # Save configurations
    model_config.save_to_json(str(model_path))
    training_config.save_to_json(str(training_path))
    
    print(f"Saved model config to: {model_path}")
    print(f"Saved training config to: {training_path}")
    
    # Show what's in the saved files
    with open(model_path, 'r') as f:
        saved_model = json.load(f)
    print(f"\nSaved model config contains:")
    print(f"  - Configuration class: {saved_model['_config_class']}")
    print(f"  - Number of parameters: {len(saved_model)}")
    print(f"  - Style: {saved_model['style']}")
    print(f"  - Model dimensions: {saved_model['d_model']}")
    
    return str(model_path), str(training_path)


def demonstrate_load_from_json(model_path, training_path):
    """Demonstrate loading configurations from JSON files."""
    print("\n" + "="*60)
    print("3. LOADING FROM JSON DEMONSTRATION")
    print("="*60)
    
    # Load configurations from JSON
    loaded_model = Config.load_from_json(model_path)
    loaded_training = Config.load_from_json(training_path)
    
    print(f"Loaded model configuration:")
    print(f"  - Type: {type(loaded_model).__name__}")
    print(f"  - Style: {loaded_model.style}")
    print(f"  - d_model: {loaded_model.d_model}")
    print(f"  - First block: {loaded_model.first_block_cfg}")
    
    print(f"\nLoaded training configuration:")
    print(f"  - Type: {type(loaded_training).__name__}")
    print(f"  - Batch size: {loaded_training.batch_size}")
    print(f"  - Learning rate: {loaded_training.learning_rate}")
    print(f"  - Mask config: {type(loaded_training.mask_config).__name__}")
    print(f"  - Loss config: {type(loaded_training.loss_config).__name__}")
    
    # Verify configurations are valid
    try:
        # The loaded configurations should pass all validation
        assert hasattr(loaded_model, 'ff_hidden_dim')
        assert loaded_model.ff_hidden_dim == loaded_model.d_model * loaded_model.ff_mult
        print("\n✓ Loaded configurations passed validation!")
    except AssertionError as e:
        print(f"\n✗ Validation failed: {e}")
    
    return loaded_model, loaded_training


def demonstrate_safe_modification():
    """Demonstrate safe configuration modification practices."""
    print("\n" + "="*60)
    print("4. SAFE MODIFICATION DEMONSTRATION")
    print("="*60)
    
    # Load original configuration
    config_loader = ConfigLoader('configs/mlm_config.yaml')
    config_loader._resolve_paths()
    original_config = config_loader.get_model_config()
    
    # Save original state
    original_dict = original_config.get_config_dict()
    
    # Create modified versions for experimentation
    experiments = []
    
    # Experiment 1: Larger model
    exp1_dict = original_dict.copy()
    exp1_dict['d_model'] = 1024
    exp1_dict['n_heads'] = 16
    exp1_dict['n_layers'] = 24
    experiments.append(('large_model', exp1_dict))
    
    # Experiment 2: Different attention
    exp2_dict = original_dict.copy()
    exp2_dict['first_block_cfg'] = {
        '_block_type': 'GeometricAttentionConfig'
    }
    experiments.append(('geometric_attention', exp2_dict))
    
    # Save all experiments
    backup_dir = Path('configs/backup/experiments')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    original_path = backup_dir / 'original_mlm.json'
    original_config.save_to_json(str(original_path))
    print(f"Saved original to: {original_path}")
    
    # Save experiments
    for exp_name, exp_dict in experiments:
        # Recreate configuration from modified dictionary
        if exp_dict.get('fsq_encoder_path'):
            # This is a TrunkConfig
            exp_config = TrunkConfig.from_dict(exp_dict)
        else:
            # This is an FSQConfig
            exp_config = FSQConfig.from_dict(exp_dict)
        
        exp_path = backup_dir / f'mlm_{exp_name}.json'
        # We save the dictionary directly since from_dict might not handle all cases
        with open(exp_path, 'w') as f:
            json.dump(exp_dict, f, indent=2)
        print(f"Saved experiment '{exp_name}' to: {exp_path}")


def demonstrate_version_control_friendly():
    """Demonstrate version control friendly practices."""
    print("\n" + "="*60)
    print("5. VERSION CONTROL BEST PRACTICES")
    print("="*60)
    
    # Create a structured backup directory
    today = datetime.now().strftime('%Y%m%d')
    experiment_name = "fsq_baseline"
    
    backup_structure = Path('configs/backup') / today / experiment_name
    backup_structure.mkdir(parents=True, exist_ok=True)
    
    # Load and save all configurations for an experiment
    config_loader = ConfigLoader('configs/fsq_stage1_config.yaml')
    config_loader._resolve_paths()
    
    # Save YAML config
    config_loader.save_config(backup_structure / 'config.yaml')
    
    # Save individual components as JSON
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    
    model_config.save_to_json(str(backup_structure / 'model.json'))
    training_config.save_to_json(str(backup_structure / 'training.json'))
    
    # Create a metadata file
    metadata = {
        'experiment_name': experiment_name,
        'date': today,
        'description': 'Baseline FSQ Stage 1 training configuration',
        'notes': 'Using self-attention with standard hyperparameters',
        'files': {
            'yaml': 'config.yaml',
            'model': 'model.json',
            'training': 'training.json'
        }
    }
    
    with open(backup_structure / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created version-controlled experiment backup:")
    print(f"  {backup_structure}/")
    print(f"  ├── config.yaml      (Complete YAML configuration)")
    print(f"  ├── model.json       (Model configuration)")
    print(f"  ├── training.json    (Training configuration)")
    print(f"  └── metadata.json    (Experiment metadata)")
    
    # Show git commands
    print(f"\nTo add to version control:")
    print(f"  git add {backup_structure}")
    print(f"  git commit -m 'Add {experiment_name} configuration backup'")


def main():
    """Run all safety feature demonstrations."""
    print("\nODYSSEY CONFIGURATION SAFETY FEATURES DEMONSTRATION")
    print("This script shows how to safely manage and backup configurations.")
    
    # 1. Demonstrate automatic backup
    demonstrate_automatic_backup()
    
    # 2. Demonstrate saving to JSON
    model_path, training_path = demonstrate_save_to_json()
    
    # 3. Demonstrate loading from JSON
    demonstrate_load_from_json(model_path, training_path)
    
    # 4. Demonstrate safe modification
    demonstrate_safe_modification()
    
    # 5. Demonstrate version control practices
    demonstrate_version_control_friendly()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Configurations automatically preserve their initial state")
    print("✓ Configurations can be saved to JSON for backup")
    print("✓ Configurations can be loaded from JSON with full validation")
    print("✓ Safe modification through dictionary copies")
    print("✓ Version control friendly structure for experiments")
    print("\nSee configs/backup/ for all generated example files.")


if __name__ == "__main__":
    main()