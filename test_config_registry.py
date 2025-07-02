#!/usr/bin/env python3
"""Test the configuration registry system without YAML dependencies."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.configurations import CONFIG_REGISTRY, Config

def test_registry():
    """Test that all expected configuration types are registered."""
    print("Configuration Registry Test")
    print("=" * 50)
    
    expected_types = [
        # Model configs
        'fsq_cfg', 'trunk_cfg',
        # Block configs
        'self_attention_cfg', 'geometric_attention_cfg', 
        'reflexive_attention_cfg', 'self_consensus_cfg',
        # Training config
        'training_cfg',
        # Loss configs
        'cross_entropy_loss_cfg', 'kabsch_rmsd_loss_cfg', 'score_entropy_loss_cfg',
        # Mask configs
        'simple_mask_cfg', 'complex_mask_cfg', 'diffusion_mask_cfg', 'no_mask_cfg'
    ]
    
    print(f"\nFound {len(CONFIG_REGISTRY)} registered configuration types:")
    for config_type in sorted(CONFIG_REGISTRY.keys()):
        config_class = CONFIG_REGISTRY[config_type]
        if config_type in expected_types:
            print(f"  ✓ {config_type:<30} -> {config_class.__name__}")
        else:
            print(f"  ? {config_type:<30} -> {config_class.__name__} (unexpected)")
    
    missing = set(expected_types) - set(CONFIG_REGISTRY.keys())
    if missing:
        print(f"\nMissing expected types:")
        for config_type in missing:
            print(f"  ✗ {config_type}")
    else:
        print(f"\n✓ All expected types are registered!")
    
    return len(missing) == 0

def test_config_building():
    """Test building configurations from type/params structure."""
    print("\n\nConfiguration Building Test")
    print("=" * 50)
    
    # Simulate what the config loader does
    def build_from_dict(config_dict):
        if 'type' in config_dict and 'params' in config_dict:
            config_type = config_dict['type']
            params = config_dict['params']
            
            if config_type not in CONFIG_REGISTRY:
                raise ValueError(f"Unknown type: {config_type}")
            
            config_class = CONFIG_REGISTRY[config_type]
            
            # Recursively build nested configs
            built_params = {}
            for key, value in params.items():
                if isinstance(value, dict) and 'type' in value and 'params' in value:
                    built_params[key] = build_from_dict(value)
                else:
                    built_params[key] = value
            
            return config_class(**built_params)
        return config_dict
    
    # Test building a simple mask config
    print("\n1. Testing SimpleMaskConfig:")
    mask_dict = {
        'type': 'simple_mask_cfg',
        'params': {
            'mask_prob_seq': 0.15,
            'mask_prob_struct': 0.15
        }
    }
    try:
        mask_cfg = build_from_dict(mask_dict)
        print(f"   ✓ Built {type(mask_cfg).__name__}")
        print(f"     - mask_prob_seq: {mask_cfg.mask_prob_seq}")
        print(f"     - mask_prob_struct: {mask_cfg.mask_prob_struct}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test building a nested config (training with loss and mask)
    print("\n2. Testing TrainingConfig with nested configs:")
    train_dict = {
        'type': 'training_cfg',
        'params': {
            'batch_size': 32,
            'max_epochs': 100,
            'learning_rate': 1e-4,
            'data_dir': 'sample_data',
            'checkpoint_dir': 'checkpoints',
            'loss_config': {
                'type': 'kabsch_rmsd_loss_cfg',
                'params': {}
            },
            'mask_config': {
                'type': 'simple_mask_cfg',
                'params': {
                    'mask_prob_seq': 0.2,
                    'mask_prob_struct': 0.2
                }
            }
        }
    }
    try:
        train_cfg = build_from_dict(train_dict)
        print(f"   ✓ Built {type(train_cfg).__name__}")
        print(f"     - batch_size: {train_cfg.batch_size}")
        print(f"     - loss_config: {type(train_cfg.loss_config).__name__}")
        print(f"     - mask_config: {type(train_cfg.mask_config).__name__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test building FSQ model config
    print("\n3. Testing FSQConfig with nested block config:")
    model_dict = {
        'type': 'fsq_cfg',
        'params': {
            'style': 'stage_1',
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 6,
            'max_len': 1024,
            'dropout': 0.1,
            'ff_mult': 4,
            'reference_model_seed': 42,
            'first_block_cfg': {
                'type': 'self_consensus_cfg',
                'params': {
                    'num_iterations': 1,
                    'connectivity_type': 'local_window',
                    'w': 2,
                    'r': 8,
                    'edge_hidden_dim': 12
                }
            },
            'latent_dim': 32,
            'fsq_levels': [7, 5, 5, 5, 5],
            'fsq_encoder_path': None
        }
    }
    try:
        model_cfg = build_from_dict(model_dict)
        print(f"   ✓ Built {type(model_cfg).__name__}")
        print(f"     - style: {model_cfg.style}")
        print(f"     - d_model: {model_cfg.d_model}")
        print(f"     - first_block: {type(model_cfg.first_block_cfg).__name__}")
        print(f"     - latent_dim: {model_cfg.latent_dim}")
        
        # Test dictionary backup
        config_dict = model_cfg.get_config_dict()
        print(f"     - Backup dict has {len(config_dict)} keys")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run tests
    registry_ok = test_registry()
    test_config_building()
    
    print("\n" + "=" * 50)
    if registry_ok:
        print("✓ Configuration registry system is working correctly!")
    else:
        print("✗ Configuration registry has issues")
    print("=" * 50)