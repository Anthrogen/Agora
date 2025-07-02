#!/usr/bin/env python3
"""Test the new registry-based configuration loading system."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.config_loader import load_config_from_args
from odyssey.src.configurations import CONFIG_REGISTRY
import argparse

def test_config(config_path, name, override_args=None):
    """Test loading a specific configuration."""
    args = argparse.Namespace(
        config=config_path,
        print_config=False,
        save_config=None,
        style=None,
        batch_size=None,
        learning_rate=None,
        epochs=None,
        device=None
    )
    
    # Apply any overrides
    if override_args:
        for key, value in override_args.items():
            setattr(args, key, value)
    
    try:
        # New loader returns 4 values
        model_cfg, train_cfg, model_dict, train_dict = load_config_from_args(args)
        
        print(f"✓ {name} Config Loaded Successfully!")
        print(f"  Model Type: {type(model_cfg).__name__}")
        print(f"  Model Style: {model_cfg.style}")
        print(f"  First Block: {type(model_cfg.first_block_cfg).__name__}")
        print(f"  Training: batch_size={train_cfg.batch_size}, epochs={train_cfg.max_epochs}, lr={train_cfg.learning_rate}")
        print(f"  Loss Type: {type(train_cfg.loss_config).__name__}")
        print(f"  Mask Type: {type(train_cfg.mask_config).__name__}")
        
        # Check FSQ-specific fields
        if hasattr(model_cfg, 'latent_dim'):
            print(f"  FSQ: latent_dim={model_cfg.latent_dim}, levels={model_cfg.fsq_levels}")
        if hasattr(model_cfg, 'fsq_encoder_path') and model_cfg.fsq_encoder_path:
            print(f"  FSQ Encoder: {model_cfg.fsq_encoder_path}")
            
        # Verify backup dictionaries
        print(f"  Backup Dicts: model={len(model_dict)} keys, train={len(train_dict)} keys")
        
        # Test that backup is immutable
        original_batch_size = model_dict.get('batch_size') or train_dict.get('batch_size')
        if original_batch_size:
            train_cfg.batch_size = 999  # Modify the object
            backup_batch_size = train_dict.get('batch_size')
            if backup_batch_size and backup_batch_size != 999:
                print(f"  ✓ Backup preserved original batch_size: {backup_batch_size}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ ERROR loading {name}: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_command_line_overrides():
    """Test that command-line arguments override config values."""
    print("Testing Command-Line Overrides:")
    print("-" * 50)
    
    # Test batch size override
    success = test_config(
        'configs/default_config.yaml', 
        'Default with batch_size=64',
        {'batch_size': 64}
    )
    
    # Test learning rate override
    success = test_config(
        'configs/fsq_stage1_config.yaml',
        'FSQ Stage 1 with lr=1e-3',
        {'learning_rate': 1e-3}
    )
    
    # Test style override (this is more complex as it requires compatible config)
    # Skip for now as it would require the YAML to be compatible
    
    print()

def test_registry():
    """Test that all expected configuration types are registered."""
    print("Configuration Registry:")
    print("-" * 50)
    
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
    
    print(f"Found {len(CONFIG_REGISTRY)} registered types:")
    for config_type in sorted(CONFIG_REGISTRY.keys()):
        if config_type in expected_types:
            print(f"  ✓ {config_type}")
        else:
            print(f"  ? {config_type} (unexpected)")
    
    missing = set(expected_types) - set(CONFIG_REGISTRY.keys())
    if missing:
        print(f"\nMissing expected types:")
        for config_type in missing:
            print(f"  ✗ {config_type}")
    
    print()

if __name__ == "__main__":
    print("Testing New Configuration System")
    print("=" * 50)
    print()
    
    # Test the registry
    test_registry()
    
    # Test different configurations
    print("Testing Configuration Files:")
    print("-" * 50)
    
    configs_to_test = [
        ('configs/default_config.yaml', 'Default'),
        ('configs/fsq_stage1_config.yaml', 'FSQ Stage 1'),
        ('configs/fsq_stage2_config.yaml', 'FSQ Stage 2'),
        ('configs/mlm_config.yaml', 'MLM'),
        ('configs/discrete_diffusion_config.yaml', 'Discrete Diffusion'),
        ('configs/fsq_stage1_geometric.yaml', 'FSQ Stage 1 w/ Geometric'),
        ('configs/mlm_reflexive.yaml', 'MLM w/ Reflexive'),
    ]
    
    success_count = 0
    for config_path, name in configs_to_test:
        if test_config(config_path, name):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(configs_to_test)} configs loaded successfully")
    
    # Test command-line overrides
    print("\n")
    test_command_line_overrides()
    
    print("\nTest complete!")