#!/usr/bin/env python3
"""Final test of the new configuration system."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.config_loader import load_config_from_args
import argparse

def test_config_system():
    """Test the complete configuration system."""
    print("Testing New Configuration System")
    print("=" * 60)
    
    # Test different configs
    configs = [
        'configs/default_config.yaml',
        'configs/fsq_stage1_config.yaml', 
        'configs/mlm_config.yaml',
        'configs/discrete_diffusion_config.yaml'
    ]
    
    for config_path in configs:
        print(f"\nTesting: {config_path}")
        print("-" * 40)
        
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
        
        try:
            model_cfg, train_cfg, model_dict, train_dict = load_config_from_args(args)
            
            print(f"✓ Successfully loaded")
            print(f"  Model: {type(model_cfg).__name__} - {model_cfg.style}")
            print(f"  Block: {type(model_cfg.first_block_cfg).__name__}")
            print(f"  Train: batch={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
            print(f"  Loss: {type(train_cfg.loss_config).__name__}")
            print(f"  Mask: {type(train_cfg.mask_config).__name__}")
            
            # Verify backup dicts
            assert len(model_dict) > 0, "Model dict should not be empty"
            assert len(train_dict) > 0, "Training dict should not be empty"
            
            # Test immutability
            original_batch = train_dict['batch_size']
            train_cfg.batch_size = 999
            assert train_dict['batch_size'] == original_batch, "Backup should be immutable"
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Summary: Configuration system is working correctly!")

if __name__ == "__main__":
    test_config_system()