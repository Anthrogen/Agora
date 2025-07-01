#!/usr/bin/env python
"""Test configuration loading with different config files."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.config_loader import load_config_from_args
import argparse

def test_config(config_path, name):
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
    
    try:
        config_loader, model_cfg, train_cfg = load_config_from_args(args)
        print(f"{name} Config Loaded Successfully!")
        print(f"  Model: {model_cfg.style} with {model_cfg.first_block_cfg.initials()}")
        print(f"  Training: batch_size={train_cfg.batch_size}, epochs={train_cfg.max_epochs}")
        if hasattr(model_cfg, 'fsq_encoder_path') and model_cfg.fsq_encoder_path:
            print(f"  FSQ Encoder: {model_cfg.fsq_encoder_path}")
        print()
    except Exception as e:
        print(f"ERROR loading {name}: {e}")
        print()

if __name__ == "__main__":
    # Test different configurations
    test_config('configs/default_config.yaml', 'Default')
    test_config('configs/fsq_stage1_config.yaml', 'FSQ Stage 1')
    test_config('configs/fsq_stage2_config.yaml', 'FSQ Stage 2')