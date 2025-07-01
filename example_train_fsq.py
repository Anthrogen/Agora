#!/usr/bin/env python
"""Example of how to use configuration files for FSQ training."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.config_loader import load_config_from_args
import argparse

def main():
    # Example 1: Use default arguments (will load configs/default_config.yaml)
    print("Example 1: Loading default configuration")
    config_loader, model_cfg, train_cfg = load_config_from_args()
    print(f"Loaded: {model_cfg.style} with {model_cfg.first_block_cfg.initials()}")
    print()
    
    # Example 2: Load specific config file
    print("Example 2: Loading FSQ Stage 1 configuration")
    args = argparse.Namespace(
        config='configs/fsq_stage1_config.yaml',
        print_config=False,
        save_config=None,
        style=None,
        batch_size=None,
        learning_rate=None,
        epochs=None,
        device=None
    )
    config_loader, model_cfg, train_cfg = load_config_from_args(args)
    print(f"Loaded: {model_cfg.style} with {model_cfg.first_block_cfg.initials()}")
    print(f"FSQ levels: {model_cfg.fsq_levels}")
    print()
    
    # Example 3: Override configuration with command-line args
    print("Example 3: Override batch size and learning rate")
    args.batch_size = 8
    args.learning_rate = 5e-5
    config_loader, model_cfg, train_cfg = load_config_from_args(args)
    print(f"Original config had batch_size=4, lr=1e-5")
    print(f"After override: batch_size={train_cfg.batch_size}, lr={train_cfg.learning_rate}")
    print()
    
    # Example 4: Print full configuration
    print("Example 4: Print configuration")
    args.print_config = True
    # This will print the config and exit, so we wrap it
    try:
        config_loader, model_cfg, train_cfg = load_config_from_args(args)
    except SystemExit:
        pass
    
    print("\nTo use with train.py:")
    print("from train.train import train")
    print("model, history = train(model_cfg, train_cfg)")

if __name__ == "__main__":
    main()