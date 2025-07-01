#!/usr/bin/env python
"""
Training script that uses YAML configuration files.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from odyssey.src.config_loader import load_config_from_args
from train.train import train

def main():
    # Load configuration from YAML and command-line args
    config_loader, model_cfg, train_cfg = load_config_from_args()
    
    # Print configuration summary
    print("="*60)
    print("Configuration Summary:")
    print("="*60)
    print(f"Model Style: {model_cfg.style}")
    print(f"Model Type: {model_cfg.first_block_cfg.initials()} - {model_cfg.first_block_cfg}")
    print(f"Model Params: d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, n_layers={model_cfg.n_layers}")
    print(f"Training: batch_size={train_cfg.batch_size}, epochs={train_cfg.max_epochs}, lr={train_cfg.learning_rate}")
    print(f"Masking: {train_cfg.mask_config}")
    print(f"Loss: {type(train_cfg.loss_config).__name__}")
    print("="*60)
    
    # Run training
    model, history = train(model_cfg, train_cfg)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()