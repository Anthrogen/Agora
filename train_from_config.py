#!/usr/bin/env python
"""
Main training script that uses YAML configuration files.
This replaces the hardcoded configurations in train.py with YAML-based configs.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.config_loader import load_config_from_args
from train.train import train

def main():
    """Load configuration and run training."""
    # Load configuration from YAML file and command-line arguments
    config_loader, model_cfg, train_cfg = load_config_from_args()
    
    # Run the training
    model, epoch_metrics = train(model_cfg, train_cfg)
    
    print("\nTraining completed successfully!")
    return model, epoch_metrics

if __name__ == "__main__":
    main()