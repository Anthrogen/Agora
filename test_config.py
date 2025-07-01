#!/usr/bin/env python
"""Simple test to verify config loader works correctly."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'odyssey'))

from src.config_loader import load_config_from_args

# Test loading configuration
config_loader, model_cfg, train_cfg = load_config_from_args()

print("Configuration loaded successfully!")
print(f"\nModel Configuration:")
print(f"  Type: {type(model_cfg).__name__}")
print(f"  Style: {model_cfg.style}")

print(f"\nTraining Configuration:")
print(f"  Batch size: {train_cfg.batch_size}")
print(f"  Mask config: {type(train_cfg.mask_config).__name__}")
print(f"  Loss config: {type(train_cfg.loss_config).__name__}")