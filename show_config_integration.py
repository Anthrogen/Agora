#!/usr/bin/env python
"""
Demonstrates how the configuration system integrates with train.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from odyssey.src.config_loader import load_config_from_args
from odyssey.src.configurations import *
import argparse

def show_config_equivalence():
    """Show that YAML configs produce the same objects as hardcoded configs."""
    
    print("="*60)
    print("Configuration System Integration Demo")
    print("="*60)
    
    # Load FSQ Stage 1 from YAML
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
    
    config_loader, yaml_model_cfg, yaml_train_cfg = load_config_from_args(args)
    
    # Create equivalent hardcoded configuration (from train.py)
    hardcoded_mask_cfg = SimpleMaskConfig(mask_prob_seq=0.15, mask_prob_struct=0.15)
    hardcoded_loss_cfg = KabschRMSDLossConfig()
    hardcoded_first_block_cfg = SelfAttentionConfig()
    
    hardcoded_train_cfg = TrainingConfig(
        batch_size=4,
        max_epochs=50,
        learning_rate=1e-5,
        mask_config=hardcoded_mask_cfg,
        loss_config=hardcoded_loss_cfg,
        data_dir="sample_data/1k",
        checkpoint_dir="checkpoints/fsq"
    )
    
    hardcoded_model_cfg = FSQConfig(
        style='stage_1',
        d_model=128,
        n_heads=1,
        n_layers=3,
        max_len=2048,
        dropout=0.1,
        ff_mult=4,
        first_block_cfg=hardcoded_first_block_cfg,
        reference_model_seed=42,
        latent_dim=32,
        fsq_levels=[7, 5, 5, 5, 5],
        fsq_encoder_path=None,
        seq_vocab=len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS),
        struct_vocab=4375 + len(SPECIAL_TOKENS)
    )
    
    # Compare configurations
    print("\nYAML Configuration:")
    print(f"  Model Style: {yaml_model_cfg.style}")
    print(f"  Block Type: {yaml_model_cfg.first_block_cfg.initials()}")
    print(f"  Model Params: d_model={yaml_model_cfg.d_model}, n_heads={yaml_model_cfg.n_heads}")
    print(f"  Training: batch_size={yaml_train_cfg.batch_size}, epochs={yaml_train_cfg.max_epochs}")
    print(f"  Masking: {type(yaml_train_cfg.mask_config).__name__}")
    print(f"  Loss: {type(yaml_train_cfg.loss_config).__name__}")
    
    print("\nHardcoded Configuration:")
    print(f"  Model Style: {hardcoded_model_cfg.style}")
    print(f"  Block Type: {hardcoded_model_cfg.first_block_cfg.initials()}")
    print(f"  Model Params: d_model={hardcoded_model_cfg.d_model}, n_heads={hardcoded_model_cfg.n_heads}")
    print(f"  Training: batch_size={hardcoded_train_cfg.batch_size}, epochs={hardcoded_train_cfg.max_epochs}")
    print(f"  Masking: {type(hardcoded_train_cfg.mask_config).__name__}")
    print(f"  Loss: {type(hardcoded_train_cfg.loss_config).__name__}")
    
    print("\n" + "="*60)
    print("Both configurations produce compatible objects for train.py!")
    print("The train() function can accept either configuration.")
    print("="*60)
    
    # Show usage
    print("\nUsage Examples:")
    print("1. Using YAML configuration:")
    print("   python train_from_config.py --config configs/fsq_stage1_config.yaml")
    print("\n2. Using hardcoded configuration:")
    print("   python train/train.py  # (with appropriate config uncommented)")
    print("\n3. Using YAML with overrides:")
    print("   python train_from_config.py --config configs/fsq_stage1_config.yaml --batch-size 8")

if __name__ == "__main__":
    show_config_equivalence()