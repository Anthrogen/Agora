"""
Configuration loader for YAML-based experiment configuration.

This module provides utilities to load and validate configuration files,
merge configurations with command-line arguments, and create configuration
objects for different components of the system.
"""

import yaml
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json
from copy import deepcopy

from src.configurations import (
    TransformerConfig, FsqConfig, TrainingConfig, 
    DiffusionConfig, ConfigurationError
)
from src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS


class ConfigLoader:
    """Handles loading and processing of YAML configuration files."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the configuration
        self.config = self._load_yaml()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _resolve_paths(self, base_dir: Optional[Path] = None) -> None:
        """
        Resolve relative paths in configuration to absolute paths.
        
        Args:
            base_dir: Base directory for relative paths. If None, uses config file directory.
        """
        if base_dir is None:
            base_dir = self.config_path.parent.parent  # Assuming config is in configs/
        
        # Resolve data paths
        if 'data' in self.config:
            for key in ['data_dir', 'csv_file']:
                if key in self.config['data'] and self.config['data'][key]:
                    path = Path(self.config['data'][key])
                    if not path.is_absolute():
                        self.config['data'][key] = str(base_dir / path)
        
        # Resolve checkpoint directory
        if 'checkpoint' in self.config and 'dir' in self.config['checkpoint']:
            path = Path(self.config['checkpoint']['dir'])
            if not path.is_absolute():
                self.config['checkpoint']['dir'] = str(base_dir / path)
    
    def get_model_config(self, model_type: Optional[str] = None) -> Union[TransformerConfig, FsqConfig]:
        """
        Create model configuration object.
        
        Args:
            model_type: Override model type from config
            
        Returns:
            TransformerConfig or FsqConfig depending on context
        """
        model_cfg = self.config['model']
        
        # Override model type if specified
        if model_type:
            model_cfg['type'] = model_type
        
        # Calculate vocabulary sizes
        seq_vocab = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)
        struct_vocab = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
        
        # Create FSQ configuration
        if 'fsq' in model_cfg:
            fsq_cfg = FsqConfig(
                fsq_dim=model_cfg['fsq']['dim'],
                fsq_levels=model_cfg['fsq']['levels'],
                model_type=model_cfg['type'],
                d_model=model_cfg['d_model'],
                latent_dim=model_cfg['fsq']['latent_dim'],
                n_heads=model_cfg['n_heads'],
                n_layers=model_cfg['n_layers'],
                seq_vocab=seq_vocab,
                struct_vocab=struct_vocab,
                max_len=model_cfg['max_len'],
                dropout=model_cfg['dropout'],
                ff_mult=model_cfg['ff_mult'],
                consensus_num_iterations=model_cfg['consensus']['num_iterations'],
                consensus_connectivity_type=model_cfg['consensus']['connectivity_type'],
                consensus_w=model_cfg['consensus']['w'],
                consensus_r=model_cfg['consensus']['r'],
                consensus_edge_hidden_dim=model_cfg['consensus']['edge_hidden_dim'],
                stage=self.config['training'].get('stage', 'stage_1')
            )
            return fsq_cfg
        
        # Create standard transformer configuration
        return TransformerConfig(
            d_model=model_cfg['d_model'],
            n_heads=model_cfg['n_heads'],
            n_layers=model_cfg['n_layers'],
            seq_vocab=seq_vocab,
            struct_vocab=struct_vocab,
            max_len=model_cfg['max_len'],
            dropout=model_cfg['dropout'],
            ff_mult=model_cfg['ff_mult'],
            consensus_num_iterations=model_cfg['consensus']['num_iterations'],
            consensus_connectivity_type=model_cfg['consensus']['connectivity_type'],
            consensus_w=model_cfg['consensus']['w'],
            consensus_r=model_cfg['consensus']['r'],
            consensus_edge_hidden_dim=model_cfg['consensus']['edge_hidden_dim']
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Create training configuration object."""
        train_cfg = self.config['training']
        mask_cfg = self.config['masking']
        checkpoint_cfg = self.config['checkpoint']
        
        # Build checkpoint patterns
        patterns = checkpoint_cfg['patterns']
        
        # Create training config
        config = TrainingConfig(
            model_types=[self.config['model']['type']],  # Can be extended for multi-model training
            batch_size=train_cfg['batch_size'],
            max_epochs=train_cfg['max_epochs'],
            learning_rate=train_cfg['learning_rate'],
            num_iter=train_cfg['num_iterations'],
            masking_strategy=mask_cfg['strategy'],
            data_dir=self.config['data']['data_dir'],
            checkpoint_dir=checkpoint_cfg['dir'],
            reference_model_seed=train_cfg.get('reference_model_seed', 1234),
            seq_loss_weight=train_cfg['loss_weights']['sequence'],
            struct_loss_weight=train_cfg['loss_weights']['structure'],
            ce_loss_function_elements=train_cfg.get('ce_loss_elements', 'masked'),
            simple_checkpoint_pattern=patterns['simple'],
            complex_checkpoint_pattern=patterns['complex'],
            discrete_diffusion_checkpoint_pattern=patterns['discrete_diffusion'],
            fsq_encoder_pattern=patterns['fsq_encoder']
        )
        
        # Add masking-specific parameters
        if mask_cfg['strategy'] == 'simple':
            config.mask_prob_seq = mask_cfg['simple']['mask_prob_seq']
            config.mask_prob_coords = mask_cfg['simple']['mask_prob_coords']
        
        return config
    
    def get_diffusion_config(self) -> Optional[DiffusionConfig]:
        """Create diffusion configuration object if using discrete diffusion."""
        if self.config['masking']['strategy'] != 'discrete_diffusion':
            return None
        
        diff_cfg = self.config['masking']['discrete_diffusion']
        
        return DiffusionConfig(
            noise_schedule=diff_cfg['noise_schedule'],
            sigma_min=diff_cfg['sigma_min'],
            sigma_max=diff_cfg['sigma_max'],
            num_timesteps=diff_cfg['num_timesteps'],
            seq_absorb_token=diff_cfg['seq_absorb_token'],
            struct_absorb_token=diff_cfg['struct_absorb_token']
        )
    
    def merge_with_args(self, args: argparse.Namespace) -> None:
        """
        Merge command-line arguments with configuration.
        Command-line args take precedence.
        
        Args:
            args: Parsed command-line arguments
        """
        # Override model type
        if hasattr(args, 'model_type') and args.model_type:
            self.config['model']['type'] = args.model_type
        
        # Override batch size
        if hasattr(args, 'batch_size') and args.batch_size:
            self.config['training']['batch_size'] = args.batch_size
        
        # Override learning rate
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.config['training']['learning_rate'] = args.learning_rate
        
        # Override epochs
        if hasattr(args, 'epochs') and args.epochs:
            self.config['training']['max_epochs'] = args.epochs
        
        # Override device
        if hasattr(args, 'device') and args.device:
            self.config['experiment']['device'] = args.device
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        print("="*60)
        print("Configuration:")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, sort_keys=False))
        print("="*60)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration override."""
    parser = argparse.ArgumentParser(description="Odyssey Training Configuration")
    
    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file (default: configs/default_config.yaml)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['SA', 'GA', 'RA', 'SC'],
        help='Model type to train (overrides config file)'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config file)'
    )
    
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate (overrides config file)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Maximum number of epochs (overrides config file)'
    )
    
    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use for training (overrides config file)'
    )
    
    # Other options
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print configuration and exit'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save effective configuration to specified path'
    )
    
    return parser


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> tuple:
    """
    Load configuration from command-line arguments.
    
    Args:
        args: Parsed arguments. If None, will parse from sys.argv
        
    Returns:
        Tuple of (config_loader, model_config, training_config, diffusion_config)
    """
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config_loader._resolve_paths()
    
    # Merge with command-line arguments
    config_loader.merge_with_args(args)
    
    # Print config if requested
    if args.print_config:
        config_loader.print_config()
        exit(0)
    
    # Save config if requested
    if args.save_config:
        config_loader.save_config(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
    
    # Create configuration objects
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    diffusion_config = config_loader.get_diffusion_config()
    
    return config_loader, model_config, training_config, diffusion_config


# Example usage in training scripts:
if __name__ == "__main__":
    # Example of how to use in a training script
    config_loader, model_cfg, train_cfg, diff_cfg = load_config_from_args()
    
    print("Model type:", model_cfg.model_type if hasattr(model_cfg, 'model_type') else 'N/A')
    print("Batch size:", train_cfg.batch_size)
    print("Learning rate:", train_cfg.learning_rate)
    print("Masking strategy:", train_cfg.masking_strategy)