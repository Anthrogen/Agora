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
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import json
from copy import deepcopy

from src.configurations import (
    Config, TransformerConfig, TrunkConfig, FSQConfig, TrainingConfig,
    LossConfig, CrossEntropyLossConfig, KabschRMSDLossConfig,
    MaskConfig, SimpleMaskConfig, ComplexMaskConfig, NoMaskConfig, DiffusionConfig,
    BlockConfig, SelfConsensusConfig, ReflexiveAttentionConfig, 
    SelfAttentionConfig, GeometricAttentionConfig,
    ConfigurationError
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
        if 'training' in self.config and 'data_dir' in self.config['training']:
            path = Path(self.config['training']['data_dir'])
            if not path.is_absolute():
                self.config['training']['data_dir'] = str(base_dir / path)
        
        # Resolve checkpoint directory
        if 'training' in self.config and 'checkpoint_dir' in self.config['training']:
            path = Path(self.config['training']['checkpoint_dir'])
            if not path.is_absolute():
                self.config['training']['checkpoint_dir'] = str(base_dir / path)
        
        # Resolve FSQ encoder path if present
        if 'model' in self.config:
            if 'fsq_encoder_path' in self.config['model'] and self.config['model']['fsq_encoder_path']:
                path = Path(self.config['model']['fsq_encoder_path'])
                if not path.is_absolute():
                    self.config['model']['fsq_encoder_path'] = str(base_dir / path)
            
            # Also check in fsq section for stage_2
            if 'fsq' in self.config['model'] and 'encoder_path' in self.config['model']['fsq']:
                if self.config['model']['fsq']['encoder_path']:
                    path = Path(self.config['model']['fsq']['encoder_path'])
                    if not path.is_absolute():
                        self.config['model']['fsq']['encoder_path'] = str(base_dir / path)
    
    def get_block_config(self, block_type: str, block_params: Dict[str, Any]) -> BlockConfig:
        """
        Create block configuration based on type.
        
        Args:
            block_type: Type of block (self_attention, geometric_attention, etc.)
            block_params: Parameters for the block
            
        Returns:
            Appropriate BlockConfig subclass
        """
        if block_type == "self_consensus":
            return SelfConsensusConfig(
                consensus_num_iterations=block_params['num_iterations'],
                consensus_connectivity_type=block_params['connectivity_type'],
                consensus_w=block_params['w'],
                consensus_r=block_params['r'],
                consensus_edge_hidden_dim=block_params['edge_hidden_dim']
            )
        elif block_type == "reflexive_attention":
            return ReflexiveAttentionConfig()
        elif block_type == "self_attention":
            return SelfAttentionConfig()
        elif block_type == "geometric_attention":
            return GeometricAttentionConfig()
        else:
            raise ValueError(f"Unknown block type: {block_type}")
    
    def get_model_config(self) -> Union[TransformerConfig, TrunkConfig, FSQConfig]:
        """
        Create model configuration object based on style.
        
        Returns:
            Appropriate model configuration
        """
        model_cfg = self.config['model']
        
        # Calculate vocabulary sizes
        seq_vocab = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)
        struct_vocab = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
        
        # Get block configuration
        block_type = model_cfg.get('block_type', 'self_attention')
        block_params = model_cfg.get('block_params', {})
        first_block_config = self.get_block_config(block_type, block_params)
        
        # Common parameters
        common_params = {
            'style': model_cfg['style'],
            'd_model': model_cfg['d_model'],
            'n_heads': model_cfg['n_heads'],
            'n_layers': model_cfg['n_layers'],
            'max_len': model_cfg['max_len'],
            'dropout': model_cfg['dropout'],
            'ff_mult': model_cfg['ff_mult'],
            'first_block_config': first_block_config,
            'seq_vocab': seq_vocab,
            'struct_vocab': struct_vocab
        }
        
        # Create appropriate config based on style
        if model_cfg['style'] in ['stage_1', 'stage_2']:
            # FSQ configuration
            fsq_params = model_cfg.get('fsq', {})
            config = FSQConfig(
                **common_params,
                latent_dim=fsq_params.get('latent_dim', 32),
                fsq_levels=fsq_params.get('levels', [7, 5, 5, 5, 5]),
                fsq_encoder_path=fsq_params.get('encoder_path') if model_cfg['style'] == 'stage_2' else None
            )
        elif model_cfg['style'] in ['mlm', 'discrete_diffusion']:
            # Trunk configuration
            config = TrunkConfig(
                **common_params,
                fsq_encoder_path=model_cfg.get('fsq_encoder_path')
            )
        else:
            # Base transformer configuration
            config = TransformerConfig(**common_params)
        
        return config
    
    def get_mask_config(self) -> MaskConfig:
        """Create mask configuration based on strategy."""
        mask_cfg = self.config['masking']
        strategy = mask_cfg['strategy']
        
        if strategy == 'simple':
            return SimpleMaskConfig(
                mask_prob_seq=mask_cfg['simple']['mask_prob_seq'],
                mask_prob_struct=mask_cfg['simple']['mask_prob_struct']
            )
        elif strategy == 'complex':
            return ComplexMaskConfig()
        elif strategy == 'none':
            return NoMaskConfig()
        elif strategy == 'discrete_diffusion':
            diff_cfg = mask_cfg['discrete_diffusion']
            return DiffusionConfig(
                noise_schedule=diff_cfg['noise_schedule'],
                sigma_min=diff_cfg['sigma_min'],
                sigma_max=diff_cfg['sigma_max'],
                num_timesteps=diff_cfg['num_timesteps']
            )
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")
    
    def get_loss_config(self) -> LossConfig:
        """Create loss configuration."""
        loss_cfg = self.config['loss']
        loss_type = loss_cfg['type']
        
        if loss_type == 'cross_entropy':
            return CrossEntropyLossConfig(
                seq_loss_weight=loss_cfg['weights']['sequence'],
                struct_loss_weight=loss_cfg['weights']['structure'],
                loss_elements=loss_cfg['loss_elements']
            )
        elif loss_type == 'kabsch_rmsd':
            return KabschRMSDLossConfig(
                rmsd_elements=loss_cfg['rmsd_elements']
            )
        elif loss_type == 'diffusion':
            # Diffusion loss is handled as part of DiffusionConfig
            return self.get_mask_config()  # Returns DiffusionConfig which is also a LossConfig
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_training_config(self) -> TrainingConfig:
        """Create training configuration object."""
        train_cfg = self.config['training']
        
        # Get mask and loss configurations
        mask_config = self.get_mask_config()
        
        # For diffusion, the mask config is also the loss config
        if isinstance(mask_config, DiffusionConfig):
            loss_config = mask_config
        else:
            loss_config = self.get_loss_config()
        
        config = TrainingConfig(
            batch_size=train_cfg['batch_size'],
            max_epochs=train_cfg['max_epochs'],
            learning_rate=float(train_cfg['learning_rate']),
            mask_config=mask_config,
            loss_config=loss_config,
            data_dir=train_cfg['data_dir'],
            checkpoint_dir=train_cfg['checkpoint_dir']
        )
        
        return config
    
    def merge_with_args(self, args: argparse.Namespace) -> None:
        """
        Merge command-line arguments with configuration.
        Command-line args take precedence.
        
        Args:
            args: Parsed command-line arguments
        """
        # Override model style
        if hasattr(args, 'style') and args.style:
            self.config['model']['style'] = args.style
        
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
        '--style',
        type=str,
        choices=['stage_1', 'stage_2', 'mlm', 'discrete_diffusion'],
        help='Model style (overrides config file)'
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
        Tuple of (config_loader, model_config, training_config)
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
    
    return config_loader, model_config, training_config


# Example usage in training scripts:
if __name__ == "__main__":
    # Example of how to use in a training script
    config_loader, model_cfg, train_cfg = load_config_from_args()
    
    print("Model style:", model_cfg.style)
    print("Batch size:", train_cfg.batch_size)
    print("Learning rate:", train_cfg.learning_rate)
    print("Mask config type:", type(train_cfg.mask_config).__name__)
    print("Loss config type:", type(train_cfg.loss_config).__name__)