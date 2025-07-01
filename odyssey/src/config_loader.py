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

from odyssey.src.configurations import (
    Config, TransformerConfig, TrunkConfig, FSQConfig, TrainingConfig,
    LossConfig, CrossEntropyLossConfig, KabschRMSDLossConfig, ScoreEntropyLossConfig,
    MaskConfig, SimpleMaskConfig, ComplexMaskConfig, NoMaskConfig, DiffusionConfig,
    BlockConfig, SelfConsensusConfig, ReflexiveAttentionConfig, 
    SelfAttentionConfig, GeometricAttentionConfig,
    ConfigurationError
)
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS


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
        
        # Validate required fields
        required_fields = ['style', 'd_model', 'n_heads', 'n_layers', 'max_len', 'dropout', 'ff_mult']
        for field in required_fields:
            if field not in model_cfg:
                raise ValueError(f"Missing required model field: {field}")
        
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
            'first_block_cfg': first_block_config,
            'reference_model_seed': model_cfg.get('reference_model_seed', 42),
            'seq_vocab': seq_vocab,
            'struct_vocab': struct_vocab
        }
        
        # Create appropriate config based on style
        if model_cfg['style'] in ['stage_1', 'stage_2']:
            # FSQ configuration
            if 'fsq' not in model_cfg:
                raise ValueError(f"FSQ parameters required for {model_cfg['style']} style")
            
            fsq_params = model_cfg['fsq']
            if 'latent_dim' not in fsq_params or 'levels' not in fsq_params:
                raise ValueError("FSQ config must include 'latent_dim' and 'levels'")
            
            config = FSQConfig(
                **common_params,
                latent_dim=fsq_params['latent_dim'],
                fsq_levels=fsq_params['levels'],
                fsq_encoder_path=fsq_params.get('encoder_path') if model_cfg['style'] == 'stage_2' else None
            )
        elif model_cfg['style'] in ['mlm', 'discrete_diffusion']:
            # Trunk configuration
            if 'fsq_encoder_path' not in model_cfg:
                raise ValueError(f"fsq_encoder_path required for {model_cfg['style']} style")
            
            seq_absorb_token = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
            struct_absorb_token = SPECIAL_TOKENS.MASK.value + 4375
            config = TrunkConfig(
                **common_params,
                fsq_encoder_path=model_cfg['fsq_encoder_path'],
                seq_absorb_token=seq_absorb_token,
                struct_absorb_token=struct_absorb_token
            )
        else:
            raise ValueError(f"Unknown model style: {model_cfg['style']}")
        
        return config
    
    def get_mask_config(self) -> MaskConfig:
        """Create mask configuration based on strategy."""
        if 'masking' not in self.config:
            raise ValueError("Configuration must include 'masking' section")
        
        mask_cfg = self.config['masking']
        if 'strategy' not in mask_cfg:
            raise ValueError("Masking configuration must specify 'strategy'")
        
        strategy = mask_cfg['strategy']
        
        if strategy == 'simple':
            if 'simple' not in mask_cfg:
                raise ValueError("Simple masking requires 'simple' configuration section")
            simple_cfg = mask_cfg['simple']
            if 'mask_prob_seq' not in simple_cfg or 'mask_prob_struct' not in simple_cfg:
                raise ValueError("Simple masking requires 'mask_prob_seq' and 'mask_prob_struct'")
            
            return SimpleMaskConfig(
                mask_prob_seq=simple_cfg['mask_prob_seq'],
                mask_prob_struct=simple_cfg['mask_prob_struct']
            )
        elif strategy == 'complex':
            return ComplexMaskConfig()
        elif strategy == 'none':
            return NoMaskConfig()
        elif strategy == 'discrete_diffusion':
            if 'discrete_diffusion' not in mask_cfg:
                raise ValueError("Discrete diffusion masking requires 'discrete_diffusion' configuration section")
            
            diff_cfg = mask_cfg['discrete_diffusion']
            required_fields = ['noise_schedule', 'sigma_min', 'sigma_max', 'num_timesteps']
            for field in required_fields:
                if field not in diff_cfg:
                    raise ValueError(f"Discrete diffusion masking requires '{field}'")
            
            return DiffusionConfig(
                noise_schedule=diff_cfg['noise_schedule'],
                sigma_min=diff_cfg['sigma_min'],
                sigma_max=diff_cfg['sigma_max'],
                num_timesteps=diff_cfg['num_timesteps']
            )
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}. Valid options: simple, complex, none, discrete_diffusion")
    
    def get_loss_config(self) -> LossConfig:
        """Create loss configuration."""
        loss_cfg = self.config['loss']
        if 'type' not in loss_cfg:
            raise ValueError("Loss configuration must specify 'type'")
        
        loss_type = loss_cfg['type']
        
        if loss_type == 'cross_entropy':
            if 'weights' not in loss_cfg:
                raise ValueError("Cross-entropy loss requires 'weights' section")
            if 'loss_elements' not in loss_cfg:
                raise ValueError("Cross-entropy loss requires 'loss_elements'")
            
            weights = loss_cfg['weights']
            if 'sequence' not in weights or 'structure' not in weights:
                raise ValueError("Loss weights must include 'sequence' and 'structure'")
            
            return CrossEntropyLossConfig(
                seq_loss_weight=weights['sequence'],
                struct_loss_weight=weights['structure'],
                loss_elements=loss_cfg['loss_elements']
            )
        elif loss_type == 'kabsch_rmsd':
            return KabschRMSDLossConfig()
        elif loss_type == 'score_entropy':
            if 'weights' not in loss_cfg:
                raise ValueError("Score entropy loss requires 'weights' section")
            
            weights = loss_cfg['weights']
            if 'sequence' not in weights or 'structure' not in weights:
                raise ValueError("Loss weights must include 'sequence' and 'structure'")
            
            return ScoreEntropyLossConfig(
                seq_loss_weight=weights['sequence'],
                struct_loss_weight=weights['structure']
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Valid options: cross_entropy, kabsch_rmsd, score_entropy")
    
    def get_training_config(self) -> TrainingConfig:
        """Create training configuration object."""
        if 'training' not in self.config:
            raise ValueError("Configuration must include 'training' section")
        
        train_cfg = self.config['training']
        
        # Validate required fields
        required_fields = ['batch_size', 'max_epochs', 'learning_rate', 'data_dir', 'checkpoint_dir']
        for field in required_fields:
            if field not in train_cfg:
                raise ValueError(f"Missing required training field: {field}")
        
        # Get mask and loss configurations
        mask_config = self.get_mask_config()
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
    
    def validate_config_consistency(self) -> None:
        """Validate that configuration combinations make sense."""
        model_style = self.config['model']['style']
        loss_type = self.config['loss']['type']
        mask_strategy = self.config['masking']['strategy']
        
        # Validate loss type for model style
        if model_style in ['stage_1', 'stage_2'] and loss_type != 'kabsch_rmsd':
            print(f"Warning: FSQ models typically use 'kabsch_rmsd' loss, but got '{loss_type}'")
        
        if model_style == 'mlm' and loss_type != 'cross_entropy':
            print(f"Warning: MLM models typically use 'cross_entropy' loss, but got '{loss_type}'")
        
        if model_style == 'discrete_diffusion':
            if loss_type != 'score_entropy':
                print(f"Warning: Discrete diffusion models should use 'score_entropy' loss, but got '{loss_type}'")
            if mask_strategy != 'discrete_diffusion':
                print(f"Warning: Discrete diffusion models should use 'discrete_diffusion' masking, but got '{mask_strategy}'")
        
        # Validate stage 2 requirements
        if model_style == 'stage_2':
            if mask_strategy != 'none':
                print(f"Warning: Stage 2 training typically uses 'none' masking, but got '{mask_strategy}'")
            fsq_params = self.config['model'].get('fsq', {})
            if 'encoder_path' not in fsq_params or not fsq_params['encoder_path']:
                raise ValueError("Stage 2 training requires fsq.encoder_path to be specified")
    
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
    
    # Validate configuration consistency
    config_loader.validate_config_consistency()
    
    # Print config if requested
    if args.print_config:
        config_loader.print_config()
        exit(0)
    
    # Save config if requested
    if args.save_config:
        config_loader.save_config(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
    
    # Create configuration objects
    try:
        model_config = config_loader.get_model_config()
        training_config = config_loader.get_training_config()
    except Exception as e:
        print(f"\nError creating configuration: {e}")
        print("\nPlease check your configuration file follows the correct format.")
        print("See configs/configuration_constructor.md for detailed documentation.")
        raise
    
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