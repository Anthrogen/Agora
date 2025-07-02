"""
Configuration loader for YAML-based experiment configuration.

This module provides utilities to load and validate configuration files using
the new registry-based type/params system for automatic configuration building.
"""

import yaml
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
import json
from copy import deepcopy

from odyssey.src.configurations import (
    Config, CONFIG_REGISTRY,
    ConfigurationError
)
from odyssey.src.vocabulary import SEQUENCE_TOKENS, SPECIAL_TOKENS


class ConfigLoader:
    """Handles loading and processing of YAML configuration files using the registry system."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the raw YAML configuration
        self.raw_config = self._load_yaml()
        
        # Build configuration objects from the raw config
        self.model_config = None
        self.training_config = None
        self._build_configs()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_config_from_dict(self, config_dict: Dict[str, Any]) -> Any:
        """
        Recursively build configuration objects from dictionary with type/params structure.
        
        Args:
            config_dict: Dictionary with 'type' and 'params' fields
            
        Returns:
            Built configuration object
        """
        if not isinstance(config_dict, dict):
            return config_dict
            
        # Check if this is a type/params structure
        if 'type' in config_dict and 'params' in config_dict:
            config_type = config_dict['type']
            params = config_dict['params']
            
            # Look up the configuration class in the registry
            if config_type not in CONFIG_REGISTRY:
                raise ConfigurationError(f"Unknown configuration type: {config_type}")
            
            config_class = CONFIG_REGISTRY[config_type]
            
            # Recursively build nested configurations
            built_params = {}
            for key, value in params.items():
                if isinstance(value, dict) and 'type' in value and 'params' in value:
                    # This is a nested configuration
                    built_params[key] = self._build_config_from_dict(value)
                else:
                    built_params[key] = value
            
            # Convert learning rate if it's a string representation of a float
            if 'learning_rate' in built_params and isinstance(built_params['learning_rate'], str):
                try:
                    built_params['learning_rate'] = float(built_params['learning_rate'])
                except ValueError:
                    pass
            
            # Add vocabulary sizes for transformer configs if not provided
            if config_type in ['fsq_cfg', 'trunk_cfg']:
                if 'seq_vocab' not in built_params:
                    built_params['seq_vocab'] = len(SEQUENCE_TOKENS) + len(SPECIAL_TOKENS)
                if 'struct_vocab' not in built_params:
                    built_params['struct_vocab'] = 4375 + len(SPECIAL_TOKENS)  # FSQ tokens + special tokens
                
                # For trunk configs, add absorb tokens if not provided
                if config_type == 'trunk_cfg':
                    if 'seq_absorb_token' not in built_params:
                        built_params['seq_absorb_token'] = SPECIAL_TOKENS.MASK.value + len(SEQUENCE_TOKENS)
                    if 'struct_absorb_token' not in built_params:
                        built_params['struct_absorb_token'] = SPECIAL_TOKENS.MASK.value + 4375
            
            # Create the configuration object
            try:
                return config_class(**built_params)
            except Exception as e:
                raise ConfigurationError(f"Failed to create {config_type}: {str(e)}")
        else:
            # Not a type/params structure, return as-is
            return config_dict
    
    def _resolve_paths(self, params: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Resolve relative paths in parameters to absolute paths.
        
        Args:
            params: Parameter dictionary
            base_dir: Base directory for relative paths
            
        Returns:
            Parameters with resolved paths
        """
        if base_dir is None:
            base_dir = self.config_path.parent.parent  # Project root
        
        resolved = deepcopy(params)
        
        # List of parameter names that should be treated as paths
        path_params = ['data_dir', 'checkpoint_dir', 'fsq_encoder_path']
        
        for key, value in resolved.items():
            if key in path_params and value is not None:
                path = Path(value)
                if not path.is_absolute():
                    resolved[key] = str(base_dir / path)
            elif isinstance(value, dict):
                # Recursively resolve paths in nested dictionaries
                resolved[key] = self._resolve_paths(value, base_dir)
        
        return resolved
    
    def _build_configs(self) -> None:
        """Build model and training configurations from raw config."""
        # Resolve paths in the raw config first
        if 'model_cfg' in self.raw_config and 'params' in self.raw_config['model_cfg']:
            self.raw_config['model_cfg']['params'] = self._resolve_paths(
                self.raw_config['model_cfg']['params']
            )
        
        if 'train_cfg' in self.raw_config and 'params' in self.raw_config['train_cfg']:
            self.raw_config['train_cfg']['params'] = self._resolve_paths(
                self.raw_config['train_cfg']['params']
            )
        
        # Build model configuration
        if 'model_cfg' not in self.raw_config:
            raise ConfigurationError("Configuration must include 'model_cfg' section")
        
        self.model_config = self._build_config_from_dict(self.raw_config['model_cfg'])
        
        # Build training configuration
        if 'train_cfg' not in self.raw_config:
            raise ConfigurationError("Configuration must include 'train_cfg' section")
        
        self.training_config = self._build_config_from_dict(self.raw_config['train_cfg'])
    
    def get_model_config(self) -> Config:
        """Get the built model configuration."""
        return self.model_config
    
    def get_training_config(self) -> Config:
        """Get the built training configuration."""
        return self.training_config
    
    def merge_with_args(self, args: argparse.Namespace) -> None:
        """
        Merge command-line arguments with configuration.
        Command-line args take precedence.
        
        Args:
            args: Parsed command-line arguments
        """
        # For the new system, we need to rebuild configs after merging
        # This is more complex with the registry system, so we'll update raw config
        # and rebuild
        
        modified = False
        
        # Override model style
        if hasattr(args, 'style') and args.style:
            if 'model_cfg' in self.raw_config and 'params' in self.raw_config['model_cfg']:
                self.raw_config['model_cfg']['params']['style'] = args.style
                modified = True
        
        # Override batch size
        if hasattr(args, 'batch_size') and args.batch_size:
            if 'train_cfg' in self.raw_config and 'params' in self.raw_config['train_cfg']:
                self.raw_config['train_cfg']['params']['batch_size'] = args.batch_size
                modified = True
        
        # Override learning rate
        if hasattr(args, 'learning_rate') and args.learning_rate:
            if 'train_cfg' in self.raw_config and 'params' in self.raw_config['train_cfg']:
                self.raw_config['train_cfg']['params']['learning_rate'] = args.learning_rate
                modified = True
        
        # Override epochs
        if hasattr(args, 'epochs') and args.epochs:
            if 'train_cfg' in self.raw_config and 'params' in self.raw_config['train_cfg']:
                self.raw_config['train_cfg']['params']['max_epochs'] = args.epochs
                modified = True
        
        # Rebuild configs if modified
        if modified:
            self._build_configs()
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.raw_config, f, default_flow_style=False, sort_keys=False)
    
    def validate_config_consistency(self) -> None:
        """Validate that configuration combinations make sense."""
        if not self.model_config or not self.training_config:
            return
        
        model_style = self.model_config.style
        
        # Get loss and mask types from training config
        loss_type = type(self.training_config.loss_config).__name__.replace('Config', '').lower()
        mask_type = type(self.training_config.mask_config).__name__.replace('Config', '').lower()
        
        # Validate loss type for model style
        if model_style in ['stage_1', 'stage_2'] and 'kabschrmsd' not in loss_type:
            print(f"Warning: FSQ models typically use 'kabsch_rmsd' loss, but got '{loss_type}'")
        
        if model_style == 'mlm' and 'crossentropy' not in loss_type:
            print(f"Warning: MLM models typically use 'cross_entropy' loss, but got '{loss_type}'")
        
        if model_style == 'discrete_diffusion':
            if 'scoreentropy' not in loss_type:
                print(f"Warning: Discrete diffusion models should use 'score_entropy' loss, but got '{loss_type}'")
            if 'diffusion' not in mask_type:
                print(f"Warning: Discrete diffusion models should use 'diffusion' masking, but got '{mask_type}'")
        
        # Validate stage 2 requirements
        if model_style == 'stage_2':
            if 'nomask' not in mask_type:
                print(f"Warning: Stage 2 training typically uses 'no' masking, but got '{mask_type}'")
            if hasattr(self.model_config, 'fsq_encoder_path'):
                if not self.model_config.fsq_encoder_path:
                    raise ValueError("Stage 2 training requires fsq_encoder_path to be specified")
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        print("="*60)
        print("Raw Configuration:")
        print("="*60)
        print(yaml.dump(self.raw_config, default_flow_style=False, sort_keys=False))
        print("="*60)
        print("\nBuilt Model Config:")
        print(self.model_config)
        print("\nBuilt Training Config:")
        print(self.training_config)
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


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> Tuple[Config, Config, Dict, Dict]:
    """
    Load configuration from command-line arguments using the new registry system.
    
    Args:
        args: Parsed arguments. If None, will parse from sys.argv
        
    Returns:
        Tuple of (model_config, training_config, model_config_dict, training_config_dict)
    """
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()
    
    # Load configuration with new loader
    config_loader = ConfigLoader(args.config)
    
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
    
    # Get configuration objects
    try:
        model_config = config_loader.get_model_config()
        training_config = config_loader.get_training_config()
        
        # Get configuration dictionaries for backup
        model_config_dict = model_config.get_config_dict()
        training_config_dict = training_config.get_config_dict()
        
    except Exception as e:
        print(f"\nError creating configuration: {e}")
        print("\nPlease check your configuration file follows the correct format.")
        print("See configs/configuration_constructor.md for detailed documentation.")
        raise
    
    return model_config, training_config, model_config_dict, training_config_dict


# Example usage in training scripts:
if __name__ == "__main__":
    # Example of how to use in a training script
    model_cfg, train_cfg, model_dict, train_dict = load_config_from_args()
    
    print("Model style:", model_cfg.style)
    print("Model type:", type(model_cfg).__name__)
    print("Batch size:", train_cfg.batch_size)
    print("Learning rate:", train_cfg.learning_rate)
    print("Mask config type:", type(train_cfg.mask_config).__name__)
    print("Loss config type:", type(train_cfg.loss_config).__name__)
    print("\nModel config dict keys:", list(model_dict.keys()))
    print("Training config dict keys:", list(train_dict.keys()))