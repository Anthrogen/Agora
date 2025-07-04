#!/usr/bin/env python3
"""
YAML Expander Script

Takes a YAML configuration file with lists and generates multiple single-value YAML files.
Handles the new simplified YAML structure where configuration types are used directly as headers.

Example input YAML:
    model_cfg:
      trunk_cfg:
        d_model: [128, 256]  # Multiple values
        n_heads: [1, 2]      # Multiple values  
        n_layers: 3          # Single value
        first_block_cfg:
          - self_attention_cfg:
          - geometric_attention_cfg:
          - self_consensus_cfg:
            consensus_w:
              - 2
              - 3
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union
import copy

# Configuration
CONFIG_FILE = "/workspace/demo/Odyssey/configs/sean.yaml"
OUTPUT_DIR = "/workspace/demo/Odyssey/configs/expanded"


def expand_config(config_dict: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Recursively expand a configuration dictionary containing lists.
    Handles both atomic values and dash-notation lists in the new YAML structure.
    
    Args:
        config_dict: Configuration dictionary that may contain lists
        
    Yields:
        Configuration dictionaries with single values for each field
    """
    # Find the first list in the config
    for key, value in config_dict.items():
        if isinstance(value, list):
            # Handle dash-notation lists - each item can be a config type or a config with params
            for list_item in value:
                new_config = copy.deepcopy(config_dict)
                new_config[key] = list_item
                # Recursively expand the rest
                yield from expand_config(new_config)
            return
        elif isinstance(value, dict):
            # Check if nested dict contains lists
            nested_expansions = list(expand_config(value))
            if len(nested_expansions) > 1:
                # Found expandable nested config
                for nested_config in nested_expansions:
                    new_config = copy.deepcopy(config_dict)
                    new_config[key] = nested_config
                    yield from expand_config(new_config)
                return
    
    # No lists found - return as-is
    yield config_dict


def expand_dash_lists(config_dict: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Handle dash-notation lists specifically for configuration sections.
    This handles cases where we have lists of configuration options.
    
    Args:
        config_dict: Configuration dictionary
        
    Yields:
        Configuration dictionaries with single config selections
    """
    # Look for dash lists at any level
    for key, value in config_dict.items():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            # This is a dash list of configuration dictionaries
            for list_item in value:
                new_config = copy.deepcopy(config_dict)
                new_config[key] = list_item
                # Recursively expand the rest
                yield from expand_dash_lists(new_config)
            return
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            nested_expansions = list(expand_dash_lists(value))
            if len(nested_expansions) > 1:
                for nested_config in nested_expansions:
                    new_config = copy.deepcopy(config_dict)
                    new_config[key] = nested_config
                    yield from expand_dash_lists(new_config)
                return
    
    # No expandable dash lists found - continue with regular expansion
    yield from expand_config(config_dict)


def expand_yaml_to_directory(config_file: str, output_dir: str) -> int:
    """
    Expand a YAML configuration file and save all resulting configurations to a directory.
    
    Args:
        config_file: Path to the input YAML configuration file
        output_dir: Directory where expanded YAML files will be saved
        
    Returns:
        Number of generated files, or -1 if there was an error
    """
    # Load the input YAML file
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} does not exist")
        return -1
    
    try:
        with open(config_path, 'r') as f:
            original_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return -1
    
    # Extract yaml name from config file path
    yaml_name = config_path.stem
    
    # Create output directory structure: output_dir/yaml_name/
    output_dir = Path(output_dir)
    yaml_main_dir = output_dir / yaml_name
    yaml_main_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save configurations
    print(f"Generating YAML files from {config_file} to {yaml_main_dir}...")
    print(f"Input config structure: {list(original_config.keys())}")
    
    generated_count = 0
    for i, config in enumerate(expand_dash_lists(original_config)):
        # Create subdirectory named after yaml file and indexed
        subdir_name = f"{yaml_name}_{i:03d}"
        subdir_path = yaml_main_dir / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)
        
        # Modify train_cfg.checkpoint_dir to include yaml name and index subfolders
        # Handle nested structure: train_cfg.training_cfg.checkpoint_dir
        if 'train_cfg' in config:
            if 'training_cfg' in config['train_cfg'] and 'checkpoint_dir' in config['train_cfg']['training_cfg']:
                base_checkpoint_dir = config['train_cfg']['training_cfg']['checkpoint_dir'].rstrip('/')
                config['train_cfg']['training_cfg']['checkpoint_dir'] = f"{base_checkpoint_dir}/{yaml_name}/{yaml_name}_{i:03d}"
            elif 'checkpoint_dir' in config['train_cfg']:
                base_checkpoint_dir = config['train_cfg']['checkpoint_dir'].rstrip('/')
                config['train_cfg']['checkpoint_dir'] = f"{base_checkpoint_dir}/{yaml_name}/{yaml_name}_{i:03d}"
        
        # Save YAML file in the subdirectory
        yaml_path = subdir_path / f"{subdir_name}.yaml"
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Generated: {yaml_path}")
            generated_count += 1
        except Exception as e:
            print(f"Error saving {yaml_path}: {e}")

        assert i < 999 # right now we print 3 digits almost everywhere.  It would be better to dynamically determine the number of digits but this would be awkward with the enumeration system.
    
    print(f"\nGenerated {generated_count} YAML files in {yaml_main_dir}")
    
    return generated_count

def main():
    """Main function to run the YAML expander with hardcoded paths."""
    return expand_yaml_to_directory(CONFIG_FILE, OUTPUT_DIR)


if __name__ == "__main__":
    exit(main())