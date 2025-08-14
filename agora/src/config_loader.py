"""
Simple configuration loader for YAML files.
"""
import glob
import yaml
from pathlib import Path
from typing import Tuple, Dict, Union, List

from agora.src.configurations import Config


def load_config(config_path: Union[str, Path]) -> Tuple[Config, Config, Dict, Dict]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Tuple of (model_config, training_config, model_config_dict, training_config_dict)
    """
    # Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Build configurations using from_dict
    model_config = Config.from_dict(config_dict['model_cfg'])
    training_config = Config.from_dict(config_dict['train_cfg'])
    
    # Get backup dictionaries
    model_config_dict = model_config.to_dict()
    training_config_dict = training_config.to_dict()

    # print("########################################################")
    # print(model_config)
    # print(training_config)
    # print("########################################################")
    
    return model_config, training_config, model_config_dict, training_config_dict


def load_multi_configs(expanded_dir: str) -> Tuple[List[Config], List[Config]]:
    """
    Load all expanded YAML configuration files from a directory.
    
    Args:
        expanded_dir: Path to directory containing expanded YAML files in subdirectories
        
    Returns:
        Tuple of (list of model_configs, list of train_configs)
    """
    # Find all YAML files in subdirectories of the expanded directory
    yaml_files = sorted(glob.glob(f"{expanded_dir}/*/*.yaml"))
    
    if not yaml_files:
        raise ValueError(f"No YAML files found in subdirectories of {expanded_dir}")
    
    model_configs = []
    train_configs = []
    
    print(f"Loading {len(yaml_files)} configuration files from {expanded_dir}")
    
    for yaml_file in yaml_files:
        print("########################################################")
        print(f"Loading: {yaml_file}")
        
        # Create checkpoint directory before loading config
        import yaml
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create checkpoint directory if it doesn't exist
        if 'train_cfg' in config_data:
            if 'training_cfg' in config_data['train_cfg']:
                checkpoint_dir = config_data['train_cfg']['training_cfg'].get('checkpoint_dir')
            else:
                checkpoint_dir = config_data['train_cfg'].get('checkpoint_dir')
            
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                print(f"Created checkpoint directory: {checkpoint_path}")
        
        model_cfg, train_cfg, _, _ = load_config(yaml_file)
        model_configs.append(model_cfg)
        train_configs.append(train_cfg)
    
    print(f"Successfully loaded {len(model_configs)} model configs and {len(train_configs)} train configs")
    
    return model_configs, train_configs