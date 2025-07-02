"""
Simple configuration loader for YAML files.
"""

import yaml
from pathlib import Path
from typing import Tuple, Dict, Union

from odyssey.src.configurations import Config


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
    model_config_dict = model_config.get_config_dict()
    training_config_dict = training_config.get_config_dict()
    
    return model_config, training_config, model_config_dict, training_config_dict