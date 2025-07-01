#!/usr/bin/env python
"""
Save Configuration to JSON

A simple utility script to convert YAML configurations to JSON format
with full safety features enabled.

Usage:
    python configs/save_config_to_json.py --config configs/fsq_stage1_config.yaml
    python configs/save_config_to_json.py --config configs/mlm_config.yaml --output my_backup.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path
from datetime import datetime

from odyssey.src.config_loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(
        description='Save Odyssey configuration to JSON format'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file path (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--separate',
        action='store_true',
        help='Save model and training configs as separate files'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config_loader = ConfigLoader(args.config)
    config_loader._resolve_paths()
    
    # Get configuration objects
    model_config = config_loader.get_model_config()
    training_config = config_loader.get_training_config()
    
    # Determine output path(s)
    if args.output:
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
    else:
        # Auto-generate filename with timestamp
        config_name = Path(args.config).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('configs/backup')
        output_base = f"{config_name}_{timestamp}"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.separate:
        # Save as separate files
        model_path = output_dir / f"{output_base}_model.json"
        training_path = output_dir / f"{output_base}_training.json"
        
        model_config.save_to_json(str(model_path))
        training_config.save_to_json(str(training_path))
        
        print(f"\nSaved configurations:")
        print(f"  Model config: {model_path}")
        print(f"  Training config: {training_path}")
    else:
        # Save complete configuration
        # For simplicity, we'll save the model config which includes most info
        output_path = output_dir / f"{output_base}.json"
        
        # Create a combined configuration dictionary
        combined_config = {
            'model': model_config.to_dict(),
            'training': training_config.to_dict(),
            '_config_file': args.config,
            '_saved_at': datetime.now().isoformat()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(combined_config, f, indent=2)
        
        print(f"\nSaved complete configuration to: {output_path}")
    
    # Show how to load it back
    print("\nTo load this configuration later:")
    if args.separate:
        print(f"  from odyssey.src.configurations import Config")
        print(f"  model = Config.load_from_json('{model_path}')")
        print(f"  training = Config.load_from_json('{training_path}')")
    else:
        print(f"  # Note: For combined configs, you'll need custom loading logic")
        print(f"  import json")
        print(f"  with open('{output_path}', 'r') as f:")
        print(f"      config = json.load(f)")


if __name__ == "__main__":
    main()