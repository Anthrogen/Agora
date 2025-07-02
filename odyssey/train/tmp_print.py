"""
Test script to load and display YAML configuration files.

Usage:
    python test.py --config file.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Test YAML configuration loader"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print greeting
    print("hello")
    print()
    
    # Check if file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{args.config}' not found!")
        sys.exit(1)
    
    if not config_path.is_file():
        print(f"Error: '{args.config}' is not a file!")
        sys.exit(1)
    
    # Read and display file contents
    print(f"Loading config from: {config_path.absolute()}")
    print("-" * 50)
    
    try:
        # Read raw file contents
        with open(config_path, 'r') as f:
            raw_contents = f.read()
        
        print("Raw file contents:")
        print(raw_contents)
        print("-" * 50)
        
        # Parse YAML
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        print("Parsed YAML structure:")
        print(yaml.dump(config_data, default_flow_style=False, sort_keys=False))
        print("-" * 50)
        
        # Pretty print with indentation
        print("Structured view:")
        print_dict(config_data)
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def print_dict(data, indent=0):
    """Recursively print dictionary with indentation."""
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print("  " * indent + f"[{i}]:")
            print_dict(item, indent + 1)
    else:
        print("  " * indent + str(data))


if __name__ == "__main__":
    main()