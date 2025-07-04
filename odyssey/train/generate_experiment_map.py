#!/usr/bin/env python3
"""
Script to generate a map.md file for expanded experiment configurations.
Shows the original list parameters that were expanded.
"""

import os
import yaml
from pathlib import Path

def read_yaml_file(file_path):
    """Read and parse a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def extract_readable_info(values):
    """Extract human-readable information from complex list structures."""
    if not isinstance(values, list):
        return str(values)
    
    readable_items = []
    for item in values:
        if isinstance(item, dict):
            # Extract the key names (like 'self_attention_cfg', 'geometric_attention_cfg')
            keys = list(item.keys())
            if len(keys) == 1:
                key_name = keys[0].replace('_cfg', '').replace('_', ' ').title()
                readable_items.append(key_name)
            else:
                readable_items.append(f"Dict with {len(keys)} keys")
        else:
            readable_items.append(str(item))
    
    return readable_items

def find_list_parameters(base_config):
    """Find all parameters that are lists in the base configuration."""
    list_info = []
    
    def traverse(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, list):
                    readable_values = extract_readable_info(value)
                    list_info.append({
                        'path': current_path,
                        'values': value,
                        'readable_values': readable_values
                    })
                    # Also traverse list items to find nested lists
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            traverse(item, current_path)
                else:
                    traverse(value, current_path)
        elif isinstance(obj, list):
            # Handle lists at the current level
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    traverse(item, path)
    
    traverse(base_config)
    return list_info

def generate_experiment_map(base_config_path, expanded_dir):
    """Generate map.md file for expanded configurations."""
    
    # Read base configuration
    base_config = read_yaml_file(base_config_path)
    base_name = os.path.basename(base_config_path).replace('.yaml', '')
    
    # Find all list parameters in the base config
    list_info = find_list_parameters(base_config)
    
    # Count experiments
    experiment_count = len([item for item in os.listdir(expanded_dir) 
                           if os.path.isdir(os.path.join(expanded_dir, item))])
    
    # Generate map.md content
    content = f"""# {base_name.title()} Experiment Map

## Experimental Parameters

This experiment tests the following parameters:

"""
    
    for list_item in list_info:
        if 'readable_values' in list_item:
            content += f"**{list_item['path']}**:\n"
            for value in list_item['readable_values']:
                content += f"- {value}\n"
        else:
            content += f"**{list_item['path']}**:\n"
            for value in list_item['values']:
                content += f"- {value}\n"
        content += "\n"
    
    content += f"""
## Experiments

Total experiments generated: {experiment_count}

Each experiment tests a different combination of the parameters listed above.
"""
    
    # Write map.md file
    map_file = os.path.join(expanded_dir, 'map.md')
    with open(map_file, 'w') as f:
        f.write(content)
    
    print(f"Generated map.md at: {map_file}")
    return map_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate experiment map for expanded configurations')
    parser.add_argument('base_config', help='Path to base YAML configuration file')
    parser.add_argument('expanded_dir', help='Directory containing expanded configurations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_config):
        print(f"Error: Base configuration file not found: {args.base_config}")
        return 1
    
    if not os.path.exists(args.expanded_dir):
        print(f"Error: Expanded directory not found: {args.expanded_dir}")
        return 1
    
    try:
        map_file = generate_experiment_map(args.base_config, args.expanded_dir)
        print(f"Successfully generated experiment map: {map_file}")
        return 0
    except Exception as e:
        print(f"Error generating map: {e}")
        return 1

if __name__ == "__main__":
    exit(main())