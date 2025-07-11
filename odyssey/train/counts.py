#!/usr/bin/env python3
"""
Script to extract and filter annotation vocabularies from protein dataset.
Saves only terms that meet minimum frequency thresholds.
"""

import json
import os
import csv
import argparse
from collections import Counter
from pathlib import Path


def parse_global_annotation(annotation_str):
    """
    Parse global annotation string and extract individual terms.
    Terms are separated by semicolons.
    """
    if not annotation_str or annotation_str.strip() == "": return []
    
    # Split by semicolon and clean each term
    terms = [term.strip() for term in annotation_str.split(';') if term.strip()]
    return terms


def parse_per_residue_annotation(annotation_dict):
    """
    Parse per-residue annotation dictionary and extract all annotation terms.
    """
    terms = []
    if not annotation_dict: return terms
    
    # Iterate through all residue positions
    for position, annotations in annotation_dict.items():
        if isinstance(annotations, list): terms.extend([term.strip() for term in annotations if term.strip()])
        elif isinstance(annotations, str): terms.append(annotations.strip())
    
    return terms


def process_dataset(csv_file_path):
    """
    Process JSON files referenced in the CSV and extract annotation terms.
    
    Args:
        csv_file_path: Path to the CSV file containing JSON file paths
    
    Returns:
        Tuple of (global_terms, per_residue_terms) lists
    """
    global_terms = []
    per_residue_terms = []
    
    # Get the directory containing the CSV file to resolve relative paths
    csv_dir = os.path.dirname(csv_file_path)
    
    # Read CSV file
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        json_files = []
        
        for row in reader:
            # Get the JSON path from rep_json_path column
            json_path = row.get('rep_json_path', '').strip()
            if json_path: json_files.append(json_path)
    
    print(f"Found {len(json_files)} files to process from CSV")
    
    # Process each JSON file
    processed = 0
    for json_path in json_files:
        full_path = os.path.join(csv_dir, json_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
            continue
            
        try:
            with open(full_path, 'r') as f: data = json.load(f)
            if 'global_annotation' in data: global_terms.extend(parse_global_annotation(data['global_annotation']))
            if 'per_residue_annotation' in data: per_residue_terms.extend(parse_per_residue_annotation(data['per_residue_annotation']))
            processed += 1
            if processed % 500 == 0: print(f"Processed {processed}/{len(json_files)} files...")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {full_path}: {e}")
            continue
    
    print(f"Successfully processed {processed} files")
    return global_terms, per_residue_terms


def filter_and_save_vocabulary(terms, min_frequency, output_file):
    """
    Filter terms by minimum frequency and save vocabulary.
    
    Args:
        terms: List of all terms
        min_frequency: Minimum frequency threshold
        output_file: Path to save the filtered vocabulary
    """
    # Count term frequencies
    term_counts = Counter(terms)
    
    # Filter by minimum frequency
    filtered_terms = [term for term, count in term_counts.items() if count >= min_frequency]
    
    # Sort by frequency (descending)
    filtered_terms.sort(key=lambda term: term_counts[term], reverse=True)
    
    # Save vocabulary (one term per line)
    with open(output_file, 'w') as f:
        for term in filtered_terms: f.write(f"{term}\n")
    
    print(f"Saved {len(filtered_terms)} terms to {output_file}")
    print(f"  (filtered from {len(term_counts)} unique terms)")
    
    return len(filtered_terms)


def main():
    """
    Main function to process annotations and create filtered vocabularies.
    """
    parser = argparse.ArgumentParser(description='Extract and filter annotation vocabularies from protein dataset')
    parser.add_argument('csv_file', help='Path to CSV file containing dataset file paths')
    parser.add_argument('--min_per_residue_freq', type=int, required=True, help='Minimum frequency for per-residue annotations')
    parser.add_argument('--min_global_freq', type=int, required=True, help='Minimum frequency for global annotations')
    parser.add_argument('--output_dir', default='.', help='Directory to save vocabulary files (default: current directory)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing dataset from CSV: {args.csv_file}")
    print(f"Min per-residue frequency: {args.min_per_residue_freq}")
    print(f"Min global frequency: {args.min_global_freq}")
    
    # Process the dataset
    global_terms, per_residue_terms = process_dataset(args.csv_file)
    
    print(f"\nFound {len(global_terms)} total global annotation occurrences")
    print(f"Found {len(per_residue_terms)} total per-residue annotation occurrences")
    
    # Filter and save vocabularies
    if per_residue_terms:
        per_residue_vocab_file = os.path.join(args.output_dir, 'vocab_per_residue_annotations.txt')
        per_residue_vocab_size = filter_and_save_vocabulary(per_residue_terms, args.min_per_residue_freq, per_residue_vocab_file)
    
    if global_terms:
        global_vocab_file = os.path.join(args.output_dir, 'vocab_global_annotations.txt')
        global_vocab_size = filter_and_save_vocabulary(global_terms, args.min_global_freq, global_vocab_file)
    
    print("\nVocabulary extraction complete!")


if __name__ == "__main__":
    # Example usage:
    # python counts.py ../../sample_data/3k.csv --min_per_residue_freq 3 --min_global_freq 3
    main() 