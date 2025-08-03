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
import re


def parse_orthologous_groups(annotation_str):
    """Parse orthologous groups string and return list of terms."""
    if not annotation_str or annotation_str.strip() == "": return []
    
    # Split by commas instead of semicolons and strip whitespace
    terms = [term.strip() for term in annotation_str.split(',') if term.strip()]
    return terms


def parse_semantic_description(description_str):
    """Parse semantic description string and return list of terms."""
    if not description_str or description_str.strip() == "": return []

    # Split on punctuation and whitespace, keeping letters and numbers together
    terms = re.split(r'[^a-zA-Z0-9]+', description_str)
    terms = [term.strip() for term in terms if term.strip()]
    return terms


def parse_domains_annotation(annotation_list):
    """
    Parse domains annotation list and extract all annotation terms.
    The domains field is a list of lists, where each inner list contains
    domain annotations for that residue position.
    """
    terms = []
    if not annotation_list: return terms
    
    # Iterate through all residue positions (list of lists)
    for position_annotations in annotation_list:
        if isinstance(position_annotations, list):
            # Extract terms from this position's annotations
            terms.extend([term.strip() for term in position_annotations if term and str(term).strip()])
        elif position_annotations:  # Handle single annotation (non-list)
            terms.append(str(position_annotations).strip())
    
    return terms


def process_dataset(csv_file_path):
    """
    Process JSON files referenced in the CSV and extract annotation terms.
    
    Args:
        csv_file_path: Path to the CSV file containing JSON file paths
    
    Returns:
        Tuple of (orthologous_groups_terms, semantic_description_terms, domains_terms) lists
    """
    orthologous_groups_terms = []
    semantic_description_terms = []
    domains_terms = []
    
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
            if 'orthologous_groups' in data: orthologous_groups_terms.extend(parse_orthologous_groups(data['orthologous_groups']))
            if 'semantic_description' in data: semantic_description_terms.extend(parse_semantic_description(data['semantic_description']))
            if 'domains' in data: domains_terms.extend(parse_domains_annotation(data['domains']))
            processed += 1
            if processed % 500 == 0: print(f"Processed {processed}/{len(json_files)} files...")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {full_path}: {e}")
            continue
    
    print(f"Successfully processed {processed} files")
    return orthologous_groups_terms, semantic_description_terms, domains_terms


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
    parser.add_argument('--min_domains_freq', type=int, required=True, help='Minimum frequency for domains')
    parser.add_argument('--min_orthologous_groups_freq', type=int, required=True, help='Minimum frequency for orthologous groups')
    parser.add_argument('--min_semantic_description_freq', type=int, required=True, help='Minimum frequency for semantic description')
    parser.add_argument('--output_dir', default='.', help='Directory to save vocabulary files (default: current directory)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing dataset from CSV: {args.csv_file}")
    print(f"Min domains frequency: {args.min_domains_freq}")
    print(f"Min orthologous groups frequency: {args.min_orthologous_groups_freq}")
    print(f"Min semantic description frequency: {args.min_semantic_description_freq}")
    
    # Process the dataset
    orthologous_groups_terms, semantic_description_terms, domains_terms = process_dataset(args.csv_file)
    
    print(f"\nFound {len(orthologous_groups_terms)} total orthologous groups occurrences")
    print(f"Found {len(semantic_description_terms)} total semantic description occurrences")
    print(f"Found {len(domains_terms)} total domains occurrences")
    
    # Filter and save vocabularies
    if domains_terms:
        domains_vocab_file = os.path.join(args.output_dir, 'vocab_domains.txt')
        domains_vocab_size = filter_and_save_vocabulary(domains_terms, args.min_domains_freq, domains_vocab_file)
    
    if orthologous_groups_terms:
        orthologous_groups_vocab_file = os.path.join(args.output_dir, 'vocab_orthologous_groups.txt')
        orthologous_groups_vocab_size = filter_and_save_vocabulary(orthologous_groups_terms, args.min_orthologous_groups_freq, orthologous_groups_vocab_file)
    
    if semantic_description_terms:
        semantic_description_vocab_file = os.path.join(args.output_dir, 'vocab_semantic_descriptions.txt')
        semantic_description_vocab_size = filter_and_save_vocabulary(semantic_description_terms, args.min_semantic_description_freq, semantic_description_vocab_file)
    
    print("\nVocabulary extraction complete!")


if __name__ == "__main__":
    # Example usage:
    # python counts.py ../../sample_data/27k.csv --min_domains_freq 100 --min_orthologous_groups_freq 3 --min_semantic_description_freq 3
    main() 