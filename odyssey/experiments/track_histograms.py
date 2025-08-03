#!/usr/bin/env python3

import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_orthologous_groups(annotation_str):
    """Parse orthologous groups string and return list of terms."""
    if not annotation_str or annotation_str.strip() == "":
        return []
    
    # Split by commas and strip whitespace
    terms = [term.strip() for term in annotation_str.split(',') if term.strip()]
    return terms

def parse_semantic_description(description_str):
    """Parse semantic description string and return list of terms."""
    if not description_str or description_str.strip() == "":
        return []
    
    # Split on punctuation and whitespace, keeping letters and numbers together
    terms = re.split(r'[^a-zA-Z0-9]+', description_str)
    terms = [term.strip() for term in terms if term.strip()]
    return terms

def parse_domains_annotation(annotation_list):
    """Parse domains annotation list and extract all annotation terms."""
    terms = []
    if not annotation_list:
        return terms
    
    # Domains is a list of lists (one list per residue position)
    if isinstance(annotation_list, list):
        for domain_list in annotation_list:
            if isinstance(domain_list, list):
                terms.extend([term.strip() for term in domain_list if term.strip()])
            elif isinstance(domain_list, str):
                terms.append(domain_list.strip())
    
    return terms

def analyze_track_in_dataset(track_name, dataset_dir='sample_data/27k/', parser_func=None):
    """
    Analyze any track in the dataset and create a frequency histogram.
    
    Args:
        track_name: Name of the track to analyze ('domains', 'orthologous_groups', 'semantic_description')
        dataset_dir: Directory containing JSON files
        parser_func: Function to parse the track data
    """
    
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    
    print(f'Analyzing {track_name} in {len(all_files)} files from {dataset_dir}...')
    print('=' * 60)
    
    all_terms = []
    files_processed = 0
    files_with_track = 0
    total_term_instances = 0
    
    # Process all files
    for i, filename in enumerate(all_files):
        if i % 1000 == 0:  # Progress update
            print(f'Progress: {i}/{len(all_files)} files processed...')
        
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            track_data = data.get(track_name, None)
            files_processed += 1
            
            if not track_data:
                continue
            
            # Parse the track data using the appropriate parser
            if parser_func:
                terms = parser_func(track_data)
            else:
                terms = track_data if isinstance(track_data, list) else [track_data]
            
            if terms:
                all_terms.extend(terms)
                total_term_instances += len(terms)
                files_with_track += 1
                
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    
    print(f'\nProcessing complete!')
    print(f'Files processed: {files_processed:,}')
    print(f'Files with {track_name}: {files_with_track:,}')
    print(f'Total {track_name} instances: {total_term_instances:,}')
    print(f'Unique {track_name} types: {len(set(all_terms)):,}')
    
    if not all_terms:
        print(f"No {track_name} found in the dataset!")
        return
    
    # Count frequencies
    term_counts = Counter(all_terms)
    
    # Sort by frequency (most common first)
    sorted_terms = term_counts.most_common()
    
    print(f'\nTop 20 most common {track_name}:')
    print('-' * 40)
    for i, (term, count) in enumerate(sorted_terms[:20]):
        percentage = (count / total_term_instances) * 100
        print(f'{i+1:2d}. {term:<30} {count:8,} ({percentage:5.2f}%)')
    
    # Create histograms
    create_histograms(track_name, sorted_terms, total_term_instances)
    
    # Save detailed results
    save_detailed_results(track_name, sorted_terms, files_processed, files_with_track, total_term_instances)
    
    return sorted_terms

def create_histograms(track_name, sorted_terms, total_instances):
    """Create log scale histogram for the track."""
    
    term_names = [item[0] for item in sorted_terms]
    term_frequencies = [item[1] for item in sorted_terms]
    
    # Create log scale histogram
    plt.figure(figsize=(20, 10))
    
    # Create x-axis values starting from 1 (rank numbers)
    x_positions = range(1, len(term_frequencies) + 1)
    bars = plt.bar(x_positions, term_frequencies, 
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    plt.title(f'{track_name.replace("_", " ").title()} Frequency Distribution\n({len(term_names)} unique terms, {total_instances:,} total instances)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'{track_name.replace("_", " ").title()} Rank (sorted by frequency)', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency (log scale)', fontsize=14, fontweight='bold')
    
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis limits and let matplotlib handle tick spacing automatically
    num_unique_terms = len(term_names)
    plt.xlim(0, num_unique_terms + 1)
    
    # Highlight top 10 terms
    for i in range(min(10, len(bars))):
        bars[i].set_color('darkred')
        bars[i].set_alpha(0.8)
    
    # Add annotations for top 5 terms
    for i in range(min(5, len(sorted_terms))):
        term, count = sorted_terms[i]
        plt.annotate(f'{term}\n({count:,})', 
                    xy=(i + 1, count),  # Adjust for 1-based indexing
                    xytext=(i + 1, count * 2),
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save log scale histogram
    output_file = f'{track_name}_frequency_histogram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    print(f"✓ Saved histogram: {output_file}")
    print(f"  - {num_unique_terms:,} unique terms plotted with rank 1 to {num_unique_terms}")
    print(f"  - X-axis intervals: automatic")

def save_detailed_results(track_name, sorted_terms, files_processed, files_with_track, total_instances):
    """Save detailed analysis results to a text file."""
    
    term_frequencies = [item[1] for item in sorted_terms]
    
    print(f'\nSummary Statistics for {track_name}:')
    print(f'Mean frequency: {np.mean(term_frequencies):.2f}')
    print(f'Median frequency: {np.median(term_frequencies):.2f}')
    print(f'Standard deviation: {np.std(term_frequencies):.2f}')
    print(f'Most common term: {sorted_terms[0][0]} ({sorted_terms[0][1]:,} occurrences)')
    print(f'Terms appearing only once: {len([x for x in term_frequencies if x == 1])}')
    
    # Save detailed results to text file
    results_file = f'{track_name}_frequency_analysis.txt'
    with open(results_file, 'w') as f:
        f.write(f'{track_name.replace("_", " ").title()} Frequency Analysis\n')
        f.write(f'=' * 50 + '\n\n')
        f.write(f'Files processed: {files_processed:,}\n')
        f.write(f'Files with {track_name}: {files_with_track:,}\n')
        f.write(f'Total {track_name} instances: {total_instances:,}\n')
        f.write(f'Unique {track_name} types: {len(set([item[0] for item in sorted_terms])):,}\n\n')
        
        f.write(f'Complete frequency list (sorted by frequency):\n')
        f.write(f'Rank\tTerm\tFrequency\tPercentage\n')
        f.write(f'-' * 60 + '\n')
        
        for i, (term, count) in enumerate(sorted_terms):
            percentage = (count / total_instances) * 100
            f.write(f'{i+1}\t{term}\t{count}\t{percentage:.3f}%\n')
    
    print(f'Detailed results saved as: {results_file}')

def analyze_domains(dataset_dir='sample_data/27k/'):
    """Analyze domains in the dataset."""
    return analyze_track_in_dataset('domains', dataset_dir, parse_domains_annotation)

def analyze_orthologous_groups(dataset_dir='sample_data/27k/'):
    """Analyze orthologous groups in the dataset."""
    return analyze_track_in_dataset('orthologous_groups', dataset_dir, parse_orthologous_groups)

def analyze_semantic_description(dataset_dir='sample_data/27k/'):
    """Analyze semantic descriptions in the dataset."""
    return analyze_track_in_dataset('semantic_description', dataset_dir, parse_semantic_description)

def main():
    """Main function to run all three analyses."""
    parser = argparse.ArgumentParser(description='Generate histograms for protein annotation tracks')
    parser.add_argument('--dataset_dir', default='/workspace/demo/Odyssey/sample_data/27k/', 
                       help='Directory containing JSON files (default: sample_data/27k/)')
    parser.add_argument('--tracks', nargs='+', 
                       choices=['domains', 'orthologous_groups', 'semantic_description', 'all'],
                       default=['all'],
                       help='Which tracks to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Determine which tracks to analyze
    if 'all' in args.tracks:
        tracks_to_analyze = ['domains', 'orthologous_groups', 'semantic_description']
    else:
        tracks_to_analyze = args.tracks
    
    print(f"Analyzing tracks: {', '.join(tracks_to_analyze)}")
    print(f"Dataset directory: {args.dataset_dir}")
    print("=" * 80)
    
    results = {}
    
    for track in tracks_to_analyze:
        print(f"\n{'='*20} ANALYZING {track.upper()} {'='*20}")
        
        if track == 'domains':
            results[track] = analyze_domains(args.dataset_dir)
        elif track == 'orthologous_groups':
            results[track] = analyze_orthologous_groups(args.dataset_dir)
        elif track == 'semantic_description':
            results[track] = analyze_semantic_description(args.dataset_dir)
    
    print(f"\n{'='*20} ANALYSIS COMPLETE {'='*20}")
    print(f"Generated log scale histograms and frequency analysis files for: {', '.join(tracks_to_analyze)}")

if __name__ == "__main__":
    main() 