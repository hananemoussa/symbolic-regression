#!/usr/bin/env python3
"""
Utility script to plot a histogram comparing score distributions between EvoTune and FunSearch.

Usage:
    python plot_score_histogram.py <evotune_json_file> <funsearch_json_file> [--output <output_file>]

Example:
    python plot_score_histogram.py \
        out/logs/evotune_sr_MatSci20_phi/sr_sr_phi_dpo_0/metrics/metrics_train_round_50.json \
        out/logs/funsearch_sr_MatSci20_phi/sr_sr_phi_none_0/metrics/metrics_train_round_50.json
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


def load_scores(json_file: str):
    """
    Load all_scores from a metrics JSON file.

    Args:
        json_file: Path to the metrics JSON file

    Returns:
        List of scores
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("all_scores", [])
        return data.get("all_scores", [])


def plot_histogram(evotune_scores, funsearch_scores, output_file=None, title=None,
                   top_percentile=70):
    """
    Plot overlapping histograms for EvoTune and FunSearch score distributions.

    Args:
        evotune_scores: List of scores from EvoTune
        funsearch_scores: List of scores from FunSearch
        output_file: Optional path to save the plot
        title: Optional custom title
        top_percentile: Only show top X% of scores (default: 70)
    """
    # Calculate the percentile threshold to keep top X% of scores
    # Higher scores are better, so we use (100 - top_percentile) as lower bound
    all_scores = evotune_scores + funsearch_scores
    lower_threshold = np.percentile(all_scores, 100 - top_percentile)

    # Filter to keep only top percentile scores
    evotune_filtered = [s for s in evotune_scores if s >= lower_threshold]
    funsearch_filtered = [s for s in funsearch_scores if s >= lower_threshold]

    # Determine bin edges based on filtered data
    all_filtered = evotune_filtered + funsearch_filtered
    min_score = min(all_filtered)
    max_score = max(all_filtered)
    # Use ~30 bins for a clean histogram
    num_bins = min(30, int(max_score - min_score) + 1)
    bins = np.linspace(min_score, max_score, num_bins + 1)

    plt.figure(figsize=(12, 6))

    # Plot EvoTune histogram with frequency (not density)
    plt.hist(evotune_filtered, bins=bins, density=False, histtype='step',
             linewidth=2, color='#2E86AB', label='EvoTune')

    # Plot FunSearch histogram with dashed line
    plt.hist(funsearch_filtered, bins=bins, density=False, histtype='step',
             linewidth=2, color='#E94F37', linestyle='--', label='FunSearch')

    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'Score Distribution: EvoTune vs FunSearch', fontsize=14)

    # Add statistics annotation (using filtered data)
    evotune_mean = np.mean(evotune_filtered) if evotune_filtered else 0
    funsearch_mean = np.mean(funsearch_filtered) if funsearch_filtered else 0
    evotune_median = np.median(evotune_filtered) if evotune_filtered else 0
    funsearch_median = np.median(funsearch_filtered) if funsearch_filtered else 0

    # stats_text = (f'EvoTune: mean={evotune_mean:.1f}, median={evotune_median:.1f}, n={len(evotune_filtered)}/{len(evotune_scores)}\n'
    #               f'FunSearch: mean={funsearch_mean:.1f}, median={funsearch_median:.1f}, n={len(funsearch_filtered)}/{len(funsearch_scores)}')
    # plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
    #              verticalalignment='top', fontsize=9,
    #              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Note about filtering
    # clip_note = f'Showing top {top_percentile}% of scores (threshold: {lower_threshold:.1f})'
    # plt.annotate(clip_note, xy=(0.98, 0.02), xycoords='axes fraction',
    #              horizontalalignment='right', fontsize=8, alpha=0.7)

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram comparing score distributions between EvoTune and FunSearch'
    )
    parser.add_argument(
        'evotune_file',
        type=str,
        help='Path to the EvoTune metrics JSON file'
    )
    parser.add_argument(
        'funsearch_file',
        type=str,
        help='Path to the FunSearch metrics JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for the plot. If not specified, saves to out/logs/plots/.'
    )
    parser.add_argument(
        '--title', '-t',
        type=str,
        default=None,
        help='Custom title for the plot'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default=None,
        help='Prefix to attach to the plot filename'
    )
    parser.add_argument(
        '--top-percentile',
        type=int,
        default=80,
        help='Only show top X%% of scores (default: 70)'
    )

    args = parser.parse_args()

    # Validate files
    if not os.path.isfile(args.evotune_file):
        print(f"Error: '{args.evotune_file}' is not a valid file")
        return 1
    if not os.path.isfile(args.funsearch_file):
        print(f"Error: '{args.funsearch_file}' is not a valid file")
        return 1

    # Load scores
    print(f"Loading EvoTune scores from: {args.evotune_file}")
    evotune_scores = load_scores(args.evotune_file)
    if not evotune_scores:
        print("Error: No scores found in EvoTune file")
        return 1
    print(f"  Found {len(evotune_scores)} scores")

    print(f"Loading FunSearch scores from: {args.funsearch_file}")
    funsearch_scores = load_scores(args.funsearch_file)
    if not funsearch_scores:
        print("Error: No scores found in FunSearch file")
        return 1
    print(f"  Found {len(funsearch_scores)} scores")

    # Determine output file path
    output_file = args.output
    if output_file is None:
        plots_dir = Path("out/logs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Extract round number from filename
        evotune_path = Path(args.evotune_file)
        filename_stem = evotune_path.stem  # e.g., metrics_train_round_50
        # Get experiment name from parent directories
        experiment_name = evotune_path.parent.parent.name  # e.g., sr_sr_phi_dpo_0

        if args.prefix:
            filename = f"{args.prefix}_histogram_{experiment_name}_{filename_stem}.png"
        else:
            filename = f"histogram_{experiment_name}_{filename_stem}.png"
        output_file = str(plots_dir / filename)

    # Plot histogram
    plot_histogram(evotune_scores, funsearch_scores, output_file, args.title, args.top_percentile)

    return 0


if __name__ == '__main__':
    exit(main())
