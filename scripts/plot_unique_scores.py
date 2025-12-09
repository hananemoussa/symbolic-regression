#!/usr/bin/env python3
"""
Utility script to plot the num_unique_scores metric across training rounds,
comparing EvoTune and FunSearch results.

Usage:
    python plot_unique_scores.py <evotune_metrics_dir> <funsearch_metrics_dir> [--output <output_file>]

Example:
    python plot_unique_scores.py out/logs/evotune_sr_CRK16_phi/sr_sr_phi_dpo_0/metrics out/logs/funsearch_sr_CRK16_phi/sr_sr_phi_none_0/metrics
    python plot_unique_scores.py evotune_metrics funsearch_metrics --output comparison.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def get_round_numbers(include_negative=False):
    """Generate the expected round numbers: 0, 50, 100, 150, ..., 1200 (optionally -1)"""
    rounds = []
    if include_negative:
        rounds.append(-1)
    rounds.append(0)
    rounds.extend(range(50, 1201, 50))
    return rounds


def load_metrics(metrics_dir: str, field: str = "num_unique_scores"):
    """
    Load metric values from metrics_train_round_X.json files.

    Args:
        metrics_dir: Path to the metrics directory
        field: The JSON field to extract (default: num_unique_scores)

    Returns:
        Tuple of (rounds, values) lists
    """
    metrics_path = Path(metrics_dir)
    rounds = []
    values = []

    expected_rounds = get_round_numbers()

    for round_num in expected_rounds:
        filename = f"metrics_train_round_{round_num}.json"
        filepath = metrics_path / filename

        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # The file contains a list with one element
                    if isinstance(data, list) and len(data) > 0:
                        value = data[0].get(field)
                    else:
                        value = data.get(field)

                    if value is not None:
                        rounds.append(round_num)
                        values.append(value)
                    else:
                        print(f"Warning: Field '{field}' not found in {filename}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read {filename}: {e}")

    return rounds, values


def plot_comparison(evotune_rounds, evotune_values, funsearch_rounds, funsearch_values,
                    output_file=None, title=None, field_name="num_unique_scores"):
    """
    Plot both EvoTune and FunSearch metrics on the same plot.

    Args:
        evotune_rounds: List of round numbers for EvoTune
        evotune_values: List of metric values for EvoTune
        funsearch_rounds: List of round numbers for FunSearch
        funsearch_values: List of metric values for FunSearch
        output_file: Optional path to save the plot
        title: Optional custom title
        field_name: Name of the field being plotted
    """
    plt.figure(figsize=(12, 6))

    plt.plot(evotune_rounds, evotune_values, marker='o', markersize=5, linewidth=2,
             color='#2E86AB', label='EvoTune')
    plt.plot(funsearch_rounds, funsearch_values, marker='s', markersize=5, linewidth=2,
             color='#E94F37', label='FunSearch')

    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Number of Unique Scores', fontsize=12)

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title('Number of unique scores across Rounds', fontsize=14)

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
        description='Plot num_unique_scores metric comparing EvoTune and FunSearch'
    )
    parser.add_argument(
        'evotune_dir',
        type=str,
        help='Path to the EvoTune metrics directory'
    )
    parser.add_argument(
        'funsearch_dir',
        type=str,
        help='Path to the FunSearch metrics directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for the plot (e.g., plot.png). If not specified, saves to out/logs/plots/.'
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
        help='Prefix to attach to the plot filename (e.g., "sr_CRK16" -> "sr_CRK16_unique_scores_...")'
    )

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.evotune_dir):
        print(f"Error: '{args.evotune_dir}' is not a valid directory")
        return 1
    if not os.path.isdir(args.funsearch_dir):
        print(f"Error: '{args.funsearch_dir}' is not a valid directory")
        return 1

    # Load EvoTune metrics
    print(f"Loading EvoTune metrics from: {args.evotune_dir}")
    evotune_rounds, evotune_values = load_metrics(args.evotune_dir)
    if not evotune_rounds:
        print("Error: No valid EvoTune data found")
        return 1
    print(f"  Found {len(evotune_rounds)} data points")
    print(f"  Rounds: {min(evotune_rounds)} to {max(evotune_rounds)}")
    print(f"  Values range: {min(evotune_values)} to {max(evotune_values)}")

    # Load FunSearch metrics
    print(f"Loading FunSearch metrics from: {args.funsearch_dir}")
    funsearch_rounds, funsearch_values = load_metrics(args.funsearch_dir)
    if not funsearch_rounds:
        print("Error: No valid FunSearch data found")
        return 1
    print(f"  Found {len(funsearch_rounds)} data points")
    print(f"  Rounds: {min(funsearch_rounds)} to {max(funsearch_rounds)}")
    print(f"  Values range: {min(funsearch_values)} to {max(funsearch_values)}")

    # Determine output file path
    output_file = args.output
    if output_file is None:
        # Generate default output path in out/logs/plots/
        plots_dir = Path("out/logs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Extract experiment name from evotune directory path
        evotune_path = Path(args.evotune_dir)
        # Try to get a meaningful name from the path (e.g., sr_sr_phi_dpo_0)
        experiment_name = evotune_path.parent.name if evotune_path.name == "metrics" else evotune_path.name
        # Add prefix if provided
        if args.prefix:
            filename = f"{args.prefix}_unique_scores_{experiment_name}.png"
        else:
            filename = f"unique_scores_{experiment_name}.png"
        output_file = str(plots_dir / filename)

    # Plot comparison
    plot_comparison(evotune_rounds, evotune_values, funsearch_rounds, funsearch_values,
                    output_file, args.title, "num_unique_scores")

    return 0


if __name__ == '__main__':
    exit(main())
