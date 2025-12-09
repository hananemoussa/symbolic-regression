#!/usr/bin/env python3
"""
Utility script to plot the best_10_scores_avg_overall metric across training rounds,
comparing EvoTune and FunSearch results.

Usage:
    python plot_metrics.py <evotune_metrics_dir> <funsearch_metrics_dir> [--output <output_file>]

Example:
    python plot_metrics.py out/logs/evotune_sr_CRK16_phi/sr_sr_phi_dpo_0/metrics out/logs/funsearch_sr_CRK16_phi/sr_sr_phi_none_0/metrics
    python plot_metrics.py evotune_metrics funsearch_metrics --output comparison.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def get_round_numbers(include_negative=True):
    """Generate the expected round numbers: -1, 0, 50, 100, 150, ..., 1200"""
    rounds = []
    if include_negative:
        rounds.append(-1)
    rounds.append(0)
    rounds.extend(range(50, 1201, 50))
    return rounds


def load_metrics(metrics_dir: str, field: str = "best_10_scores_avg_overall"):
    """
    Load metric values from metrics_train_round_X.json files.

    Args:
        metrics_dir: Path to the metrics directory
        field: The JSON field to extract (default: best_10_scores_avg_overall)

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
                    output_file=None, title=None, field_name="best_10_scores_avg_overall"):
    """
    Plot both EvoTune and FunSearch metrics on the same plot.

    If round -1 has very negative values (outliers), they are shown as annotated points
    at the bottom of the plot with their actual values displayed, preventing y-axis distortion.

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

    # Identify outlier threshold - values that would distort the plot
    all_values = evotune_values + funsearch_values
    # Use values from round 0 onwards to determine reasonable y-axis range
    non_neg1_values = []
    for r, v in zip(evotune_rounds, evotune_values):
        if r >= 0:
            non_neg1_values.append(v)
    for r, v in zip(funsearch_rounds, funsearch_values):
        if r >= 0:
            non_neg1_values.append(v)

    if non_neg1_values:
        y_min_main = min(non_neg1_values)
        y_max_main = max(non_neg1_values)
        # Add padding
        y_range = y_max_main - y_min_main
        y_min_plot = y_min_main - y_range * 0.15
        y_max_plot = y_max_main + y_range * 0.1
    else:
        y_min_plot = min(all_values)
        y_max_plot = max(all_values)

    # Separate round -1 data (starting points) from rest
    evotune_start_round = None
    evotune_start_value = None
    funsearch_start_round = None
    funsearch_start_value = None

    evotune_main_rounds = []
    evotune_main_values = []
    funsearch_main_rounds = []
    funsearch_main_values = []

    for r, v in zip(evotune_rounds, evotune_values):
        if r == -1:
            evotune_start_round = r
            evotune_start_value = v
        else:
            evotune_main_rounds.append(r)
            evotune_main_values.append(v)

    for r, v in zip(funsearch_rounds, funsearch_values):
        if r == -1:
            funsearch_start_round = r
            funsearch_start_value = v
        else:
            funsearch_main_rounds.append(r)
            funsearch_main_values.append(v)

    # Plot main data
    plt.plot(evotune_main_rounds, evotune_main_values, marker='o', markersize=5, linewidth=2,
             color='#2E86AB', label='EvoTune')
    plt.plot(funsearch_main_rounds, funsearch_main_values, marker='s', markersize=5, linewidth=2,
             color='#E94F37', label='FunSearch')

    # Handle round -1 starting points
    # Check if they're outliers (significantly below the main plot range)
    outlier_threshold = y_min_plot - (y_max_plot - y_min_plot) * 0.5

    has_evotune_outlier = evotune_start_value is not None and evotune_start_value < outlier_threshold
    has_funsearch_outlier = funsearch_start_value is not None and funsearch_start_value < outlier_threshold

    if has_evotune_outlier or has_funsearch_outlier:
        # Place outlier points at a visible position near the bottom with annotation
        outlier_y_position = y_min_plot - (y_max_plot - y_min_plot) * 0.08

        # Draw a subtle break indicator
        break_y = y_min_plot - (y_max_plot - y_min_plot) * 0.02
        plt.axhline(y=break_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        if has_evotune_outlier:
            # Plot the starting point at the adjusted position
            plt.plot(evotune_start_round, outlier_y_position, marker='o', markersize=8,
                    color='#2E86AB', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            # Add annotation with actual value
            plt.annotate(f'{evotune_start_value:.1f}',
                        xy=(evotune_start_round, outlier_y_position),
                        xytext=(evotune_start_round + 50, outlier_y_position),
                        fontsize=9, color='#2E86AB', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1),
                        verticalalignment='center')
            # Draw dashed line connecting to first main point
            if evotune_main_rounds:
                plt.plot([evotune_start_round, evotune_main_rounds[0]],
                        [outlier_y_position, evotune_main_values[0]],
                        linestyle=':', linewidth=1.5, color='#2E86AB', alpha=0.6)
        elif evotune_start_value is not None:
            # Not an outlier, plot normally
            plt.plot(evotune_start_round, evotune_start_value, marker='o', markersize=5,
                    color='#2E86AB')
            if evotune_main_rounds:
                plt.plot([evotune_start_round, evotune_main_rounds[0]],
                        [evotune_start_value, evotune_main_values[0]],
                        linestyle='-', linewidth=2, color='#2E86AB')

        if has_funsearch_outlier:
            # Offset slightly to avoid overlap
            fs_y_offset = outlier_y_position - (y_max_plot - y_min_plot) * 0.04
            plt.plot(funsearch_start_round, fs_y_offset, marker='s', markersize=8,
                    color='#E94F37', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            plt.annotate(f'{funsearch_start_value:.1f}',
                        xy=(funsearch_start_round, fs_y_offset),
                        xytext=(funsearch_start_round + 50, fs_y_offset),
                        fontsize=9, color='#E94F37', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1),
                        verticalalignment='center')
            if funsearch_main_rounds:
                plt.plot([funsearch_start_round, funsearch_main_rounds[0]],
                        [fs_y_offset, funsearch_main_values[0]],
                        linestyle=':', linewidth=1.5, color='#E94F37', alpha=0.6)
        elif funsearch_start_value is not None:
            plt.plot(funsearch_start_round, funsearch_start_value, marker='s', markersize=5,
                    color='#E94F37')
            if funsearch_main_rounds:
                plt.plot([funsearch_start_round, funsearch_main_rounds[0]],
                        [funsearch_start_value, funsearch_main_values[0]],
                        linestyle='-', linewidth=2, color='#E94F37')

        # Extend y-axis to show the outlier region
        y_min_plot = outlier_y_position - (y_max_plot - y_min_plot) * 0.08

        # # Add text indicating both start from same point if values are close
        # if evotune_start_value is not None and funsearch_start_value is not None:
        #     if abs(evotune_start_value - funsearch_start_value) < abs(evotune_start_value) * 0.1:
        #         plt.text(-1, y_max_plot - (y_max_plot - y_min_plot) * 0.05,
        #                 'Both methods start\nfrom same baseline',
        #                 fontsize=9, style='italic', color='gray',
        #                 horizontalalignment='left', verticalalignment='top')
    else:
        # No outliers - plot starting points normally
        if evotune_start_value is not None:
            full_evotune_rounds = [evotune_start_round] + evotune_main_rounds
            full_evotune_values = [evotune_start_value] + evotune_main_values
            plt.plot(full_evotune_rounds, full_evotune_values, marker='o', markersize=5, linewidth=2,
                    color='#2E86AB')
        if funsearch_start_value is not None:
            full_funsearch_rounds = [funsearch_start_round] + funsearch_main_rounds
            full_funsearch_values = [funsearch_start_value] + funsearch_main_values
            plt.plot(full_funsearch_rounds, full_funsearch_values, marker='s', markersize=5, linewidth=2,
                    color='#E94F37')

    # Annotate final values (last round) for both methods
    if evotune_main_rounds and evotune_main_values:
        last_evotune_round = evotune_main_rounds[-1]
        last_evotune_value = evotune_main_values[-1]
        plt.annotate(f'{last_evotune_value:.1f}',
                    xy=(last_evotune_round, last_evotune_value),
                    xytext=(last_evotune_round - 80, last_evotune_value + (y_max_plot - y_min_plot) * 0.05),
                    fontsize=9, color='#2E86AB', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1),
                    horizontalalignment='right', verticalalignment='bottom')

    if funsearch_main_rounds and funsearch_main_values:
        last_funsearch_round = funsearch_main_rounds[-1]
        last_funsearch_value = funsearch_main_values[-1]
        # Offset annotation to avoid overlap if values are close
        y_offset = -0.08 if evotune_main_values and abs(last_funsearch_value - evotune_main_values[-1]) < (y_max_plot - y_min_plot) * 0.1 else 0.05
        plt.annotate(f'{last_funsearch_value:.1f}',
                    xy=(last_funsearch_round, last_funsearch_value),
                    xytext=(last_funsearch_round - 80, last_funsearch_value + (y_max_plot - y_min_plot) * y_offset),
                    fontsize=9, color='#E94F37', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1),
                    horizontalalignment='right', verticalalignment='top' if y_offset < 0 else 'bottom')

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Average Score (Best 10 Programs)', fontsize=12)

    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'Average score of best 10 programs across rounds', fontsize=14)

    plt.ylim(y_min_plot, y_max_plot)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot best_10_scores_avg_overall metric comparing EvoTune and FunSearch'
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
        '--field', '-f',
        type=str,
        default='best_10_scores_avg_overall',
        help='JSON field to plot (default: best_10_scores_avg_overall)'
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
        help='Prefix to attach to the plot filename (e.g., "sr_CRK16" -> "sr_CRK16_evotune_vs_funsearch_...")'
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
    evotune_rounds, evotune_values = load_metrics(args.evotune_dir, args.field)
    if not evotune_rounds:
        print("Error: No valid EvoTune data found")
        return 1
    print(f"  Found {len(evotune_rounds)} data points")
    print(f"  Rounds: {min(evotune_rounds)} to {max(evotune_rounds)}")
    print(f"  Values range: {min(evotune_values):.4f} to {max(evotune_values):.4f}")

    # Load FunSearch metrics
    print(f"Loading FunSearch metrics from: {args.funsearch_dir}")
    funsearch_rounds, funsearch_values = load_metrics(args.funsearch_dir, args.field)
    if not funsearch_rounds:
        print("Error: No valid FunSearch data found")
        return 1
    print(f"  Found {len(funsearch_rounds)} data points")
    print(f"  Rounds: {min(funsearch_rounds)} to {max(funsearch_rounds)}")
    print(f"  Values range: {min(funsearch_values):.4f} to {max(funsearch_values):.4f}")

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
            filename = f"{args.prefix}_evotune_vs_funsearch_{experiment_name}.png"
        else:
            filename = f"evotune_vs_funsearch_{experiment_name}.png"
        output_file = str(plots_dir / filename)

    # Plot comparison
    plot_comparison(evotune_rounds, evotune_values, funsearch_rounds, funsearch_values,
                    output_file, args.title, args.field)

    return 0


if __name__ == '__main__':
    exit(main())
