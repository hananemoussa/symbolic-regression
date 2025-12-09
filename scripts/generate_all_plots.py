#!/usr/bin/env python3
"""
Batch script to generate comparison plots for multiple EvoTune/FunSearch experiment pairs.

Usage:
    python generate_all_plots.py <evotune_dir1> <funsearch_dir1> [<evotune_dir2> <funsearch_dir2> ...]

Example:
    python generate_all_plots.py \
        out/logs/evotune_sr_CRK16_phi/sr_sr_phi_dpo_0/metrics \
        out/logs/funsearch_sr_CRK16_phi/sr_sr_phi_none_0/metrics \
        out/logs/evotune_sr_CRK16_phi/sr_sr_phi_dpo_1/metrics \
        out/logs/funsearch_sr_CRK16_phi/sr_sr_phi_none_1/metrics

The script expects pairs of directories: evotune first, then funsearch.
All plots will be saved to out/logs/plots/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison plots for multiple EvoTune/FunSearch experiment pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'directories',
        nargs='+',
        help='Pairs of directories: evotune1 funsearch1 evotune2 funsearch2 ...'
    )
    parser.add_argument(
        '--field', '-f',
        type=str,
        default='best_10_scores_avg_overall',
        help='JSON field to plot (default: best_10_scores_avg_overall)'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default=None,
        help='Prefix to attach to all plot filenames'
    )

    args = parser.parse_args()

    # Validate we have pairs
    if len(args.directories) % 2 != 0:
        print("Error: Must provide pairs of directories (evotune, funsearch)")
        print(f"Got {len(args.directories)} directories, expected an even number")
        return 1

    # Get the script directory to find plot_metrics.py
    script_dir = Path(__file__).parent
    plot_script = script_dir / "plot_metrics.py"

    if not plot_script.exists():
        print(f"Error: Could not find plot_metrics.py at {plot_script}")
        return 1

    # Process pairs
    num_pairs = len(args.directories) // 2
    success_count = 0
    failed_pairs = []

    print(f"Processing {num_pairs} experiment pair(s)...\n")

    for i in range(num_pairs):
        evotune_dir = args.directories[i * 2]
        funsearch_dir = args.directories[i * 2 + 1]

        print(f"[{i + 1}/{num_pairs}] Processing pair:")
        print(f"  EvoTune:   {evotune_dir}")
        print(f"  FunSearch: {funsearch_dir}")

        # Build command
        cmd = [
            sys.executable,
            str(plot_script),
            evotune_dir,
            funsearch_dir,
            "--field", args.field
        ]
        if args.prefix:
            cmd.extend(["--prefix", args.prefix])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                success_count += 1
            else:
                print(f"  Error: {result.stderr}")
                failed_pairs.append((evotune_dir, funsearch_dir))
        except Exception as e:
            print(f"  Exception: {e}")
            failed_pairs.append((evotune_dir, funsearch_dir))

        print()

    # Summary
    print("=" * 60)
    print(f"Summary: {success_count}/{num_pairs} plots generated successfully")
    if failed_pairs:
        print("\nFailed pairs:")
        for evotune_dir, funsearch_dir in failed_pairs:
            print(f"  - {evotune_dir}")
            print(f"    {funsearch_dir}")

    return 0 if not failed_pairs else 1


if __name__ == '__main__':
    exit(main())
