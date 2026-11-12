"""Optimiser comparison on shifted (and rotated) benchmark functions.

Usage (from the repository root):
    python experiments/run_benchmarks.py            # full study, resumable
    python experiments/run_benchmarks.py --smoke    # pipeline check
    python experiments/run_benchmarks.py --summarize-only

Writes results/benchmarks/: raw/<problem>/<method>/seed_XX.json, manifest.json,
summary.csv, tests.csv, wtl.csv, ranks.csv, curves.csv, curves_diversity.csv, report.md.
"""

from _cli import main

if __name__ == "__main__":
    main("configs/experiments/benchmarks.yaml", __doc__)
