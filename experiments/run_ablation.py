"""Ablation: harmonic term vs SBOA, matched noise, momentum and shuffled history.

Usage (from the repository root):
    python experiments/run_ablation.py              # full study, resumable
    python experiments/run_ablation.py --smoke      # pipeline check
    python experiments/run_ablation.py --summarize-only

Writes results/ablation/ (same files as run_benchmarks.py; report.md also
lists the size of the added term relative to the SBOA escape move).
"""

from _cli import main

if __name__ == "__main__":
    main("configs/experiments/ablation.yaml", __doc__)
