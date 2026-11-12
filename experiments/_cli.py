"""Shared command line for the benchmark-style experiments (see hsboa.experiment)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsboa.experiment import jobs, load_experiment, run_experiment, summarize  # noqa: E402


def main(default_config: str, doc: str) -> None:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / default_config)
    ap.add_argument("--out", help="output folder (default results/<experiment>[_smoke])")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (default: all cores)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (2 seeds, D=10, 500 x D evals) to check the pipeline")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    name = cfg["experiment"]
    if args.smoke:
        cfg.update(seeds=cfg["seeds"][:2], dims=[10], budget_per_dim=500)
        cfg["_config_text"] += "\n# --smoke: seeds[:2], dims [10], budget_per_dim 500\n"
        name += "_smoke"
    out = Path(args.out) if args.out else ROOT / "results" / name
    print(f"{cfg['experiment']}: {len(jobs(cfg))} runs -> {out}")
    if not args.summarize_only:
        man = run_experiment(cfg, out, n_jobs=args.n_jobs)
        print(json.dumps({k: man[k] for k in ("n_runs", "n_run_this_call", "wall_s_this_call")}))
    tables = summarize(cfg, out)
    print(tables["wtl"].to_string(index=False))
    print(tables["ranks"].round(2).to_string(index=False))
    print(f"report: {out / 'report.md'}")
