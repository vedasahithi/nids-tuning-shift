"""Shift diagnostic: how different is the official test partition, and is the shift in a few features?

Usage (from the repository root):
    python scripts/diagnose_shift.py --smoke                 # ~1 minute: checks the pipeline
    python scripts/diagnose_shift.py --workers 6 --threads 16   # full run (PC)

Validation macro-F1 is ~0.64 but test macro-F1 ~0.53 for every tuned model, and the
two-stage diagnostic showed that the higher Normal share of the test partition explains
only ~0.013 of that gap. This measures the covariate shift directly (hsboa.shift).

Unsupervised domain adaptation: the test partition's FEATURES are used, its LABELS are
not - they are discarded when the file is read and never reach any function here.

    domain AUC  out-of-fold AUC (5 folds, HGB), seeds 0-2, of separating
                  train_vs_test    the training partition from the test partition;
                  matched_vs_test  training rows resampled to the test partition's published
                                   class counts (aggregate metadata, hsboa.shift) from the
                                   test partition - the shift beyond the class mix;
                  control          the seed's validation part from its training part (both
                                   from the training partition; no shift expected)
    univariate  per feature: domain AUC of the raw value, KS statistic, PSI (test vs matched)
    greedy      matched-vs-test AUC after removing 0..15 features, each time the one the
                domain classifier relies on most (permutation importance), seed 0
    duplicates  duplicate and conflicting-label rows in the training partition; share of
                test rows whose exact feature vector occurs in the training partition

Decision rule, fixed before the run:
    S1 covariate shift:   matched-vs-test AUC (mean of seeds) >= 0.70 and control AUC <= 0.55.
    S2 concentrated:      the greedy curve reaches AUC <= 0.5 + (AUC_0 - 0.5) / 2 within
                          10 removed features (of 42).
    S1 and S2  -> step 3: shift-aware feature selection with HSBOA.
    S1, not S2 -> shift spread over many features: selection alone is unlikely to remove
                  it; the paper goes with contributions 1 + 2 and DANN / importance
                  weighting as the adaptation comparison.
    not S1     -> the inputs look alike; the test drop is a change in P(label | features)
                  (same-looking traffic, different classes), reported as such.
Writes results/diag_shift[_smoke]/: domain_auc.csv, univariate.csv, greedy.csv,
duplicates.json, report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per fit in the workers; --threads for the greedy phase
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hsboa.data import make_split  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.shift import (PUBLISHED_TEST_COUNTS, domain_auc, duplicate_stats, encode, greedy_removal,  # noqa: E402
                         match_class_mix, univariate)

FULL = {"seeds": [0, 1, 2], "folds": 5, "max_iter": 200, "steps": 15, "greedy_folds": 3, "rows": None}
SMOKE = {"seeds": [0], "folds": 3, "max_iter": 30, "steps": 3, "greedy_folds": 3, "rows": 4000}
S1_AUC, S1_CONTROL, S2_MAX_REMOVED = 0.70, 0.55, 10


def load(rows: int | None, seed: int = 0):
    """Encoded training partition (A, y), its class-mix-matched rows (indices into A) and the test features B.

    The test labels are dropped as the file is read. Matching uses the published
    test class counts (hsboa.shift.PUBLISHED_TEST_COUNTS). rows (smoke only) subsamples each side.
    """
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, _ = ds.load("test")               # features only, labels discarded
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    matched = match_class_mix(y, PUBLISHED_TEST_COUNTS, rng)
    if rows is not None:
        i = np.sort(rng.choice(len(X), rows, replace=False))
        matched = np.sort(rng.choice(matched, rows, replace=False))
        X_te = X_te.iloc[np.sort(rng.choice(len(X_te), rows, replace=False))]
        keep = np.union1d(i, matched)       # re-index into the kept rows
        matched = np.searchsorted(keep, matched)
        X, y = X.iloc[keep], y[keep]
    A, B = encode(X, X_te, ds.categorical_columns)
    return A, y, matched, B, list(ds.feature_columns)


def auc_task(kind: str, seed: int, grid: dict) -> dict:
    A, y, matched, B, _ = load(grid["rows"], seed)
    t0 = time.perf_counter()
    if kind == "train_vs_test":
        auc = domain_auc(A, B, folds=grid["folds"], seed=seed, max_iter=grid["max_iter"])
    elif kind == "matched_vs_test":         # training rows with the test partition's class mix
        auc = domain_auc(A[matched], B, folds=grid["folds"], seed=seed, max_iter=grid["max_iter"])
    else:                                   # control: validation part vs training part of the seed's split
        sp = make_split(y, seed=seed)
        auc = domain_auc(A[sp.train_idx], A[sp.val_idx], folds=grid["folds"], seed=seed, max_iter=grid["max_iter"])
    print(f"{kind} seed {seed}: AUC {auc:.4f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return {"comparison": kind, "seed": seed, "auc": auc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for the domain-AUC fits")
    ap.add_argument("--threads", type=int, default=1, help="threads for the greedy phase (runs after the pool)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    grid = SMOKE if args.smoke else FULL
    out = ROOT / "results" / ("diag_shift_smoke" if args.smoke else "diag_shift")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    tasks = [(k, s) for k in ("train_vs_test", "matched_vs_test", "control") for s in grid["seeds"]]
    print(f"{len(tasks)} domain-AUC fits, {args.workers} workers; results in {out}", flush=True)
    if args.workers > 1:
        from joblib import Parallel, delayed
        aucs = Parallel(n_jobs=args.workers)(delayed(auc_task)(k, s, grid) for k, s in tasks)
    else:
        aucs = [auc_task(k, s, grid) for k, s in tasks]
    auc_df = pd.DataFrame(aucs)
    auc_df.to_csv(out / "domain_auc.csv", index=False)

    A, y, matched, B, names = load(grid["rows"], 0)
    uni = univariate(A[matched], B, names)
    uni.to_csv(out / "univariate.csv", index=False)
    dup = duplicate_stats(A, y, B)
    key_b = pd.util.hash_pandas_object(pd.DataFrame(B), index=False)
    dup["other_dup_row_share"] = float(key_b.duplicated(keep=False).mean())
    (out / "duplicates.json").write_text(json.dumps(dup, indent=2), encoding="utf-8")

    from threadpoolctl import threadpool_limits
    print(f"greedy removal: {grid['steps']} steps, {args.threads} threads", flush=True)
    with threadpool_limits(limits=args.threads):
        greedy = greedy_removal(A[matched], B, names, steps=grid["steps"], folds=grid["greedy_folds"], seed=0,
                                max_iter=grid["max_iter"])
    greedy.to_csv(out / "greedy.csv", index=False)

    write_report(out, auc_df, uni, greedy, dup, time.perf_counter() - t0, grid["steps"])
    print((out / "report.md").read_text(encoding="utf-8"))


def write_report(out: Path, auc_df, uni, greedy, dup, wall_s: float, steps: int) -> None:
    m = auc_df.groupby("comparison")["auc"].agg(["mean", "std", "count"])
    auc_raw = float(m.loc["train_vs_test", "mean"])
    auc_tt, auc_c = float(m.loc["matched_vs_test", "mean"]), float(m.loc["control", "mean"])
    s1 = auc_tt >= S1_AUC and auc_c <= S1_CONTROL
    auc0 = float(greedy.auc.iloc[0])
    target = 0.5 + (auc0 - 0.5) / 2
    hit = greedy[(greedy.auc <= target) & (greedy.removed <= S2_MAX_REMOVED)]
    s2 = len(hit) > 0
    k2 = int(hit.removed.iloc[0]) if s2 else None
    s2_text = (f"met at {k2} removed" if s2 else
               "not met" if steps >= S2_MAX_REMOVED else f"inconclusive (only {steps} steps run)")
    if s1 and not s2 and steps < S2_MAX_REMOVED:
        outcome = "S1 met; S2 needs the full greedy curve (smoke run)."
    elif s1 and s2:
        outcome = "S1 and S2: covariate shift concentrated in a few features -> step 3 (shift-aware HSBOA selection)."
    elif s1:
        outcome = ("S1 only: covariate shift spread over many features -> contributions 1 + 2, "
                   "DANN / importance weighting as the adaptation comparison.")
    else:
        outcome = "Not S1: inputs look alike; the test drop is a change in P(label | features)."

    def md(df):
        def cell(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                         + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])

    lines = ["# Shift diagnostic (test features, no test labels)", "",
             f"Wall time {wall_s / 60:.1f} min.", "",
             "## Decision rule", "",
             f"- S1 covariate shift: class-mix-matched train-vs-test AUC {auc_tt:.4f} (>= {S1_AUC}), control AUC "
             f"{auc_c:.4f} (<= {S1_CONTROL}) -> **{'met' if s1 else 'not met'}** "
             f"(unmatched train-vs-test AUC {auc_raw:.4f}; the difference is what the class mix alone explains)",
             f"- S2 concentrated: greedy AUC_0 {auc0:.4f}, target <= {target:.4f} within {S2_MAX_REMOVED} removed "
             f"-> **{s2_text}**",
             f"- Outcome: **{outcome}**", "",
             "## Domain AUC by comparison and seed", "", md(auc_df), "",
             "## Greedy removal (features the domain classifier relies on most)", "",
             md(greedy.fillna("")), "",
             "## Univariate drift, top 20 features (test vs training partition)", "", md(uni.head(20)), "",
             "## Duplicates", "", "```", json.dumps(dup, indent=2), "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
