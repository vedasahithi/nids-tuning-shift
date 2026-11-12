"""Domain AUC of the train/test shift with duplicates controlled.

The shift diagnostic (scripts/diagnose_shift.py) trains a classifier to tell training records from test records
with ordinary stratified folds. About half of the training records have an exact copy, so such a classifier could
partly memorise clusters of identical records. This script repeats the measurement in three ways:

    stratified   the original protocol (reproduces results/diag_shift/domain_auc.csv)
    grouped      folds that keep every group of identical feature vectors together (StratifiedGroupKFold)
    dedup        identical feature vectors are removed within each partition before the usual stratified folds

for the whole training partition and for the class-mix-matched training rows, against the test features (test
labels are not used).

Usage (from the repository root):
    python scripts/domain_auc_grouped.py --workers 2 --threads 4
    python scripts/domain_auc_grouped.py --smoke

Writes results/domain_auc_grouped/runs.csv and summary.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.shift import PUBLISHED_TEST_COUNTS, _model, encode, match_class_mix  # noqa: E402

FULL = {"seeds": [0, 1, 2], "folds": 5, "max_iter": 200, "rows": None}
SMOKE = {"seeds": [0], "folds": 3, "max_iter": 30, "rows": 4000}
KINDS = ("train_vs_test", "matched_vs_test")
VARIANTS = ("stratified", "grouped", "dedup")


def load(rows, seed):
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
        keep = np.union1d(i, matched)
        matched = np.searchsorted(keep, matched)
        X = X.iloc[keep]
    A, B = encode(X, X_te, ds.categorical_columns)
    return A, matched, B


def vector_keys(M: np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.DataFrame(M), index=False).to_numpy()


def oof_auc(X, d, groups, folds, seed, max_iter) -> float:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    oof = np.zeros(len(d))
    if groups is None:
        splits = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d)
    else:
        splits = StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, d, groups)
    for tr, te in splits:
        oof[te] = _model(seed, max_iter).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(d, oof))


def task(kind: str, variant: str, seed: int, grid: dict) -> dict:
    A, matched, B = load(grid["rows"], seed)
    if kind == "matched_vs_test":
        A = A[matched]
    t0 = time.perf_counter()
    if variant == "dedup":
        A = A[~pd.Series(vector_keys(A)).duplicated().to_numpy()]
        B = B[~pd.Series(vector_keys(B)).duplicated().to_numpy()]
    X = np.vstack([A, B])
    d = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
    groups = None
    if variant == "grouped":
        groups = pd.factorize(vector_keys(X))[0]
    auc = oof_auc(X, d, groups, grid["folds"], seed, grid["max_iter"])
    print(f"{kind:16s} {variant:10s} seed {seed}: AUC {auc:.4f}  (rows {len(A)} vs {len(B)}, "
          f"{time.perf_counter() - t0:.0f} s)", flush=True)
    return {"comparison": kind, "variant": variant, "seed": seed, "auc": auc, "n_train_side": len(A),
            "n_test_side": len(B)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="threads per worker")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    grid = SMOKE if args.smoke else FULL
    out = ROOT / "results" / ("domain_auc_grouped_smoke" if args.smoke else "domain_auc_grouped")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, v, s, grid) for k in KINDS for v in VARIANTS for s in grid["seeds"]]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(task, *zip(*jobs)))
    else:
        rows = [task(*j) for j in jobs]
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    summ = df.groupby(["comparison", "variant"])["auc"].agg(["mean", "std", "min", "max"]).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
