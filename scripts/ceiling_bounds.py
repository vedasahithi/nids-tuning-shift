"""Metric-specific macro-F1 bounds for classifiers that see only the 42 features.

Any classifier of the features assigns one label to every group of identical feature vectors
(deterministic case). Given the true labels of a record set, we bound the best macro-F1 such a
classifier can reach on that set:

  majority   group-majority labels (minimises errors; the earlier "ceiling"; NOT macro-F1 optimal)
  achievable best macro-F1 found by coordinate ascent over group->class assignments (a real
             assignment, so a valid LOWER bound on the true maximum), several starts
  upper      sum over classes of the exact per-class maximum F1 divided by K. For one class c,
             F1 = 2a/(a+b+P) (a = true positives, b = records predicted c, P = true c); by the
             Dinkelbach argument an optimal prediction set takes the groups whose share of class c
             is above a threshold, so the optimum is a prefix of the groups sorted by that share.
             Each class is maximised separately, so the sum bounds macro-F1 of ANY assignment.

Writes results/ceiling_bounds/{bounds.csv,per_class.csv,report.md}.
Usage (from the repository root):  python scripts/ceiling_bounds.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def group_counts(X: pd.DataFrame, y: np.ndarray, k: int) -> np.ndarray:
    """(groups x classes) count matrix for groups of identical feature vectors."""
    key = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy()
    codes, _ = pd.factorize(key)
    g = codes.max() + 1
    M = np.zeros((g, k), dtype=np.int64)
    np.add.at(M, (codes, np.asarray(y, dtype=int)), 1)
    return M


def macro_f1_from(tp: np.ndarray, pred: np.ndarray, sup: np.ndarray) -> float:
    den = pred + sup
    return float(np.mean(np.where(den > 0, 2.0 * tp / np.maximum(den, 1), 0.0)))


def assignment_scores(M: np.ndarray, a: np.ndarray) -> tuple[float, np.ndarray]:
    k = M.shape[1]
    sup = M.sum(0)
    tp = np.zeros(k)
    pred = np.zeros(k)
    N = M.sum(1)
    for c in range(k):
        s = a == c
        tp[c] = M[s, c].sum()
        pred[c] = N[s].sum()
    f1 = np.where(pred + sup > 0, 2 * tp / np.maximum(pred + sup, 1), 0.0)
    return float(f1.mean()), f1


def majority(M: np.ndarray) -> np.ndarray:
    return M.argmax(1)            # ties -> lowest class index (same rule as final_eval.ceiling)


def coordinate_ascent(M: np.ndarray, a0: np.ndarray, rng, max_sweeps: int = 50) -> np.ndarray:
    """Move one group at a time to the class that raises macro-F1 most, until no move helps."""
    g, k = M.shape
    sup = M.sum(0).astype(float)
    N = M.sum(1).astype(float)
    a = a0.copy()
    tp = np.array([M[a == c, c].sum() for c in range(k)], dtype=float)
    pred = np.array([N[a == c].sum() for c in range(k)], dtype=float)
    den = pred + sup
    f = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    for _ in range(max_sweeps):
        moved = 0
        for i in rng.permutation(g):
            c0, n, Ni = a[i], M[i], N[i]
            # F1 of c0 after removing the group
            f0_new = _f1(tp[c0] - n[c0], pred[c0] - Ni, sup[c0])
            best_gain, best_c = 1e-12, c0
            for c in range(k):
                if c == c0:
                    continue
                fc_new = _f1(tp[c] + n[c], pred[c] + Ni, sup[c])
                gain = (f0_new - f[c0]) + (fc_new - f[c])
                if gain > best_gain:
                    best_gain, best_c = gain, c
            if best_c != c0:
                c = best_c
                tp[c0] -= n[c0]; pred[c0] -= Ni
                tp[c] += n[c]; pred[c] += Ni
                f[c0] = _f1(tp[c0], pred[c0], sup[c0]); f[c] = _f1(tp[c], pred[c], sup[c])
                a[i] = c
                moved += 1
        if moved == 0:
            break
    return a


def _f1(tp: float, pred: float, sup: float) -> float:
    d = pred + sup
    return 2.0 * tp / d if d > 0 else 0.0


def per_class_upper(M: np.ndarray) -> np.ndarray:
    """Exact max F1 for each class when only that class is optimised (prefix by class share)."""
    g, k = M.shape
    N = M.sum(1).astype(float)
    out = np.zeros(k)
    for c in range(k):
        P = M[:, c].sum()
        if P == 0:
            continue
        share = M[:, c] / N
        order = np.argsort(-share, kind="stable")
        tp = np.cumsum(M[order, c])
        pr = np.cumsum(N[order])
        out[c] = float(np.max(2.0 * tp / (pr + P)))
    return out


def bounds(X: pd.DataFrame, y: np.ndarray, classes, restarts: int = 5, seed: int = 0) -> dict:
    k = len(classes)
    M = group_counts(X, y, k)
    a_maj = majority(M)
    maj, maj_f1 = assignment_scores(M, a_maj)
    rng = np.random.default_rng(seed)
    best, best_f1 = maj, maj_f1
    starts = [a_maj] + [np.where(rng.random(len(M)) < 0.5, a_maj, rng.integers(0, k, len(M)))
                        for _ in range(restarts - 1)]
    for a0 in starts:
        a = coordinate_ascent(M, a0, rng)
        s, f1 = assignment_scores(M, a)
        if s > best:
            best, best_f1 = s, f1
    ub = per_class_upper(M)
    return {"rows": int(M.sum()), "groups": int(len(M)), "majority": maj, "achievable": best,
            "upper": float(ub.mean()),
            "per_class": pd.DataFrame({"class": classes, "support": M.sum(0), "majority_f1": maj_f1,
                                       "achievable_f1": best_f1, "upper_f1": ub})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=5)
    args = ap.parse_args()
    from hsboa.datasets import get_dataset
    ds = get_dataset()
    X, y = ds.load("train")
    Xt, yt = ds.load("test")
    sets = {"training partition": (X, y), "test partition": (Xt, yt),
            "both (merged)": (pd.concat([X, Xt], ignore_index=True), np.r_[y, yt])}
    out = ROOT / "results" / "ceiling_bounds"
    out.mkdir(parents=True, exist_ok=True)
    rows, pcs = [], []
    for name, (Xs, ys) in sets.items():
        r = bounds(Xs, ys, ds.classes, args.restarts)
        pc = r.pop("per_class")
        pc.insert(0, "set", name)
        pcs.append(pc)
        rows.append({"set": name, **r})
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    b = pd.DataFrame(rows)
    p = pd.concat(pcs, ignore_index=True)
    b.to_csv(out / "bounds.csv", index=False)
    p.to_csv(out / "per_class.csv", index=False)
    (out / "report.md").write_text(
        "# Macro-F1 bounds for classifiers of the 42 features\n\n"
        "majority = group-majority labels; achievable = best assignment found (lower bound on the true "
        "maximum); upper = sum of exact per-class maxima / K (upper bound).\n\n"
        + b.round(4).to_string(index=False) + "\n\n"
        + p.round(4).to_string(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
