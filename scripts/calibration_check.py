"""Does calibrating the domain classifier change the importance-weighted validation score?

The importance weights of the shift-aware fitness come from the raw probabilities of a histogram
gradient-boosting domain classifier (training part vs test records). A boosted classifier can be
over-confident, which would inflate the weights. For every seed we compare
  raw        the weights of gate_iw.py (as used in the paper)
  isotonic   probabilities recalibrated by isotonic regression
  platt      probabilities recalibrated by logistic regression on the log-odds
The calibrators are fitted on out-of-fold probabilities (5 folds over the domain data: training part and test
records, never the validation records that are scored). We report the Kish effective sample size, the
weighted macro-F1 of the stored E2 LightGBM validation predictions (as in clip_sensitivity.py), and the
expected calibration error (ECE, 10 equal-mass bins) of the raw and recalibrated out-of-fold probabilities.

Writes results/calibration_check[_smoke]/{runs.csv,summary.csv,report.md}.
Usage (from the repository root):  python scripts/calibration_check.py --workers 10      (about 3 minutes)
                                   python scripts/calibration_check.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hsboa.data import CLASSES, make_split  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.shift import density_ratio_weights, encode, ess, weighted_macro_f1  # noqa: E402

PROBA = "results/diag_twostage/proba/lgbm_s{seed}.npz"
FULL = {"seeds": list(range(10)), "max_iter": 200}
SMOKE = {"seeds": [0], "max_iter": 20}


def ece(p: np.ndarray, d: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - d[c].mean()) for c in chunks))


def seed_task(seed: int, grid: dict) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, _ = ds.load("test")
    y = np.asarray(y)
    A, B = encode(X, X_te, ds.categorical_columns)
    sp = make_split(y, seed=seed)
    Xd = np.vstack([A[sp.train_idx], B])
    d = np.r_[np.zeros(len(sp.train_idx), int), np.ones(len(B), int)]
    make = lambda: HGB(max_iter=grid["max_iter"], learning_rate=0.1, max_leaf_nodes=31, early_stopping=False,
                       random_state=seed)
    p_val = make().fit(Xd, d).predict_proba(A[sp.val_idx])[:, 1]
    oof = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, d):
        oof[te] = make().fit(Xd[tr], d[tr]).predict_proba(Xd[te])[:, 1]
    eps = 1e-6
    logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    iso = IsotonicRegression(y_min=eps, y_max=1 - eps, out_of_bounds="clip").fit(oof, d)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof).reshape(-1, 1), d)
    versions = {"raw": (p_val, oof),
                "isotonic": (iso.predict(p_val), iso.predict(oof)),
                "platt": (platt.predict_proba(logit(p_val).reshape(-1, 1))[:, 1],
                          platt.predict_proba(logit(oof).reshape(-1, 1))[:, 1])}
    z = np.load(ROOT / PROBA.format(seed=seed), allow_pickle=False)
    yv, pred = y[sp.val_idx], z["P_flat"].argmax(1)
    k = len(CLASSES)
    base = weighted_macro_f1(yv, pred, k)
    out = []
    for name, (pv, po) in versions.items():
        w = density_ratio_weights(pv, len(sp.train_idx), len(B), clip_quantile=0.99)
        out.append({"seed": seed, "version": name, "ess_ratio": ess(w) / len(w), "max_weight": float(w.max()),
                    "macro_f1_ordinary": base, "macro_f1_weighted": weighted_macro_f1(yv, pred, k, w),
                    "ece_oof": ece(po, d)})
    print(f"seed {seed} done", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    grid = SMOKE if args.smoke else FULL
    if args.workers > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.workers)(delayed(seed_task)(s, grid) for s in grid["seeds"])
    else:
        res = [seed_task(s, grid) for s in grid["seeds"]]
    df = pd.DataFrame([r for rs in res for r in rs])
    out = ROOT / "results" / ("calibration_check_smoke" if args.smoke else "calibration_check")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    s = df.groupby("version", sort=False)[["ess_ratio", "max_weight", "macro_f1_ordinary", "macro_f1_weighted",
                                           "ece_oof"]].agg(["mean", "std"]).round(4)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.to_csv(out / "summary.csv")
    (out / "report.md").write_text("# Calibration check of the domain classifier\n\n" + s.to_string() + "\n",
                                   encoding="utf-8")
    print(s.to_string())


if __name__ == "__main__":
    main()
