"""Sensitivity of the importance-weighted validation score to the weight cap (gate_iw settings).

Same domain classifier, weights and stored E2 LightGBM validation predictions as gate_iw.py; only the cap
changes (none, 99.9th, 99th, 95th percentile). Reports Kish effective sample size ratio and the weighted
macro-F1 (a self-normalised plug-in ratio estimate, not an unbiased finite-sample score).
Usage (from the repository root): python scripts/clip_sensitivity.py   -> results/clip_sensitivity/
"""
import os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from hsboa.data import CLASSES, make_split
from hsboa.datasets import get_dataset
from hsboa.shift import density_ratio_weights, encode, ess, weighted_macro_f1

ds = get_dataset(); X, y = ds.load("train"); X_te, _ = ds.load("test"); y = np.asarray(y)
A, B = encode(X, X_te, ds.categorical_columns)
rows = []
for seed in range(10):
    sp = make_split(y, seed=seed)
    Xd = np.vstack([A[sp.train_idx], B]); d = np.r_[np.zeros(len(sp.train_idx), int), np.ones(len(B), int)]
    dom = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False,
                                         random_state=seed).fit(Xd, d)
    p = dom.predict_proba(A[sp.val_idx])[:, 1]
    z = np.load(ROOT / f"results/diag_twostage/proba/lgbm_s{seed}.npz", allow_pickle=False)
    yv, pred = y[sp.val_idx], z["P_flat"].argmax(1)
    for name, q in (("none", None), ("99.9%", 0.999), ("99%", 0.99), ("95%", 0.95)):
        w = density_ratio_weights(p, len(sp.train_idx), len(B), clip_quantile=q)
        rows.append({"seed": seed, "cap": name, "ess_ratio": ess(w) / len(w), "max_weight": float(w.max()),
                     "macro_f1_ordinary": weighted_macro_f1(yv, pred, len(CLASSES)),
                     "macro_f1_weighted": weighted_macro_f1(yv, pred, len(CLASSES), w)})
    print("seed", seed, flush=True)
df = pd.DataFrame(rows)
out = ROOT / "results" / "clip_sensitivity"; out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "runs.csv", index=False)
s = df.groupby("cap", sort=False)[["ess_ratio", "max_weight", "macro_f1_ordinary", "macro_f1_weighted"]].agg(["mean", "std"]).round(4)
s.columns = [f"{a}_{b}" for a, b in s.columns]; s.to_csv(out / "summary.csv")
(out / "report.md").write_text(s.to_string() + "\n", encoding="utf-8"); print(s.to_string())
