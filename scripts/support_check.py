"""Support check: do test records lie farther from the training records than validation records do?

Support mismatch (test regions without training support) predicts that test records are far from every
training record. For seed 0 we take the training part (140,272 records), encode the 42 features (ordinal
codes for the three categorical ones), map every feature to uniform marginals with a quantile transform fitted
on the training part, and compute the Euclidean distance of each query record to its nearest training record.
Queries: validation records (the within-partition control) and test records, 10,000 each, random.
Records with an exact training twin have distance 0; because duplicates are common inside the training
partition, we report the share of twins and the distance quantiles among records WITHOUT a twin.

Writes results/support_check/{summary.csv,by_class.csv,report.md}.
Usage (from the repository root):  python scripts/support_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

from hsboa.data import make_split  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.shift import encode  # noqa: E402

N_QUERY = 10_000
SEED = 0


def nn_distance(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1).fit(ref)
    return nn.kneighbors(q)[0][:, 0]


def main() -> None:
    ds = get_dataset()
    X, y = ds.load("train")
    Xt, yt = ds.load("test")
    y, yt = np.asarray(y, int), np.asarray(yt, int)
    A, B = encode(X, Xt, ds.categorical_columns)
    sp = make_split(y, seed=SEED)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", random_state=SEED)
    ref = qt.fit_transform(A[sp.train_idx])
    rng = np.random.default_rng(SEED)
    iv = rng.choice(sp.val_idx, N_QUERY, replace=False)
    it = rng.choice(len(B), N_QUERY, replace=False)
    dv = nn_distance(ref, qt.transform(A[iv]))
    dt = nn_distance(ref, qt.transform(B[it]))
    rows = []
    for name, d in (("validation (control)", dv), ("test", dt)):
        nt = d > 1e-9
        q = np.quantile(d[nt], [0.5, 0.9, 0.99])
        rows.append({"query set": name, "records": len(d), "share with exact training twin": float(1 - nt.mean()),
                     "median distance (no twin)": q[0], "p90 (no twin)": q[1], "p99 (no twin)": q[2],
                     "share beyond validation p99 (no twin)": np.nan})
    thr = rows[0]["p99 (no twin)"]
    for r, d in zip(rows, (dv, dt)):
        nt = d > 1e-9
        r["share beyond validation p99 (no twin)"] = float((d[nt] > thr).mean())
    out = ROOT / "results" / "support_check"
    out.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    # by test class: median distance among records without a twin
    by = []
    for c, name in enumerate(ds.classes):
        m = (yt[it] == c) & (dt > 1e-9)
        mv = (y[iv] == c) & (dv > 1e-9)
        by.append({"class": name, "test records (no twin)": int(m.sum()),
                   "test median distance": float(np.median(dt[m])) if m.sum() else np.nan,
                   "validation records (no twin)": int(mv.sum()),
                   "validation median distance": float(np.median(dv[mv])) if mv.sum() else np.nan})
    b = pd.DataFrame(by)
    b.to_csv(out / "by_class.csv", index=False)
    (out / "report.md").write_text("# Support check (seed 0, 10,000 queries each)\n\n" + s.round(4).to_string(index=False)
                                   + "\n\nBy class (records without a training twin):\n\n" + b.round(4).to_string(index=False)
                                   + "\n", encoding="utf-8")
    print(s.round(4).to_string(index=False))
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
