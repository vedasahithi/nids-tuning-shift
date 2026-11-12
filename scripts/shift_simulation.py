"""Controlled simulation: what can domain AUC and importance-weighted validation identify?

Question: a high domain AUC shows separability but not a changed labelling rule, and weighted
validation beating ordinary validation does not isolate label shift. We build synthetic targets from
the TRAINING partition (labels known) with exactly known shift and run the same diagnostics.

Source S and target pool T0 are disjoint random halves of the training partition. Scenarios:
  none        T = uniform thinning of T0
  covariate   T = T0 thinned with keep-probability proportional to exp(beta * z(x)); labels untouched
              (P(y|x) identical). z = mean rank-gauss of sttl, ct_state_ttl, ct_dst_src_ltm. beta is
              calibrated so the held-out domain AUC is close to the observed 0.89.
  label_rule  T = uniform thinning of T0, then a fraction q of Backdoor and Analysis rows is relabelled
              Exploits/DoS/Fuzzers (x unchanged, so P(x) identical; P(y|x) changed). This mimics the
              pattern in results/label_agreement/twin_label_matrix.csv.
  both        covariate thinning + relabelling.
For each scenario and seed: domain AUC (held out), ordinary validation macro-F1 of a model trained on
S_train, importance-weighted validation macro-F1 (the gate_iw procedure: Bayes weights from a domain
classifier S_train vs T features, 99th-percentile cap, mean 1), and the model's actual macro-F1 on T.
Key output: estimate - actual for ordinary and weighted validation.

Writes results/shift_simulation[_smoke]/{runs.csv,summary.csv,report.md}.
Usage (from the repository root):  python scripts/shift_simulation.py --smoke   (small, minutes)
                     python scripts/shift_simulation.py           (full)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier as HGB  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.shift import density_ratio_weights, encode, weighted_macro_f1  # noqa: E402

FULL = {"seeds": [0, 1, 2, 3, 4], "n_src": 60000, "n_tgt": 30000, "iters": 100, "dom_iters": 100}
SMOKE = {"seeds": [0], "n_src": 8000, "n_tgt": 4000, "iters": 20, "dom_iters": 20}
Q = 0.5                                # fraction of Backdoor/Analysis rows relabelled
TARGET_AUC = 0.89
Z_FEATURES = ("sttl", "ct_state_ttl", "ct_dst_src_ltm")


def zscore(X: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in Z_FEATURES:
        r = rankdata(X[c].to_numpy(float), method="average") / (len(X) + 1)
        cols.append(norm.ppf(r))
    z = np.mean(cols, axis=0)
    return (z - z.mean()) / z.std()


def thin(z: np.ndarray, beta: float, n: int, rng) -> np.ndarray:
    """Bernoulli thinning with keep probability proportional to exp(beta z), about n rows kept."""
    w = np.exp(beta * z) if beta else np.ones(len(z))
    p = np.minimum(1.0, w * n / w.sum())
    for _ in range(5):                                   # re-scale after capping so the size is near n
        p = np.minimum(1.0, p * n / p.sum())
    return np.flatnonzero(rng.random(len(z)) < p)


def relabel(y: np.ndarray, classes, rng) -> np.ndarray:
    y = y.copy()
    idx = {c: i for i, c in enumerate(classes)}
    src = np.flatnonzero(np.isin(y, [idx["Backdoor"], idx["Analysis"]]) & (rng.random(len(y)) < Q))
    y[src] = rng.choice([idx["Exploits"], idx["DoS"], idx["Fuzzers"]], len(src))
    return y


def domain_auc_and_weights(A_src_tr, A_src_val, A_tgt, iters, seed):
    """Held-out domain AUC and Bayes weights for the source validation rows."""
    Xd = np.vstack([A_src_tr, A_tgt])
    d = np.r_[np.zeros(len(A_src_tr), int), np.ones(len(A_tgt), int)]
    tr, te = train_test_split(np.arange(len(d)), test_size=0.3, stratify=d, random_state=seed)
    auc = roc_auc_score(d[te], HGB(max_iter=iters, early_stopping=False, random_state=seed)
                        .fit(Xd[tr], d[tr]).predict_proba(Xd[te])[:, 1])
    dom = HGB(max_iter=iters, early_stopping=False, random_state=seed).fit(Xd, d)
    p = dom.predict_proba(A_src_val)[:, 1]
    return auc, density_ratio_weights(p, len(A_src_tr), len(A_tgt), clip_quantile=0.99)


def calibrate_beta(z_pool, A_pool, A_src, rng, iters, seed):
    lo, hi = 0.0, 4.0
    for _ in range(6):
        mid = (lo + hi) / 2
        idx = thin(z_pool, mid, min(len(z_pool) // 2, 4000), rng)
        d = np.r_[np.zeros(len(A_src), int), np.ones(len(idx), int)]
        Xd = np.vstack([A_src, A_pool[idx]])
        tr, te = train_test_split(np.arange(len(d)), test_size=0.3, stratify=d, random_state=seed)
        auc = roc_auc_score(d[te], HGB(max_iter=iters, early_stopping=False, random_state=seed)
                            .fit(Xd[tr], d[tr]).predict_proba(Xd[te])[:, 1])
        lo, hi = (mid, hi) if auc < TARGET_AUC else (lo, mid)
    return (lo + hi) / 2


def one_seed(seed: int, g: dict, ds, A_all, X_all, y_all) -> list[dict]:
    rng = np.random.default_rng(seed)
    k = len(ds.classes)
    perm = rng.permutation(len(y_all))
    src = perm[: min(g["n_src"], len(perm) // 2)]
    pool = perm[len(perm) // 2:]
    pool = pool[: min(len(pool), g["n_tgt"] * 6)]        # target pool; thinning keeps ~n_tgt of it
    s_tr, s_val = train_test_split(src, test_size=0.3, stratify=y_all[src], random_state=seed)
    model = HGB(max_iter=g["iters"], early_stopping=False, random_state=seed).fit(A_all[s_tr], y_all[s_tr])
    pv = model.predict(A_all[s_val])
    f_val = weighted_macro_f1(y_all[s_val], pv, k)
    z_pool = zscore(X_all.iloc[pool])
    beta = calibrate_beta(z_pool, A_all[pool], A_all[s_tr], rng, max(10, g["dom_iters"] // 2), seed)
    out = []
    for scen, (cov, lab) in {"none": (False, False), "covariate": (True, False),
                             "label_rule": (False, True), "both": (True, True)}.items():
        idx = thin(z_pool, beta if cov else 0.0, g["n_tgt"], rng)
        t = pool[idx]
        y_t = relabel(y_all[t], ds.classes, rng) if lab else y_all[t]
        auc, w = domain_auc_and_weights(A_all[s_tr], A_all[s_val], A_all[t], g["dom_iters"], seed)
        pt = model.predict(A_all[t])
        actual = weighted_macro_f1(y_t, pt, k)
        f_w = weighted_macro_f1(y_all[s_val], pv, k, w)
        out.append({"seed": seed, "scenario": scen, "beta": beta if cov else 0.0, "domain_auc": auc,
                    "val_ordinary": f_val, "val_weighted": f_w, "target_actual": actual,
                    "err_ordinary": f_val - actual, "err_weighted": f_w - actual,
                    "target_rows": len(t), "relabelled": int((y_t != y_all[t]).sum())})
        print(f"seed {seed} {scen:10s} AUC {auc:.3f} ord {f_val:.3f} wtd {f_w:.3f} actual {actual:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    g = SMOKE if args.smoke else FULL
    t0 = time.perf_counter()
    ds = get_dataset()
    X, y = ds.load("train")
    A_all, _ = encode(X, X.iloc[:5], ds.categorical_columns)
    y = np.asarray(y, int)
    rows = []
    for s in g["seeds"]:
        rows += one_seed(s, g, ds, A_all, X, y)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / ("shift_simulation_smoke" if args.smoke else "shift_simulation")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    summ = df.groupby("scenario")[["domain_auc", "val_ordinary", "val_weighted", "target_actual",
                                   "err_ordinary", "err_weighted"]].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(out / "summary.csv")
    (out / "report.md").write_text(f"# Shift simulation ({'smoke' if args.smoke else 'full'})\n\nWall time "
        f"{(time.perf_counter() - t0) / 60:.1f} min. err = validation estimate - actual target macro-F1.\n\n"
        + summ.T.to_string() + "\n", encoding="utf-8")
    print(summ.T.to_string())


if __name__ == "__main__":
    main()
