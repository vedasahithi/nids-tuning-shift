"""Feasibility gate for shift-aware (importance-weighted) model selection with HSBOA.

Usage (from the repository root):
    python scripts/gate_iw.py --smoke             # ~1 minute: checks the pipeline
    python scripts/gate_iw.py --workers 10        # full run (PC)

Idea (2026-10-30): the shift diagnostic found the test partition separable from the
training partition (class-mix-matched AUC 0.897) with the shift spread over many
features, so feature selection cannot remove it. Instead the optimizer's fitness would
become validation macro-F1 with each validation row weighted by how test-like it is,
w(x) = p_test(x) / p_train(x) (importance-weighted validation, Sugiyama et al. 2007).
With little overlap such weights can be extreme and make the fitness noisy - the very
problem that made all optimizers tie - so this checks first whether the weights leave
enough effective data. Test features are used, test labels are not (they are
discarded when the file is read).

Per seed 0-9: a domain classifier (HGB) separates the seed's training part (domain 0)
from the test partition's features (domain 1) and scores the seed's validation rows
(never seen by it); weights from Bayes' rule, capped at their 99th percentile, mean 1.
The stored E2 LightGBM validation predictions (results/diag_twostage/proba) are scored
unweighted and weighted, with bootstrap standard deviations (200 resamples).

Decision rule, fixed before the run (capped weights):
    G1 effective sample size / validation rows >= 0.20 (mean of seeds, and on >= 8/10 seeds)
    G2 for every class, mean over seeds of its own ESS ratio >= 0.10
    G3 bootstrap sd of the weighted macro-F1 <= 2 x the unweighted sd (mean of seeds)
    All met -> build shift-aware HSBOA (step 3). Otherwise the weighted fitness is too noisy
    and the paper goes with the audit + benchmark + conflict-aware ceiling.
The weighted estimates are not compared with test scores (no test labels in method choices).
Writes results/gate_iw[_smoke]/: seeds.csv, classes.csv, weights_s<seed>.npy, report.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per fit; parallelism comes from the workers
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hsboa.data import CLASSES, make_split  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.shift import density_ratio_weights, encode, ess, weighted_macro_f1  # noqa: E402

PROBA = "results/diag_twostage/proba/lgbm_s{seed}.npz"   # stored E2 LightGBM validation probabilities
FULL = {"seeds": list(range(10)), "max_iter": 200, "boot": 200}
SMOKE = {"seeds": [0], "max_iter": 30, "boot": 20}
CLIP = 0.99
G1, G1_SEEDS, G2, G3 = 0.20, 8, 0.10, 2.0


def gate_task(seed: int, grid: dict, out: Path) -> tuple[dict, list[dict]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.perf_counter()
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, _ = ds.load("test")               # features only, labels discarded
    y = np.asarray(y)
    A, B = encode(X, X_te, ds.categorical_columns)
    sp = make_split(y, seed=seed)
    Xd = np.vstack([A[sp.train_idx], B])
    d = np.r_[np.zeros(len(sp.train_idx), int), np.ones(len(B), int)]
    dom = HistGradientBoostingClassifier(max_iter=grid["max_iter"], learning_rate=0.1, max_leaf_nodes=31,
                                         early_stopping=False, random_state=seed).fit(Xd, d)
    p = dom.predict_proba(A[sp.val_idx])[:, 1]
    w_raw = density_ratio_weights(p, len(sp.train_idx), len(B), clip_quantile=None)
    w = density_ratio_weights(p, len(sp.train_idx), len(B), clip_quantile=CLIP)
    np.save(out / f"weights_s{seed}.npy", w)

    z = np.load(ROOT / PROBA.format(seed=seed), allow_pickle=False)
    yv = y[sp.val_idx]
    if not np.array_equal(z["y"], yv):
        raise RuntimeError(f"seed {seed}: stored validation labels do not match the split")
    k = len(CLASSES)
    pred = z["P_flat"].argmax(axis=1)
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(yv), len(yv)) for _ in range(grid["boot"])]
    sd_u = float(np.std([weighted_macro_f1(yv[b], pred[b], k) for b in boots]))
    sd_w = float(np.std([weighted_macro_f1(yv[b], pred[b], k, w[b]) for b in boots]))
    n = len(yv)
    rec = {"seed": seed, "val_rows": n, "ess_ratio": ess(w) / n, "ess_ratio_uncapped": ess(w_raw) / n,
           "w_p50": float(np.median(w)), "w_p99_uncapped": float(np.quantile(w_raw, 0.99)),
           "w_max_uncapped": float(w_raw.max()), "macro_f1": weighted_macro_f1(yv, pred, k),
           "macro_f1_weighted": weighted_macro_f1(yv, pred, k, w), "sd": sd_u, "sd_weighted": sd_w,
           "sd_ratio": sd_w / sd_u if sd_u > 0 else np.nan, "wall_s": time.perf_counter() - t0}
    cls = [{"seed": seed, "class": c, "val_rows": int((yv == i).sum()),
            "ess_ratio": ess(w[yv == i]) / max(1, int((yv == i).sum())),
            "mean_weight": float(w[yv == i].mean()) if (yv == i).any() else np.nan}
           for i, c in enumerate(CLASSES)]
    print(f"seed {seed}: ESS ratio {rec['ess_ratio']:.3f} (uncapped {rec['ess_ratio_uncapped']:.3f}), "
          f"sd ratio {rec['sd_ratio']:.2f} ({rec['wall_s']:.0f} s)", flush=True)
    return rec, cls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    grid = SMOKE if args.smoke else FULL
    out = ROOT / "results" / ("gate_iw_smoke" if args.smoke else "gate_iw")
    out.mkdir(parents=True, exist_ok=True)
    missing = [s for s in grid["seeds"] if not (ROOT / PROBA.format(seed=s)).exists()]
    if missing:
        raise SystemExit(f"stored LightGBM validation probabilities missing for seeds {missing} "
                         f"(results/diag_twostage/proba from the two-stage diagnostic)")
    t0 = time.perf_counter()
    print(f"{len(grid['seeds'])} seeds, {args.workers} workers; results in {out}", flush=True)
    if args.workers > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.workers)(delayed(gate_task)(s, grid, out) for s in grid["seeds"])
    else:
        res = [gate_task(s, grid, out) for s in grid["seeds"]]
    seeds = pd.DataFrame([r for r, _ in res])
    classes = pd.DataFrame([c for _, cl in res for c in cl])
    seeds.to_csv(out / "seeds.csv", index=False)
    classes.to_csv(out / "classes.csv", index=False)
    write_report(out, seeds, classes, time.perf_counter() - t0)
    print((out / "report.md").read_text(encoding="utf-8"))


def write_report(out: Path, seeds: pd.DataFrame, classes: pd.DataFrame, wall_s: float) -> None:
    n = len(seeds)
    need = min(G1_SEEDS, n)
    g1_mean = float(seeds.ess_ratio.mean())
    g1_seeds = int((seeds.ess_ratio >= G1).sum())
    ok1 = g1_mean >= G1 and g1_seeds >= need
    per_class = classes.groupby("class", sort=False).agg(val_rows=("val_rows", "mean"), ess_ratio=("ess_ratio", "mean"),
                                                          mean_weight=("mean_weight", "mean")).reset_index()
    worst = per_class.loc[per_class.ess_ratio.idxmin()]
    ok2 = bool((per_class.ess_ratio >= G2).all())
    g3 = float(seeds.sd_ratio.mean())
    ok3 = g3 <= G3
    verdict = ("PASS -> build shift-aware HSBOA (step 3)" if ok1 and ok2 and ok3 else
               "FAIL -> weighted fitness too noisy; audit + benchmark + conflict-aware ceiling")

    def md(df):
        def cell(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                         + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])

    lines = ["# Feasibility gate: importance-weighted validation", "", f"Wall time {wall_s / 60:.1f} min.",
             "", "## Decision rule (weights capped at the 99th percentile)", "",
             f"- G1 ESS ratio: mean {g1_mean:.4f} (>= {G1}), seeds >= {G1}: {g1_seeds}/{n} (need {need}) "
             f"-> **{'met' if ok1 else 'not met'}**",
             f"- G2 per-class ESS ratio: lowest {worst['class']} {worst['ess_ratio']:.4f} (>= {G2}) "
             f"-> **{'met' if ok2 else 'not met'}**",
             f"- G3 noise: weighted / unweighted bootstrap sd {g3:.2f} (<= {G3}) -> **{'met' if ok3 else 'not met'}**",
             f"- Verdict: **{verdict}**", "",
             "## Per seed", "", md(seeds), "", "## Per class (mean over seeds)", "", md(per_class), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
