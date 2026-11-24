"""Unsearched full-feature baseline: library settings, all 42 features, no class weights.

The untuned arms of the paper (HGB default, LightGBM default) use the 20 most informative features and a
class-weight power of 0.5 (the best cell of a validation-only pilot). This script scores the plain library
configuration on the same ten seeds, splits and test partition, so that the effect of those two choices can
be read separately:

  hgb_plain   HistGradientBoosting, the settings of the HGB default (learning rate 0.1, 200 iterations, 31 leaves,
              minimum 20 records per leaf, no L2), all 42 features, class-weight power 0
  lgbm_plain  LightGBM library defaults (learning rate 0.1, 100 trees, 31 leaves, minimum 20 records per leaf),
              all 42 features, class-weight power 0

Writes results/library_baseline/{rows/<arm>/seed_XX.json, summary.csv, contrasts.csv, report.md}.
Usage (from the repository root):
    python scripts/library_baseline.py --workers 10        (full; LightGBM needed for lgbm_plain)
    python scripts/library_baseline.py --arms hgb_plain --seeds 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "results" / "library_baseline"
SETTINGS = {
    "hgb_plain": ("hgb", {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                          "l2_regularization": 0.0, "class_weight_power": 0.0}),
    "lgbm_plain": ("lgbm", {"learning_rate": 0.1, "n_estimators": 100, "num_leaves": 31, "min_child_samples": 20,
                            "reg_lambda": 0.0, "subsample": 1.0, "colsample_bytree": 1.0, "class_weight_power": 0.0}),
}
# the paper's untuned arms (20 features, class-weight power 0.5) that each plain arm is compared with
REFERENCE = {"hgb_plain": ("e1b_unsw_gbdt", "default"), "lgbm_plain": ("e6_unsw_shift", "default")}


def run_one(arm: str, seed: int) -> dict:
    path = OUT / "rows" / arm / f"seed_{seed:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from hsboa.baselines import FIT
    from hsboa.datasets import get_dataset
    from hsboa.folds import load_config, prepare_fold
    from hsboa.metrics import evaluate, predict_labels
    t0 = time.perf_counter()
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    cfg["features"].update(n_features=42, selection="mutual_info")
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=cfg, X_test=X_te, y_test=np.asarray(y_te, int),
                        dataset=ds)
    assert fold.X_train.shape[1] == 42, fold.X_train.shape
    model, s = SETTINGS[arm]
    predict, _ = FIT[model]({**s, "n_classes": len(ds.classes)}, fold.X_train, fold.y_train, fold.X_val,
                           fold.y_val, seed)
    rec = {"arm": arm, "seed": seed, "settings": s, "n_features": int(fold.X_train.shape[1])}
    for part, Xp, yp in (("val", fold.X_val, fold.y_val), ("test", fold.X_test, fold.y_test)):
        rep = evaluate(yp, predict_labels(predict(Xp)), classes=fold.classes)
        rec[part] = {"macro_f1": rep.summary["macro_f1"], "accuracy": rep.summary["accuracy"],
                     **{f"f1_{c}": float(rep.per_class.loc[c, "f1"]) for c in fold.classes}}
    rec["wall_s"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    print(f"{arm} seed {seed}: val {rec['val']['macro_f1']:.4f} test {rec['test']['macro_f1']:.4f} "
          f"({rec['wall_s'] / 60:.1f} min)", flush=True)
    return rec


def _task(args):
    return run_one(*args)


def summarize() -> None:
    from hsboa.stats import bootstrap_ci, wilcoxon
    rows, cons, lines = [], [], ["# Unsearched full-feature baseline", ""]
    for arm in SETTINGS:
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "rows" / arm).glob("seed_*.json"))]
        if not recs:
            continue
        exp, sub = REFERENCE[arm]
        ref = {}
        for p in sorted((ROOT / "results" / exp / sub).glob("seed_*/result.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            ref[r["seed"]] = r
        for part in ("val", "test"):
            x = np.array([r[part]["macro_f1"] for r in recs])
            seeds = [r["seed"] for r in recs]
            base = np.array([ref[s][part]["macro_f1"] for s in seeds])
            d = x - base
            lo, hi = bootstrap_ci(d)
            rows.append({"arm": arm, "part": part, "n_seeds": len(x), "macro_f1": x.mean(), "sd": x.std(ddof=1),
                         "reference_macro_f1": base.mean()})
            cons.append({"arm": arm, "part": part, "reference": f"{exp}/{sub}", "diff": d.mean(), "ci95_lo": lo,
                         "ci95_hi": hi, "seeds_better": int((d > 0).sum()), "wilcoxon_p": wilcoxon(x, base)})
    summ, con = pd.DataFrame(rows), pd.DataFrame(cons)
    summ.to_csv(OUT / "summary.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)
    lines += [summ.round(4).to_string(index=False), "", "Plain minus the paper's untuned arm (20 features, alpha 0.5):", "",
              con.round(4).to_string(index=False)]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", choices=list(SETTINGS), default=list(SETTINGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    jobs = [(a, s) for s in args.seeds for a in args.arms]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(_task, jobs))
    else:
        for j in jobs:
            run_one(*j)
    summarize()


if __name__ == "__main__":
    main()
