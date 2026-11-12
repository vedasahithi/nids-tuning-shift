"""Headroom diagnostic: which pipeline factor limits macro-F1 on UNSW-NB15?

Usage (from the repository root):
    python scripts/diagnose_headroom.py --smoke              # ~1 minute: checks the pipeline
    python scripts/diagnose_headroom.py --workers 20         # full grid, resumable
    python scripts/diagnose_headroom.py --summarize-only

This measures how far fast tree models move validation
macro-F1 when one pipeline factor changes, with fixed, untuned settings:
    d            number of MI-ranked features: 5, 10 (current), 15, 20, 30, all (42)
    train rows   30000 (the E1/E2 search subsample) or the full training part
    class weight power 0 (none), 0.5, 1 (balanced); see hsboa.baselines.sample_weight
    model        rf, hgb
on seeds 0-2. Everything is scored on the validation split of the official
training partition: the test partition is never loaded (no selection on test).
Writes results/diag_headroom[_smoke]/: rows/<task>.json (one per fit, so a rerun
resumes), runs.csv, summary.csv, report.md.
"""

from __future__ import annotations

import argparse
import copy
import json
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

from hsboa.baselines import FIT  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.fitness import stratified_subsample  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate, predict_labels  # noqa: E402

GRID = {"seeds": [0, 1, 2], "d": [5, 10, 15, 20, 30, "all"], "train_rows": [30000, None],
        "class_weight_power": [0.0, 0.5, 1.0], "models": ["rf", "hgb"]}
SMOKE = {"seeds": [0], "d": [5, "all"], "train_rows": [3000], "class_weight_power": [0.0, 1.0],
         "models": ["rf", "hgb"]}
SETTINGS = {    # fixed, common defaults (no tuning): the diagnostic compares factors, not models
    "rf": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "hgb": {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20,
            "l2_regularization": 0.0},
}
SMOKE_SETTINGS = {"rf": {"n_estimators": 20}, "hgb": {"max_iter": 20}}
REFERENCE = "E1 QRNN (d = 10, tuned): val macro-F1 HSBOA 0.432, random search 0.449 (10 seeds)"


def tag(model: str, seed: int, d, rows, cw: float) -> str:
    return f"{model}_s{seed}_d{d}_n{rows or 'full'}_cw{cw:g}"


def run_task(model: str, seed: int, d, grid: dict, settings: dict, out: Path) -> int:
    """All fits for one (model, seed, d): the fold is prepared once, then train rows x class weights."""
    todo = [(rows, cw) for rows in grid["train_rows"] for cw in grid["class_weight_power"]
            if not (out / "rows" / f"{tag(model, seed, d, rows, cw)}.json").exists()]
    if not todo:
        return 0
    ds = get_dataset()
    X, y = ds.load("train")                 # the test partition is not loaded
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    if d == "all":
        dcfg["features"]["selection"] = "all"
    else:
        dcfg["features"]["n_features"] = int(d)
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, dataset=ds)
    for rows, cw in todo:
        idx = stratified_subsample(fold.y_train, rows, np.random.SeedSequence([seed, 2]))
        s = {**settings[model], "class_weight_power": cw, "n_classes": len(fold.classes)}
        t0 = time.perf_counter()
        predict, _ = FIT[model](s, fold.X_train[idx], fold.y_train[idx], fold.X_val, fold.y_val, seed)
        rep = evaluate(fold.y_val, predict_labels(predict(fold.X_val)), classes=fold.classes)
        rec = {"model": model, "seed": seed, "d": str(d), "n_features": fold.X_train.shape[1],
               "train_rows": len(idx), "train_rows_setting": rows or "full", "class_weight_power": cw,
               "fit_s": time.perf_counter() - t0, "features": fold.feature_names,
               **{f"val_{k}": float(v) for k, v in rep.summary.items()},
               **{f"val_f1_{c}": float(rep.per_class.loc[c, "f1"]) for c in fold.classes}}
        (out / "rows" / f"{tag(model, seed, d, rows, cw)}.json").write_text(json.dumps(rec, indent=2),
                                                                           encoding="utf-8")
        print(f"{tag(model, seed, d, rows, cw)}: val macro-F1 {rec['val_macro_f1']:.4f} "
              f"({rec['fit_s']:.0f} s)", flush=True)
    return len(todo)


def summarize(out: Path) -> pd.DataFrame:
    runs = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "rows").glob("*.json"))])
    if runs.empty:
        print("no finished fits yet")
        return runs
    runs.drop(columns="features").to_csv(out / "runs.csv", index=False)
    keys = ["model", "d", "n_features", "train_rows_setting", "class_weight_power"]
    f1_cols = [c for c in runs if c.startswith("val_f1_")]
    summ = (runs.groupby(keys)
            .agg(seeds=("seed", "nunique"), macro_f1=("val_macro_f1", "mean"),
                 macro_f1_std=("val_macro_f1", "std"), mcc=("val_mcc", "mean"),
                 fit_s=("fit_s", "mean"), **{c[4:]: (c, "mean") for c in f1_cols})
            .reset_index().sort_values("macro_f1", ascending=False))
    summ.to_csv(out / "summary.csv", index=False)

    lines = ["# Headroom diagnostic (validation only)", "",
             f"Fixed, untuned rf/hgb settings; seeds {sorted(runs['seed'].unique().tolist())}; "
             "mean validation macro-F1 over seeds. The test partition was not used.", "",
             f"Reference: {REFERENCE}.", ""]
    for model in sorted(runs["model"].unique()):
        sub = summ[summ["model"] == model]
        piv = sub.pivot_table(index="n_features", values="macro_f1",
                              columns=["train_rows_setting", "class_weight_power"])
        piv.columns = [f"rows {r}, cw {c:g}" for r, c in piv.columns]
        lines += [f"## {model}: macro-F1 by d (rows) and training rows / class weight (columns)", "",
                  _md(piv.reset_index().rename(columns={"n_features": "d"})), ""]
    show = ["model", "n_features", "train_rows_setting", "class_weight_power", "seeds", "macro_f1",
            "macro_f1_std", "mcc", "fit_s"] + [c[4:] for c in f1_cols]
    lines += ["## Top 10 settings", "", _md(summ[show].head(10)), "",
              "## Current pipeline (d = 10, 30000 rows)", "",
              _md(summ[(summ["n_features"] == 10) & (summ["train_rows_setting"] == 30000)][show]), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summ


def _md(df: pd.DataFrame) -> str:
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    return "\n".join([head, rule] + ["| " + " | ".join(cell(v) for v in row) + " |"
                                     for row in df.itertuples(index=False)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (one task = model x seed x d)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    grid = SMOKE if args.smoke else GRID
    settings = copy.deepcopy(SETTINGS)
    if args.smoke:
        for m, extra in SMOKE_SETTINGS.items():
            settings[m].update(extra)
    out = ROOT / "results" / ("diag_headroom_smoke" if args.smoke else "diag_headroom")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        tasks = [(m, s, d) for m in grid["models"] for s in grid["seeds"] for d in grid["d"]]
        tasks.sort(key=lambda t: (t[2] != "all", t[0] != "rf"))      # costliest first
        print(f"{len(tasks)} tasks, {args.workers} workers; results in {out}", flush=True)
        t0 = time.perf_counter()
        if args.workers > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.workers, verbose=5)(delayed(run_task)(m, s, d, grid, settings, out)
                                                     for m, s, d in tasks)
        else:
            for m, s, d in tasks:
                run_task(m, s, d, grid, settings, out)
        print(f"all tasks finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    summ = summarize(out)
    if not summ.empty:
        print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
