"""Preprocessing diagnostic: do correlation-aware selection, resampling, de-duplication or log scaling help?

Usage (from the repository root):
    python scripts/diagnose_preprocess.py --smoke              # ~2-3 minutes: checks the pipeline
    python scripts/diagnose_preprocess.py --workers 16         # full run (PC), resumable
    python scripts/diagnose_preprocess.py --summarize-only

The current pipeline (hsboa.preprocess) is: ordinal encoding -> median imputation
-> min-max scaling (fitted on the training part) -> top-d mutual-information
features; class imbalance is handled by class weights with power p
(hsboa.baselines.sample_weight). This checks, one factor at a time and on
validation data only, the steps it does not have (hsboa.prep_variants):

    arm         selection (d = 20)            imbalance                     other
    baseline    MI top 20 (current)           the model's own weight power  -
    sel_corr    MI order, |Spearman| <= 0.95  own                           -
    sel_mrmr    mRMR (MI - mean |Spearman|)   own                           -
    sel_all     all 42 columns                own                           -
    cw0/0.5/1   MI top 20                     class weight power 0/0.5/1    -
    ros, smote  MI top 20                     oversampling to the power-0.5 class sizes, no weights
    dedup       MI top 20                     own                           training duplicates removed
    log         MI top 20                     own                           signed log1p before scaling (MLP only;
                                                                            trees are invariant to it)

Models, with their stored per-seed settings held fixed (no re-tuning): lgbm and rf
(E2), hgb_default (E1b) on seeds 0-9 with every arm but log; mlp (E2) on seeds 0-4
with baseline and log. The baseline arm repeats the stored model (determinism
check). Also writes the Spearman correlation analysis of the 42 inputs (seed 0).
The test partition is never loaded.

Decision rule, fixed before the run:
    A step joins the final pipeline only if it raises validation macro-F1 by
    >= +0.005 over the baseline for at least 2 of the 3 tree models, positive on
    >= 7/10 seeds for each of them. log joins the neural pipeline if it raises the
    MLP by >= +0.005, positive on >= 4/5 seeds. Otherwise the current pipeline
    stays and the variant is reported as an ablation. (A change that joins means
    refitting the E2 models with it.)
Writes results/diag_preprocess[_smoke]/: rows/<model>_s<seed>_<arm>.json, runs.csv,
summary.csv, corr_spearman.csv, high_corr_pairs.csv, corr_spearman.png, report.md.
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

from hsboa.baselines import FIT, available  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.fitness import stratified_subsample  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate  # noqa: E402
from hsboa.prep_variants import (corr_filter, drop_duplicates, mrmr, random_oversample, signed_log1p,  # noqa: E402
                                 smote, spearman, target_counts)
from hsboa.stats import bootstrap_ci, wilcoxon  # noqa: E402

D = 20                                   # features per model, as E2 and the E1b default arm
CORR_ROWS = 20000                        # stratified training rows for the Spearman matrix
# name: (folder with per-seed result.json, model)
SOURCES = {"lgbm": ("results/e2_unsw_baselines/lgbm", "lgbm"), "rf": ("results/e2_unsw_baselines/rf", "rf"),
           "hgb_default": ("results/e1b_unsw_gbdt/default", "hgb"), "mlp": ("results/e2_unsw_baselines/mlp", "mlp")}
TREE_ARMS = ["baseline", "sel_corr", "sel_mrmr", "sel_all", "cw0", "cw0.5", "cw1", "ros", "smote", "dedup"]
NEURAL_ARMS = ["baseline", "log"]
TREES = ["lgbm", "rf", "hgb_default"]
FULL = {"tree_seeds": list(range(10)), "neural_seeds": list(range(5)), "models": TREES + ["mlp"]}
SMOKE = {"tree_seeds": [0], "neural_seeds": [0], "models": ["rf", "hgb_default", "mlp"]}
SMOKE_SETTINGS = {"hgb": {"max_iter": 20}, "rf": {"n_estimators": 20}, "mlp": {"epochs": 2}}
REPORT_CLASSES = ["DoS", "Analysis", "Backdoor", "Shellcode", "Worms"]


def arms_for(name: str) -> list[str]:
    return NEURAL_ARMS if SOURCES[name][1] == "mlp" else TREE_ARMS


def _fold(seed: int, log: bool):
    ds = get_dataset()
    X, y = ds.load("train")                 # the test partition is not loaded
    if log:
        X = signed_log1p(X, [c for c in ds.feature_columns if c not in ds.categorical_columns])
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    dcfg["features"]["selection"] = "all"   # every column, scaled, in descending MI order
    return prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, dataset=ds), ds


def run_task(name: str, seed: int, arm: str, smoke: bool, out: Path) -> int:
    path = out / "rows" / f"{name}_s{seed}_{arm}.json"
    if path.exists():
        return 0
    folder, model = SOURCES[name]
    rec = json.loads((ROOT / folder / f"seed_{seed:02d}" / "result.json").read_text(encoding="utf-8"))
    s = {k: v for k, v in rec["settings"].items() if k not in ("features", "n_selected")}
    if smoke:
        s.update(SMOKE_SETTINGS.get(model, {}))
    fold, ds = _fold(seed, log=(arm == "log"))
    names = fold.feature_names
    Xtr, ytr = np.asarray(fold.X_train, dtype=float), np.asarray(fold.y_train)
    Xva, yva = np.asarray(fold.X_val, dtype=float), np.asarray(fold.y_val)
    k = len(fold.classes)

    if arm in ("sel_corr", "sel_mrmr"):
        rows = stratified_subsample(ytr, CORR_ROWS, np.random.SeedSequence([seed, 7]))
        C = spearman(Xtr[rows])
        mi = fold.pipeline.named_steps["select"].mutual_info_
        cols = corr_filter(C, D) if arm == "sel_corr" else mrmr([mi[n] for n in names], C, D)
    elif arm == "sel_all":
        cols = list(range(len(names)))
    else:
        cols = list(range(D))
    Xtr, Xva = Xtr[:, cols], Xva[:, cols]
    if arm.startswith("cw"):
        s["class_weight_power"] = float(arm[2:])
    elif arm in ("ros", "smote"):
        s["class_weight_power"] = 0.0
        t = target_counts(ytr, k, power=0.5)
        rng = np.random.default_rng([seed, 11])
        if arm == "ros":
            Xtr, ytr = random_oversample(Xtr, ytr, t, rng)
        else:
            cat = [i for i, c in enumerate(cols) if names[c] in ds.categorical_columns]
            Xtr, ytr = smote(Xtr, ytr, t, rng, categorical=cat)
    elif arm == "dedup":
        Xtr, ytr = drop_duplicates(Xtr, ytr)

    t0 = time.perf_counter()
    predict, _ = FIT[model](s, Xtr, ytr, Xva, yva, seed)
    rep = evaluate(yva, predict(Xva).argmax(axis=1), classes=fold.classes)
    row = {"model": name, "seed": seed, "arm": arm, "n_features": len(cols), "features": [names[c] for c in cols],
           "train_rows": int(len(ytr)), "fit_s": time.perf_counter() - t0,
           "class_weight_power": s.get("class_weight_power", 0.0),
           "stored_val_macro_f1": rec["val"]["macro_f1"],
           **{f"val_{m}": float(v) for m, v in rep.summary.items()},
           **{f"val_f1_{c}": float(rep.per_class.loc[c, "f1"]) for c in fold.classes}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"{name} s{seed} {arm}: val macro-F1 {row['val_macro_f1']:.4f} ({row['fit_s']:.0f} s)", flush=True)
    return 1


def correlation_analysis(out: Path, smoke: bool) -> None:
    """Spearman matrix of all 42 scaled inputs on the seed-0 training part (stratified subsample)."""
    if (out / "corr_spearman.csv").exists():
        return
    fold, _ = _fold(0, log=False)
    X, y = np.asarray(fold.X_train, dtype=float), np.asarray(fold.y_train)
    rows = stratified_subsample(y, 3000 if smoke else CORR_ROWS, np.random.SeedSequence([0, 7]))
    names = fold.feature_names
    C = pd.DataFrame(spearman(X[rows]), index=names, columns=names)
    C.to_csv(out / "corr_spearman.csv")
    pairs = [{"a": names[i], "b": names[j], "spearman": C.iat[i, j], "mi_rank_a": i + 1, "mi_rank_b": j + 1}
             for i in range(len(names)) for j in range(i + 1, len(names)) if abs(C.iat[i, j]) >= 0.9]
    pd.DataFrame(pairs, columns=["a", "b", "spearman", "mi_rank_a", "mi_rank_b"]) \
        .sort_values("spearman", key=np.abs, ascending=False).to_csv(out / "high_corr_pairs.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(C.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title("Spearman correlation of the 42 inputs (training part, seed 0; MI order)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / "corr_spearman.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def summarize(out: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "rows").glob("*.json"))]
    if not rows:
        print("no finished fits yet")
        return pd.DataFrame()
    runs = pd.DataFrame(rows)
    runs.drop(columns="features").to_csv(out / "runs.csv", index=False)
    f1 = [f"val_f1_{c}" for c in REPORT_CLASSES if f"val_f1_{c}" in runs]
    out_rows, ref = [], {}
    for (model, arm), g in runs.groupby(["model", "arm"]):
        base = runs[(runs.model == model) & (runs.arm == "baseline")].set_index("seed")
        g = g.set_index("seed")
        common = sorted(set(g.index) & set(base.index))
        d = (g.loc[common, "val_macro_f1"] - base.loc[common, "val_macro_f1"]).to_numpy()
        lo, hi = bootstrap_ci(d) if len(d) else (np.nan, np.nan)
        out_rows.append({"model": model, "arm": arm, "seeds": len(g), "val_macro_f1": g["val_macro_f1"].mean(),
                         "sd": g["val_macro_f1"].std(), "diff_vs_baseline": d.mean() if len(d) else np.nan,
                         "ci95_low": lo, "ci95_high": hi, "seeds_positive": int((d > 0).sum()),
                         "p": wilcoxon(g.loc[common, "val_macro_f1"], base.loc[common, "val_macro_f1"])
                         if arm != "baseline" and len(common) > 1 else np.nan,
                         "n_features": g["n_features"].mean(), "train_rows": g["train_rows"].mean(),
                         "fit_s": g["fit_s"].mean(),
                         **{c[4:] + "_diff": (g.loc[common, c] - base.loc[common, c]).mean() for c in f1}})
        if arm == "baseline":
            ref[model] = (g["stored_val_macro_f1"] - g["val_macro_f1"]).abs().max()
    summ = pd.DataFrame(out_rows).sort_values(["model", "diff_vs_baseline"], ascending=[True, False])
    summ.to_csv(out / "summary.csv", index=False)

    # Decision rule (fixed before the run; see the module docstring).
    verdict = []
    trees = [m for m in TREES if m in set(runs.model)]
    for arm in [a for a in TREE_ARMS if a != "baseline"]:
        sub = summ[(summ.arm == arm) & summ.model.isin(trees)]
        good = sub[(sub.diff_vs_baseline >= 0.005) & (sub.seeds_positive >= np.minimum(7, sub.seeds))]
        verdict.append({"arm": arm, "models_meeting": ", ".join(good.model) or "-",
                        "joins_pipeline": "yes" if len(good) >= min(2, len(trees)) else "no"})
    sub = summ[(summ.arm == "log") & (summ.model == "mlp")]
    if not sub.empty:
        r = sub.iloc[0]
        verdict.append({"arm": "log (mlp)", "models_meeting": "mlp" if r.diff_vs_baseline >= 0.005
                        and r.seeds_positive >= min(4, r.seeds) else "-",
                        "joins_pipeline": "yes" if r.diff_vs_baseline >= 0.005 and r.seeds_positive >= min(4, r.seeds)
                        else "no"})
    verdict = pd.DataFrame(verdict)

    feats = runs[runs.arm.isin(["sel_corr", "sel_mrmr", "baseline"]) & runs.model.isin(trees[:1])]
    freq = (feats.explode("features").groupby(["arm", "features"]).size().rename("seeds").reset_index()
            .sort_values(["arm", "seeds"], ascending=[True, False]))
    freq.to_csv(out / "features.csv", index=False)
    det = pd.DataFrame([{"model": m, "max_abs_diff_stored_vs_baseline": v} for m, v in ref.items()])

    lines = ["# Preprocessing diagnostic (validation only)", "",
             "One factor at a time against the current pipeline (baseline: MI top 20, the model's own class-weight "
             "power), stored per-seed settings held fixed. Differences are paired by seed against the baseline arm. "
             "The test partition was not used.", "",
             "## Decision rule (>= +0.005 on >= 2 of 3 tree models, >= 7/10 seeds; MLP log >= +0.005, >= 4/5)", "",
             _md(verdict), "", "## Validation macro-F1 by model and arm", "", _md(summ), ""]
    if (out / "high_corr_pairs.csv").exists():
        hp = pd.read_csv(out / "high_corr_pairs.csv")
        lines += [f"## Correlation analysis: {len(hp)} input pairs with |Spearman| >= 0.9 (seed 0, training part)",
                  "", _md(hp) if not hp.empty else "(none)", "",
                  "Full matrix: corr_spearman.csv; heat map: corr_spearman.png.", ""]
    lines += ["## Determinism check: baseline arm vs the stored E1b/E2 validation macro-F1", "", _md(det), ""]
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
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (one task = model x seed x arm)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    grid = copy.deepcopy(SMOKE if args.smoke else FULL)
    out = ROOT / "results" / ("diag_preprocess_smoke" if args.smoke else "diag_preprocess")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        missing = [n for n in grid["models"] if not available(SOURCES[n][1])]
        if missing and not args.smoke:
            raise SystemExit(f"models {missing} need libraries that are not installed (see SETUP_NEW_MACHINE.md)")
        grid["models"] = [n for n in grid["models"] if n not in missing]
        absent = [n for n in grid["models"] if not (ROOT / SOURCES[n][0]).exists()]
        if absent:
            raise SystemExit(f"stored results missing for {absent}: run E1b/E2 first")
        correlation_analysis(out, args.smoke)
        tasks = [(n, s, a) for n in grid["models"]
                 for s in (grid["neural_seeds"] if SOURCES[n][1] == "mlp" else grid["tree_seeds"])
                 for a in arms_for(n)]
        tasks.sort(key=lambda t: (t[0] != "mlp", t[2] not in ("smote", "ros", "sel_all")))   # slowest first
        print(f"{len(tasks)} tasks, {args.workers} workers; results in {out}", flush=True)
        t0 = time.perf_counter()
        if args.workers > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.workers, verbose=5)(delayed(run_task)(n, s, a, args.smoke, out)
                                                     for n, s, a in tasks)
        else:
            for n, s, a in tasks:
                run_task(n, s, a, args.smoke, out)
        print(f"all tasks finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    summ = summarize(out)
    if not summ.empty:
        print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
