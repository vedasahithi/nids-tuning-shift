"""Tuning ablation (post hoc): does a better-designed search beat the default?

Motivation. In the main experiment the tuned HGB arms score below the default on validation as well as on test
(0.626-0.634 against 0.645). That could mean "tuning does not help" or only "our search setup is weak": the
fitness used models trained on a 30,000-record sample, the search covered a 42-dimensional feature mask, and
the defaults were not among the candidates. This ablation removes those three handicaps:
  * hyper-parameters only (6 settings, ranges as in the main experiment) on the top-20 mutual-information
    features, the same features as the default arm;
  * fitness = validation macro-F1 of a model trained on the FULL training part (140,272 records);
  * the default settings are the first candidate of every search (x0), so a search cannot end below the default
    on validation (up to the L2 term, which the log-scale range cannot set to exactly 0: 1e-6 instead of 0).
Arms: default (no search), random search, SBOA; 60 trainings each; population 10; seeds 0-9 (paired).
SBOA is used, not HSBOA: with 60 evaluations and a population of 10 there are only 3 iterations, so the
harmonic term (which needs 10 iterations of history) would never start.

Decision rule, fixed before the run. Primary outcome: test macro-F1 of the tuned arm minus the default arm, paired
by seed, mean difference with a 95% percentile bootstrap interval over seeds and a two-sided exact Wilcoxon
test, Holm over the two arms. "Tuning helps in this setup" if the interval lies above 0 and the Holm p < 0.05;
"no benefit" if the 90% interval lies inside +-0.01; otherwise "inconclusive". Validation is secondary. The
test labels are read once per run after the search (post hoc analysis, not part of the pre-specified study).

Writes results/ablation_tuning[_smoke]/{rows/*.json, trials/*.csv, runs.csv, summary.csv, report.md}.
Usage (from the repository root):  python scripts/ablation_tuning.py --smoke
                                   python scripts/ablation_tuning.py --workers 20
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
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hsboa.baselines import FIT, BaselineFitness  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate, predict_labels  # noqa: E402
from hsboa.optimizers import get_optimizer  # noqa: E402
from hsboa.stats import bootstrap_ci, holm, wilcoxon  # noqa: E402

N_FEATURES = 20
SPACE = {"log10_learning_rate": [-2.5, -0.5], "int_max_iter": [100, 400], "int_max_leaf_nodes": [15, 127],
         "int_min_samples_leaf": [5, 100], "log10_l2_regularization": [-6, 0], "class_weight_power": [0, 1]}
DEFAULT = {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20,
           "l2_regularization": 0.0, "class_weight_power": 0.5}
FULL = {"seeds": list(range(10)), "budget": 60, "pop": 10, "arms": ["random", "sboa"], "rows": None}
SMOKE = {"seeds": [0], "budget": 12, "pop": 4, "arms": ["random", "sboa"], "rows": 4000}
MARGINS = (0.005, 0.01)


def default_point(space: dict) -> np.ndarray:
    """The default settings as a point of [0, 1]^k (the L2 term maps to the lower end, 1e-6)."""
    x = []
    for k, (lo, hi) in space.items():
        name = k.split("_", 1)[1] if k.startswith(("log10_", "int_")) else k
        v = DEFAULT[name]
        if k.startswith("log10_"):
            v = np.log10(max(v, 10 ** lo))
        x.append((v - lo) / (hi - lo))
    return np.array(x)


def macro_f1(y, proba, classes) -> tuple[float, float]:
    rep = evaluate(y, predict_labels(proba), classes=classes)
    return rep.summary["macro_f1"], rep.summary["accuracy"]


def task(seed: int, arm: str, grid: dict, out: Path) -> dict:
    path = out / "rows" / f"{arm}_seed{seed:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    dcfg["features"]["selection"] = "all"
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, X_test=X_te, y_test=y_te, dataset=ds)
    Xtr, Xva, Xte = (np.asarray(a)[:, :N_FEATURES] for a in (fold.X_train, fold.X_val, fold.X_test))
    ytr, yva, yte = fold.y_train, fold.y_val, fold.y_test
    space, default, budget, pop = SPACE, dict(DEFAULT), grid["budget"], grid["pop"]
    if grid["rows"]:                                        # smoke: tiny data and a tiny model
        rng = np.random.default_rng(0)
        i, j, k = (np.sort(rng.choice(len(a), min(grid["rows"], len(a)), replace=False)) for a in (Xtr, Xva, Xte))
        Xtr, ytr, Xva, yva, Xte, yte = Xtr[i], ytr[i], Xva[j], yva[j], Xte[k], yte[k]
        space = {**SPACE, "int_max_iter": [10, 20]}
        default["max_iter"] = 20
    classes, kcls = fold.classes, len(fold.classes)
    fixed = {"n_classes": kcls}
    rec = {"arm": arm, "seed": seed, "budget": 0 if arm == "default" else budget}
    if arm == "default":
        best, trials = {**default, **fixed}, pd.DataFrame()
        predict, _ = FIT["hgb"](best, Xtr, ytr, Xva, yva, seed)
        rec.update(distinct_trainings=1, search_best_val=None)
    else:
        fit = BaselineFitness("hgb", space, fixed, Xtr, ytr, Xva, yva, seed, cache_path=None, classes=classes)
        x0 = default_point(space)
        res = get_optimizer(arm, x0=x0)(fit, np.zeros(len(space)), np.ones(len(space)), pop, budget, seed)
        best = fit.settings(res.best_x)
        trials = pd.DataFrame(fit.trials)
        keys = [c for c in trials.columns if c not in {"val_macro_f1", "fitness", "wall_s", "gradient_steps", "trial", "cached"}]
        hist = 1.0 - np.asarray(res.history)
        rec.update(distinct_trainings=int((~trials.duplicated(subset=keys)).sum()), search_best_val=float(hist[-1]),
                   first_candidate_val=float(trials["val_macro_f1"].iloc[0]),
                   best_after={str(n): float(hist[min(n, len(hist)) - 1]) for n in (1, 10, 20, 40, 60)})
        predict, _ = FIT["hgb"](best, Xtr, ytr, Xva, yva, seed)        # refit: deterministic, identical model
    rec["val_macro_f1"], rec["val_accuracy"] = macro_f1(yva, predict(Xva), classes)
    rec["test_macro_f1"], rec["test_accuracy"] = macro_f1(yte, predict(Xte), classes)
    rec["settings"] = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in best.items()
                       if k != "n_classes"}
    rec["wall_min"] = (time.perf_counter() - t0) / 60
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if len(trials):
        (out / "trials").mkdir(exist_ok=True)
        trials.drop(columns=[c for c in ("features",) if c in trials.columns]).to_csv(out / "trials" / f"{arm}_seed{seed:02d}.csv", index=False)
    print(f"{arm} seed {seed}: val {rec['val_macro_f1']:.4f} test {rec['test_macro_f1']:.4f} "
          f"({rec['wall_min']:.1f} min, {rec['distinct_trainings']} trainings)", flush=True)
    return rec


def summarise(out: Path, arms: list[str]) -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "rows").glob("*.json"))]
    df = pd.DataFrame(rows)
    df.drop(columns=["settings", "best_after"], errors="ignore").to_csv(out / "runs.csv", index=False)
    base = df[df.arm == "default"].set_index("seed")
    res = []
    for metric in ("test_macro_f1", "val_macro_f1"):
        for arm in arms:
            a = df[df.arm == arm].set_index("seed")
            seeds = sorted(set(a.index) & set(base.index))
            if not seeds:
                continue
            d = (a.loc[seeds, metric] - base.loc[seeds, metric]).to_numpy()
            lo, hi = bootstrap_ci(d)
            l90, h90 = bootstrap_ci(d, level=0.90)
            res.append({"metric": metric, "arm": arm, "n_seeds": len(seeds), "arm_mean": a.loc[seeds, metric].mean(),
                        "default_mean": base.loc[seeds, metric].mean(), "mean_diff": d.mean(), "ci95_low": lo,
                        "ci95_high": hi, "seeds_arm_better": int((d > 0).sum()),
                        "p_two_sided": wilcoxon(a.loc[seeds, metric], base.loc[seeds, metric]),
                        **{f"equiv_within_{m}": bool(l90 > -m and h90 < m) for m in MARGINS}})
    s = pd.DataFrame(res)
    for metric in s.metric.unique() if len(s) else []:
        m = s.metric == metric
        s.loc[m, "p_holm"] = holm(s.loc[m, "p_two_sided"].to_numpy())
    s.to_csv(out / "summary.csv", index=False)
    lines = ["# Tuning ablation: hyper-parameter-only search, defaults as first candidate, full-data fitness", "",
             "Arms minus default (paired by seed). Decision rule in the script header.", "", s.round(4).to_string(index=False)]
    if "distinct_trainings" in df:
        lines += ["", "Distinct trainings and minutes per run:", "",
                  df.groupby("arm")[["distinct_trainings", "wall_min"]].mean().round(1).to_string()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(s.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    grid = SMOKE if args.smoke else FULL
    out = ROOT / "results" / ("ablation_tuning_smoke" if args.smoke else "ablation_tuning")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, a) for a in grid["arms"] + ["default"] for s in grid["seeds"]]       # searches first (longest)
    t0 = time.perf_counter()
    if args.workers > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.workers)(delayed(task)(s, a, grid, out) for s, a in jobs)
    else:
        for s, a in jobs:
            task(s, a, grid, out)
    summarise(out, grid["arms"])
    print(f"wall {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
