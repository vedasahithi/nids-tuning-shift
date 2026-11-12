"""Decision-layer diagnostic (direction B): do per-class biases raise held-out macro-F1?

Usage (from the repository root):
    python scripts/diagnose_decision.py --smoke              # ~1-2 minutes: checks the pipeline
    python scripts/diagnose_decision.py --workers 16         # full run (PC), resumable
    python scripts/diagnose_decision.py --summarize-only

After E1/E1b/E2 (2026-10-28: hyper-parameter tuning by HSBOA gave nothing over
random search or a good default), this checks, on validation data only,
whether the decision layer has headroom (hsboa.decision): the trained model
is fixed and the predicted class becomes argmax(log p + b).

For each model and seed, the model is refitted exactly as in E1b/E2 from the
stored per-seed settings (a determinism check compares the plain-argmax
validation macro-F1 with the stored one). The validation probabilities are
then split into two stratified halves, 5 times (cross-fitting): b is searched
on one half and scored on the other. The held-out gain is
macro-F1(half B, b) - macro-F1(half B, b = 0). The in-sample gain on half A
shows how much of the search result is overfitting.

Models: hgb_default and hgb_hsboa (E1b), lgbm, rf, xgb (E2; lgbm/xgb need
their libraries). Methods: hsboa, sboa, pso, de, random, coord
(hsboa.decision.METHODS), each with the same budget of evaluations.
The test partition is never loaded.

Decision rule, fixed before the run:
    Direction B is worth pursuing if, for the best E2 model (lgbm), the best
    method's mean held-out gain is >= +0.010 macro-F1 and positive on >= 8 of
    10 seeds. HSBOA has a role only if its held-out gain is not below random
    search and coordinate search (paired over seeds).
Writes results/diag_decision[_smoke]/: proba/<model>_s<seed>.npz, rows/<model>_s<seed>.json,
runs.csv, summary.csv, report.md.
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
import yaml  # noqa: E402

from hsboa.baselines import FIT, available  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.decision import METHODS, apply_bias, log_proba, macro_f1, per_class_f1, tune_bias  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.gbdt import masked  # noqa: E402
from hsboa.stats import a12, bootstrap_ci, wilcoxon  # noqa: E402

# name: (folder with per-seed result.json, model, feature selection of that experiment)
SOURCES = {
    "lgbm": ("results/e2_unsw_baselines/lgbm", "lgbm", 20),
    "rf": ("results/e2_unsw_baselines/rf", "rf", 20),
    "xgb": ("results/e2_unsw_baselines/xgb", "xgb", 20),
    "hgb_default": ("results/e1b_unsw_gbdt/default", "hgb", "all"),
    "hgb_hsboa": ("results/e1b_unsw_gbdt/hsboa", "hgb", "all"),
}
FULL = {"seeds": list(range(10)), "models": list(SOURCES), "methods": list(METHODS),
        "max_evals": 2000, "pop_size": 20, "bound": 3.0, "repeats": 5}
SMOKE = {"seeds": [0], "models": ["hgb_default", "rf"], "methods": list(METHODS),
         "max_evals": 200, "pop_size": 10, "bound": 3.0, "repeats": 1}
SMOKE_SETTINGS = {"hgb": {"max_iter": 20}, "rf": {"n_estimators": 20}}
BEST_MODEL = "lgbm"


def probabilities(name: str, seed: int, grid: dict, smoke: bool, out: Path) -> dict:
    """Validation probabilities of one model and seed (cached in proba/), refitted from the stored settings."""
    path = out / "proba" / f"{name}_s{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    folder, model, sel = SOURCES[name]
    rec = json.loads((ROOT / folder / f"seed_{seed:02d}" / "result.json").read_text(encoding="utf-8"))
    s = dict(rec["settings"])
    if smoke:
        s.update(SMOKE_SETTINGS.get(model, {}))
    ds = get_dataset()
    X, y = ds.load("train")                 # the test partition is not loaded
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    if sel == "all":
        dcfg["features"]["selection"] = "all"
    else:
        dcfg["features"]["n_features"] = int(sel)
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, dataset=ds)
    fit = masked(FIT[model]) if sel == "all" else FIT[model]
    t0 = time.perf_counter()
    predict, _ = fit(s, fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed)
    P = predict(fold.X_val)
    k = len(fold.classes)
    res = {"P": P, "y": np.asarray(fold.y_val), "k": np.array(k), "fit_s": np.array(time.perf_counter() - t0),
           "stored_val_macro_f1": np.array(rec["val"]["macro_f1"]),
           "refit_val_macro_f1": np.array(macro_f1(fold.y_val, P.argmax(axis=1), k)),
           "classes": np.array(fold.classes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return res


def run_task(name: str, seed: int, grid: dict, smoke: bool, hcfg: dict, out: Path) -> int:
    row_path = out / "rows" / f"{name}_s{seed}.json"
    if row_path.exists():
        return 0
    pr = probabilities(name, seed, grid, smoke, out)
    L, y, k = log_proba(pr["P"]), pr["y"], int(pr["k"])
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for r in range(grid["repeats"]):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1000 * seed + r)
        for h, (a, b_idx) in enumerate(skf.split(L, y)):
            base_b = macro_f1(y[b_idx], L[b_idx].argmax(axis=1), k)
            base_a = macro_f1(y[a], L[a].argmax(axis=1), k)
            pc_base = per_class_f1(y[b_idx], L[b_idx].argmax(axis=1), k)
            for m in grid["methods"]:
                t0 = time.perf_counter()
                t = tune_bias(m, L[a], y[a], k, max_evals=grid["max_evals"], pop_size=grid["pop_size"],
                              bound=grid["bound"], seed=[seed, r, h, METHODS.index(m)],
                              optimizer_params={"harmonic": hcfg} if m == "hsboa" else {})
                pred_b = apply_bias(L[b_idx], t["b"])
                rows.append({"model": name, "seed": seed, "method": m, "repeat": r, "half": h,
                             "base_heldout": base_b, "tuned_heldout": macro_f1(y[b_idx], pred_b, k),
                             "gain_heldout": macro_f1(y[b_idx], pred_b, k) - base_b,
                             "gain_insample": t["macro_f1"] - base_a, "n_evals": t["n_evals"],
                             "search_s": time.perf_counter() - t0, "b": t["b"].tolist(),
                             "dpc_f1": (per_class_f1(y[b_idx], pred_b, k) - pc_base).tolist()})
    rec = {"model": name, "seed": seed, "classes": pr["classes"].tolist(), "fit_s": float(pr["fit_s"]),
           "stored_val_macro_f1": float(pr["stored_val_macro_f1"]),
           "refit_val_macro_f1": float(pr["refit_val_macro_f1"]), "rows": rows}
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    best = max(grid["methods"], key=lambda m: np.mean([x["gain_heldout"] for x in rows if x["method"] == m]))
    print(f"{name} s{seed}: base {float(pr['refit_val_macro_f1']):.4f}; best held-out gain {best} "
          f"{np.mean([x['gain_heldout'] for x in rows if x['method'] == best]):+.4f}", flush=True)
    return 1


def summarize(out: Path) -> pd.DataFrame:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "rows").glob("*.json"))]
    if not recs:
        print("no finished tasks yet")
        return pd.DataFrame()
    runs = pd.DataFrame([r for rec in recs for r in rec["rows"]])
    runs.drop(columns=["b", "dpc_f1"]).to_csv(out / "runs.csv", index=False)
    classes = recs[0]["classes"]
    per_seed = runs.groupby(["model", "method", "seed"])[["gain_heldout", "gain_insample", "base_heldout"]] \
                   .mean().reset_index()
    summ = (per_seed.groupby(["model", "method"])
            .agg(seeds=("seed", "nunique"), gain_heldout=("gain_heldout", "mean"),
                 gain_heldout_sd=("gain_heldout", "std"),
                 seeds_positive=("gain_heldout", lambda g: int((g > 0).sum())),
                 gain_insample=("gain_insample", "mean"))
            .reset_index().sort_values(["model", "gain_heldout"], ascending=[True, False]))
    summ["p_vs_zero"] = [wilcoxon(per_seed[(per_seed.model == r.model) & (per_seed.method == r.method)]
                                  .sort_values("seed")["gain_heldout"].to_numpy(), np.zeros(int(r.seeds)))
                         if r.seeds > 1 else np.nan for r in summ.itertuples()]
    summ.to_csv(out / "summary.csv", index=False)

    tests = []
    for model in sorted(per_seed.model.unique()):
        g = per_seed[per_seed.model == model].pivot(index="seed", columns="method", values="gain_heldout")
        for a in ("hsboa",):
            for b in [m for m in g.columns if m != a]:
                if a in g and g[[a, b]].dropna().shape[0] > 1:
                    d = (g[a] - g[b]).dropna().to_numpy()
                    lo, hi = bootstrap_ci(d)
                    tests.append({"model": model, "a": a, "b": b, "mean_diff": d.mean(), "ci95_low": lo,
                                  "ci95_high": hi, "p": wilcoxon(g[a].to_numpy(), g[b].to_numpy()),
                                  # hsboa.stats.a12 counts lower as better; a gain is better when higher
                                  "a12_a_better": a12(g[b].to_numpy(), g[a].to_numpy())})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "tests.csv", index=False)

    det = pd.DataFrame([{"model": r["model"], "seed": r["seed"], "stored": r["stored_val_macro_f1"],
                         "refit": r["refit_val_macro_f1"],
                         "abs_diff": abs(r["stored_val_macro_f1"] - r["refit_val_macro_f1"])} for r in recs])
    det_s = det.groupby("model").agg(seeds=("seed", "nunique"), stored=("stored", "mean"),
                                     refit=("refit", "mean"), max_abs_diff=("abs_diff", "max")).reset_index()

    dpc = pd.DataFrame([dict(zip(classes, r["dpc_f1"]), model=r["model"], method=r["method"])
                        for r in runs.to_dict("records")]).groupby(["model", "method"])[classes].mean()

    lines = ["# Decision-layer diagnostic (validation only)", "",
             "Fixed models (E1b/E2 settings per seed); biases b on log-probabilities searched on one stratified "
             "half of the validation split and scored on the other (5 repeats x 2 halves, averaged per seed). "
             "Gains are macro-F1 differences against the plain argmax on the same held-out half. "
             "The test partition was not used.", ""]
    best = summ[summ.model == BEST_MODEL].head(1)
    if not best.empty:
        r = best.iloc[0]
        go = r.gain_heldout >= 0.010 and r.seeds_positive >= min(8, int(r.seeds))
        lines += [f"**Decision rule ({BEST_MODEL}):** best method {r.method}, held-out gain {r.gain_heldout:+.4f}, "
                  f"positive on {r.seeds_positive}/{r.seeds} seeds -> "
                  f"{'meets' if go else 'does NOT meet'} the threshold (+0.010, >= 8/10 seeds).", ""]
    lines += ["## Held-out gain by model and method", "", _md(summ), "",
              "## HSBOA vs the other methods (held-out gain, paired over seeds)", "",
              _md(tests) if not tests.empty else "(fewer than 2 seeds)", "",
              "## Mean per-class held-out F1 change", "", _md(dpc.reset_index()), "",
              "## Determinism check: plain-argmax validation macro-F1, refit vs stored (E1b/E2)", "",
              _md(det_s), ""]
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
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (one task = model x seed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    grid = copy.deepcopy(SMOKE if args.smoke else FULL)
    out = ROOT / "results" / ("diag_decision_smoke" if args.smoke else "diag_decision")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    hcfg = yaml.safe_load((ROOT / "configs" / "hsboa.yaml").read_text(encoding="utf-8"))["harmonic"]
    if not args.summarize_only:
        missing = [n for n in grid["models"] if not available(SOURCES[n][1])]
        if missing and not args.smoke:
            raise SystemExit(f"models {missing} need libraries that are not installed (see SETUP_NEW_MACHINE.md)")
        grid["models"] = [n for n in grid["models"] if n not in missing]
        absent = [n for n in grid["models"] if not (ROOT / SOURCES[n][0]).exists()]
        if absent:
            raise SystemExit(f"stored results missing for {absent}: run E1b/E2 first")
        tasks = [(n, s) for n in grid["models"] for s in grid["seeds"]]
        tasks.sort(key=lambda t: t[0] != "rf")                 # slowest refits first
        print(f"{len(tasks)} tasks, {args.workers} workers; results in {out}", flush=True)
        t0 = time.perf_counter()
        if args.workers > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.workers, verbose=5)(delayed(run_task)(n, s, grid, args.smoke, hcfg, out)
                                                     for n, s in tasks)
        else:
            for n, s in tasks:
                run_task(n, s, grid, args.smoke, hcfg, out)
        print(f"all tasks finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    summ = summarize(out)
    if not summ.empty:
        print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
