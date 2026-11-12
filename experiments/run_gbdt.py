"""E1b: joint feature and hyper-parameter selection for gradient boosting.

Usage (from the repository root):
    python experiments/run_gbdt.py --smoke             # ~1 minute: pipeline check
    python experiments/run_gbdt.py                     # full E1b, resumable
    python experiments/run_gbdt.py --summarize-only

For each seed and tuner (HSBOA, SBOA, random search): search jointly over a
feature mask on all 42 MI-ordered columns and the gradient-boosting settings
(fitness 1 - validation macro-F1 on the 30k training subsample), refit the
best setting on the full training part, and score validation and the official
test partition. The ``default`` arm fits the fixed diagnostic pipeline (top-20
MI features) without search. Writes results/e1b_unsw_gbdt[_smoke]/:
    manifest.json
    <tuner>/seed_XX/trials.jsonl     trial checkpoint (a rerun resumes mid-search)
                    trials.csv, result.json (settings, selected features, metrics, cost)
    runs.csv, summary.csv, tests.csv, curves.csv, features.csv, report.md
Final models are not saved: gradient boosting without early stopping is
deterministic, so test scores can be added later by refitting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hsboa.datasets import DEFAULT_DATASET, get_dataset, load_parts  # noqa: E402
from _jobs import claim, taken  # noqa: E402
from hsboa.experiment import code_hash  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate, predict_labels  # noqa: E402
from hsboa.stats import a12, bootstrap_ci, holm, wilcoxon  # noqa: E402
from hsboa.utils import peak_memory_mb  # noqa: E402

E1B_MODULES = ["data", "datasets", "preprocess", "folds", "metrics", "harmonic", "optimizers", "models",
               "fitness", "baselines", "gbdt"]
METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "mcc")      # plus every per-class f1_<class>
SETTINGS = ("n_selected", "class_weight_power", "learning_rate", "max_iter", "max_leaf_nodes",
            "min_samples_leaf", "l2_regularization")


def load(path: Path, smoke: bool) -> tuple[dict, dict, dict]:
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    dcfg["features"]["selection"] = cfg["features"]["selection"]
    hcfg = yaml.safe_load((ROOT / cfg["harmonic_config"]).read_text(encoding="utf-8"))["harmonic"]
    cfg["seeds"] = dcfg["seeds"][:cfg["n_seeds"]]
    if smoke:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["hpo"].update(pop_size=4, train_samples=3000,
                          max_evals={t: (8 if n else 0) for t, n in cfg["hpo"]["max_evals"].items()})
        cfg["hpo"]["space"]["int_max_iter"] = [10, 20]
        cfg["default"]["settings"]["max_iter"] = 20
        text += "\n# --smoke\n"
    blob = json.dumps({"e1b": text, "harmonic": hcfg, "features": dcfg["features"]}, sort_keys=True)
    cfg["_config_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return cfg, dcfg, hcfg


def check_manifest(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    man = {"experiment": cfg["experiment"], "config_sha256": cfg["_config_sha256"],
           "code_sha256": code_hash(E1B_MODULES), "modules": E1B_MODULES,
           "seeds": cfg["seeds"], "tuners": cfg["tuners"]}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "code_sha256"):
            if old[key] != man[key]:
                raise RuntimeError(f"{out} was made with a different {key.split('_')[0]}; use a new folder")
        man["started"] = old.get("started")
    else:
        man["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")


def scores(proba, y, classes) -> dict:
    rep = evaluate(y, predict_labels(proba), classes=classes)
    out = dict(rep.summary)
    out.update({f"binary_{k}": v for k, v in rep.binary.items()})
    out.update({f"f1_{c}": float(rep.per_class.loc[c, "f1"]) for c in classes})
    out.update({f"recall_{c}": float(rep.per_class.loc[c, "recall"]) for c in classes})
    return out


def fit_arm(tuner: str, seed: int, fold, cfg: dict, hcfg: dict, run_dir: Path | None) -> dict:
    from hsboa.gbdt import fit_default, tune_joint
    h = cfg["hpo"]
    if tuner == "default":
        return fit_default(cfg["model"], fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed,
                           fixed=cfg["default"]["settings"], n_selected=cfg["default"]["n_selected"],
                           classes=fold.classes)
    return tune_joint(cfg["model"], fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed, tuner=tuner,
                      hp_space=h["space"], fixed={}, pop_size=h["pop_size"], max_evals=h["max_evals"][tuner],
                      train_samples=h["train_samples"], min_features=cfg["features"]["min_features"],
                      optimizer_params={"harmonic": hcfg} if tuner == "hsboa" else {},
                      cache_path=run_dir / "trials.jsonl" if run_dir else None, classes=fold.classes)


def run_one(tuner: str, seed: int, fold, cfg: dict, hcfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    r = fit_arm(tuner, seed, fold, cfg, hcfg, run_dir)
    if r["trials"] is not None:
        r["trials"].to_csv(run_dir / "trials.csv", index=False)
    names = fold.feature_names
    rec = {"tuner": tuner, "seed": seed, "model": cfg["model"], "settings": r["settings"],
           "selected_features": [names[i] for i in r["settings"]["features"]],
           "search_best_val_macro_f1": r["search_best_val_macro_f1"], "search_rows": r["search_rows"],
           "distinct_trials": r["distinct_trials"], "wall_s": r["wall_s"], "peak_memory_mb": peak_memory_mb(),
           "dataset": cfg.get("dataset", DEFAULT_DATASET),
           "val": scores(r["predict"](fold.X_val), fold.y_val, fold.classes),
           "test": None if fold.X_test is None else scores(r["predict"](fold.X_test), fold.y_test, fold.classes)}
    (run_dir / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


def add_test_scores(rd: Path, fold, cfg: dict) -> None:
    """Refit the final model from the saved settings and seed (deterministic) and score the test partition."""
    from hsboa.baselines import FIT
    from hsboa.gbdt import masked
    rec = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    predict, _ = masked(FIT[cfg["model"]])(rec["settings"], fold.X_train, fold.y_train, fold.X_val,
                                          fold.y_val, rec["seed"])
    val = scores(predict(fold.X_val), fold.y_val, fold.classes)
    rec["refit_val_macro_f1_diff"] = val["macro_f1"] - rec["val"]["macro_f1"]   # ~0 if deterministic
    rec["test"] = scores(predict(fold.X_test), fold.y_test, fold.classes)
    (rd / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(out: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    rows, curves, feats = [], [], []
    for p in sorted(out.glob("*/seed_*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"tuner": r["tuner"], "seed": r["seed"], "search_val_macro_f1": r["search_best_val_macro_f1"],
               "wall_min": r["wall_s"] / 60, "distinct_trials": r["distinct_trials"],
               "peak_memory_mb": r["peak_memory_mb"],
               **{f"setting_{k}": r["settings"][k] for k in SETTINGS if k in r["settings"]}}
        for part in ("val", "test"):
            if r[part]:
                row.update({f"{part}_{k}": v for k, v in r[part].items() if k in METRICS or k.startswith("f1_")})
        rows.append(row)
        feats += [{"tuner": r["tuner"], "seed": r["seed"], "feature": f} for f in r["selected_features"]]
        if (p.parent / "trials.csv").exists():
            t = pd.read_csv(p.parent / "trials.csv")
            best = (1 - t["fitness"]).cummax()
            curves += [{"tuner": r["tuner"], "seed": r["seed"], "trial": i + 1, "best_val_macro_f1": v}
                       for i, v in enumerate(best)]
    if not rows:
        raise FileNotFoundError(f"no finished runs under {out}")
    runs = pd.DataFrame(rows).sort_values(["seed", "tuner"])
    num = runs.select_dtypes("number").drop(columns=["seed"])
    summary = num.groupby(runs["tuner"]).agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    tests = []
    ref = cfg["tuners"][0]
    for other in cfg["tuners"][1:]:
        for col in [c for c in runs.columns if c.startswith(("val_", "test_"))]:
            wide = runs.pivot(index="seed", columns="tuner", values=col)
            if ref not in wide or other not in wide:
                continue
            wide = wide[[ref, other]].dropna()
            if len(wide) == 0:
                continue
            a, b = wide[ref].to_numpy(), wide[other].to_numpy()
            lo, hi = bootstrap_ci(a - b)
            tests.append({"metric": col, "a": ref, "b": other, "n_seeds": len(wide),
                          "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": (a - b).mean(),
                          "ci95_low": lo, "ci95_high": hi, "p": wilcoxon(a, b),
                          "a12_a_better": a12(-a, -b)})      # higher is better: A12 on negated scores
    tests = pd.DataFrame(tests)
    if not tests.empty:
        primary = tests["metric"].isin(["val_macro_f1", "test_macro_f1"])
        tests["p_holm_primary"] = np.nan
        tests.loc[primary, "p_holm_primary"] = holm(tests.loc[primary, "p"].to_numpy())
    feats = pd.DataFrame(feats)
    freq = (feats.groupby(["tuner", "feature"]).size().unstack("tuner", fill_value=0)
            .div(runs.groupby("tuner").size()).round(2))
    freq = freq.reindex(freq.mean(axis=1).sort_values(ascending=False).index).reset_index()
    curves = pd.DataFrame(curves)
    for name, df in (("runs", runs), ("summary", summary), ("tests", tests), ("curves", curves),
                     ("features", freq)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "report.md").write_text(_report(cfg, runs, tests, freq), encoding="utf-8")
    return {"runs": runs, "summary": summary, "tests": tests, "curves": curves, "features": freq}


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _report(cfg, runs, tests, freq) -> str:
    n_done = runs.groupby("tuner").size().to_dict()
    h = cfg["hpo"]
    lines = ["# E1b gradient boosting, joint feature + hyper-parameter selection: summary", "",
             f"Finished runs per tuner: {n_done} (planned {len(cfg['seeds'])} seeds). Budget (evaluations, "
             f"matched on distinct trainings) {h['max_evals']} (pop {h['pop_size']}); search on "
             f"{h['train_samples']} training rows; final fit on the full training part. Default arm: top "
             f"{cfg['default']['n_selected']} MI features, {cfg['default']['settings']}. Wilcoxon signed-rank "
             "paired by seed; bootstrap 95% CI of the mean difference; Holm over the primary metrics "
             "(validation and test macro-F1).", ""]
    cols = ["tuner", "seed"] + [c for c in ("search_val_macro_f1", "val_macro_f1", "test_macro_f1",
                                            "setting_n_selected", "setting_class_weight_power",
                                            "distinct_trials", "wall_min") if c in runs]
    lines += ["## Per run", "", _md(runs[cols]), ""]
    if not tests.empty:
        prim = tests["metric"].isin(["val_macro_f1", "test_macro_f1", "val_balanced_accuracy",
                                     "test_balanced_accuracy", "test_mcc"])
        lines += ["## Paired comparisons (main metrics; all metrics in tests.csv)", "",
                  _md(tests[prim].round(4)), ""]
    lines += ["## How often each feature was selected (fraction of seeds)", "", _md(freq), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "experiments" / "e1b_unsw_gbdt.yaml", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="parallel worker (tools/run_parallel.py): claim runs, skip the summary")
    args = ap.parse_args()

    cfg, dcfg, hcfg = load(args.config, args.smoke)
    out = ROOT / "results" / (cfg["experiment"] + ("_smoke" if args.smoke else ""))
    if not args.summarize_only:
        check_manifest(out, cfg)
        ds = get_dataset(cfg.get("dataset", DEFAULT_DATASET))
        X, y, X_te, y_te = load_parts(ds)
        print(f"{cfg['experiment']}: seeds {cfg['seeds']}, tuners {cfg['tuners']}, "
              f"{ds.name}, test part {'found' if X_te is not None else 'MISSING (validation only)'}", flush=True)
        for seed in cfg["seeds"]:
            todo = []
            for tuner in cfg["tuners"]:
                rd = out / tuner / f"seed_{seed:02d}"
                done = (rd / "result.json").exists()
                needs_test = done and X_te is not None and \
                    json.loads((rd / "result.json").read_text(encoding="utf-8"))["test"] is None
                if (not done or needs_test) and not taken(rd, args.worker):
                    todo.append((tuner, rd, done))
            if not todo:
                continue
            fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, X_test=X_te, y_test=y_te,
                                dataset=ds)
            for tuner, rd, done in todo:
                if not claim(rd, args.worker):    # another worker took it meanwhile
                    continue
                t0 = time.perf_counter()
                if done:
                    add_test_scores(rd, fold, cfg)
                    print(f"seed {seed} {tuner}: test scores added", flush=True)
                    continue
                run_one(tuner, seed, fold, cfg, hcfg, rd)
                r = json.loads((rd / "result.json").read_text(encoding="utf-8"))
                test = f", test {r['test']['macro_f1']:.4f}" if r["test"] else ""
                print(f"seed {seed} {tuner}: val macro-F1 {r['val']['macro_f1']:.4f}{test}, "
                      f"{r['settings']['n_selected']} features ({(time.perf_counter() - t0) / 60:.1f} min)",
                      flush=True)
    if args.worker:
        return
    t = summarize(out, cfg)
    print(t["runs"][[c for c in ("tuner", "seed", "val_macro_f1", "test_macro_f1", "setting_n_selected")
                     if c in t["runs"]]].round(4).to_string(index=False))
    if not t["tests"].empty:
        print(t["tests"][t["tests"]["metric"].isin(["val_macro_f1", "test_macro_f1"])]
              [["metric", "b", "mean_diff", "ci95_low", "ci95_high", "p", "p_holm_primary"]]
              .round(4).to_string(index=False))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
