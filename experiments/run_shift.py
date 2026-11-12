"""E6 (step 3): shift-aware joint feature + hyper-parameter selection for LightGBM (configs/experiments/e6_unsw_shift.yaml).

Usage (from the repository root):
    python experiments/run_shift.py --smoke                # ~2 minutes: pipeline check
    python tools/run_parallel.py e6 --workers 20           # full run (PC), resumable
    python experiments/run_shift.py --summarize-only       # validation summary while it runs
    python experiments/run_shift.py --final-test --workers 10   # ONCE, after all runs finished

Unsupervised domain adaptation: the importance weights use the test partition's features (hsboa.shift_tune);
its labels are not read in the run phase. ``--final-test`` refits every finished run
(deterministic), checks the refit against the stored validation score, scores the test
partition, and writes final_test.json; after that the run phase refuses to start, so
nothing can be re-tuned once the test labels have been seen. Writes results/e6_unsw_shift[_smoke]/:
    manifest.json, weights/seed_XX.npz (importance weights of the seed's validation and training rows)
    <arm>/seed_XX/trials.jsonl (checkpoint), trials.csv, result.json
    runs.csv, summary.csv, tests.csv (after the final test), report.md, final_test.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
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

from _jobs import claim, taken  # noqa: E402
from hsboa.datasets import DEFAULT_DATASET, get_dataset  # noqa: E402
from hsboa.experiment import code_hash  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate, predict_labels  # noqa: E402
from hsboa.shift import encode, weighted_macro_f1  # noqa: E402
from hsboa.stats import a12, bootstrap_ci, holm  # noqa: E402
from hsboa.utils import peak_memory_mb  # noqa: E402

E6_MODULES = ["data", "datasets", "preprocess", "folds", "metrics", "harmonic", "optimizers", "models",
              "fitness", "baselines", "gbdt", "shift", "shift_tune"]
METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "mcc", "binary_accuracy", "binary_f1", "binary_fpr")
LOCK = "final_test.json"
# Smoke run on a machine without LightGBM: E1b's HGB space and default, smoke only.
HGB_SPACE = {"log10_learning_rate": [-2.5, -0.5], "int_max_iter": [10, 20], "int_max_leaf_nodes": [15, 127],
             "int_min_samples_leaf": [5, 100], "log10_l2_regularization": [-6, 0], "class_weight_power": [0, 1]}
HGB_DEFAULT = {"learning_rate": 0.1, "max_iter": 20, "max_leaf_nodes": 31, "min_samples_leaf": 20,
               "l2_regularization": 0.0, "class_weight_power": 0.5}


def load(path: Path, smoke: bool) -> tuple[dict, dict, dict]:
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    dcfg["features"]["selection"] = cfg["features"]["selection"]
    hcfg = yaml.safe_load((ROOT / cfg["harmonic_config"]).read_text(encoding="utf-8"))["harmonic"]
    cfg["seeds"] = dcfg["seeds"][:cfg["n_seeds"]]
    if smoke:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["hpo"].update(pop_size=4, train_samples=3000, max_evals={t: 8 for t in cfg["hpo"]["max_evals"]})
        cfg["weights"].update(domain_max_iter=30, cross_fit_folds=2)
        if importlib.util.find_spec("lightgbm") is None:
            cfg["model"], cfg["hpo"]["space"], cfg["default"]["settings"] = "hgb", dict(HGB_SPACE), dict(HGB_DEFAULT)
        else:
            cfg["hpo"]["space"]["int_n_estimators"] = [10, 20]
            cfg["default"]["settings"]["n_estimators"] = 20
        text += f"\n# --smoke ({cfg['model']})\n"
    blob = json.dumps({"e6": text, "harmonic": hcfg, "features": dcfg["features"]}, sort_keys=True)
    cfg["_config_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return cfg, dcfg, hcfg


def check_manifest(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    man = {"experiment": cfg["experiment"], "model": cfg["model"], "config_sha256": cfg["_config_sha256"],
           "code_sha256": code_hash(E6_MODULES), "modules": E6_MODULES, "seeds": cfg["seeds"],
           "arms": list(cfg["arms"])}
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
    return out


def get_weights(out: Path, seed: int, X, X_te, split, cfg: dict, ds) -> dict:
    """Importance weights of the seed's validation and training rows (cached; deterministic)."""
    from hsboa.shift_tune import importance_weights
    path = out / "weights" / f"seed_{seed:02d}.npz"
    if path.exists():
        z = np.load(path)
        return {"val": z["val"], "train": z["train"]}
    A, B = encode(X, X_te, ds.categorical_columns)
    w = importance_weights(A[split.train_idx], A[split.val_idx], B, seed, max_iter=cfg["weights"]["domain_max_iter"],
                           folds=cfg["weights"]["cross_fit_folds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, val=w["val"], train=w["train"])
    os.replace(tmp, path)                   # several workers may compute the same seed: identical results
    return w


def run_one(arm: str, seed: int, fold, w: dict, cfg: dict, hcfg: dict, run_dir: Path) -> dict:
    from hsboa.shift_tune import fit_default, tune_shift
    run_dir.mkdir(parents=True, exist_ok=True)
    spec, h = cfg["arms"][arm], cfg["hpo"]
    if spec["tuner"] == "none":
        r = fit_default(cfg["model"], fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed,
                        fixed=cfg["default"]["settings"], n_selected=cfg["default"]["n_selected"], classes=fold.classes)
    else:
        r = tune_shift(cfg["model"], fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed, tuner=spec["tuner"],
                       hp_space=h["space"], pop_size=h["pop_size"], max_evals=h["max_evals"][spec["tuner"]],
                       train_samples=h["train_samples"], w_val=w["val"] if spec["fitness"] == "iw" else None,
                       w_train=w["train"] if spec["train"] == "iw" else None,
                       min_features=cfg["features"]["min_features"],
                       optimizer_params={"harmonic": hcfg} if spec["tuner"] == "hsboa" else {},
                       cache_path=run_dir / "trials.jsonl", classes=fold.classes)
        r["trials"].to_csv(run_dir / "trials.csv", index=False)
    proba = r["predict"](fold.X_val)
    names = fold.feature_names
    rec = {"arm": arm, **spec, "seed": seed, "model": cfg["model"], "settings": r["settings"],
           "selected_features": [names[i] for i in r["settings"]["features"]],
           "search_best_fitness_f1": r["search_best_fitness_f1"], "search_rows": r["search_rows"],
           "distinct_trials": r["distinct_trials"], "wall_s": r["wall_s"], "peak_memory_mb": peak_memory_mb(),
           "dataset": cfg.get("dataset", DEFAULT_DATASET), "val": scores(proba, fold.y_val, fold.classes),
           "val_macro_f1_weighted": weighted_macro_f1(fold.y_val, np.asarray(proba).argmax(1), len(fold.classes),
                                                      w["val"]),
           "test": None}
    (run_dir / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
    return rec


# ---------------------------------------------------------------------------
# Final test (once)
# ---------------------------------------------------------------------------

def final_test_seed(seed: int, cfg: dict, dcfg: dict, out: Path) -> list[dict]:
    from hsboa.shift_tune import refit
    ds = get_dataset(cfg.get("dataset", DEFAULT_DATASET))
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")            # the only place the test labels are read
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, X_test=X_te, y_test=y_te, dataset=ds)
    w = get_weights(out, seed, X, X_te, fold.split, cfg, ds)
    done = []
    for arm, spec in cfg["arms"].items():
        rd = out / arm / f"seed_{seed:02d}"
        rec = json.loads((rd / "result.json").read_text(encoding="utf-8"))
        if rec["test"] is not None:
            continue
        predict = refit(cfg["model"], rec["settings"], fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed,
                        w["train"] if spec["train"] == "iw" else None)
        rec["refit_val_macro_f1_diff"] = scores(predict(fold.X_val), fold.y_val, fold.classes)["macro_f1"] \
            - rec["val"]["macro_f1"]
        rec["test"] = scores(predict(fold.X_test), fold.y_test, fold.classes)
        (rd / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")
        done.append({"arm": arm, "seed": seed, "refit_diff": rec["refit_val_macro_f1_diff"]})
    print(f"seed {seed}: test scored for {len(done)} arms", flush=True)
    return done


def final_test(out: Path, cfg: dict, dcfg: dict, workers: int) -> None:
    missing = [f"{a}/seed_{s:02d}" for a in cfg["arms"] for s in cfg["seeds"]
               if not (out / a / f"seed_{s:02d}" / "result.json").exists()]
    if missing:
        raise SystemExit(f"final test refused: {len(missing)} runs unfinished (e.g. {missing[:3]})")
    if workers > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=workers)(delayed(final_test_seed)(s, cfg, dcfg, out) for s in cfg["seeds"])
    else:
        res = [final_test_seed(s, cfg, dcfg, out) for s in cfg["seeds"]]
    refits = [r for rs in res for r in rs]
    lock = {"done": time.strftime("%Y-%m-%d %H:%M:%S"), "runs_scored": len(refits),
            "max_abs_refit_val_diff": max((abs(r["refit_diff"]) for r in refits), default=0.0)}
    (out / LOCK).write_text(json.dumps(lock, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _wilcoxon(a, b, alternative: str) -> float:
    from scipy import stats as st
    if np.all(a == b):
        return 1.0
    return float(st.wilcoxon(a, b, zero_method="pratt", alternative=alternative).pvalue)


def summarize(out: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    rows = []
    for p in sorted(out.glob("*/seed_*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"arm": r["arm"], "seed": r["seed"], "search_fitness_f1": r["search_best_fitness_f1"],
               "val_macro_f1_weighted": r["val_macro_f1_weighted"], "n_selected": r["settings"]["n_selected"],
               "class_weight_power": r["settings"].get("class_weight_power"), "distinct_trials": r["distinct_trials"],
               "wall_min": r["wall_s"] / 60}
        for part in ("val", "test"):
            if r[part]:
                row.update({f"{part}_{k}": v for k, v in r[part].items() if k in METRICS or k.startswith("f1_")})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no finished runs under {out}")
    runs = pd.DataFrame(rows).sort_values(["seed", "arm"])
    final = (out / LOCK).exists() and "test_macro_f1" in runs
    if final and cfg.get("e2_reference") and (ROOT / cfg["e2_reference"]).exists():
        for s in cfg["seeds"]:
            p = ROOT / cfg["e2_reference"] / f"seed_{s:02d}" / "result.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                row = {"arm": "e2_lgbm (reference)", "seed": s}
                for part in ("val", "test"):
                    if r.get(part):
                        row.update({f"{part}_{k}": v for k, v in r[part].items() if k in METRICS})
                runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    num = runs.select_dtypes("number").drop(columns=["seed"])
    summary = num.groupby(runs["arm"]).agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    tests = []
    metrics = ["test_macro_f1", "val_macro_f1", "val_macro_f1_weighted"] if final else \
        ["val_macro_f1", "val_macro_f1_weighted"]
    for name, a, b, alt in cfg["hypotheses"]:
        for col in metrics:
            wide = runs.pivot_table(index="seed", columns="arm", values=col)
            if a not in wide or b not in wide:
                continue
            wide = wide[[a, b]].dropna()
            if len(wide) < 2:
                continue
            x, z = wide[a].to_numpy(), wide[b].to_numpy()
            lo, hi = bootstrap_ci(x - z)
            tests.append({"hypothesis": name, "metric": col, "a": a, "b": b, "alternative": alt,
                          "n_seeds": len(wide), "mean_a": x.mean(), "mean_b": z.mean(), "mean_diff": (x - z).mean(),
                          "ci95_low": lo, "ci95_high": hi, "seeds_a_better": int((x > z).sum()),
                          "p": _wilcoxon(x, z, alt), "a12_a_better": a12(-x, -z)})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["p_holm"] = np.nan
        for col in tests["metric"].unique():
            m = tests["metric"] == col
            tests.loc[m, "p_holm"] = holm(tests.loc[m, "p"].to_numpy())
    for name, df in (("runs", runs), ("summary", summary), ("tests", tests)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "report.md").write_text(_report(cfg, runs, tests, final), encoding="utf-8")
    return {"runs": runs, "summary": summary, "tests": tests}


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _report(cfg, runs, tests, final: bool) -> str:
    h = cfg["hpo"]
    cols = [c for c in ("search_fitness_f1", "val_macro_f1", "val_macro_f1_weighted", "test_macro_f1",
                        "test_accuracy", "test_binary_accuracy", "test_f1_Normal", "test_f1_Fuzzers", "n_selected",
                        "distinct_trials", "wall_min") if c in runs]
    agg = runs.groupby("arm")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    lines = [f"# E6 shift-aware joint selection ({cfg['model']}): summary", "",
             ("Final test done (test labels read once)." if final else
              "**Validation only: the final test has not been run.**"), "",
             f"Finished runs per arm: {runs.groupby('arm').size().to_dict()} (planned {len(cfg['seeds'])} seeds). "
             f"Budget {h['max_evals']} evaluations, pop {h['pop_size']}, search on {h['train_samples']} training "
             "rows, final fit on the full training part. Importance weights from a domain classifier "
             "(training part vs test features, no test labels), capped at the 99th percentile. Hypotheses H5-H10 "
             "fixed in the config; Wilcoxon signed-rank paired by seed in the stated direction, Holm over H5-H10 "
             "per metric.", "", "## Mean (sd) by arm", "", _md(agg.reset_index()), ""]
    if not tests.empty:
        lines += ["## Hypotheses", "", _md(tests.round(4)), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "experiments" / "e6_unsw_shift.yaml", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--worker", action="store_true", help="parallel worker (tools/run_parallel.py)")
    ap.add_argument("--final-test", action="store_true", help="refit all runs and score the test partition (once)")
    ap.add_argument("--workers", type=int, default=1, help="processes for --final-test")
    args = ap.parse_args()

    cfg, dcfg, hcfg = load(args.config, args.smoke)
    out = ROOT / "results" / (cfg["experiment"] + ("_smoke" if args.smoke else ""))
    if args.final_test:
        if (out / LOCK).exists():
            raise SystemExit(f"the final test was already done ({out / LOCK})")
        final_test(out, cfg, dcfg, args.workers)
    elif not args.summarize_only:
        if (out / LOCK).exists():
            raise SystemExit(f"refused: the final test was done ({out / LOCK}); runs cannot change afterwards")
        check_manifest(out, cfg)
        ds = get_dataset(cfg.get("dataset", DEFAULT_DATASET))
        X, y = ds.load("train")
        X_te, _ = ds.load("test")           # features only, labels discarded
        print(f"{cfg['experiment']} ({cfg['model']}): seeds {cfg['seeds']}, arms {list(cfg['arms'])}", flush=True)
        for seed in cfg["seeds"]:
            todo = [(a, out / a / f"seed_{seed:02d}") for a in cfg["arms"]]
            todo = [(a, rd) for a, rd in todo if not (rd / "result.json").exists() and not taken(rd, args.worker)]
            if not todo:
                continue
            fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, dataset=ds)
            w = None
            for arm, rd in todo:
                if not claim(rd, args.worker):
                    continue
                w = w or get_weights(out, seed, X, X_te, fold.split, cfg, ds)
                t0 = time.perf_counter()
                r = run_one(arm, seed, fold, w, cfg, hcfg, rd)
                print(f"seed {seed} {arm}: val macro-F1 {r['val']['macro_f1']:.4f}, weighted "
                      f"{r['val_macro_f1_weighted']:.4f}, {r['settings']['n_selected']} features "
                      f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    if args.worker:
        return
    t = summarize(out, cfg)
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
