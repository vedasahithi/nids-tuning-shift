"""E2: model comparison on UNSW-NB15.

Usage (from the repository root):
    python experiments/run_baselines.py --smoke
    python experiments/run_baselines.py                      # all models, resumable
    python experiments/run_baselines.py --models logreg rf hgb   # light models only
    python experiments/run_baselines.py --summarize-only

The sklearn models (logreg, rf, hgb) are light and can run alongside E1; the
neural ones (mlp, cnn_bilstm, qrnn_untuned) need ~1 GB each and are best run
after E1 on this machine. Writes results/e2_unsw_baselines[_smoke]/:
    <model>/seed_XX/result.json, trials.jsonl/csv, history.csv
    runs.csv, summary.csv, tests.csv, report.md
The summary adds the tuned QRNN runs from E1 (qrnn_hsboa, qrnn_random; d = 10) and
the gradient-boosting arms from E1b (hgb_hsboa, ...); hgb_hsboa is the reference.
xgb and lgbm need their libraries: a full run refuses to start without
them, a smoke run skips them. If
the test partition arrives later, rerunning refits each finished final model from its
saved settings and seed (deterministic) and adds test scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from hsboa.baselines import FIT, NEURAL, available, tune_and_fit  # noqa: E402
from hsboa.datasets import DEFAULT_DATASET, get_dataset, load_parts  # noqa: E402
from _jobs import claim, taken  # noqa: E402
from hsboa.experiment import code_hash  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate, predict_labels  # noqa: E402
from hsboa.stats import a12, bootstrap_ci, holm, wilcoxon  # noqa: E402
from hsboa.utils import peak_memory_mb  # noqa: E402

E2_MODULES = ["data", "datasets", "preprocess", "folds", "metrics", "harmonic", "optimizers", "models", "fitness", "train",
              "baselines"]
METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "mcc")      # plus every per-class f1_<class>


def load(path: Path, smoke: bool):
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    dcfg = load_config(ROOT / "configs" / "data.yaml")
    dcfg["features"].update(cfg.get("features", {}))      # E2 uses d = 20
    cfg["seeds"] = dcfg["seeds"][:cfg["n_seeds"]]
    if smoke:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["search"].update(train_samples=3000, pop_size=4)
        for m in cfg["models"].values():
            if m.get("tune") == "random":
                m["max_evals"] = 4
            if m.get("tune") == "qrnn_hpo":
                m.update(max_evals=4, pop_size=4, epochs=2, final_epochs=2)
            if "epochs" in m.get("fixed", {}):
                m["fixed"]["epochs"] = 2
            if "final_epochs" in m:
                m["final_epochs"] = 2
        text += "\n# --smoke\n"
    # The hash covers this config and the QRNN search space it points to (qrnn_hpo models).
    blob = text + "".join((ROOT / m["model_config"]).read_text(encoding="utf-8")
                          for m in cfg["models"].values() if "model_config" in m)
    cfg["_config_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return cfg, dcfg


def check_manifest(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    man = {"experiment": cfg["experiment"], "config_sha256": cfg["_config_sha256"],
           "code_sha256": code_hash(E2_MODULES), "modules": E2_MODULES, "seeds": cfg["seeds"]}
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


def latency_ms(predict, X, reps: int = 7) -> dict:
    """Inference time per row at batch 1024 and for a single row: median of ``reps`` timed calls."""
    Xb, x1 = X[:1024], X[:1]
    predict(Xb)                                               # warm-up
    predict(x1)

    def median_ms(Z):
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            predict(Z)
            times.append((time.perf_counter() - t) * 1000)
        return float(np.median(times))

    return {"ms_per_row_batch1024": median_ms(Xb) / len(Xb), "ms_single_row": median_ms(x1)}


def run_qrnn_hpo(model: str, seed: int, fold, cfg: dict, rd: Path) -> None:
    """The E1 QRNN search (configs/model.yaml) on this experiment's features."""
    from hsboa.models import forward
    from hsboa.train import train_hpo
    mc = cfg["models"][model]
    mcfg = yaml.safe_load((ROOT / mc["model_config"]).read_text(encoding="utf-8"))
    m = mcfg["model"]
    base = {"epochs": mc["epochs"], "patience": None, "pooling": m["pooling"],
            "attention_units": m["attention_units"]}
    t0 = time.perf_counter()
    res = train_hpo(fold.X_train.shape[1], fold.X_train, fold.y_train, fold.X_val, fold.y_val,
                    space=mcfg["hpo"]["space"], base=base, optimizer=mc["tuner"], pop_size=mc["pop_size"],
                    max_evals=mc["max_evals"], train_samples=cfg["search"]["train_samples"], seed=seed,
                    cache_path=rd / "trials.jsonl", final_epochs=mc["final_epochs"], final_patience=None,
                    classes=fold.classes)
    res.extra["trials"].to_csv(rd / "trials.csv", index=False)
    res.history.to_csv(rd / "history.csv", index=False)
    np.savez(rd / "weights.npz", weights=res.weights, spec=json.dumps(res.spec.to_dict()))
    predict = lambda Z: forward(res.spec, res.weights, Z)
    rec = {"model": model, "seed": seed, "settings": res.extra["best_settings"],
           "wall_s": time.perf_counter() - t0, "distinct_trials": res.extra["distinct_trials"],
           "n_params": res.spec.n_params, "gradient_steps": res.n_gradient_steps,
           "peak_memory_mb": peak_memory_mb(), "latency": latency_ms(predict, fold.X_val),
           "dataset": cfg.get("dataset", DEFAULT_DATASET),
           "val": scores(predict(fold.X_val), fold.y_val, fold.classes),
           "test": None if fold.X_test is None else scores(predict(fold.X_test), fold.y_test, fold.classes)}
    (rd / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


def run_one(model: str, seed: int, fold, cfg: dict, rd: Path) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    if cfg["models"][model].get("tune") == "qrnn_hpo":
        return run_qrnn_hpo(model, seed, fold, cfg, rd)
    s = cfg["search"]
    r = tune_and_fit(model, cfg["models"][model], fold.X_train, fold.y_train, fold.X_val, fold.y_val,
                     seed, train_samples=s["train_samples"], pop_size=s["pop_size"], run_dir=rd,
                     classes=fold.classes)
    if not r["trials"].empty:
        r["trials"].to_csv(rd / "trials.csv", index=False)
    if "history" in r["info"]:
        pd.DataFrame(r["info"]["history"]).to_csv(rd / "history.csv", index_label="epoch")
    trials = r["trials"]
    rec = {"model": model, "seed": seed, "settings": r["settings"], "wall_s": r["wall_s"],
           "distinct_trials": int((~trials["cached"]).sum()) if not trials.empty else 0,
           "n_params": r["info"].get("n_params"),
           "gradient_steps": int(r["info"].get("gradient_steps", 0)
                                 + (trials.loc[~trials["cached"], "gradient_steps"].sum() if not trials.empty else 0)),
           "peak_memory_mb": peak_memory_mb(), "latency": latency_ms(r["predict"], fold.X_val),
           "dataset": cfg.get("dataset", DEFAULT_DATASET),
           "val": scores(r["predict"](fold.X_val), fold.y_val, fold.classes),
           "test": None if fold.X_test is None else scores(r["predict"](fold.X_test), fold.y_test, fold.classes)}
    (rd / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


def add_test_scores(model: str, rd: Path, fold) -> None:
    """Refit the final model from saved settings and seed (deterministic) and score the test partition."""
    rec = json.loads((rd / "result.json").read_text(encoding="utf-8"))
    if (rd / "weights.npz").exists():                         # tuned QRNN: saved weights, no refit
        from hsboa.models import QRNNSpec, forward
        z = np.load(rd / "weights.npz")
        spec = QRNNSpec(**json.loads(str(z["spec"])))
        predict = lambda Z: forward(spec, z["weights"], Z)
    else:
        predict, _ = FIT[model](rec["settings"], fold.X_train, fold.y_train, fold.X_val, fold.y_val,
                                rec["seed"])
    val = scores(predict(fold.X_val), fold.y_val, fold.classes)
    rec["refit_val_macro_f1_diff"] = val["macro_f1"] - rec["val"]["macro_f1"]   # ~0 if deterministic
    rec["test"] = scores(predict(fold.X_test), fold.y_test, fold.classes)
    (rd / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


def _rows(out: Path, e1: Path | None, e1b: Path | None = None) -> list[dict]:
    rows = []
    for p in sorted(out.glob("*/seed_*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"model": r["model"], "seed": r["seed"], "wall_min": r["wall_s"] / 60,
                     "distinct_trials": r["distinct_trials"], "n_params": r["n_params"],
                     "gradient_steps": r["gradient_steps"], "peak_memory_mb": r["peak_memory_mb"],
                     **{f"latency_{k}": v for k, v in r["latency"].items()},
                     **{f"{part}_{k}": v for part in ("val", "test") if r[part] for k, v in r[part].items()
                        if k in METRICS or k.startswith("f1_")}})
    if e1 and e1.exists():
        for p in sorted(e1.glob("*/seed_*/result.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"model": f"qrnn_{r['tuner']}", "seed": r["seed"], "wall_min": r["wall_s"] / 60,
                         "distinct_trials": r["distinct_trials"], "n_params": r["n_params"],
                         "gradient_steps": r["n_gradient_steps"], "peak_memory_mb": r["peak_memory_mb"],
                         **{f"{part}_{k}": v for part in ("val", "test") if r[part] for k, v in r[part].items()
                            if k in METRICS or k.startswith("f1_")}})
    if e1b and e1b.exists():
        for p in sorted(e1b.glob("*/seed_*/result.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"model": f"{r['model']}_{r['tuner']}", "seed": r["seed"], "wall_min": r["wall_s"] / 60,
                         "distinct_trials": r["distinct_trials"], "peak_memory_mb": r["peak_memory_mb"],
                         **{f"{part}_{k}": v for part in ("val", "test") if r[part] for k, v in r[part].items()
                            if k in METRICS or k.startswith("f1_")}})
    return rows


def summarize(out: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    rows = _rows(out, ROOT / cfg["e1_results"] if "e1_results" in cfg else None,
                 ROOT / cfg["e1b_results"] if "e1b_results" in cfg else None)
    ref = cfg.get("reference", "qrnn_hsboa")
    if not rows:
        raise FileNotFoundError(f"no finished runs under {out}")
    runs = pd.DataFrame(rows).sort_values(["model", "seed"])
    num = runs.select_dtypes("number").drop(columns=["seed"])
    summary = num.groupby(runs["model"]).agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    tests = []
    if ref in set(runs["model"]):
        for other in sorted(set(runs["model"]) - {ref}):
            for col in [c for c in runs.columns if c.startswith(("val_", "test_"))]:
                wide = runs.pivot(index="seed", columns="model", values=col)[[ref, other]].dropna()
                if len(wide) == 0:
                    continue
                a, b = wide[ref].to_numpy(), wide[other].to_numpy()
                lo, hi = bootstrap_ci(a - b)
                tests.append({"metric": col, "a": ref, "b": other, "n_seeds": len(wide),
                              "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": (a - b).mean(),
                              "ci95_low": lo, "ci95_high": hi, "p": wilcoxon(a, b), "a12_a_better": a12(-a, -b)})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        primary = tests["metric"].isin(["val_macro_f1", "test_macro_f1"])
        tests["p_holm_primary"] = np.nan
        tests.loc[primary, "p_holm_primary"] = holm(tests.loc[primary, "p"].to_numpy())
    for name, df in (("runs", runs), ("summary", summary), ("tests", tests)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "report.md").write_text(_report(runs, summary, tests, ref), encoding="utf-8")
    return {"runs": runs, "summary": summary, "tests": tests}


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _report(runs, summary, tests, ref: str) -> str:
    keep = ["model"] + [c for c in summary.columns if c.endswith(("_mean", "_std")) and c.split("_")[0] in ("val", "test")
                        and any(m in c for m in ("macro_f1", "balanced_accuracy", "f1_Shellcode", "f1_Worms"))] \
        + [c for c in ("wall_min_mean", "count") if c in summary.columns]
    keep = [c for c in keep if c in summary.columns]
    lines = ["# E2 model comparison: summary", "",
             f"Mean and SD over seeds. Reference for paired tests: {ref}. qrnn_hsboa / qrnn_random come "
             "from E1 (d = 10), hgb_* from E1b (joint feature selection); all other models use d = 20. "
             "Wilcoxon signed-rank paired by seed; bootstrap 95% CI of the mean difference; "
             "Holm over the primary metrics (validation and test macro-F1).", "",
             "## Models", "", _md(summary[keep].sort_values(keep[1], ascending=False)), ""]
    if not tests.empty:
        prim = tests[tests["metric"].isin(["val_macro_f1", "test_macro_f1"])]
        lines += ["## Paired comparisons (primary metrics)", "", _md(prim.round(4)), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "experiments" / "e2_unsw_baselines.yaml", type=Path)
    ap.add_argument("--models", nargs="*", help="subset of models to run (default: all in the config)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="parallel worker (tools/run_parallel.py): claim runs, skip the summary")
    args = ap.parse_args()

    cfg, dcfg = load(args.config, args.smoke)
    out = ROOT / "results" / (cfg["experiment"] + ("_smoke" if args.smoke else ""))
    models = args.models or list(cfg["models"])
    unknown = set(models) - set(cfg["models"])
    if unknown:
        raise SystemExit(f"unknown models {sorted(unknown)}; choose from {list(cfg['models'])}")
    missing = [m for m in models if not available(m)]
    if missing and not args.summarize_only:
        if not args.smoke:
            raise SystemExit(f"models {missing} need libraries that are not installed "
                             "(see SETUP_NEW_MACHINE.md); install them or pass --models without them")
        print(f"smoke: skipping {missing} (library not installed)", flush=True)
        models = [m for m in models if m not in missing]
    if not args.summarize_only:
        check_manifest(out, cfg)
        ds = get_dataset(cfg.get("dataset", DEFAULT_DATASET))
        X, y, X_te, y_te = load_parts(ds)
        print(f"{cfg['experiment']}: seeds {cfg['seeds']}, models {models}, "
              f"{ds.name}, test part {'found' if X_te is not None else 'MISSING (validation only)'}", flush=True)
        for seed in cfg["seeds"]:
            todo = []
            for model in models:
                rd = out / model / f"seed_{seed:02d}"
                done = (rd / "result.json").exists()
                needs_test = done and X_te is not None and \
                    json.loads((rd / "result.json").read_text(encoding="utf-8"))["test"] is None
                if (not done or needs_test) and not taken(rd, args.worker):
                    todo.append((model, rd, done))
            if not todo:
                continue
            fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, X_test=X_te, y_test=y_te,
                                dataset=ds)
            for model, rd, done in todo:
                if not claim(rd, args.worker):    # another worker took it meanwhile
                    continue
                t0 = time.perf_counter()
                if done:
                    add_test_scores(model, rd, fold)
                    print(f"seed {seed} {model}: test scores added", flush=True)
                    continue
                run_one(model, seed, fold, cfg, rd)
                r = json.loads((rd / "result.json").read_text(encoding="utf-8"))
                test = f", test {r['test']['macro_f1']:.4f}" if r["test"] else ""
                print(f"seed {seed} {model}: val macro-F1 {r['val']['macro_f1']:.4f}{test} "
                      f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    if args.worker:
        return
    t = summarize(out, cfg)
    cols = ["model", "val_macro_f1_mean", "val_macro_f1_std", "wall_min_mean", "val_macro_f1_count"]
    print(t["summary"][[c for c in cols if c in t["summary"]]].round(4).to_string(index=False))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
