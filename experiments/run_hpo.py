"""E1: tuner comparison on UNSW-NB15.

Usage (from the repository root):
    python experiments/run_hpo.py --smoke             # minutes: pipeline check
    python experiments/run_hpo.py                     # full E1, resumable
    python experiments/run_hpo.py --summarize-only

For each seed (outer loop, so partial results stay paired) and each tuner:
search on a training subsample (fitness 1 - validation macro-F1), retrain
the best settings on the full training part, and evaluate on validation and
the official test partition. Writes results/e1_unsw_tuners[_smoke]/:
    manifest.json
    <tuner>/seed_XX/trials.jsonl     trial checkpoint (a rerun resumes mid-search)
                    trials.csv, history.csv (final-model loss curve)
                    weights.npz      final weights + spec (re-evaluation without retraining)
                    result.json      settings, validation/test metrics, cost
    runs.csv, summary.csv, tests.csv, curves.csv, report.md

If the test partition is missing, runs are scored on validation only; once the
file is in data/raw/, rerunning adds the test scores from the saved weights.
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

from hsboa.datasets import DEFAULT_DATASET, get_dataset, load_parts  # noqa: E402
from _jobs import claim, taken  # noqa: E402
from hsboa.experiment import code_hash  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import evaluate, predict_labels  # noqa: E402
from hsboa.models import QRNNSpec, forward  # noqa: E402
from hsboa.stats import a12, bootstrap_ci, holm, wilcoxon  # noqa: E402
from hsboa.utils import peak_memory_mb  # noqa: E402

# Modules E1 depends on; only these guard a resume (unrelated new code does not).
E1_MODULES = ["data", "datasets", "preprocess", "folds", "metrics", "harmonic", "optimizers", "models",
              "fitness", "train"]


def load(path: Path, smoke: bool) -> tuple[dict, dict, dict, dict]:
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    dcfg = load_config(ROOT / "configs" / "data.yaml")
    mcfg = yaml.safe_load((ROOT / cfg["model_config"]).read_text(encoding="utf-8"))
    hcfg = yaml.safe_load((ROOT / cfg["harmonic_config"]).read_text(encoding="utf-8"))["harmonic"]
    cfg["seeds"] = dcfg["seeds"][:cfg["n_seeds"]]
    if smoke:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["hpo"].update(pop_size=4, max_evals={t: 8 for t in cfg["tuners"]},
                          train_samples=3000, epochs=2)
        cfg["final"].update(epochs=2)
        text += "\n# --smoke\n"
    # The hash covers every input that changes results: this config, the search space and the harmonic settings.
    blob = json.dumps({"e1": text, "space": mcfg["hpo"]["space"], "model": mcfg["model"],
                       "harmonic": hcfg}, sort_keys=True)
    cfg["_config_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return cfg, dcfg, mcfg, hcfg


def check_manifest(out: Path, cfg: dict, allow_code_change: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    man = {"experiment": cfg["experiment"], "config_sha256": cfg["_config_sha256"],
           "code_sha256": code_hash(E1_MODULES), "modules": E1_MODULES,
           "seeds": cfg["seeds"], "tuners": cfg["tuners"]}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["config_sha256"] != man["config_sha256"]:
            raise RuntimeError(f"{out} was made with a different configuration; use a new folder")
        if old["code_sha256"] != man["code_sha256"]:
            if not allow_code_change:
                raise RuntimeError(f"{out} was made with different code in {E1_MODULES}; "
                                   "pass --allow-code-change only if the change cannot affect results")
            man["code_changes"] = old.get("code_changes", []) + [
                {"from": old["code_sha256"], "to": man["code_sha256"],
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}]
        man["started"] = old.get("started")
    else:
        man["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")


def scores(spec: QRNNSpec, w: np.ndarray, X, y, classes) -> dict:
    rep = evaluate(y, predict_labels(forward(spec, w, X)), classes=classes)
    out = dict(rep.summary)
    out.update({f"binary_{k}": v for k, v in rep.binary.items()})
    out.update({f"f1_{c}": float(rep.per_class.loc[c, "f1"]) for c in classes})
    out.update({f"recall_{c}": float(rep.per_class.loc[c, "recall"]) for c in classes})
    return out


def run_one(tuner: str, seed: int, fold, cfg: dict, mcfg: dict, hcfg: dict, run_dir: Path) -> None:
    from hsboa.train import train_hpo

    h, m = cfg["hpo"], mcfg["model"]
    base = {"epochs": h["epochs"], "patience": h["patience"], "pooling": m["pooling"],
            "attention_units": m["attention_units"]}
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = train_hpo(fold.X_train.shape[1], fold.X_train, fold.y_train, fold.X_val, fold.y_val,
                    space=mcfg["hpo"]["space"], base=base, optimizer=tuner,
                    optimizer_params={"harmonic": hcfg} if tuner == "hsboa" else {},
                    pop_size=h["pop_size"], max_evals=h["max_evals"][tuner],
                    train_samples=h["train_samples"], seed=seed,
                    cache_path=run_dir / "trials.jsonl",
                    final_epochs=cfg["final"]["epochs"], final_patience=cfg["final"]["patience"],
                    classes=fold.classes)
    trials = res.extra["trials"]
    trials.to_csv(run_dir / "trials.csv", index=False)
    res.history.to_csv(run_dir / "history.csv", index=False)
    np.savez(run_dir / "weights.npz", weights=res.weights, spec=json.dumps(res.spec.to_dict()))
    rec = {"tuner": tuner, "seed": seed, "spec": res.spec.to_dict(), "n_params": res.spec.n_params,
           "settings": res.extra["best_settings"], "search_best_val_macro_f1": 1 - res.extra["search_best_fitness"],
           "distinct_trials": res.extra["distinct_trials"], "cached_trials": int(trials["cached"].sum()),
           "wall_s": time.perf_counter() - t0, "n_fitness_evals": res.n_fitness_evals,
           "n_gradient_steps": res.n_gradient_steps, "final_epochs_run": len(res.history),
           "peak_memory_mb": peak_memory_mb(),
           "dataset": cfg.get("dataset", DEFAULT_DATASET),
           "val": scores(res.spec, res.weights, fold.X_val, fold.y_val, fold.classes),
           "test": None if fold.X_test is None else scores(res.spec, res.weights, fold.X_test, fold.y_test,
                                                             fold.classes)}
    (run_dir / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


def add_test_scores(run_dir: Path, fold) -> None:
    """Score a finished run on the test partition from its saved weights (no retraining)."""
    rec = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    z = np.load(run_dir / "weights.npz")
    spec = QRNNSpec(**json.loads(str(z["spec"])))
    rec["test"] = scores(spec, z["weights"], fold.X_test, fold.y_test, fold.classes)
    (run_dir / "result.json").write_text(json.dumps(rec, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "mcc")      # plus every per-class f1_<class>


def summarize(out: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    rows, curves = [], []
    for p in sorted(out.glob("*/seed_*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"tuner": r["tuner"], "seed": r["seed"], "search_val_macro_f1": r["search_best_val_macro_f1"],
               "wall_min": r["wall_s"] / 60, "distinct_trials": r["distinct_trials"],
               "gradient_steps": r["n_gradient_steps"], "peak_memory_mb": r["peak_memory_mb"],
               **{f"setting_{k}": v for k, v in r["settings"].items()
                  if k in ("learning_rate", "units", "window", "dropout", "l2", "batch_size", "class_weight")}}
        for part in ("val", "test"):
            if r[part]:
                row.update({f"{part}_{k}": v for k, v in r[part].items() if k in METRICS or k.startswith("f1_")})
        rows.append(row)
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
    if len(cfg["tuners"]) >= 2:
        ref = cfg["tuners"][0]
        for other in cfg["tuners"][1:]:
            for col in [c for c in runs.columns if c.startswith(("val_", "test_", "search_"))]:
                wide = runs.pivot(index="seed", columns="tuner", values=col).dropna()
                if ref not in wide or other not in wide or len(wide) == 0:
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
    curves = pd.DataFrame(curves)
    for name, df in (("runs", runs), ("summary", summary), ("tests", tests), ("curves", curves)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "report.md").write_text(_report(cfg, runs, tests), encoding="utf-8")
    return {"runs": runs, "summary": summary, "tests": tests, "curves": curves}


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _report(cfg, runs, tests) -> str:
    n_done = runs.groupby("tuner").size().to_dict()
    lines = [f"# E1 tuner comparison: summary", "",
             f"Finished runs per tuner: {n_done} (planned {len(cfg['seeds'])} seeds). "
             f"Budget (evaluations per tuner, matched on distinct trainings) {cfg['hpo']['max_evals']} "
             f"(pop {cfg['hpo']['pop_size']}), "
             f"{cfg['hpo']['epochs']} epochs per trial, {cfg['final']['epochs']} final. "
             "Wilcoxon signed-rank paired by seed; bootstrap 95% CI of the mean difference; "
             "Holm over the primary metrics (validation and test macro-F1).", ""]
    cols = ["tuner", "seed"] + [c for c in ("search_val_macro_f1", "val_macro_f1", "test_macro_f1",
                                            "val_f1_Shellcode", "val_f1_Worms", "wall_min") if c in runs]
    lines += ["## Per run", "", _md(runs[cols]), ""]
    if not tests.empty:
        lines += ["## Paired comparisons", "", _md(tests.round(4)), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "experiments" / "e1_unsw_tuners.yaml", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="parallel worker (tools/run_parallel.py): claim runs, skip the summary")
    ap.add_argument("--allow-code-change", action="store_true")
    args = ap.parse_args()

    cfg, dcfg, mcfg, hcfg = load(args.config, args.smoke)
    out = ROOT / "results" / (cfg["experiment"] + ("_smoke" if args.smoke else ""))
    if not args.summarize_only:
        check_manifest(out, cfg, args.allow_code_change)
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
                    add_test_scores(rd, fold)
                    print(f"seed {seed} {tuner}: test scores added", flush=True)
                    continue
                run_one(tuner, seed, fold, cfg, mcfg, hcfg, rd)
                r = json.loads((rd / "result.json").read_text(encoding="utf-8"))
                test = f", test {r['test']['macro_f1']:.4f}" if r["test"] else ""
                print(f"seed {seed} {tuner}: val macro-F1 {r['val']['macro_f1']:.4f}{test} "
                      f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    if args.worker:
        return
    t = summarize(out, cfg)
    print(t["runs"].round(4).to_string(index=False))
    if not t["tests"].empty:
        print(t["tests"][["metric", "mean_diff", "ci95_low", "ci95_high", "p", "a12_a_better"]]
              .round(4).to_string(index=False))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
