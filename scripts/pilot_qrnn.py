"""Pilot: weight-space swarm training (a) vs hyper-parameter tuning of Adam (b).

Usage (from the repository root):
    python scripts/pilot_qrnn.py --smoke          # minutes: checks the pipeline
    python scripts/pilot_qrnn.py                  # full pilot, seed 0, resumable
    python scripts/pilot_qrnn.py --arms adam_default swarm_hsboa

Arms (configs/model.yaml):
    adam_default, adam_balanced     gradient training, untuned (balanced = class-weighted loss)
    swarm_hsboa, swarm_sboa, swarm_pso    option (a): optimiser searches the flat weight vector
    hpo_hsboa, hpo_sboa, hpo_random       option (b): optimiser tunes Adam's hyper-parameters

Dataset: UNSW-NB15. Everything is
scored on the validation split of the official training partition (model
selection; the test partition stays untouched). Writes
results/pilot_qrnn_unsw[_smoke]/seed_XX/:
<arm>.json (metrics, work, time, settings), <arm>_history.csv (loss curve), <arm>_trials.csv (HPO arms), and summary.csv / summary.md. HPO arms
checkpoint each trial in <arm>_trials.jsonl, so a rerun resumes mid-search.
"""

from __future__ import annotations

import argparse
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

from hsboa.data import CLASSES  # noqa: E402
from hsboa.datasets import get_dataset, load_parts  # noqa: E402
from hsboa.experiment import code_hash  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.models import QRNNSpec  # noqa: E402
from hsboa.train import train_gradient, train_hpo, train_swarm  # noqa: E402

ARMS = ("adam_default", "adam_balanced", "swarm_hsboa", "swarm_sboa", "swarm_pso",
        "hpo_hsboa", "hpo_sboa", "hpo_random")


def smoke(mcfg: dict) -> dict:
    mcfg["swarm"].update(max_evals=300, fitness_samples=2000, pop_size=10)
    mcfg["gradient"].update(epochs=2)
    mcfg["hpo"].update(max_evals=4, pop_size=4, train_samples=3000, epochs=2)
    return mcfg


def run_arm(arm: str, fold, mcfg: dict, hcfg: dict, seed: int, out: Path):
    m, g, s, h = mcfg["model"], mcfg["gradient"], mcfg["swarm"], mcfg["hpo"]
    d = fold.X_train.shape[1]
    spec = QRNNSpec(d, m["units"], m["window"], m["pooling"], m["attention_units"])
    opt_params = lambda name: {"harmonic": hcfg} if name == "hsboa" else {}
    if arm.startswith("adam"):
        return train_gradient(spec, fold.X_train, fold.y_train, fold.X_val, fold.y_val,
                              learning_rate=g["learning_rate"], batch_size=g["batch_size"],
                              epochs=g["epochs"], patience=g["early_stopping_patience"],
                              dropout=g["dropout"], l2=g["l2"],
                              class_weight="balanced" if arm == "adam_balanced" else g["class_weight"],
                              seed=seed)
    name = arm.split("_", 1)[1]
    if arm.startswith("swarm"):
        return train_swarm(spec, fold.X_train, fold.y_train, optimizer=name,
                           optimizer_params=opt_params(name), pop_size=s["pop_size"],
                           max_evals=s["max_evals"], bound=s["bound"],
                           fitness_samples=s["fitness_samples"], fitness=s["fitness"], seed=seed)
    base = {"epochs": h["epochs"], "patience": h.get("early_stopping_patience", g["early_stopping_patience"]),
            "class_weight": g["class_weight"], "pooling": m["pooling"],
            "attention_units": m["attention_units"]}
    return train_hpo(d, fold.X_train, fold.y_train, fold.X_val, fold.y_val, space=h["space"],
                     base=base, optimizer=name, optimizer_params=opt_params(name),
                     pop_size=h["pop_size"], max_evals=h["max_evals"],
                     train_samples=h["train_samples"], seed=seed,
                     cache_path=out / f"{arm}_trials.jsonl")


def record(arm: str, res, fold) -> dict:
    rep = res.evaluate(fold.X_val, fold.y_val)
    tr = res.evaluate(fold.X_train, fold.y_train).summary
    extra = {k: v for k, v in res.extra.items() if k not in ("trials", "diversity")}
    for k in ("h_norm", "move_norm"):
        if k in extra:
            extra[k] = float(np.median(extra[k])) if len(extra[k]) else 0.0
    return {"arm": arm, "mode": res.mode, "spec": res.spec.to_dict(), "n_params": res.spec.n_params,
            "val": rep.summary, "val_binary": rep.binary,
            "val_per_class": {c: {"f1": float(rep.per_class.loc[c, "f1"]),
                                  "recall": float(rep.per_class.loc[c, "recall"])} for c in CLASSES},
            "train_macro_f1": tr["macro_f1"], "train_accuracy": tr["accuracy"],
            "wall_s": res.wall_s, "n_fitness_evals": res.n_fitness_evals,
            "n_gradient_steps": res.n_gradient_steps, "extra": extra}


def summarize(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out.glob("*.json")):
        if p.name == "manifest.json":
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"arm": r["arm"], "params": r["n_params"], "val_macro_f1": r["val"]["macro_f1"],
               "val_accuracy": r["val"]["accuracy"], "val_bal_acc": r["val"]["balanced_accuracy"],
               "val_mcc": r["val"]["mcc"], "train_macro_f1": r["train_macro_f1"]}
        row.update({f"recall_{c}": r["val_per_class"][c]["recall"] for c in CLASSES})
        row.update(wall_min=r["wall_s"] / 60, fitness_evals=r["n_fitness_evals"],
                   gradient_steps=r["n_gradient_steps"])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(out / "summary.csv", index=False)
    cols = [str(c) for c in df.columns]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
           for r in df.itertuples(index=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dcfg = load_config(ROOT / "configs" / "data.yaml")
    mcfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text(encoding="utf-8"))
    hcfg = yaml.safe_load((ROOT / "configs" / "hsboa.yaml").read_text(encoding="utf-8"))["harmonic"]
    if args.smoke:
        mcfg = smoke(mcfg)
    out = ROOT / "results" / ("pilot_qrnn_unsw_smoke" if args.smoke else "pilot_qrnn_unsw") / f"seed_{args.seed:02d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(
        {"seed": args.seed, "code_sha256": code_hash(), "model_config": mcfg, "harmonic": hcfg,
         "data_config": {k: v for k, v in dcfg.items() if k != "paths"},
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2, default=str), encoding="utf-8")

    t0 = time.perf_counter()
    ds = get_dataset()
    X, y, _, _ = load_parts(ds)    # the test partition is not used here
    fold = prepare_fold(X, y, seed=args.seed, train_fraction=1.0, cfg=dcfg, dataset=ds)
    print(f"fold ready in {time.perf_counter() - t0:.0f}s: train {len(fold.y_train)}, "
          f"val {len(fold.y_val)}, d = {fold.X_train.shape[1]}", flush=True)

    for arm in args.arms:
        if (out / f"{arm}.json").exists():
            print(f"{arm}: done already, skipped", flush=True)
            continue
        t1 = time.perf_counter()
        res = run_arm(arm, fold, mcfg, hcfg, args.seed, out)
        rec = record(arm, res, fold)
        res.history.to_csv(out / f"{arm}_history.csv", index=False)
        if "trials" in res.extra:
            res.extra["trials"].to_csv(out / f"{arm}_trials.csv", index=False)
        (out / f"{arm}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        print(f"{arm}: val macro-F1 {rec['val']['macro_f1']:.4f}, accuracy {rec['val']['accuracy']:.4f} "
              f"({(time.perf_counter() - t1) / 60:.1f} min)", flush=True)
    print(summarize(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
