"""Final evaluation bundle (reads the test labels; run once, with `run_shift.py --final-test`).

Usage (from the repository root):
    python scripts/final_eval.py --smoke              # ~1 minute: checks the pipeline
    python scripts/final_eval.py --workers 10         # full run (PC), resumable per seed

All settings were fixed before any of these numbers were seen (2026-10-30). Four parts:

A  Binary LightGBM on the official split (answers binary-only papers such as ISBOA-BiLSTM,
   ICCCS 2026, 90.35% accuracy): Normal vs attack, top-20 MI features (E2 pipeline), seeds 0-9,
   two settings: LightGBM defaults (class weight power 0.5, as the E6 default arm) and each
   seed's E2-tuned settings. Test accuracy, precision, recall (detection rate), F1, FPR.
B  Protocol effect: the same 10-class LightGBM (defaults; E2 settings) on (i) the official
   split and (ii) a stratified random 70/30 split of the merged partitions (257,673 records),
   the protocol of many published papers. Same pipeline, fitted on the training side only.
C  Conflict-aware ceiling: records with identical 42 features but different labels cannot all
   be classified correctly. Predicting each identical-feature group's majority label gives the
   best accuracy any classifier of the 42 features can reach on that set (in-sample bound) and a
   near-best macro-F1. Reported for the training partition, the test partition and both.
D  E2 statistics: Holm over the H1-H4 family of DESIGN_OPTION_A.md (16 macro-F1 comparisons:
   hgb_hsboa vs qrnn_hsboa, qrnn_random, qrnn_random_d20, cnn_bilstm, mlp (H1), hgb_random (H2),
   hgb_sboa (H3), hgb_default (H4); validation and test), replacing the 28-comparison family of
   the run_baselines.py summary.
Writes results/final_eval[_smoke]/: rows/seed_XX.json, binary.csv, protocol.csv, ceiling.csv,
e2_holm_family.csv, report.md.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
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

from hsboa.baselines import FIT  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.metrics import binary_view, evaluate  # noqa: E402
from hsboa.stats import holm  # noqa: E402

E2 = "results/e2_unsw_baselines"
LGBM_DEFAULT = {"learning_rate": 0.1, "n_estimators": 100, "num_leaves": 31, "min_child_samples": 20,
                "reg_lambda": 0.0, "subsample": 1.0, "colsample_bytree": 1.0, "class_weight_power": 0.5}
HGB_DEFAULT = {"learning_rate": 0.1, "max_iter": 20, "max_leaf_nodes": 31, "min_samples_leaf": 20,
               "l2_regularization": 0.0, "class_weight_power": 0.5}
H1_H4 = {"qrnn_hsboa": "H1", "qrnn_random": "H1", "qrnn_random_d20": "H1", "cnn_bilstm": "H1", "mlp": "H1",
         "hgb_random": "H2", "hgb_sboa": "H3", "hgb_default": "H4"}
FULL = {"seeds": list(range(10))}
SMOKE = {"seeds": [0], "rows": 20000}


def _cfg() -> dict:
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    dcfg["features"].update(n_features=20, selection="mutual_info")      # as E2
    return dcfg


def _settings(model: str, seed: int, smoke: bool) -> dict[str, dict]:
    """{name: settings} for the default and (LightGBM only) the seed's E2-tuned settings."""
    if model == "hgb":
        return {"default": dict(HGB_DEFAULT)}
    out = {"default": dict(LGBM_DEFAULT)}
    p = ROOT / E2 / "lgbm" / f"seed_{seed:02d}" / "result.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))["settings"]
        out["e2_tuned"] = {k: v for k, v in s.items() if k != "n_classes"}
    if smoke:
        for s in out.values():
            s["n_estimators"] = 20
    return out


def _multi(proba, y, classes) -> dict:
    rep = evaluate(y, proba.argmax(1), classes=classes)
    return {"macro_f1": rep.summary["macro_f1"], "accuracy": rep.summary["accuracy"],
            "binary_accuracy": rep.binary["accuracy"], "binary_fpr": rep.binary["fpr"]}


def seed_task(seed: int, model: str, smoke: bool, out: Path) -> dict:
    path = out / "rows" / f"seed_{seed:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from sklearn.model_selection import train_test_split
    t0 = time.perf_counter()
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")
    if smoke:
        rng = np.random.default_rng(0)
        i, j = rng.choice(len(X), SMOKE["rows"], replace=False), rng.choice(len(X_te), SMOKE["rows"] // 2, replace=False)
        X, y, X_te, y_te = X.iloc[np.sort(i)], y[np.sort(i)], X_te.iloc[np.sort(j)], y_te[np.sort(j)]
    dcfg, k = _cfg(), len(ds.classes)
    fit = FIT[model]
    settings = _settings(model, seed, smoke)
    rows = []

    # A + B(i): official split
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, X_test=X_te, y_test=y_te, dataset=ds)
    for name, s in settings.items():
        pr, _ = fit({**s, "n_classes": k}, fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed)
        rows.append({"part": "protocol", "split": "official", "settings": name,
                     **_multi(pr(fold.X_test), fold.y_test, fold.classes)})
        yb_tr, yb_va, yb_te = (fold.y_train != 0).astype(int), (fold.y_val != 0).astype(int), (fold.y_test != 0).astype(int)
        pb, _ = fit({**s, "n_classes": 2}, fold.X_train, yb_tr, fold.X_val, yb_va, seed)
        bv = binary_view(yb_te, pb(fold.X_test).argmax(1))
        rows.append({"part": "binary", "split": "official", "settings": name,
                     **{k2: bv[k2] for k2 in ("accuracy", "precision", "tpr", "f1", "fpr")}})

    # B(ii): stratified random 70/30 split of the merged partitions
    Xm = pd.concat([X, X_te], ignore_index=True)
    ym = np.r_[y, y_te]
    tr, te = train_test_split(np.arange(len(ym)), test_size=0.3, stratify=ym, random_state=seed)
    fold_r = prepare_fold(Xm.iloc[tr], ym[tr], seed=seed, train_fraction=1.0, cfg=dcfg, X_test=Xm.iloc[te],
                          y_test=ym[te], dataset=ds)
    for name, s in settings.items():
        pr, _ = fit({**s, "n_classes": k}, fold_r.X_train, fold_r.y_train, fold_r.X_val, fold_r.y_val, seed)
        rows.append({"part": "protocol", "split": "random_70_30", "settings": name,
                     **_multi(pr(fold_r.X_test), fold_r.y_test, fold_r.classes)})
    rec = {"seed": seed, "model": model, "wall_s": time.perf_counter() - t0, "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"seed {seed}: done ({rec['wall_s'] / 60:.1f} min)", flush=True)
    return rec


def ceiling(X: pd.DataFrame, y: np.ndarray, classes) -> dict:
    """Group-majority oracle on identical 42-feature records."""
    key = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy()
    df = pd.DataFrame({"key": key, "y": y})
    maj = df.groupby(["key", "y"]).size().reset_index(name="n").sort_values(["key", "n", "y"],
                                                                              ascending=[True, False, True])
    maj = maj.drop_duplicates("key").set_index("key")["y"]
    pred = df["key"].map(maj).to_numpy()
    rep = evaluate(y, pred, classes=classes)
    out = {"rows": len(y), "unique_feature_vectors": int(df["key"].nunique()),
           "rows_unavoidably_wrong": float((pred != y).mean()), "ceiling_accuracy": rep.summary["accuracy"],
           "ceiling_macro_f1": rep.summary["macro_f1"]}
    out.update({f"ceiling_f1_{c}": float(rep.per_class.loc[c, "f1"]) for c in classes})
    return out


def e2_family() -> pd.DataFrame:
    p = ROOT / E2 / "tests.csv"
    if not p.exists():
        return pd.DataFrame()
    t = pd.read_csv(p)
    t = t[t["metric"].isin(["val_macro_f1", "test_macro_f1"]) & t["b"].isin(H1_H4)].copy()
    t["hypothesis"] = t["b"].map(H1_H4)
    t["p_holm_h1_h4"] = holm(t["p"].to_numpy())
    return t[["hypothesis", "metric", "a", "b", "mean_diff", "ci95_low", "ci95_high", "p", "p_holm_primary",
              "p_holm_h1_h4"]].rename(columns={"p_holm_primary": "p_holm_28_old"})


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    model = "lgbm" if importlib.util.find_spec("lightgbm") else "hgb"
    if model == "hgb" and not args.smoke:
        raise SystemExit("LightGBM is not installed (it is needed for the full run)")
    grid = SMOKE if args.smoke else FULL
    out = ROOT / "results" / ("final_eval_smoke" if args.smoke else "final_eval")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"final evaluation ({model}): {len(grid['seeds'])} seeds, {args.workers} workers; results in {out}",
          flush=True)
    if args.workers > 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=args.workers)(delayed(seed_task)(s, model, args.smoke, out) for s in grid["seeds"])
    else:
        recs = [seed_task(s, model, args.smoke, out) for s in grid["seeds"]]
    rows = pd.DataFrame([{"seed": r["seed"], **row} for r in recs for row in r["rows"]])

    binary = rows[rows.part == "binary"].dropna(axis=1, how="all")
    protocol = rows[rows.part == "protocol"].dropna(axis=1, how="all")
    binary.to_csv(out / "binary.csv", index=False)
    protocol.to_csv(out / "protocol.csv", index=False)
    bsum = binary.groupby("settings")[["accuracy", "precision", "tpr", "f1", "fpr"]].agg(["mean", "std"])
    psum = protocol.groupby(["settings", "split"])[["macro_f1", "accuracy", "binary_accuracy",
                                                    "binary_fpr"]].agg(["mean", "std"])
    for df in (bsum, psum):
        df.columns = [f"{a}_{b}" for a, b in df.columns]
    gap = protocol.pivot_table(index=["settings", "seed"], columns="split", values=["macro_f1", "accuracy"])
    gap = pd.DataFrame({"macro_f1_gain_random_vs_official": gap[("macro_f1", "random_70_30")]
                        - gap[("macro_f1", "official")],
                        "accuracy_gain_random_vs_official": gap[("accuracy", "random_70_30")]
                        - gap[("accuracy", "official")]}).groupby("settings").agg(["mean", "min", "max"])
    gap.columns = [f"{a}_{b}" for a, b in gap.columns]

    ds = get_dataset()
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")
    if args.smoke:                          # the files are sorted by class: subsample at random
        rng = np.random.default_rng(1)
        i = np.sort(rng.choice(len(X), SMOKE["rows"], replace=False))
        j = np.sort(rng.choice(len(X_te), SMOKE["rows"] // 2, replace=False))
        X, y, X_te, y_te = X.iloc[i], y[i], X_te.iloc[j], y_te[j]
    ceil = pd.DataFrame([{"set": "training partition", **ceiling(X, y, ds.classes)},
                         {"set": "test partition", **ceiling(X_te, y_te, ds.classes)},
                         {"set": "both (merged)", **ceiling(pd.concat([X, X_te], ignore_index=True), np.r_[y, y_te],
                                                            ds.classes)}])
    ceil.to_csv(out / "ceiling.csv", index=False)
    fam = e2_family()
    fam.to_csv(out / "e2_holm_family.csv", index=False)

    lines = [f"# Final evaluation ({model})", "", f"Wall time {(time.perf_counter() - t0) / 60:.1f} min; "
             f"seeds {grid['seeds']}. Settings fixed before the run (script docstring).", "",
             "## A. Binary Normal-vs-attack LightGBM, official split (test partition)", "", _md(bsum.reset_index()), "",
             "## B. Protocol effect: official split vs random 70/30 split of the merged data (10 classes)", "",
             _md(psum.reset_index()), "", "Gain of the random split over the official split (per-seed differences):",
             "", _md(gap.reset_index()), "",
             "## C. Conflict-aware ceiling (group-majority oracle on identical 42-feature records)", "",
             _md(ceil[["set", "rows", "unique_feature_vectors", "rows_unavoidably_wrong", "ceiling_accuracy",
                       "ceiling_macro_f1"]]), "",
             "Per-class ceiling F1 in ceiling.csv. In-sample bound for accuracy; macro-F1 of the majority oracle "
             "is near the bound, not exactly it.", ""]
    if not fam.empty:
        lines += ["## D. E2: Holm over the H1-H4 design family (16 comparisons)", "", _md(fam), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
