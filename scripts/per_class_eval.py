"""Per-class results on the official test partition, with intervals that include test-sample uncertainty.

Why: seed-level intervals condition on one fixed test set. For rare classes (Analysis 677, Backdoor 583,
Shellcode 378, Worms 44 records) the test sample itself is a large source of uncertainty. This script
  1. refits the final model of each main arm from its stored settings and seed (deterministic; each refit
     is checked against the stored test macro-F1) and keeps the test predictions,
  2. reports class support and per-class F1 / recall with two intervals:
       seed interval     percentile bootstrap over the ten seeds (the paper's usual interval; fixed test set)
       grouped interval  percentile bootstrap over GROUPS of identical 42-feature test records (the unit that
                         cannot be split between samples), applied to the seed-averaged score; the same
                         resamples are used for every arm, so paired differences between arms are exact
  3. writes the confusion counts of every arm and seed (reproduction package).

Arms (experiment: folder): lgbm_default (e6: default), hgb_default / hgb_hsboa / hgb_sboa / hgb_random (e1b),
lgbm_rs, xgb_rs, rf_rs, mlp, cnn_bilstm (e2).

Usage (from the repository root; LightGBM and XGBoost needed for the lgbm / xgb arms):
    python scripts/per_class_eval.py refit --workers 10            # refit + predictions (resumable)
    python scripts/per_class_eval.py refit --arms hgb_default --seeds 0      # quick check
    python scripts/per_class_eval.py summarize [--boot 2000]
Writes results/per_class_eval/{predictions/<arm>/seed_XX.npy, y_test.npy, groups.npy, confusions.csv,
refit_check.csv, support.csv, per_class.csv, paired_vs_default.csv, report.md}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = ROOT / "results" / "per_class_eval"
EXP = {"e1b": ("run_gbdt.py", "e1b_unsw_gbdt.yaml"), "e6": ("run_shift.py", "e6_unsw_shift.yaml"),
       "e2": ("run_baselines.py", "e2_unsw_baselines.yaml")}
ARMS = {"lgbm_default": ("e6", "default"), "hgb_default": ("e1b", "default"), "hgb_hsboa": ("e1b", "hsboa"),
        "hgb_sboa": ("e1b", "sboa"), "hgb_random": ("e1b", "random"), "lgbm_rs": ("e2", "lgbm"),
        "xgb_rs": ("e2", "xgb"), "rf_rs": ("e2", "rf"), "mlp": ("e2", "mlp"), "cnn_bilstm": ("e2", "cnn_bilstm")}
LABEL = {"lgbm_default": "LightGBM default", "hgb_default": "HGB default", "hgb_hsboa": "HGB+HSBOA",
         "hgb_sboa": "HGB+SBOA", "hgb_random": "HGB+RS", "lgbm_rs": "LightGBM+RS", "xgb_rs": "XGBoost+RS",
         "rf_rs": "RF+RS", "mlp": "MLP", "cnn_bilstm": "CNN-BiLSTM"}
REF = "lgbm_default"


def _module(exp: str):
    path = ROOT / "experiments" / EXP[exp][0]
    spec = importlib.util.spec_from_file_location(f"_pc_{exp}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def group_codes(X: pd.DataFrame) -> np.ndarray:
    key = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy()
    return pd.factorize(key)[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

def refit_seed(seed: int, arms: list[str]) -> list[dict]:
    from hsboa.baselines import FIT
    from hsboa.datasets import get_dataset
    from hsboa.folds import prepare_fold
    from hsboa.metrics import evaluate, predict_labels
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")
    y_te = np.asarray(y_te, int)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "y_test.npy").exists():
        np.save(OUT / "y_test.npy", y_te.astype(np.uint8))
        np.save(OUT / "groups.npy", group_codes(X_te))
        (OUT / "classes.json").write_text(json.dumps(list(ds.classes)), encoding="utf-8")
    rows = []
    for exp in sorted({ARMS[a][0] for a in arms}):
        todo = [a for a in arms if ARMS[a][0] == exp and not (OUT / "predictions" / a / f"seed_{seed:02d}.npy").exists()]
        if not todo:
            continue
        mod = _module(exp)
        loaded = mod.load(ROOT / "configs" / "experiments" / EXP[exp][1], False)
        cfg, dcfg = loaded[0], loaded[1]
        fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, X_test=X_te, y_test=y_te, dataset=ds)
        assert np.array_equal(np.asarray(fold.y_test), y_te), "fold reordered the test records"
        for arm in todo:
            t0 = time.perf_counter()
            rd = ROOT / "results" / cfg["experiment"] / ARMS[arm][1] / f"seed_{seed:02d}"
            rec = json.loads((rd / "result.json").read_text(encoding="utf-8"))
            if exp == "e1b":
                from hsboa.gbdt import masked
                predict, _ = masked(FIT[cfg["model"]])(rec["settings"], fold.X_train, fold.y_train, fold.X_val,
                                                      fold.y_val, seed)
            elif exp == "e6":
                from hsboa.shift_tune import refit
                predict = refit(cfg["model"], rec["settings"], fold.X_train, fold.y_train, fold.X_val,
                                fold.y_val, seed, None)
            else:
                predict, _ = FIT[ARMS[arm][1]](rec["settings"], fold.X_train, fold.y_train, fold.X_val,
                                               fold.y_val, seed)
            pred = predict_labels(predict(fold.X_test)).astype(np.uint8)
            f1 = evaluate(y_te, pred, classes=fold.classes).summary["macro_f1"]
            stored = rec["test"]["macro_f1"]
            d = OUT / "predictions" / arm
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"seed_{seed:02d}.npy", pred)
            rows.append({"arm": arm, "seed": seed, "stored_test_macro_f1": stored, "refit_test_macro_f1": f1,
                         "abs_diff": abs(f1 - stored), "min": (time.perf_counter() - t0) / 60})
            print(f"seed {seed} {arm}: refit {f1:.6f} stored {stored:.6f} diff {abs(f1 - stored):.2e}", flush=True)
    return rows


def cmd_refit(args) -> None:
    arms = args.arms or list(ARMS)
    seeds = args.seeds if args.seeds else list(range(10))
    if args.workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            res = [r for rs in ex.map(refit_seed, seeds, [arms] * len(seeds)) for r in rs]
    else:
        res = [r for s in seeds for r in refit_seed(s, arms)]
    if res:
        p = OUT / "refit_check.csv"
        new = pd.DataFrame(res)
        if p.exists():
            new = pd.concat([pd.read_csv(p), new]).drop_duplicates(["arm", "seed"], keep="last")
        new.sort_values(["arm", "seed"]).to_csv(p, index=False)
        bad = new[new.abs_diff > 1e-6]
        print(f"refit check: {len(new)} runs, max |diff| {new.abs_diff.max():.2e}, {len(bad)} above 1e-6", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def f1_from(tp, pc, tc):
    den = pc + tc
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def recall_from(tp, tc):
    return np.where(tc > 0, tp / np.maximum(tc, 1e-12), 0.0)


def _md(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def cmd_summarize(args) -> None:
    from hsboa.stats import bootstrap_ci
    classes = json.loads((OUT / "classes.json").read_text(encoding="utf-8"))
    k = len(classes)
    y = np.load(OUT / "y_test.npy").astype(int)
    g = np.load(OUT / "groups.npy").astype(int)
    ng = g.max() + 1
    tc_g = np.zeros((ng, k), np.float32)
    np.add.at(tc_g, (g, y), 1)
    arms = [a for a in ARMS if len(list((OUT / "predictions" / a).glob("seed_*.npy"))) > 0]
    seeds = {a: sorted(int(p.stem.split("_")[1]) for p in (OUT / "predictions" / a).glob("seed_*.npy")) for a in arms}
    support = np.bincount(y, minlength=k)

    # confusion counts and per-seed per-class scores
    conf_rows, seed_f1, seed_rec, tp_g, pc_g = [], {}, {}, {}, {}
    for a in arms:
        f1s, recs, tps, pcs = [], [], [], []
        for s in seeds[a]:
            pred = np.load(OUT / "predictions" / a / f"seed_{s:02d}.npy").astype(int)
            cm = np.zeros((k, k), int)
            np.add.at(cm, (y, pred), 1)
            for i in range(k):
                for j in range(k):
                    if cm[i, j]:
                        conf_rows.append({"arm": a, "seed": s, "true": classes[i], "pred": classes[j], "n": int(cm[i, j])})
            tp = np.diag(cm).astype(float)
            f1s.append(f1_from(tp, cm.sum(0), cm.sum(1)))
            recs.append(recall_from(tp, cm.sum(1)))
            tpm = np.zeros((ng, k), np.float32)
            pcm = np.zeros((ng, k), np.float32)
            ok = pred == y
            np.add.at(tpm, (g[ok], y[ok]), 1)
            np.add.at(pcm, (g, pred), 1)
            tps.append(tpm)
            pcs.append(pcm)
        seed_f1[a], seed_rec[a] = np.array(f1s), np.array(recs)
        tp_g[a], pc_g[a] = tps, pcs
    pd.DataFrame(conf_rows).to_csv(OUT / "confusions.csv", index=False)

    # grouped bootstrap: shared resamples of the test groups
    rng = np.random.default_rng(2026)
    B, chunk = args.boot, 250
    F1 = {a: np.zeros((B, k)) for a in arms}
    REC = {a: np.zeros((B, k)) for a in arms}
    for b0 in range(0, B, chunk):
        n = min(chunk, B - b0)
        W = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=n).astype(np.float32)
        TC = W @ tc_g
        for a in arms:
            f_acc, r_acc = 0.0, 0.0
            for tpm, pcm in zip(tp_g[a], pc_g[a]):
                TP, PC = W @ tpm, W @ pcm
                f_acc = f_acc + f1_from(TP, PC, TC)
                r_acc = r_acc + recall_from(TP, TC)
            F1[a][b0:b0 + n] = f_acc / len(seeds[a])
            REC[a][b0:b0 + n] = r_acc / len(seeds[a])
        print(f"bootstrap {b0 + n}/{B}", flush=True)

    rows = []
    pd.DataFrame({"class": classes, "support": support, "share_percent": 100 * support / support.sum()}).to_csv(
        OUT / "support.csv", index=False)
    for a in arms:
        for j, c in enumerate(classes + ["macro-F1"]):
            if j < k:
                x, br, xr, brr = seed_f1[a][:, j], F1[a][:, j], seed_rec[a][:, j], REC[a][:, j]
            else:
                x, br = seed_f1[a].mean(1), F1[a].mean(1)
                xr, brr = seed_rec[a].mean(1), REC[a].mean(1)
            lo, hi = bootstrap_ci(x)
            glo, ghi = np.quantile(br, [0.025, 0.975])
            rlo, rhi = bootstrap_ci(xr)
            grlo, grhi = np.quantile(brr, [0.025, 0.975])
            rows.append({"arm": a, "model": LABEL[a], "class": c, "support": int(support[j]) if j < k else int(support.sum()),
                         "n_seeds": len(x), "f1": x.mean(), "f1_seed_lo": lo, "f1_seed_hi": hi,
                         "f1_grouped_lo": glo, "f1_grouped_hi": ghi, "recall": xr.mean(), "recall_seed_lo": rlo,
                         "recall_seed_hi": rhi, "recall_grouped_lo": grlo, "recall_grouped_hi": grhi})
    pcdf = pd.DataFrame(rows)
    pcdf.to_csv(OUT / "per_class.csv", index=False)

    # paired differences against the reference arm (same resamples)
    prow = []
    if REF in arms:
        for a in arms:
            if a == REF:
                continue
            for j, c in enumerate(classes + ["macro-F1"]):
                if j < k:
                    d_seed = seed_f1[a][:, j] - seed_f1[REF][:, j]
                    d_boot = F1[a][:, j] - F1[REF][:, j]
                else:
                    d_seed = seed_f1[a].mean(1) - seed_f1[REF].mean(1)
                    d_boot = F1[a].mean(1) - F1[REF].mean(1)
                lo, hi = bootstrap_ci(d_seed)
                glo, ghi = np.quantile(d_boot, [0.025, 0.975])
                prow.append({"arm": a, "reference": REF, "class": c, "diff": d_seed.mean(), "seed_lo": lo, "seed_hi": hi,
                             "grouped_lo": glo, "grouped_hi": ghi})
    pd.DataFrame(prow).to_csv(OUT / "paired_vs_default.csv", index=False)

    # report
    lines = ["# Per-class test results (official partition)", "",
             f"Grouped bootstrap: {B} resamples of the {ng:,} groups of identical 42-feature test records "
             "(shared by all arms). 'seed' = percentile bootstrap over seeds; 'grouped' = over test groups.", "",
             "## Support", "", _md(pd.read_csv(OUT / "support.csv").round(2)), ""]
    for a in arms:
        t = pcdf[pcdf.arm == a]
        t = t.assign(F1=[f"{r.f1:.3f}" for r in t.itertuples()],
                     seed_CI=[f"{r.f1_seed_lo:.3f}-{r.f1_seed_hi:.3f}" for r in t.itertuples()],
                     grouped_CI=[f"{r.f1_grouped_lo:.3f}-{r.f1_grouped_hi:.3f}" for r in t.itertuples()])
        lines += [f"## {LABEL[a]} ({t.n_seeds.iloc[0]} seeds)", "",
                  _md(t[["class", "support", "F1", "seed_CI", "grouped_CI"]]), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refit")
    r.add_argument("--arms", nargs="*", choices=list(ARMS))
    r.add_argument("--seeds", nargs="*", type=int)
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summarize")
    s.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    (cmd_refit if args.cmd == "refit" else cmd_summarize)(args)


if __name__ == "__main__":
    main()
