"""Two-stage diagnostic: does Normal-vs-attack then attack-type classification, or label-shift correction, help?

Usage (from the repository root):
    python scripts/diagnose_twostage.py --smoke              # ~1 minute: checks the pipeline
    python scripts/diagnose_twostage.py --workers 16         # full run (PC), resumable
    python scripts/diagnose_twostage.py --summarize-only

In E2, LightGBM flags 99% of attacks on validation and test, but its false-alarm
rate (Normal flagged as attack) is 15% on validation and 36% on test, where Normal
is 45% of the traffic instead of 32%. This checks, on validation data only
(hsboa.twostage):

    pipelines  flat   the stored E2 model (lgbm, rf; settings per seed, d = 20)
               two    stage 1 Normal vs attack + stage 2 attack type (attack rows only),
                      same model family and settings, combined into ten-class probabilities
    decision   argmax plain argmax
               t      Normal if P(attack) < t, else the most likely attack; t chosen on half A
               em     EM label-shift correction (Saerens et al. 2002) on the unlabelled
                      evaluation rows, source prior = mean probabilities on half A; then argmax
               em_t   EM, then the threshold t
    shift      none   the held-out validation half as it is
               0.45   the held-out half resampled to 45% Normal (attack mix kept), as the
                      official test partition; 0.55 a stronger shift

Validation probabilities are split into stratified halves 5 times: t and the source
prior come from half A, every score from half B. The flat baseline repeats the stored
E2 model (determinism check). The test partition is never loaded.

Decision rule, fixed before the run (lgbm; macro-F1, mean of 10 seeds):
    R1 two-stage model: two_t - flat_t >= +0.005 at shift none, positive on >= 8/10 seeds.
    R2 threshold:       flat_t - flat_argmax >= +0.005 at shift none, on >= 8/10 seeds.
    R3 label shift:     for the pipeline chosen by R1/R2, em(_t) - its non-EM variant >= +0.005
                        at shift 0.45 on >= 8/10 seeds, and no worse than -0.002 at shift none.
    A step that fails its rule is reported as an ablation and does not enter the final pipeline.
Writes results/diag_twostage[_smoke]/: proba/<model>_s<seed>.npz, rows/<model>_s<seed>.json,
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

from hsboa.baselines import FIT, available  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.decision import macro_f1, per_class_f1  # noqa: E402
from hsboa.folds import load_config, prepare_fold  # noqa: E402
from hsboa.stats import wilcoxon  # noqa: E402
from hsboa.twostage import (NORMAL, best_threshold, binary_rates, combine, decide, em_label_shift,  # noqa: E402
                            shift_sample)

SOURCES = {"lgbm": "results/e2_unsw_baselines/lgbm", "rf": "results/e2_unsw_baselines/rf"}
FULL = {"seeds": list(range(10)), "models": ["lgbm", "rf"], "repeats": 5}
SMOKE = {"seeds": [0], "models": ["rf"], "repeats": 1}
SMOKE_SETTINGS = {"rf": {"n_estimators": 20}, "lgbm": {"n_estimators": 20}}
SHIFTS = [None, 0.45, 0.55]
VARIANTS = ["argmax", "t", "em", "em_t"]
RULE_MODEL = "lgbm"


def probabilities(model: str, seed: int, smoke: bool, out: Path) -> dict:
    """Validation probabilities of the flat model and of the two-stage model (cached in proba/)."""
    path = out / "proba" / f"{model}_s{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    rec = json.loads((ROOT / SOURCES[model] / f"seed_{seed:02d}" / "result.json").read_text(encoding="utf-8"))
    s = dict(rec["settings"])
    if smoke:
        s.update(SMOKE_SETTINGS.get(model, {}))
    ds = get_dataset()
    X, y = ds.load("train")                 # the test partition is not loaded
    dcfg = copy.deepcopy(load_config(ROOT / "configs" / "data.yaml"))
    dcfg["features"]["n_features"] = 20     # as E2
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=dcfg, dataset=ds)
    Xtr, ytr = np.asarray(fold.X_train, dtype=float), np.asarray(fold.y_train)
    Xva, yva = np.asarray(fold.X_val, dtype=float), np.asarray(fold.y_val)
    k = len(fold.classes)
    if fold.classes[NORMAL] != "Normal":
        raise RuntimeError(f"expected Normal as class 0, got {fold.classes}")
    t0 = time.perf_counter()
    flat, _ = FIT[model](s, Xtr, ytr, Xva, yva, seed)
    P_flat = flat(Xva)
    ybin = (ytr != NORMAL).astype(int)
    stage1, _ = FIT[model]({**s, "n_classes": 2}, Xtr, ybin, Xva, (yva != NORMAL).astype(int), seed)
    att = ytr != NORMAL
    stage2, _ = FIT[model]({**s, "n_classes": k - 1}, Xtr[att], ytr[att] - 1, Xva, yva, seed)
    P_two = combine(stage1(Xva)[:, 1], stage2(Xva))
    res = {"P_flat": P_flat, "P_two": P_two, "y": yva, "k": np.array(k), "classes": np.array(fold.classes),
           "fit_s": np.array(time.perf_counter() - t0), "stored_val_macro_f1": np.array(rec["val"]["macro_f1"]),
           "refit_val_macro_f1": np.array(macro_f1(yva, P_flat.argmax(axis=1), k))}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return res


def run_task(model: str, seed: int, grid: dict, smoke: bool, out: Path) -> int:
    row_path = out / "rows" / f"{model}_s{seed}.json"
    if row_path.exists():
        return 0
    pr = probabilities(model, seed, smoke, out)
    y, k, classes = pr["y"], int(pr["k"]), [str(c) for c in pr["classes"]]
    fuzz = classes.index("Fuzzers")
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for r in range(grid["repeats"]):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1000 * seed + r)
        for h, (a, b) in enumerate(skf.split(np.zeros(len(y)), y)):
            for pipe in ("flat", "two"):
                P = pr[f"P_{pipe}"]
                t = best_threshold(P[a], y[a], k)
                src = P[a].mean(axis=0)
                for sh in SHIFTS:
                    idx = b if sh is None else b[shift_sample(y[b], sh, [seed, r, h, int(sh * 100)])]
                    Pe, ye = P[idx], y[idx]
                    W, pi = em_label_shift(Pe, src)
                    preds = {"argmax": decide(Pe), "t": decide(Pe, t), "em": decide(W), "em_t": decide(W, t)}
                    for v, pred in preds.items():
                        pc = per_class_f1(ye, pred, k)
                        rows.append({"model": model, "seed": seed, "repeat": r, "half": h, "pipeline": pipe,
                                     "variant": v, "shift": "none" if sh is None else f"{sh:.2f}", "t": t,
                                     "macro_f1": macro_f1(ye, pred, k), **binary_rates(ye, pred),
                                     "f1_Normal": float(pc[NORMAL]), "f1_Fuzzers": float(pc[fuzz]),
                                     "normal_share_true": float((ye == NORMAL).mean()),
                                     "normal_share_em": float(pi[NORMAL])})
    rec = {"model": model, "seed": seed, "fit_s": float(pr["fit_s"]),
           "stored_val_macro_f1": float(pr["stored_val_macro_f1"]),
           "refit_val_macro_f1": float(pr["refit_val_macro_f1"]), "rows": rows}
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    g = pd.DataFrame(rows)
    m = g.groupby(["pipeline", "variant", "shift"])["macro_f1"].mean()
    print(f"{model} s{seed}: flat {m[('flat', 'argmax', 'none')]:.4f}, two_t {m[('two', 't', 'none')]:.4f}; "
          f"shift 0.45: flat {m[('flat', 'argmax', '0.45')]:.4f}, flat_em {m[('flat', 'em', '0.45')]:.4f}", flush=True)
    return 1


def summarize(out: Path) -> pd.DataFrame:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "rows").glob("*.json"))]
    if not recs:
        print("no finished tasks yet")
        return pd.DataFrame()
    runs = pd.DataFrame([r for rec in recs for r in rec["rows"]])
    runs["arm"] = runs["pipeline"] + "_" + runs["variant"]
    runs.to_csv(out / "runs.csv", index=False)
    cols = ["macro_f1", "tpr", "fpr", "f1_Normal", "f1_Fuzzers", "normal_share_true", "normal_share_em", "t"]
    ps = runs.groupby(["model", "shift", "arm", "seed"])[cols].mean().reset_index()

    def paired(model, shift, a, b):
        x = ps[(ps.model == model) & (ps["shift"] == shift) & (ps.arm == a)].set_index("seed")["macro_f1"]
        z = ps[(ps.model == model) & (ps["shift"] == shift) & (ps.arm == b)].set_index("seed")["macro_f1"]
        common = sorted(set(x.index) & set(z.index))
        d = (x.loc[common] - z.loc[common]).to_numpy()
        p = wilcoxon(x.loc[common].to_numpy(), z.loc[common].to_numpy()) if len(common) > 1 else np.nan
        return (d.mean() if len(d) else np.nan), int((d > 0).sum()), len(d), p

    summ = ps.groupby(["model", "shift", "arm"])[cols].mean().reset_index()
    stats = [paired(r.model, r.shift, r.arm, "flat_argmax") for r in summ.itertuples()]
    summ["diff_vs_flat_argmax"] = [s[0] for s in stats]
    summ["seeds_positive"] = [s[1] for s in stats]
    summ["seeds"] = [s[2] for s in stats]
    summ["p"] = [s[3] if a != "flat_argmax" else np.nan for s, a in zip(stats, summ.arm)]
    summ = summ.sort_values(["model", "shift", "macro_f1"], ascending=[True, True, False])
    summ.to_csv(out / "summary.csv", index=False)

    lines = ["# Two-stage diagnostic (validation only)", "",
             "Flat model = stored E2 model; two = stage 1 Normal vs attack + stage 2 attack type, same settings. "
             "Threshold t and the EM source prior from one stratified validation half, scores on the other "
             "(5 repeats x 2 halves, averaged per seed). Shift 0.45/0.55: held-out half resampled to that Normal "
             "share (the official test partition has 45%). The test partition was not used.", ""]
    if RULE_MODEL in set(runs.model):
        m = RULE_MODEL
        n = int(ps[ps.model == m].seed.nunique())
        need = min(8, n)
        r1 = paired(m, "none", "two_t", "flat_t")
        r2 = paired(m, "none", "flat_t", "flat_argmax")
        ok1 = r1[0] >= 0.005 and r1[1] >= need
        ok2 = r2[0] >= 0.005 and r2[1] >= need
        pipe = "two" if ok1 else "flat"
        base_v = "t" if (ok1 or ok2) else "argmax"
        em_v = "em_t" if base_v == "t" else "em"
        r3 = paired(m, "0.45", f"{pipe}_{em_v}", f"{pipe}_{base_v}")
        r3n = paired(m, "none", f"{pipe}_{em_v}", f"{pipe}_{base_v}")
        ok3 = r3[0] >= 0.005 and r3[1] >= need and r3n[0] >= -0.002
        verdict = pd.DataFrame([
            {"rule": "R1 two-stage model (two_t - flat_t, shift none)", "diff": r1[0], "seeds_positive": r1[1],
             "p": r1[3], "met": "yes" if ok1 else "no"},
            {"rule": "R2 threshold (flat_t - flat_argmax, shift none)", "diff": r2[0], "seeds_positive": r2[1],
             "p": r2[3], "met": "yes" if ok2 else "no"},
            {"rule": f"R3 label shift ({pipe}_{em_v} - {pipe}_{base_v}, shift 0.45)", "diff": r3[0],
             "seeds_positive": r3[1], "p": r3[3], "met": "yes" if ok3 else "no"},
            {"rule": f"R3 side condition (same, shift none; must be >= -0.002)", "diff": r3n[0],
             "seeds_positive": r3n[1], "p": r3n[3], "met": "yes" if r3n[0] >= -0.002 else "no"}])
        lines += [f"## Decision rule ({m}, {n} seeds; >= +0.005 on >= {need} seeds)", "", _md(verdict), "",
                  f"Pipeline by R1/R2: **{pipe}_{base_v}**; label-shift correction "
                  f"{'enters' if ok3 else 'does not enter'} the adaptive setting.", ""]
    show = ["model", "shift", "arm", "macro_f1", "diff_vs_flat_argmax", "seeds_positive", "seeds", "p", "tpr", "fpr",
            "f1_Normal", "f1_Fuzzers", "normal_share_true", "normal_share_em", "t"]
    lines += ["## Macro-F1 and Normal-vs-attack rates by model, shift and arm", "", _md(summ[show]), ""]
    det = pd.DataFrame([{"model": r["model"], "seed": r["seed"],
                         "abs_diff": abs(r["stored_val_macro_f1"] - r["refit_val_macro_f1"])} for r in recs])
    det = det.groupby("model").agg(seeds=("seed", "nunique"), max_abs_diff=("abs_diff", "max")).reset_index()
    lines += ["## Determinism check: flat refit vs stored E2 validation macro-F1", "", _md(det), ""]
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
    out = ROOT / "results" / ("diag_twostage_smoke" if args.smoke else "diag_twostage")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        missing = [m for m in grid["models"] if not available(m)]
        if missing and not args.smoke:
            raise SystemExit(f"models {missing} need libraries that are not installed (see SETUP_NEW_MACHINE.md)")
        grid["models"] = [m for m in grid["models"] if m not in missing]
        absent = [m for m in grid["models"] if not (ROOT / SOURCES[m]).exists()]
        if absent:
            raise SystemExit(f"stored E2 results missing for {absent}")
        tasks = [(m, s) for m in grid["models"] for s in grid["seeds"]]
        print(f"{len(tasks)} tasks, {args.workers} workers; results in {out}", flush=True)
        t0 = time.perf_counter()
        if args.workers > 1:
            from joblib import Parallel, delayed
            Parallel(n_jobs=args.workers, verbose=5)(delayed(run_task)(m, s, grid, args.smoke, out) for m, s in tasks)
        else:
            for m, s in tasks:
                run_task(m, s, grid, args.smoke, out)
        print(f"all tasks finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    summ = summarize(out)
    if not summ.empty:
        print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
