"""Per-class version of the split control: does the effect of the random split persist in every class?

Same four protocols and the same splits as scripts/split_control.py (official, R random, G group-disjoint,
P group-disjoint with the official class counts), default LightGBM only, seeds 0-9. For every seed and
protocol the 10 x 10 confusion counts of the test side are stored, from which the per-class F1 and recall,
the test support and the per-class differences from the official split follow.

Note on intervals: on R, G and P every seed draws a different test set, so the seed interval of those
protocols already includes test-sample variation; on the official split the test set is fixed (see
scripts/per_class_eval.py for grouped intervals there).

Usage (from the repository root):
    python scripts/split_control_by_class.py run --workers 10      (full; needs LightGBM)
    python scripts/split_control_by_class.py run --smoke           (20,000-row smoke test)
    python scripts/split_control_by_class.py summarize
Writes results/split_control_by_class/{rows/seed_XX.json, confusions.csv, per_class.csv, differences.csv, report.md}.
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import final_eval as fe  # noqa: E402
import split_control as sc  # noqa: E402
from hsboa.baselines import FIT  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.folds import prepare_fold  # noqa: E402
from hsboa.stats import bootstrap_ci  # noqa: E402

SPLITS = ("official", "R", "G", "P")


def out_dir(smoke: bool) -> Path:
    return ROOT / "results" / ("split_control_by_class_smoke" if smoke else "split_control_by_class")


def seed_task(seed: int, model: str, smoke: bool) -> dict:
    out = out_dir(smoke)
    path = out / "rows" / f"seed_{seed:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    ds = get_dataset()
    X, y = ds.load("train")
    X_te, y_te = ds.load("test")
    y, y_te = np.asarray(y, int), np.asarray(y_te, int)
    if smoke:
        rng = np.random.default_rng(0)
        i = np.sort(rng.choice(len(X), fe.SMOKE["rows"], replace=False))
        j = np.sort(rng.choice(len(X_te), fe.SMOKE["rows"] // 2, replace=False))
        X, y, X_te, y_te = X.iloc[i], y[i], X_te.iloc[j], y_te[j]
    k = len(ds.classes)
    dcfg, fit = fe._cfg(), FIT[model]
    settings = fe._settings(model, seed, smoke)["default"]
    Xm, ym = pd.concat([X, X_te], ignore_index=True), np.r_[y, y_te]
    codes = sc.group_codes(Xm)
    off_tr, off_te = np.bincount(y, minlength=k), np.bincount(y_te, minlength=k)
    splits = sc.make_splits(Xm, ym, codes, len(y), off_tr, off_te, k, seed)
    splits["official"] = (np.arange(len(y)), len(y) + np.arange(len(y_te)))
    rows = []
    for name in SPLITS:
        tr, te = splits[name]
        fold = prepare_fold(Xm.iloc[tr], ym[tr], seed=seed, train_fraction=1.0, cfg=dcfg, X_test=Xm.iloc[te],
                            y_test=ym[te], dataset=ds)
        pr, _ = fit({**settings, "n_classes": k}, fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed)
        pred = pr(fold.X_test).argmax(1)
        cm = np.zeros((k, k), int)
        np.add.at(cm, (np.asarray(fold.y_test, int), pred), 1)
        rows.append({"seed": seed, "split": name, "n_test": int(len(pred)), "confusion": cm.tolist()})
    rec = {"seed": seed, "model": model, "classes": list(ds.classes), "rows": rows, "wall_s": time.perf_counter() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec), encoding="utf-8")
    print(f"seed {seed}: done ({rec['wall_s'] / 60:.1f} min)", flush=True)
    return rec


def cmd_run(args) -> None:
    model = "lgbm" if importlib.util.find_spec("lightgbm") else "hgb"
    if model == "hgb" and not args.smoke:
        raise SystemExit("LightGBM is not installed (the full run needs it)")
    seeds = fe.SMOKE["seeds"] if args.smoke else (args.seeds or fe.FULL["seeds"])
    if args.workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(seed_task, seeds, [model] * len(seeds), [args.smoke] * len(seeds)))
    else:
        for s in seeds:
            seed_task(s, model, args.smoke)
    cmd_summarize(args)


def _md(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def cmd_summarize(args) -> None:
    out = out_dir(args.smoke)
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "rows").glob("seed_*.json"))]
    classes = recs[0]["classes"]
    k = len(classes)
    f1 = {s: [] for s in SPLITS}
    rc = {s: [] for s in SPLITS}
    sup = {s: [] for s in SPLITS}
    conf = []
    for r in recs:
        for row in r["rows"]:
            cm = np.array(row["confusion"], float)
            tp = np.diag(cm)
            f1[row["split"]].append(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-12))
            rc[row["split"]].append(tp / np.maximum(cm.sum(1), 1e-12))
            sup[row["split"]].append(cm.sum(1))
            for i in range(k):
                for j in range(k):
                    if cm[i, j]:
                        conf.append({"seed": r["seed"], "split": row["split"], "true": classes[i], "pred": classes[j],
                                     "n": int(cm[i, j])})
    pd.DataFrame(conf).to_csv(out / "confusions.csv", index=False)
    f1, rc, sup = ({s: np.array(v) for s, v in d.items()} for d in (f1, rc, sup))
    rows, drows = [], []
    for j, c in enumerate(classes + ["macro-F1"]):
        for s in SPLITS:
            x = f1[s][:, j] if j < k else f1[s].mean(1)
            lo, hi = bootstrap_ci(x)
            rows.append({"class": c, "split": s, "n_seeds": len(x),
                         "test_support_mean": sup[s][:, j].mean() if j < k else sup[s].sum(1).mean(),
                         "f1": x.mean(), "f1_lo": lo, "f1_hi": hi,
                         "recall": (rc[s][:, j] if j < k else rc[s].mean(1)).mean()})
        for s in SPLITS[1:]:
            a = f1[s][:, j] if j < k else f1[s].mean(1)
            b = f1["official"][:, j] if j < k else f1["official"].mean(1)
            lo, hi = bootstrap_ci(a - b)
            drows.append({"class": c, "contrast": f"{s} - official", "diff": (a - b).mean(), "lo": lo, "hi": hi,
                          "seeds_positive": int((a - b > 0).sum())})
    pc = pd.DataFrame(rows)
    pc.to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(drows).to_csv(out / "differences.csv", index=False)
    show = pc.pivot(index="class", columns="split", values="f1").reindex(classes + ["macro-F1"])[list(SPLITS)].round(3)
    diff = pd.DataFrame(drows)
    diff["text"] = [f"{r.diff:+.3f} ({r.lo:+.3f}, {r.hi:+.3f})" for r in diff.itertuples()]
    dshow = diff.pivot(index="class", columns="contrast", values="text").reindex(classes + ["macro-F1"])
    lines = ["# Split control by class (default LightGBM)", "", f"{len(recs)} seeds. F1 per class by protocol:", "",
             _md(show.reset_index()), "", "Differences from the official split (mean, 95% seed interval):", "",
             _md(dshow.reset_index())]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seeds", nargs="*", type=int)
    s = sub.add_parser("summarize")
    s.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (cmd_run if args.cmd == "run" else cmd_summarize)(args)


if __name__ == "__main__":
    main()
