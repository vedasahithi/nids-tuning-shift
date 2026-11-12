"""Decomposition of the random-split gain: size, duplicate exposure, class prevalence, partition shift.

Merged data (train + test partitions, 257,673 records). Same 10-class pipeline as final_eval.py
(top-20 mutual-information features, LightGBM defaults and the seed's E2-tuned settings; fitted on the
training side only; seeds 0-9). Protocols:

  official  the official partitions (reference)
  R         random split, sizes exactly the official ones (175,341 / 82,332), class-stratified.
            Identical feature vectors may fall on both sides (duplicate exposure), class mix equal on both sides.
  G         group-disjoint split: all records with an identical 42-feature vector go to the same side, the
            same train fraction in every class (by group majority class), so size and class mix match R but no
            twin crosses the boundary.
  P         group-disjoint split whose per-class training and test counts are the OFFICIAL ones (train 175,341
            and test 82,332 in the official class mix): removes duplicate exposure and the class-mix difference
            between official train and test, keeps nothing of the partition shift.

Reading the differences (macro-F1, paired by seed):
  R - official   total gain of random splitting at matched size
  R - G          contribution of duplicate exposure
  G - P          contribution of the official train/test class-mix difference (prevalence)
  P - official   what remains: partition shift (and, with opposite sign, the 10% duplicate exposure that the
                 official split has)
Writes results/split_control[_smoke]/{rows/seed_XX.json,protocol.csv,split_stats.csv,decomposition.csv,report.md}.
Usage (from the repository root):  python scripts/split_control.py --smoke
                     python scripts/split_control.py --workers 10        (full; LightGBM)
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import final_eval as fe  # noqa: E402
from hsboa.baselines import FIT  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.folds import prepare_fold  # noqa: E402
from hsboa.stats import bootstrap_ci  # noqa: E402


def group_codes(X: pd.DataFrame) -> np.ndarray:
    key = pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy()
    return pd.factorize(key)[0]


def group_split(codes: np.ndarray, y: np.ndarray, k: int, train_ratio: np.ndarray, rng) -> np.ndarray:
    """Boolean mask 'goes to train'. Groups stay whole; per majority class the first groups (random order)
    are sent to train until that class has about train_ratio[c] of its rows."""
    g = codes.max() + 1
    cnt = np.zeros((g, k), dtype=np.int64)
    np.add.at(cnt, (codes, y), 1)
    maj, size = cnt.argmax(1), cnt.sum(1)
    to_train = np.zeros(g, dtype=bool)
    for c in range(k):
        ids = np.flatnonzero(maj == c)
        ids = ids[rng.permutation(len(ids))]
        goal = train_ratio[c] * size[ids].sum()
        to_train[ids[np.cumsum(size[ids]) - size[ids] < goal]] = True      # groups starting before the goal
    return to_train[codes]


def subsample_to_counts(idx: np.ndarray, y: np.ndarray, target: np.ndarray, rng) -> np.ndarray:
    keep = []
    for c, n in enumerate(target):
        ic = idx[y[idx] == c]
        keep.append(rng.choice(ic, size=min(int(n), len(ic)), replace=False))
    return np.sort(np.concatenate(keep))


def make_splits(Xm, ym, codes, n_tr_off, off_tr_cnt, off_te_cnt, k, seed):
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    n, n_te_off = len(ym), int(off_te_cnt.sum())
    out = {}
    tr, te = train_test_split(np.arange(n), train_size=n_tr_off, test_size=n_te_off, stratify=ym, random_state=seed)
    out["R"] = (np.sort(tr), np.sort(te))
    total = np.bincount(ym, minlength=k).astype(float)
    mask = group_split(codes, ym, k, np.full(k, n_tr_off / (n_tr_off + n_te_off)), rng)
    out["G"] = (np.flatnonzero(mask), np.flatnonzero(~mask))
    ratio = np.minimum(1.0, off_tr_cnt / total * 1.02)                  # a little slack, then subsample
    mask = group_split(codes, ym, k, ratio, rng)
    tr_i, te_i = np.flatnonzero(mask), np.flatnonzero(~mask)
    out["P"] = (subsample_to_counts(tr_i, ym, off_tr_cnt, rng), subsample_to_counts(te_i, ym, off_te_cnt, rng))
    return out


def split_stats(name, tr, te, ym, codes, k, seed):
    tr_groups = np.zeros(codes.max() + 1, dtype=bool)
    tr_groups[codes[tr]] = True
    twin = tr_groups[codes[te]]
    cnt = np.zeros((codes.max() + 1, k))
    np.add.at(cnt, (codes[tr], ym[tr]), 1)
    conflict = twin & (cnt[codes[te]].argmax(1) != ym[te])
    prev_tr = np.bincount(ym[tr], minlength=k) / len(tr)
    prev_te = np.bincount(ym[te], minlength=k) / len(te)
    return {"seed": seed, "split": name, "n_train": len(tr), "n_test": len(te),
            "test_rows_with_train_twin": float(twin.mean()), "test_rows_twin_other_label": float(conflict.mean()),
            "class_mix_tv_distance": float(0.5 * np.abs(prev_tr - prev_te).sum())}


def seed_task(seed: int, model: str, smoke: bool, out: Path) -> dict:
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
    settings = fe._settings(model, seed, smoke)
    Xm, ym = pd.concat([X, X_te], ignore_index=True), np.r_[y, y_te]
    codes = group_codes(Xm)
    off_tr, off_te = np.bincount(y, minlength=k), np.bincount(y_te, minlength=k)
    splits = make_splits(Xm, ym, codes, len(y), off_tr, off_te, k, seed)
    splits["official"] = (np.arange(len(y)), len(y) + np.arange(len(y_te)))
    rows, stats = [], []
    for name in ("official", "R", "G", "P"):
        tr, te = splits[name]
        stats.append(split_stats(name, tr, te, ym, codes, k, seed))
        fold = prepare_fold(Xm.iloc[tr], ym[tr], seed=seed, train_fraction=1.0, cfg=dcfg, X_test=Xm.iloc[te],
                            y_test=ym[te], dataset=ds)
        for sname, s in settings.items():
            pr, _ = fit({**s, "n_classes": k}, fold.X_train, fold.y_train, fold.X_val, fold.y_val, seed)
            rows.append({"seed": seed, "split": name, "settings": sname,
                         **fe._multi(pr(fold.X_test), fold.y_test, fold.classes)})
    rec = {"seed": seed, "rows": rows, "stats": stats, "wall_s": time.perf_counter() - t0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"seed {seed}: done ({rec['wall_s'] / 60:.1f} min)", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    model = "lgbm" if importlib.util.find_spec("lightgbm") else "hgb"
    if model == "hgb" and not args.smoke:
        raise SystemExit("LightGBM is not installed (the full run needs it)")
    seeds = fe.SMOKE["seeds"] if args.smoke else fe.FULL["seeds"]
    out = ROOT / "results" / ("split_control_smoke" if args.smoke else "split_control")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.workers > 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=args.workers)(delayed(seed_task)(s, model, args.smoke, out) for s in seeds)
    else:
        recs = [seed_task(s, model, args.smoke, out) for s in seeds]
    prot = pd.DataFrame([r for rec in recs for r in rec["rows"]])
    stats = pd.DataFrame([r for rec in recs for r in rec["stats"]])
    prot.to_csv(out / "protocol.csv", index=False)
    stats.to_csv(out / "split_stats.csv", index=False)

    piv = prot.pivot_table(index=["settings", "seed"], columns="split", values="macro_f1")
    dec = pd.DataFrame({"total_gain_R_minus_official": piv["R"] - piv["official"],
                        "duplicate_exposure_R_minus_G": piv["R"] - piv["G"],
                        "class_mix_G_minus_P": piv["G"] - piv["P"],
                        "residual_shift_P_minus_official": piv["P"] - piv["official"]})
    rows = []
    for (sname, col), d in ((k_, dec.loc[k_[0], k_[1]]) for k_ in
                            [(s, c) for s in dec.index.get_level_values(0).unique() for c in dec.columns]):
        lo, hi = bootstrap_ci(d.to_numpy())
        rows.append({"settings": sname, "component": col, "mean": d.mean(), "ci95_low": lo, "ci95_high": hi})
    decs = pd.DataFrame(rows)
    decs.to_csv(out / "decomposition.csv", index=False)
    means = prot.groupby(["settings", "split"])["macro_f1"].agg(["mean", "std"]).round(4)
    ss = stats.groupby("split")[["n_train", "n_test", "test_rows_with_train_twin", "test_rows_twin_other_label",
                                 "class_mix_tv_distance"]].mean().round(4)
    (out / "report.md").write_text(f"# Split control ({model}, wall {(time.perf_counter() - t0) / 60:.1f} min)\n\n"
        + means.to_string() + "\n\nSplit statistics (mean over seeds):\n\n" + ss.to_string()
        + "\n\nDecomposition (paired by seed, 95% bootstrap CI):\n\n" + decs.round(4).to_string(index=False) + "\n",
        encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
