"""Reproduction package: the data checksums, groups, indices and settings behind the tables.

Writes results/package/:
  data_checksums.csv          SHA-256, size, records and columns of the two UNSW-NB15 files as used
  record_groups.csv.gz        for every training and test record: file row and the key of its group of
                              identical 42-feature vectors (hash of the raw values; same key = same vector)
  indices/seed_XX.npz         per seed 0-9: positions in the training partition of the training part
                              (train_idx), the validation part (val_idx) and the 30,000-record search
                              sample (search_idx); positions in the merged data (training rows first, then
                              test rows) of the train / test sides of the official split and of the controlled splits R, G and P
  preprocessing_per_seed.json per seed: feature order and mutual-information scores, scaler minima and
                              maxima, median-imputation values, class counts of the parts
  test_metrics_all_arms.csv   every stored run (E1, E1b, E2, E6): validation and test metrics, per-class F1
                              and recall, binary counts (the full confusion matrices of ten arms are in
                              results/per_class_eval and results/split_control_by_class)
  README.md                   what each file is and how it was made

Usage (from the repository root):
    python scripts/export_package.py             write everything
    python scripts/export_package.py --verify    rebuild and compare with the stored files (exit 1 on a mismatch)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import split_control as sc  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402
from hsboa.fitness import stratified_subsample  # noqa: E402
from hsboa.folds import load_config, prepare_fold, sha256  # noqa: E402

OUT = ROOT / "results" / "package"
SEEDS = list(range(10))
RAW = ROOT / "data" / "raw" / "unsw_nb15"
EXPERIMENTS = ["e1_unsw_tuners", "e1b_unsw_gbdt", "e2_unsw_baselines", "e6_unsw_shift"]


def group_key(X: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy()


def data_checksums(ds) -> pd.DataFrame:
    rows = []
    for p in sorted(RAW.glob("*.csv")):
        df = pd.read_csv(p, nrows=None)
        rows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p), "records": len(df),
                     "columns": df.shape[1]})
    return pd.DataFrame(rows)


def record_groups(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> pd.DataFrame:
    keys = np.concatenate([group_key(Xtr), group_key(Xte)])
    ids = pd.factorize(keys)[0]
    return pd.DataFrame({"partition": ["train"] * len(Xtr) + ["test"] * len(Xte),
                         "row": np.r_[np.arange(len(Xtr)), np.arange(len(Xte))],
                         "group_key": [f"{k:016x}" for k in keys], "group_id": ids})


def seed_indices(seed: int, ds, X, y, Xte, yte, codes, Xm, ym, cfg) -> dict:
    fold = prepare_fold(X, y, seed=seed, train_fraction=1.0, cfg=cfg, X_test=Xte, y_test=yte, dataset=ds)
    pos = stratified_subsample(fold.y_train, 30000, np.random.SeedSequence([seed, 2]))
    k = len(ds.classes)
    off_tr, off_te = np.bincount(y, minlength=k), np.bincount(yte, minlength=k)
    splits = sc.make_splits(Xm, ym, codes, len(y), off_tr, off_te, k, seed)
    arrays = {"train_idx": fold.split.train_idx.astype(np.int32), "val_idx": fold.split.val_idx.astype(np.int32),
              "search_idx": fold.split.train_idx[pos].astype(np.int32)}
    splits["official"] = (np.arange(len(y)), len(y) + np.arange(len(yte)))
    for name, (tr, te) in splits.items():
        arrays[f"{name}_train"] = np.asarray(tr, np.int32)
        arrays[f"{name}_test"] = np.asarray(te, np.int32)
    sel = fold.pipeline.named_steps["select"]
    sc_ = fold.pipeline.named_steps["scale"]
    summ = fold.summary()
    prep = {"seed": seed, "order": list(summ["order"]) if not isinstance(summ["order"], str) else summ["order"],
            "mutual_info": summ["mutual_info"], "class_counts": summ["class_counts"],
            "scaler_feature_names": [str(c) for c in sc_.feature_names_in_],
            "scaler_data_min": [float(v) for v in sc_.data_min_], "scaler_data_max": [float(v) for v in sc_.data_max_]}
    enc = fold.pipeline.named_steps["encode"]
    imp = {}
    for name, trans, cols in enc.transformers_:
        if hasattr(trans, "named_steps") and "imputer" in trans.named_steps:
            stats = trans.named_steps["imputer"].statistics_
            imp.update({str(c): float(v) for c, v in zip(cols, stats)})
        elif hasattr(trans, "statistics_"):
            imp.update({str(c): float(v) for c, v in zip(cols, trans.statistics_)})
    prep["median_imputation"] = imp
    prep["selected_features_all_ranked"] = [str(c) for c in sel.selected_features_]
    return {"arrays": arrays, "prep": prep}


def flatten_runs() -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        for p in sorted((ROOT / "results" / exp).glob("*/seed_*/result.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            row = {"experiment": exp, "arm": p.parts[-3], "seed": rec.get("seed"),
                   "settings": json.dumps(rec.get("settings", {}), sort_keys=True, default=str),
                   "distinct_trials": rec.get("distinct_trials"), "wall_s": rec.get("wall_s")}
            for part in ("val", "test"):
                for k, v in (rec.get(part) or {}).items():
                    if isinstance(v, (int, float)):
                        row[f"{part}_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def build(write: bool) -> dict:
    ds = get_dataset()
    X, y = ds.load("train")
    Xte, yte = ds.load("test")
    y, yte = np.asarray(y, int), np.asarray(yte, int)
    cfg = load_config(ROOT / "configs" / "data.yaml")
    Xm, ym = pd.concat([X, Xte], ignore_index=True), np.r_[y, yte]
    codes = sc.group_codes(Xm)
    out = {"data_checksums": data_checksums(ds), "record_groups": record_groups(X, Xte), "seeds": {}, "prep": []}
    for s in SEEDS:
        r = seed_indices(s, ds, X, y, Xte, yte, codes, Xm, ym, cfg)
        out["seeds"][s] = r["arrays"]
        out["prep"].append(r["prep"])
        print(f"seed {s}: indices built", flush=True)
    out["metrics"] = flatten_runs()
    if write:
        (OUT / "indices").mkdir(parents=True, exist_ok=True)
        out["data_checksums"].to_csv(OUT / "data_checksums.csv", index=False)
        out["record_groups"].to_csv(OUT / "record_groups.csv.gz", index=False, compression="gzip")
        for s, arr in out["seeds"].items():
            np.savez_compressed(OUT / "indices" / f"seed_{s:02d}.npz", **arr)
        (OUT / "preprocessing_per_seed.json").write_text(json.dumps(out["prep"], indent=1), encoding="utf-8")
        out["metrics"].to_csv(OUT / "test_metrics_all_arms.csv", index=False)
    return out


def verify() -> int:
    new = build(write=False)
    bad = []
    old = pd.read_csv(OUT / "data_checksums.csv")
    if not (old.sort_values("file").reset_index(drop=True)[["file", "bytes", "sha256", "records", "columns"]]
            .equals(new["data_checksums"].sort_values("file").reset_index(drop=True)[["file", "bytes", "sha256", "records", "columns"]])):
        bad.append("data_checksums.csv")
    og = pd.read_csv(OUT / "record_groups.csv.gz")
    ng = new["record_groups"]
    if not (og.group_key.astype(str).tolist() == ng.group_key.tolist() and og.group_id.tolist() == ng.group_id.tolist()):
        bad.append("record_groups.csv.gz")
    for s, arr in new["seeds"].items():
        z = np.load(OUT / "indices" / f"seed_{s:02d}.npz")
        for k, v in arr.items():
            if not np.array_equal(z[k], v):
                bad.append(f"indices/seed_{s:02d}.npz:{k}")
    ck = pd.read_csv(ROOT / "results" / "manifest" / "split_checksums.csv")
    for r in ck.itertuples():
        arr = new["seeds"][r.seed]
        name = "val" if r.split == "validation" else r.split
        tr = arr["train_idx"] if r.split == "validation" else arr[f"{name}_train"]
        te = arr["val_idx"] if r.split == "validation" else arr[f"{name}_test"]
        if hashlib.sha256(np.ascontiguousarray(np.sort(tr).astype("<i8")).tobytes()).hexdigest() != r.sha256_train \
                or hashlib.sha256(np.ascontiguousarray(np.sort(te).astype("<i8")).tobytes()).hexdigest() != r.sha256_test:
            bad.append(f"split_checksums.csv seed {r.seed} {r.split}")
    print("VERIFY:", "all files reproduce" if not bad else f"{len(bad)} mismatches: {bad[:5]}")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        sys.exit(verify())
    OUT.mkdir(parents=True, exist_ok=True)
    out = build(write=True)
    print(out["data_checksums"].to_string(index=False))
    print(f"{len(out['record_groups']):,} record keys; {len(out['metrics']):,} stored runs; files in {OUT}")


if __name__ == "__main__":
    main()
