"""Checksums (and optionally the indices) of every data split used in the paper.

The splits are deterministic functions of the seed, so the repository stores checksums instead of index
files: anyone can regenerate the indices and compare. For each seed 0-9 this records
  validation  the stratified 20% validation split of the training partition (hsboa.data.make_split)
  R, G, P     the three controlled splits of the merged partitions used in the split decomposition
              (scripts/split_control.py)
as SHA-256 of the sorted index arrays, with sizes and, for the controlled splits, the share of test records
that have an identical training record.

Writes results/manifest/split_checksums.csv. With --write-indices it also writes results/manifest/indices/
as compressed .npz files (about 25 MB; not stored in the repository).
Usage (from the repository root):  python scripts/export_splits.py [--write-indices]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import split_control as sc  # noqa: E402
from hsboa.data import make_split  # noqa: E402
from hsboa.datasets import get_dataset  # noqa: E402


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.sort(a).astype("<i8")).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-indices", action="store_true")
    args = ap.parse_args()
    ds = get_dataset()
    X, y = ds.load("train")
    Xt, yt = ds.load("test")
    y, yt = np.asarray(y, int), np.asarray(yt, int)
    k = len(ds.classes)
    Xm, ym = pd.concat([X, Xt], ignore_index=True), np.r_[y, yt]
    codes = sc.group_codes(Xm)
    off_tr, off_te = np.bincount(y, minlength=k), np.bincount(yt, minlength=k)
    out = ROOT / "results" / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(10):
        sp = make_split(y, seed=seed)
        rows.append({"seed": seed, "split": "validation", "n_train": len(sp.train_idx), "n_test": len(sp.val_idx),
                     "sha256_train": digest(sp.train_idx), "sha256_test": digest(sp.val_idx),
                     "test_with_train_twin": ""})
        splits = sc.make_splits(Xm, ym, codes, len(y), off_tr, off_te, k, seed)
        splits["official"] = (np.arange(len(y)), len(y) + np.arange(len(yt)))
        for name, (tr, te) in splits.items():
            st = sc.split_stats(name, tr, te, ym, codes, k, seed)
            rows.append({"seed": seed, "split": name, "n_train": len(tr), "n_test": len(te),
                         "sha256_train": digest(tr), "sha256_test": digest(te),
                         "test_with_train_twin": round(st["test_rows_with_train_twin"], 6)})
            if args.write_indices:
                d = out / "indices"
                d.mkdir(exist_ok=True)
                np.savez_compressed(d / f"{name}_seed{seed:02d}.npz", train=np.sort(tr), test=np.sort(te))
        print("seed", seed, flush=True)
    pd.DataFrame(rows).to_csv(out / "split_checksums.csv", index=False)
    print(f"wrote {out / 'split_checksums.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
