"""UNSW-NB15 loading, ten-class labels and stratified splits.

UNSW-NB15 (Moustafa & Slay 2015) is the dataset of the study.

* The official partitions are used: training 175,341 records, testing 82,332.
  In the distributed files the names are swapped relative to the dataset
  documentation (``UNSW_NB15_training-set.csv`` holds the 82,332-record
  testing partition), so the files are identified by their record counts;
  :func:`load_part` checks the counts and raises if they do not match.
* Target: ``attack_cat`` (10 classes, "Normal" first so the binary view
  works). ``id`` and the binary ``label`` are not features. The partitions
  contain many duplicate rows (a known property of the dataset); they are
  kept as published and reported.
* Every class name maps explicitly; an unknown name raises.
* Splits are seeded and stratified, and the training-size sweep subsamples
  a fixed training partition, so the validation and test sets stay the
  same across the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 42 features, in the order of the official partition files.
FEATURE_COLUMNS: tuple[str, ...] = (
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb",
    "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = ("proto", "service", "state")
NUMERIC_COLUMNS: tuple[str, ...] = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS)
LABEL_COLUMN = "attack_cat"

# Class index = position in this tuple (Normal=0 ... Worms=9).
CLASSES: tuple[str, ...] = ("Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
                            "Analysis", "Backdoor", "Shellcode", "Worms")

# Documented record counts of the official partitions.
COUNTS = {"train": 175341, "test": 82332}
FILES = {"train": "data/raw/unsw_nb15/UNSW_NB15_testing-set.csv",     # 175,341 records (names swapped)
         "test": "data/raw/unsw_nb15/UNSW_NB15_training-set.csv"}     # 82,332 records


def to_class_index(labels: pd.Series | np.ndarray) -> np.ndarray:
    """Map ``attack_cat`` names to class indices; raise on unknown names."""
    labels = pd.Series(labels, dtype=str).str.strip()
    y = labels.map({c: i for i, c in enumerate(CLASSES)})
    unknown = sorted(labels[y.isna()].unique())
    if unknown:
        raise ValueError(f"unmapped UNSW-NB15 classes: {unknown}")
    return y.to_numpy(dtype=np.int64)


def load_xy(path: str | Path, expected_records: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features DataFrame with the 42 columns, class indices) from a partition CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if expected_records is not None and len(df) != expected_records:
        raise ValueError(f"{path}: {len(df)} records, expected {expected_records}")
    missing = set(FEATURE_COLUMNS + (LABEL_COLUMN,)) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    X = df[list(FEATURE_COLUMNS)].copy()
    for c in CATEGORICAL_COLUMNS:
        X[c] = X[c].astype(str).str.strip()
    return X, to_class_index(df[LABEL_COLUMN])


def load_part(part: str) -> tuple[pd.DataFrame, np.ndarray]:
    """The official ``"train"`` or ``"test"`` partition, checked by its record count."""
    return load_xy(PROJECT_ROOT / FILES[part], COUNTS[part])


def class_counts(y: np.ndarray) -> pd.Series:
    """Per-class record counts, indexed by class name (all ten classes)."""
    counts = np.bincount(y, minlength=len(CLASSES))
    return pd.Series(counts, index=list(CLASSES), name="records")


@dataclass(frozen=True)
class Split:
    """Indices into the training-partition rows for one (seed, train_fraction)."""

    seed: int
    train_fraction: float
    train_idx: np.ndarray
    val_idx: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, seed=self.seed, train_fraction=self.train_fraction,
            train_idx=self.train_idx, val_idx=self.val_idx,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        z = np.load(path)
        return cls(int(z["seed"]), float(z["train_fraction"]),
                   z["train_idx"], z["val_idx"])


def make_split(y: np.ndarray, seed: int, val_size: float = 0.2,
               train_fraction: float = 1.0) -> Split:
    """Stratified train/validation split of the training partition, then a training subsample.

    The validation set depends only on ``seed``. For a given seed, the
    training subsets are nested (the 60% subset is contained in the 70%
    subset, and so on), so the sweep changes only the amount of training
    data. The test partition is never touched here.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    all_idx = np.arange(len(y))
    train_full, val_idx = train_test_split(
        all_idx, test_size=val_size, stratify=y, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[train_full]):
        members = rng.permutation(train_full[y[train_full] == c])
        keep.append(members[: max(1, round(train_fraction * len(members)))])
    train_idx = np.sort(np.concatenate(keep))
    return Split(seed, float(train_fraction), train_idx, np.sort(val_idx))
