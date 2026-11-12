"""Config loading and per-(seed, train_fraction) data folds.

A fold bundles the split indices, the fitted training-only pipeline and
the transformed arrays. Experiments call :func:`prepare_fold` in memory.
``Fold.save`` writes the reproducibility artefacts: split indices,
fitted pipeline, selected features and order, and class counts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from hsboa.data import CLASSES, Split, make_split
from hsboa.preprocess import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Read a YAML config. Relative paths under ``paths`` resolve against the project root."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (PROJECT_ROOT / v if v else None)
                    for k, v in cfg.get("paths", {}).items()}
    return cfg


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Fold:
    split: Split
    pipeline: Pipeline
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    classes: tuple[str, ...] = CLASSES

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.named_steps["select"].selected_features_)

    def summary(self) -> dict:
        sel = self.pipeline.named_steps["select"]
        parts = {"train": self.y_train, "val": self.y_val, "test": self.y_test}
        return {
            "seed": self.split.seed,
            "train_fraction": self.split.train_fraction,
            "selection": sel.method,
            "order": sel.order,
            "selected_features": self.feature_names,
            "mutual_info": {k: float(v) for k, v in sel.mutual_info_.items()},
            "class_counts": {
                name: dict(zip(self.classes, np.bincount(y, minlength=len(self.classes)).tolist()))
                for name, y in parts.items() if y is not None
            },
        }

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.split.save(out / "split.npz")
        joblib.dump(self.pipeline, out / "pipeline.joblib")
        (out / "features.json").write_text(json.dumps(self.summary(), indent=2))
        return out


def prepare_fold(X_train_full: pd.DataFrame, y_train_full: np.ndarray,
                 seed: int, train_fraction: float, cfg: dict,
                 X_test: pd.DataFrame | None = None,
                 y_test: np.ndarray | None = None,
                 dataset=None) -> Fold:
    """Split the training file, fit the pipeline on the training part only, transform all parts.

    ``dataset`` (an hsboa.datasets.Dataset) supplies the raw feature order,
    categorical columns and class names; the default is UNSW-NB15.
    """
    cols = {} if dataset is None else {"feature_columns": dataset.feature_columns,
                                       "categorical_columns": dataset.categorical_columns}
    split = make_split(y_train_full, seed=seed,
                       val_size=cfg["split"]["val_size"],
                       train_fraction=train_fraction)
    pp, ft = cfg["preprocess"], cfg["features"]
    pipe = build_pipeline(
        n_features=ft["n_features"], selection=ft["selection"],
        order=ft["order"], max_samples=ft.get("max_samples"),
        categorical_encoding=pp["categorical_encoding"],
        impute_strategy=pp["impute_strategy"], random_state=seed, **cols,
    )
    Xtr = X_train_full.iloc[split.train_idx]
    ytr = y_train_full[split.train_idx]
    pipe.fit(Xtr, ytr)

    def tf(X):
        return pipe.transform(X).to_numpy(dtype=np.float32)

    return Fold(
        split=split, pipeline=pipe,
        X_train=tf(Xtr), y_train=ytr,
        X_val=tf(X_train_full.iloc[split.val_idx]),
        y_val=y_train_full[split.val_idx],
        X_test=None if X_test is None else tf(X_test),
        y_test=y_test,
        classes=CLASSES if dataset is None else tuple(dataset.classes),
    )
