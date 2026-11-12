"""Dataset registry. The study uses UNSW-NB15 only (``hsboa.data``).

A :class:`Dataset` describes the raw feature order, the categorical columns,
the class names (index = class id, "Normal" first so the binary view works)
and how to load the official training and test partitions. Everything else
(split, training-only pipeline, models, metrics) is shared. The registry is
kept so the experiment configs name their dataset explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from hsboa import data

DEFAULT_DATASET = "unsw_nb15"


@dataclass(frozen=True)
class Dataset:
    name: str
    classes: tuple[str, ...]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    load: Callable[[str], tuple[pd.DataFrame, np.ndarray]]    # part ("train" | "test") -> (X, y)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


DATASETS: dict[str, Dataset] = {
    "unsw_nb15": Dataset("unsw_nb15", data.CLASSES, data.FEATURE_COLUMNS, data.CATEGORICAL_COLUMNS,
                         data.load_part),
}


def load_parts(ds: Dataset):
    """(X_train, y_train, X_test, y_test); the test part is (None, None) if its file is missing."""
    X, y = ds.load("train")
    try:
        X_te, y_te = ds.load("test")
    except FileNotFoundError:
        X_te, y_te = None, None
    return X, y, X_te, y_te


def get_dataset(name: str = DEFAULT_DATASET) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}") from None
