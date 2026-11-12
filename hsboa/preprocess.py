"""Training-only preprocessing, feature selection and feature ordering.

Everything is one sklearn ``Pipeline`` that is fitted on the training
partition only, then applied unchanged to validation and test:

    encode categoricals -> impute -> min-max scale -> select d -> order

* Selection: SVM-RFE with a linear SVM, or
  mutual-information ranking.
* Ordering: the QRNN reads the selected features as a sequence, so
  the order is an explicit, recorded choice. The default ranks features by
  mutual information on the training data. "original", "reverse" and
  seeded "random" orders support the permutation study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import LinearSVC

from hsboa.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

SELECTION_METHODS = ("svm_rfe", "mutual_info", "all")
ORDER_METHODS = ("mutual_info", "original", "reverse", "random")


def _original_position(name: str, feature_columns=FEATURE_COLUMNS,
                       categorical_columns=CATEGORICAL_COLUMNS) -> int:
    """Position of a (possibly one-hot expanded) column in the dataset's raw feature order."""
    feature_columns = tuple(feature_columns)
    if name in feature_columns:
        return feature_columns.index(name)
    for cat in categorical_columns:
        if name.startswith(cat + "_"):
            return feature_columns.index(cat)
    raise KeyError(name)


def _stratified_subsample(y: np.ndarray, max_samples: int | None,
                          rng: np.random.Generator,
                          min_per_class: int = 100) -> np.ndarray:
    """Row indices of a class-stratified subsample used only to speed up selection.

    Classes are sampled proportionally, but each class keeps at least
    ``min(n_c, min_per_class)`` rows, so rare classes (Worms) still influence
    the selection.
    """
    if max_samples is None or len(y) <= max_samples:
        return np.arange(len(y))
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        k = max(round(max_samples * len(members) / len(y)),
                min(len(members), min_per_class))
        idx.append(rng.choice(members, size=min(k, len(members)), replace=False))
    return np.sort(np.concatenate(idx))


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Select ``n_features`` columns and fix their sequence order (fit on train only).

    Fitted attributes:
        selected_indices_ : column positions in the input, in sequence order
        selected_features_ : matching column names
        mutual_info_ : MI score of every input column (training subsample)
    """

    def __init__(self, n_features: int = 10, method: str = "svm_rfe",
                 order: str = "mutual_info", max_samples: int | None = 20000,
                 random_state: int = 0, feature_columns=FEATURE_COLUMNS,
                 categorical_columns=CATEGORICAL_COLUMNS):
        self.n_features = n_features
        self.method = method
        self.order = order
        self.max_samples = max_samples
        self.random_state = random_state
        self.feature_columns = feature_columns          # raw order, for order="original"/"reverse"
        self.categorical_columns = categorical_columns

    def fit(self, X, y):
        if self.method not in SELECTION_METHODS:
            raise ValueError(f"method must be one of {SELECTION_METHODS}")
        if self.order not in ORDER_METHODS:
            raise ValueError(f"order must be one of {ORDER_METHODS}")
        X = pd.DataFrame(X)
        y = np.asarray(y).ravel()
        names = np.asarray(X.columns, dtype=object)
        n = len(names) if self.method == "all" else self.n_features
        if not 0 < n <= len(names):
            raise ValueError(f"n_features={n} not in 1..{len(names)}")

        rng = np.random.default_rng(self.random_state)
        rows = _stratified_subsample(y, self.max_samples, rng)
        Xs, ys = X.to_numpy(dtype=float)[rows], y[rows]

        mi = mutual_info_classif(Xs, ys, random_state=self.random_state)
        self.mutual_info_ = dict(zip(names, mi.round(6)))

        if self.method == "svm_rfe":
            svm = LinearSVC(dual="auto", max_iter=5000,
                            random_state=self.random_state)
            rfe = RFE(svm, n_features_to_select=n, step=1).fit(Xs, ys)
            selected = np.flatnonzero(rfe.support_)
        elif self.method == "mutual_info":
            selected = np.argsort(-mi, kind="stable")[:n]
        else:
            selected = np.arange(len(names))

        if self.order == "mutual_info":
            ordered = selected[np.argsort(-mi[selected], kind="stable")]
        else:
            ordered = np.array(sorted(selected, key=lambda i: _original_position(
                names[i], self.feature_columns, self.categorical_columns)))
            if self.order == "reverse":
                ordered = ordered[::-1]
            elif self.order == "random":
                ordered = rng.permutation(ordered)

        self.feature_names_in_ = names
        self.n_features_in_ = len(names)
        self.selected_indices_ = ordered
        self.selected_features_ = [str(names[i]) for i in ordered]
        return self

    def transform(self, X):
        return pd.DataFrame(X).iloc[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


def build_pipeline(n_features: int = 10, selection: str = "svm_rfe",
                   order: str = "mutual_info",
                   categorical_encoding: str = "ordinal",
                   impute_strategy: str = "median",
                   max_samples: int | None = 20000,
                   random_state: int = 0,
                   feature_columns=FEATURE_COLUMNS,
                   categorical_columns=CATEGORICAL_COLUMNS) -> Pipeline:
    """Build the unfitted preprocessing + selection pipeline.

    ``categorical_encoding="ordinal"`` keeps one column per categorical
    feature (42 inputs in total for UNSW-NB15), as integer
    codes. Categories unseen in training map to -1. ``"onehot"`` expands
    them instead. ``feature_columns`` / ``categorical_columns`` default to
    UNSW-NB15 (``hsboa.data``).
    """
    categorical_columns = tuple(categorical_columns)
    numeric_columns = tuple(c for c in feature_columns if c not in categorical_columns)
    if tuple(feature_columns) == FEATURE_COLUMNS:
        numeric_columns = NUMERIC_COLUMNS
    if categorical_encoding == "ordinal":
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1)
    elif categorical_encoding == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore",
                                    sparse_output=False)
    else:
        raise ValueError("categorical_encoding must be 'ordinal' or 'onehot'")

    encode = ColumnTransformer(
        [("cat", cat_encoder, list(categorical_columns)),
         ("num", SimpleImputer(strategy=impute_strategy), list(numeric_columns))],
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([
        ("encode", encode),
        ("scale", MinMaxScaler()),
        ("select", FeatureSelector(n_features=n_features, method=selection,
                                   order=order, max_samples=max_samples,
                                   random_state=random_state,
                                   feature_columns=tuple(feature_columns),
                                   categorical_columns=categorical_columns)),
    ])
    return pipe.set_output(transform="pandas")
