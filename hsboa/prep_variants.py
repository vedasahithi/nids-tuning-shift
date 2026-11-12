"""Preprocessing variants for the preprocessing diagnostic (scripts/diagnose_preprocess.py, 2026-10-29).

Each variant works on the arrays of an already prepared fold (selection
"all": every column, scaled and in descending mutual-information order) and
is fitted on training rows only:

* selection: correlation filter over the MI order, or mRMR (normalised MI
  relevance minus mean |Spearman| redundancy), both returning column indices;
* imbalance: random oversampling or SMOTE to the class sizes that class
  weighting with power 0.5 implies (``target_counts``), so they are directly
  comparable with ``class_weight_power`` 0.5;
* duplicate removal on the training rows; signed log1p for numeric columns.

Kept apart from hsboa.preprocess so that the code hashes of E1, E1b and E2
are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def signed_log1p(X: pd.DataFrame, columns) -> pd.DataFrame:
    """sign(x) * log(1 + |x|) on the given (numeric) columns; the others are unchanged."""
    X = X.copy()
    for c in columns:
        v = X[c].to_numpy(dtype=float)
        X[c] = np.sign(v) * np.log1p(np.abs(v))
    return X


def spearman(X: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix of the columns; constant columns get 0 off the diagonal."""
    R = np.column_stack([rankdata(col) for col in np.asarray(X, dtype=float).T])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(np.atleast_2d(C), nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def corr_filter(corr: np.ndarray, n: int, threshold: float = 0.95) -> list[int]:
    """Walk the columns in their given (MI) order; keep one only if |rho| <= threshold to every kept one."""
    keep: list[int] = []
    for j in range(corr.shape[0]):
        if all(abs(corr[j, i]) <= threshold for i in keep):
            keep.append(j)
        if len(keep) == n:
            break
    return keep


def mrmr(relevance, corr: np.ndarray, n: int) -> list[int]:
    """Greedy mRMR: start with the most relevant column, then maximise relevance - mean |rho| to the chosen.

    Relevance (mutual information) is divided by its maximum so both terms lie in [0, 1].
    """
    rel = np.asarray(relevance, dtype=float)
    rel = rel / rel.max() if rel.max() > 0 else rel
    chosen = [int(np.argmax(rel))]
    while len(chosen) < n:
        score = rel - np.abs(corr[:, chosen]).mean(axis=1)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


def target_counts(y: np.ndarray, k: int, power: float = 0.5) -> np.ndarray:
    """Class sizes n_max * (n_c / n_max)^power: the class mass that class weighting with this power gives."""
    n = np.bincount(np.asarray(y), minlength=k).astype(float)
    t = np.where(n > 0, np.round(n.max() * (n / n.max()) ** power), 0)
    return np.maximum(t, n).astype(int)


def random_oversample(X: np.ndarray, y: np.ndarray, targets, rng) -> tuple[np.ndarray, np.ndarray]:
    """Add rows drawn with replacement from each class until it reaches its target size."""
    rng = np.random.default_rng(rng)
    extra = [rng.choice(np.flatnonzero(y == c), size=int(t) - int((y == c).sum()), replace=True)
             for c, t in enumerate(targets) if t > (y == c).sum() > 0]
    idx = np.concatenate([np.arange(len(y))] + extra) if extra else np.arange(len(y))
    return X[idx], y[idx]


def smote(X: np.ndarray, y: np.ndarray, targets, rng, k_neighbors: int = 5,
          categorical: list[int] | tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE (Chawla et al. 2002) to the target sizes, within each class.

    A synthetic row is x + u (z - x) for a random same-class neighbour z among the k nearest; for
    categorical (ordinal-coded) columns it takes the value of x or z instead of interpolating.
    Classes with fewer than 2 rows are left as they are.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[list(categorical)] = True
    new_X, new_y = [X], [y]
    for c, t in enumerate(targets):
        members = np.flatnonzero(y == c)
        m = int(t) - len(members)
        if m <= 0 or len(members) < 2:
            continue
        kk = min(k_neighbors, len(members) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(X[members])
        _, nbr = nn.kneighbors(X[members])
        base = rng.integers(len(members), size=m)
        pick = nbr[base, 1 + rng.integers(kk, size=m)]
        xa, xb = X[members[base]], X[members[pick]]
        u = rng.random((m, 1))
        syn = xa + u * (xb - xa)
        syn[:, cat] = np.where(rng.random((m, int(cat.sum()))) < 0.5, xa[:, cat], xb[:, cat])
        new_X.append(syn)
        new_y.append(np.full(m, c, dtype=y.dtype))
    return np.vstack(new_X), np.concatenate(new_y)


def drop_duplicates(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove repeated (row, label) pairs, keeping the first; identical rows with different labels stay."""
    df = pd.DataFrame(np.asarray(X))
    df["_y"] = np.asarray(y)
    keep = ~df.duplicated().to_numpy()
    return np.asarray(X)[keep], np.asarray(y)[keep]
