"""Train/test shift measurement (shift diagnostic).

Unsupervised domain adaptation: the features of the official test partition may
be used without its labels, as in unsupervised domain adaptation; the test
labels are used once, for the final score. Nothing here takes a label of the
"other" (test) side.

* match_class_mix: training rows resampled to the test partition's published class
  counts, so the remaining separability is not just the different class mix.
* encode: ordinal codes for the categorical columns from the reference side's
  categories (a category unseen there gets -1, itself a shift signal).
* domain_auc: out-of-fold ROC AUC of a classifier that separates reference rows
  (domain 0) from other rows (domain 1). 0.5 = indistinguishable.
* univariate: per feature, domain AUC of the raw value (folded to >= 0.5),
  Kolmogorov-Smirnov statistic and population stability index (PSI).
* greedy_removal: drop the feature the domain classifier relies on most
  (permutation importance on held-out rows), refit, repeat; the AUC curve shows
  whether the shift sits in a few features or is spread over many.
* density_ratio_weights, ess, weighted_macro_f1: importance weighting of
  validation rows towards the test distribution (importance-weighted
  validation, Sugiyama, Krauledat & Mueller 2007) and its effective sample size.
* duplicate_stats: exact duplicate and conflicting-label rates on the labelled
  side, and the share of other-side rows whose feature vector occurs there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Published per-class record counts of the official test partition (Moustafa & Slay 2015/2016; the
# same table appears in most UNSW-NB15 papers). Aggregate dataset metadata, not per-record labels:
# used only to give the training rows the test partition's class mix, so that a domain classifier
# cannot separate the partitions by class mix alone. Order = hsboa.data.CLASSES.
PUBLISHED_TEST_COUNTS = (37000, 18871, 11132, 6062, 4089, 3496, 677, 583, 378, 44)


def match_class_mix(y: np.ndarray, target_counts, rng) -> np.ndarray:
    """Row indices of y with class proportions equal to target_counts, as many rows as possible, no repeats."""
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    have = np.bincount(y, minlength=len(target_counts)).astype(float)
    want = np.asarray(target_counts, float)
    f = np.min(have[want > 0] / want[want > 0])
    keep = [rng.choice(np.flatnonzero(y == c), size=int(np.floor(f * want[c])), replace=False)
            for c in range(len(want)) if want[c] > 0]
    return np.sort(np.concatenate(keep))


def encode(ref: pd.DataFrame, other: pd.DataFrame, categorical) -> tuple[np.ndarray, np.ndarray]:
    """Float arrays of both frames; categorical columns as codes of the reference side's categories."""
    ref, other = ref.copy(), other.copy()
    for c in categorical:
        cats = pd.Index(sorted(ref[c].astype(str).unique()))
        ref[c] = cats.get_indexer(ref[c].astype(str))
        other[c] = cats.get_indexer(other[c].astype(str))
    return ref.to_numpy(dtype=float), other.to_numpy(dtype=float)


def _model(seed: int, max_iter: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                          early_stopping=False, random_state=seed)


def _domain_xy(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([A, B]), np.r_[np.zeros(len(A), int), np.ones(len(B), int)]


def domain_auc(A: np.ndarray, B: np.ndarray, *, folds: int = 5, seed: int = 0, max_iter: int = 200,
               columns=None) -> float:
    """Out-of-fold AUC of separating rows of A (domain 0) from rows of B (domain 1)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X, d = _domain_xy(A, B)
    if columns is not None:
        X = X[:, list(columns)]
    oof = np.zeros(len(d))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d):
        oof[te] = _model(seed, max_iter).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(d, oof))


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (largest gap between the empirical CDFs)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                        - np.searchsorted(b, grid, side="right") / len(b)).max())


def psi(ref: np.ndarray, other: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with bins at the reference quantiles (ties merged)."""
    ref, other = np.asarray(ref, float), np.asarray(other, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    q = np.bincount(np.searchsorted(edges, other, side="right"), minlength=len(edges) + 1) / len(other)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def univariate(A: np.ndarray, B: np.ndarray, names) -> pd.DataFrame:
    """Per-feature domain AUC (folded), KS and PSI of B against reference A, sorted by AUC."""
    from sklearn.metrics import roc_auc_score
    d = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
    rows = []
    for j, name in enumerate(names):
        v = np.r_[A[:, j], B[:, j]]
        auc = 0.5 if np.all(v == v[0]) else roc_auc_score(d, v)
        rows.append({"feature": name, "auc": max(auc, 1 - auc), "ks": ks_stat(A[:, j], B[:, j]),
                     "psi": psi(A[:, j], B[:, j]),
                     "mean_ref": float(A[:, j].mean()), "mean_other": float(B[:, j].mean()),
                     "median_ref": float(np.median(A[:, j])), "median_other": float(np.median(B[:, j]))})
    return pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)


def greedy_removal(A: np.ndarray, B: np.ndarray, names, *, steps: int = 15, folds: int = 3, seed: int = 0,
                   max_iter: int = 200, perm_rows: int = 20000, perm_repeats: int = 3) -> pd.DataFrame:
    """AUC after removing 0, 1, ..., steps features, each time the one with the largest permutation importance.

    Importance comes from a model fitted on a stratified half of the domain data and
    permuted on the other half (at most perm_rows rows).
    """
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    X, d = _domain_xy(A, B)
    keep = list(range(len(names)))
    rows = []
    for k in range(steps + 1):
        auc = domain_auc(A, B, folds=folds, seed=seed, max_iter=max_iter, columns=keep)
        rec = {"removed": k, "auc": auc, "dropped": ""}
        if k < steps and len(keep) > 1:
            tr, te = train_test_split(np.arange(len(d)), test_size=0.5, stratify=d, random_state=seed)
            if len(te) > perm_rows:
                te = train_test_split(te, train_size=perm_rows, stratify=d[te], random_state=seed)[0]
            m = _model(seed, max_iter).fit(X[np.ix_(tr, keep)], d[tr])
            imp = permutation_importance(m, X[np.ix_(te, keep)], d[te], scoring="roc_auc",
                                         n_repeats=perm_repeats, random_state=seed).importances_mean
            j = keep[int(np.argmax(imp))]
            rec["dropped"] = names[j]
            rec["importance"] = float(imp.max())
            keep.remove(j)
        rows.append(rec)
    return pd.DataFrame(rows)


def density_ratio_weights(p_other: np.ndarray, n_ref: int, n_other: int, clip_quantile: float | None = 0.99
                          ) -> np.ndarray:
    """Importance weights p_other(x) / p_ref(x) from a domain classifier's P(other | x), mean 1.

    Bayes: p_other(x) / p_ref(x) = [P(other|x) / P(ref|x)] * (n_ref / n_other). Weights above the
    clip_quantile are capped (then renormalised) to limit the variance of weighted estimates.
    """
    p = np.clip(np.asarray(p_other, float), 1e-6, 1 - 1e-6)
    w = p / (1 - p) * (n_ref / n_other)
    if clip_quantile is not None:
        w = np.minimum(w, np.quantile(w, clip_quantile))
    return w / w.mean()


def ess(w: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w > 0) else 0.0


def weighted_macro_f1(y: np.ndarray, pred: np.ndarray, k: int, w=None) -> float:
    """Macro-F1 over the k classes with each row counted with weight w (unweighted if w is None)."""
    y, pred = np.asarray(y), np.asarray(pred)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    tp = np.bincount(y[y == pred], weights=w[y == pred], minlength=k)
    support = np.bincount(y, weights=w, minlength=k)
    predicted = np.bincount(pred, weights=w, minlength=k)
    denom = support + predicted
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)))


def duplicate_stats(X: np.ndarray, y: np.ndarray, X_other: np.ndarray) -> dict:
    """Duplicate structure of the labelled side (X, y) and the overlap of X_other with it (no labels of X_other)."""
    df = pd.DataFrame(X)
    key = pd.util.hash_pandas_object(df, index=False).to_numpy()
    s = pd.Series(key)
    dup = s.duplicated(keep=False).to_numpy()
    labels_per_key = pd.Series(np.asarray(y)).groupby(key).nunique()
    conflict = s.map(labels_per_key).to_numpy() > 1
    key_o = pd.util.hash_pandas_object(pd.DataFrame(X_other), index=False).to_numpy()
    seen = np.isin(key_o, key)
    conflict_keys = set(labels_per_key.index[labels_per_key.to_numpy() > 1])
    return {"rows": int(len(X)), "unique_rows": int(s.nunique()), "dup_row_share": float(dup.mean()),
            "conflict_row_share": float(conflict.mean()),
            "other_rows": int(len(X_other)), "other_seen_share": float(seen.mean()),
            "other_seen_conflicting_share": float(np.isin(key_o, list(conflict_keys)).mean())
            if conflict_keys else 0.0}
