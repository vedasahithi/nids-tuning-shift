"""Tests for the preprocessing-diagnostic variants (hsboa.prep_variants)."""

import numpy as np
import pandas as pd
import pytest

from hsboa.prep_variants import (corr_filter, drop_duplicates, mrmr, random_oversample, signed_log1p, smote,
                                 spearman, target_counts)


def _corr_toy(n=500, seed=0):
    """Column 1 is a monotone copy of column 0; column 2 is independent."""
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    return np.column_stack([a, a ** 3, rng.random(n)])


def test_spearman_detects_monotone_copy_and_constant_column():
    X = np.column_stack([_corr_toy(), np.ones(500)])
    C = spearman(X)
    assert C[0, 1] == pytest.approx(1.0) and abs(C[0, 2]) < 0.2
    assert C[3, 3] == 1.0 and C[0, 3] == 0.0


def test_corr_filter_skips_the_redundant_column():
    C = spearman(_corr_toy())
    assert corr_filter(C, 2) == [0, 2]
    assert corr_filter(C, 3, threshold=1.01) == [0, 1, 2]


def test_mrmr_prefers_the_non_redundant_column():
    C = spearman(_corr_toy())
    assert mrmr([1.0, 0.9, 0.6], C, 2) == [0, 2]
    assert len(set(mrmr([1.0, 0.9, 0.6], C, 3))) == 3


def test_target_counts_match_class_weight_power():
    y = np.repeat([0, 1, 2], [400, 100, 4])
    t = target_counts(y, 4, power=0.5)
    assert t.tolist() == [400, 200, 40, 0]
    assert target_counts(y, 3, power=0.0).tolist() == [400, 400, 400]
    assert target_counts(y, 3, power=1.0).tolist() == [400, 100, 4]


def test_random_oversample_reaches_targets_and_keeps_originals():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], [50, 5])
    X = rng.random((55, 3))
    Xo, yo = random_oversample(X, y, [50, 20], rng=1)
    assert np.bincount(yo).tolist() == [50, 20]
    assert np.array_equal(Xo[:55], X)
    assert all(any(np.array_equal(r, x) for x in X[y == 1]) for r in Xo[55:])


def test_smote_interpolates_within_class_and_copies_categoricals():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], [60, 6])
    X = rng.random((66, 3))
    X[:, 2] = rng.integers(0, 4, 66)                      # ordinal-coded categorical column
    Xs, ys = smote(X, y, [60, 30], rng=2, categorical=[2])
    assert np.bincount(ys).tolist() == [60, 30]
    syn = Xs[66:]
    lo, hi = X[y == 1, :2].min(axis=0), X[y == 1, :2].max(axis=0)
    assert np.all(syn[:, :2] >= lo - 1e-12) and np.all(syn[:, :2] <= hi + 1e-12)
    assert set(syn[:, 2]) <= set(X[y == 1, 2])


def test_drop_duplicates_keeps_conflicting_labels():
    X = np.array([[1, 2], [1, 2], [1, 2], [3, 4]], dtype=float)
    y = np.array([0, 0, 1, 1])
    Xd, yd = drop_duplicates(X, y)
    assert len(yd) == 3 and sorted(yd.tolist()) == [0, 1, 1]


def test_signed_log1p_only_touches_given_columns():
    X = pd.DataFrame({"a": [0.0, 9.0, -9.0], "b": [5.0, 6.0, 7.0]})
    Z = signed_log1p(X, ["a"])
    assert np.allclose(Z["a"], [0.0, np.log(10), -np.log(10)]) and Z["b"].tolist() == [5.0, 6.0, 7.0]
    assert X["a"].tolist() == [0.0, 9.0, -9.0]
