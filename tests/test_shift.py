"""Tests for train/test shift measurement (hsboa.shift)."""

import numpy as np
import pandas as pd
import pytest

from hsboa.shift import (PUBLISHED_TEST_COUNTS, density_ratio_weights, domain_auc, duplicate_stats, encode, ess,
                         greedy_removal, ks_stat, match_class_mix, psi, univariate, weighted_macro_f1)


def _domains(n=1500, seed=0, shift=2.0):
    """Column 0 shifts between domains, columns 1-2 do not."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, 3))
    B = rng.standard_normal((n, 3))
    B[:, 0] += shift
    return A, B


def test_encode_uses_reference_categories_and_flags_unseen():
    ref = pd.DataFrame({"p": ["tcp", "udp", "tcp"], "x": [1.0, 2.0, 3.0]})
    other = pd.DataFrame({"p": ["udp", "icmp"], "x": [4.0, 5.0]})
    A, B = encode(ref, other, ["p"])
    assert A[:, 0].tolist() == [0, 1, 0] and B[:, 0].tolist() == [1, -1]
    assert B[:, 1].tolist() == [4.0, 5.0] and ref["p"].tolist() == ["tcp", "udp", "tcp"]


def test_domain_auc_detects_shift_and_not_noise():
    A, B = _domains()
    assert domain_auc(A, B, folds=3, max_iter=30) > 0.8
    A2, B2 = _domains(shift=0.0)
    assert domain_auc(A2, B2, folds=3, max_iter=30) < 0.56
    assert domain_auc(A, B, folds=3, max_iter=30, columns=[1, 2]) < 0.56


def test_ks_and_psi():
    a = np.arange(1000.0)
    assert ks_stat(a, a) == 0.0 and ks_stat(a, a + 2000) == 1.0
    assert ks_stat(a, a + 500) == pytest.approx(0.5)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, a + 500) > 1.0
    assert psi(np.zeros(100), np.zeros(50)) == pytest.approx(0.0, abs=1e-9)   # constant column


def test_univariate_ranks_the_shifted_feature_first():
    A, B = _domains()
    u = univariate(A, B, ["s", "n1", "n2"])
    assert u.feature[0] == "s" and u.auc[0] > 0.85 and u.auc[1:].max() < 0.56
    assert set(u.columns) >= {"auc", "ks", "psi", "mean_ref", "mean_other"}


def test_greedy_removal_drops_the_shifted_feature_first():
    A, B = _domains(n=800)
    g = greedy_removal(A, B, ["s", "n1", "n2"], steps=1, folds=3, max_iter=30, perm_rows=500, perm_repeats=2)
    assert g.dropped[0] == "s" and g.auc[0] > 0.8 and g.auc[1] < 0.56
    assert g.removed.tolist() == [0, 1]


def test_match_class_mix_hits_target_proportions_without_repeats():
    y = np.repeat([0, 1, 2], [300, 600, 90])
    idx = match_class_mix(y, [50, 25, 25], rng=0)
    assert len(set(idx)) == len(idx)
    assert np.bincount(y[idx]).tolist() == [180, 90, 90]          # class 2 limits: 90 / 25 = 3.6 per unit
    assert np.array_equal(match_class_mix(y, [300, 600, 90], rng=0), np.arange(len(y)))
    assert len(PUBLISHED_TEST_COUNTS) == 10 and sum(PUBLISHED_TEST_COUNTS) == 82332


def test_density_ratio_weights_and_ess():
    w = density_ratio_weights(np.array([0.5, 0.5, 0.5]), n_ref=100, n_other=100, clip_quantile=None)
    assert np.allclose(w, 1.0) and ess(w) == pytest.approx(3.0)
    w = density_ratio_weights(np.array([0.2, 0.5, 0.8]), n_ref=1, n_other=1, clip_quantile=None)
    assert np.allclose(w / w[1], [0.25, 1.0, 4.0]) and w.mean() == pytest.approx(1.0)
    p = np.r_[np.full(99, 0.5), 0.999]
    wc, wu = density_ratio_weights(p, 1, 1, clip_quantile=0.99), density_ratio_weights(p, 1, 1, clip_quantile=None)
    assert wc.max() / wc.min() < 20 < wu.max() / wu.min() and ess(wc) > 3 * ess(wu)
    assert ess(np.r_[1.0, np.zeros(9)]) == pytest.approx(1.0)


def test_weighted_macro_f1():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    assert weighted_macro_f1(y, pred, 2) == pytest.approx((2 / 3 + 0.8) / 2)
    # all weight on the correct rows -> perfect score
    assert weighted_macro_f1(y, pred, 2, w=[1, 0, 1, 1]) == pytest.approx(1.0)
    assert weighted_macro_f1(y, pred, 3) == pytest.approx((2 / 3 + 0.8) / 3)   # absent class scores 0


def test_duplicate_stats():
    X = np.array([[1, 1], [1, 1], [1, 1], [2, 2], [3, 3]], dtype=float)
    y = np.array([0, 0, 1, 1, 2])
    Xo = np.array([[1, 1], [9, 9], [3, 3], [8, 8]], dtype=float)
    s = duplicate_stats(X, y, Xo)
    assert s["unique_rows"] == 3 and s["dup_row_share"] == pytest.approx(0.6)
    assert s["conflict_row_share"] == pytest.approx(0.6)
    assert s["other_seen_share"] == pytest.approx(0.5) and s["other_seen_conflicting_share"] == pytest.approx(0.25)
