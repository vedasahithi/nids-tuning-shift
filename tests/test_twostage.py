"""Tests for two-stage classification and label-shift correction (hsboa.twostage)."""

import numpy as np
import pytest

from hsboa.twostage import (NORMAL, best_threshold, binary_rates, combine, decide, em_label_shift, shift_sample,
                            split)


def test_combine_and_split_round_trip():
    p_a = np.array([0.2, 0.9])
    Q = np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])
    P = combine(p_a, Q)
    assert np.allclose(P.sum(axis=1), 1) and np.allclose(P[:, NORMAL], [0.8, 0.1])
    p2, Q2 = split(P)
    assert np.allclose(p2, p_a) and np.allclose(Q2, Q)


def test_decide_threshold_rule():
    P = combine(np.array([0.3, 0.6, 0.6]), np.array([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]))
    assert decide(P).tolist() == [0, 1, 2]
    assert decide(P, 0.5).tolist() == [0, 1, 2]
    assert decide(P, 0.7).tolist() == [0, 0, 0]
    assert decide(P, 0.2).tolist() == [2, 1, 2]


def _shifted(seed=0, n=4000, src=(0.5, 0.3, 0.2), tgt=(0.8, 0.12, 0.08)):
    """Calibrated probabilities under source priors, rows drawn under target priors."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=tgt)
    mu = np.array([0.0, 1.5, 3.0])
    x = mu[y] + rng.standard_normal(n)
    lik = np.exp(-0.5 * (x[:, None] - mu[None]) ** 2)
    P = lik * np.array(src)
    return P / P.sum(axis=1, keepdims=True), y


def test_em_recovers_target_prior():
    P, y = _shifted()
    W, pi = em_label_shift(P, [0.5, 0.3, 0.2])
    assert pi[0] == pytest.approx(0.8, abs=0.04)
    assert np.allclose(W.sum(axis=1), 1)
    assert (W.argmax(1) == y).mean() > (P.argmax(1) == y).mean()


def test_em_is_a_no_op_without_shift():
    P, _ = _shifted(tgt=(0.5, 0.3, 0.2))
    W, pi = em_label_shift(P, P.mean(axis=0))
    assert np.allclose(pi, P.mean(axis=0), atol=1e-6) and np.allclose(W, P, atol=1e-6)


def test_shift_sample_sets_the_normal_share_and_keeps_every_class():
    y = np.repeat([0, 1, 2, 3], [320, 500, 170, 10])
    idx = shift_sample(y, 0.45, rng=0)
    ys = y[idx]
    assert (ys == 0).sum() == 320 and (ys == 0).mean() == pytest.approx(0.45, abs=0.01)
    assert set(ys) == {0, 1, 2, 3} and len(set(idx)) == len(idx)
    assert np.array_equal(np.sort(shift_sample(y, 0.2, rng=0)), np.arange(len(y)))   # cannot add attacks


def test_binary_rates_and_threshold_search():
    y = np.array([0, 0, 0, 1, 2])
    assert binary_rates(y, np.array([0, 1, 0, 1, 0])) == {"tpr": 0.5, "fpr": pytest.approx(1 / 3)}
    P = combine(np.array([0.4, 0.45, 0.1, 0.9, 0.8]), np.array([[0.5, 0.5]] * 3 + [[0.9, 0.1], [0.2, 0.8]]))
    assert decide(P, best_threshold(P, y, 3)).tolist() == [0, 0, 0, 1, 2]
