"""Tests for decision-layer bias tuning (direction B, hsboa.decision)."""

import numpy as np
import pytest

from hsboa.decision import (METHODS, BiasObjective, apply_bias, coordinate_search, log_proba, macro_f1,
                            per_class_f1, tune_bias)
from hsboa.metrics import evaluate

CLASSES = ("a", "b", "c", "d")


def _toy(n=2000, seed=0):
    """Probabilities that under-predict the rare class 3, so a positive bias on it helps macro-F1."""
    rng = np.random.default_rng(seed)
    y = rng.choice(4, size=n, p=[0.55, 0.25, 0.15, 0.05])
    logits = rng.normal(0, 1, (n, 4))
    logits[np.arange(n), y] += 1.5
    logits[:, 3] -= 1.5
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return P, y


def test_macro_f1_matches_metrics():
    P, y = _toy()
    pred = P.argmax(axis=1)
    rep = evaluate(y, pred, classes=CLASSES)
    assert macro_f1(y, pred, 4) == pytest.approx(rep.summary["macro_f1"])
    assert np.allclose(per_class_f1(y, pred, 4), rep.per_class["f1"].to_numpy())


def test_zero_bias_is_argmax_and_absent_class_counts_zero():
    P, y = _toy()
    assert np.array_equal(apply_bias(log_proba(P), np.zeros(4)), P.argmax(axis=1))
    y2 = np.where(y == 3, 0, y)                       # class 3 absent from the truth
    pc = per_class_f1(y2, np.zeros_like(y2), 4)
    assert pc[3] == 0 and macro_f1(y2, np.zeros_like(y2), 4) == pytest.approx(pc.mean())


def test_objective_counts_evaluations():
    P, y = _toy()
    f = BiasObjective(log_proba(P), y, 4)
    v = f(np.zeros((3, 4)))
    assert f.n_evals == 3 and np.allclose(v, 1 - macro_f1(y, P.argmax(axis=1), 4))


def test_coordinate_search_improves_and_respects_budget():
    P, y = _toy()
    f = BiasObjective(log_proba(P), y, 4)
    b, v, n = coordinate_search(f, 4, bound=3.0, max_evals=200)
    assert n <= 200 and f.n_evals == n
    assert 1 - v > macro_f1(y, P.argmax(axis=1), 4) and b[3] > 0


@pytest.mark.parametrize("method", METHODS)
def test_every_method_never_worse_than_argmax(method):
    P, y = _toy()
    L = log_proba(P)
    hp = {"harmonic": {"L": 4, "F": 1}} if method == "hsboa" else {}
    t = tune_bias(method, L, y, 4, max_evals=120, pop_size=6, bound=3.0, seed=1, optimizer_params=hp)
    assert t["b"].shape == (4,) and t["n_evals"] <= 121
    assert t["macro_f1"] >= macro_f1(y, P.argmax(axis=1), 4) - 1e-12
    assert t["macro_f1"] == pytest.approx(macro_f1(y, apply_bias(L, t["b"]), 4))


def test_hsboa_differs_from_sboa_at_the_diagnostic_budget():
    """At pop 20 and 2000 evaluations (~50 iterations > L = 10) the harmonic term is active."""
    P, y = _toy(seed=3)
    L = log_proba(P)
    h =tune_bias("hsboa", L, y, 4, max_evals=2000, pop_size=20, bound=3.0, seed=5,
                  optimizer_params={"harmonic": {"L": 10, "F": 2, "kappa0": 0.1}})
    s = tune_bias("sboa", L, y, 4, max_evals=2000, pop_size=20, bound=3.0, seed=5)
    assert not np.allclose(h["b"], s["b"])
