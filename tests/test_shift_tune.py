"""Tests for shift-aware joint selection (hsboa.shift_tune); HGB stands in for LightGBM."""

import numpy as np
import pytest

from hsboa.shift import weighted_macro_f1
from hsboa.shift_tune import FIT_WEIGHTED, ShiftFitness, fit_default, importance_weights, refit, tune_shift

SPACE = {"log10_learning_rate": [-1.5, -0.5], "int_max_iter": [5, 10], "int_max_leaf_nodes": [7, 15],
         "int_min_samples_leaf": [5, 20], "log10_l2_regularization": [-6, 0], "class_weight_power": [0, 1]}
DEFAULT = {"learning_rate": 0.1, "max_iter": 10, "max_leaf_nodes": 15, "min_samples_leaf": 10,
           "l2_regularization": 0.0, "class_weight_power": 0.5}


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.standard_normal((n, 5)) + y[:, None] * np.array([1.0, 0.5, 0, 0, 0])
    return X, y


def test_importance_weights_favour_test_like_rows():
    rng = np.random.default_rng(0)
    A_tr, A_val = rng.standard_normal((800, 2)), rng.standard_normal((400, 2))
    B = rng.standard_normal((800, 2))
    B[:, 0] += 1.5
    w = importance_weights(A_tr, A_val, B, seed=0, max_iter=30, folds=2)
    assert w["val"].shape == (400,) and w["train"].shape == (800,)
    assert w["val"].mean() == pytest.approx(1.0) and w["train"].mean() == pytest.approx(1.0)
    hi = A_val[:, 0] > 1
    assert w["val"][hi].mean() > 2 * w["val"][~hi].mean()     # test-like rows (large x0) weigh more
    hi_tr = A_tr[:, 0] > 1
    assert w["train"][hi_tr].mean() > 2 * w["train"][~hi_tr].mean()


def test_weighted_fit_changes_the_model_and_none_matches_plain():
    X, y = _data()
    s = {**DEFAULT, "n_classes": 3}
    p0 = FIT_WEIGHTED["hgb"](None)(s, X, y, X, y, 0)[0](X)
    p1 = FIT_WEIGHTED["hgb"](np.ones(len(y)))(s, X, y, X, y, 0)[0](X)
    p2 = FIT_WEIGHTED["hgb"](np.where(y == 2, 5.0, 1.0))(s, X, y, X, y, 0)[0](X)
    assert np.allclose(p0, p1) and not np.allclose(p0, p2)
    assert p2[:, 2].mean() > p0[:, 2].mean()


def test_shift_fitness_uses_weights_and_records_both_scores(tmp_path):
    X, y = _data()
    Xv, yv = _data(seed=1)
    w = np.where(yv == 0, 3.0, 0.5)
    f = ShiftFitness("hgb", 5, SPACE, {"n_classes": 3}, X, y, Xv, yv, 0, w_val=w, w_train=None,
                     cache_path=tmp_path / "t.jsonl")
    x = np.full(f.dim, 0.6)
    val = f(x[None])[0]
    t = f.trials[-1]
    assert val == pytest.approx(1 - t["val_macro_f1_weighted"])
    assert t["val_macro_f1"] != pytest.approx(t["val_macro_f1_weighted"])
    f2 = ShiftFitness("hgb", 5, SPACE, {"n_classes": 3}, X, y, Xv, yv, 0, w_val=w, w_train=None,
                      cache_path=tmp_path / "t.jsonl")
    assert f2(x[None])[0] == pytest.approx(val) and f2.trials[-1]["cached"]
    std = ShiftFitness("hgb", 5, SPACE, {"n_classes": 3}, X, y, Xv, yv, 0, w_val=None, w_train=None)
    assert std(x[None])[0] == pytest.approx(1 - std.trials[-1]["val_macro_f1"])


def test_tune_shift_and_refit_are_deterministic():
    X, y = _data()
    Xv, yv = _data(seed=1)
    wv, wt = np.linspace(0.5, 1.5, len(yv)), np.linspace(1.5, 0.5, len(y))
    r = tune_shift("hgb", X, y, Xv, yv, 0, tuner="random", hp_space=SPACE, pop_size=3, max_evals=6,
                   train_samples=None, w_val=wv, w_train=wt, classes=("a", "b", "c"))
    assert r["distinct_trials"] == 6 and 3 <= r["settings"]["n_selected"] <= 5
    p = refit("hgb", r["settings"], X, y, Xv, yv, 0, wt)(Xv)
    assert np.allclose(p, r["predict"](Xv))
    assert 0 <= weighted_macro_f1(yv, p.argmax(1), 3, wv) <= 1


def test_fit_default_uses_top_columns():
    X, y = _data()
    r = fit_default("hgb", X, y, X, y, 0, fixed=DEFAULT, n_selected=2, classes=("a", "b", "c"))
    assert r["settings"]["features"] == [0, 1] and r["predict"](X).shape == (len(y), 3)
