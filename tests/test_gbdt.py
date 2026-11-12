"""Tests for joint feature and hyper-parameter selection (E1b, hsboa.gbdt) and the optional E2 models."""

import numpy as np
import pytest

from hsboa.baselines import FIT, available, sample_weight
from hsboa.gbdt import (JointFitness, decode_joint, decode_mask, default_settings, fit_default, joint_space,
                        masked, tune_joint)

HP = {"log10_learning_rate": [-1.5, -0.5], "int_max_iter": [20, 40], "int_max_leaf_nodes": [7, 15],
      "int_min_samples_leaf": [5, 20], "log10_l2_regularization": [-6, 0], "class_weight_power": [0, 1]}
CLASSES = ("a", "b", "c", "d")


def _toy(n=800, seed=0):
    """Four classes separated by columns 0 and 1 only; columns 2-5 are noise."""
    rng = np.random.default_rng(seed)
    y = rng.choice(4, size=n, p=[0.5, 0.25, 0.15, 0.1])
    X = rng.random((n, 6))
    X[:, 0] = (y % 2) + 0.1 * rng.standard_normal(n)
    X[:, 1] = (y // 2) + 0.1 * rng.standard_normal(n)
    return X.astype(np.float32), y


def test_mask_decoding_and_repair():
    assert decode_mask([0.9, 0.1, 0.7, 0.2], min_features=1) == [0, 2]
    assert decode_mask([0.1, 0.1, 0.9, 0.1], min_features=3) == [0, 1, 2]      # top-MI columns added first
    assert decode_mask([0.0] * 5, min_features=2) == [0, 1]
    with pytest.raises(ValueError):
        decode_mask([0.5, 0.5], min_features=3)


def test_joint_space_and_decode():
    sp = joint_space(6, HP)
    assert list(sp)[:6] == [f"mask_{i:02d}" for i in range(6)] and len(sp) == 6 + len(HP)
    x = np.r_[[1, 0, 1, 0, 0, 0], np.zeros(len(HP))]
    s = decode_joint(x, 6, HP, min_features=2)
    assert s["features"] == [0, 2] and s["n_selected"] == 2
    assert s["max_iter"] == 20 and s["class_weight_power"] == 0
    with pytest.raises(ValueError):
        decode_joint(x[:-1], 6, HP)
    with pytest.raises(ValueError):
        joint_space(2, {"mask_00": [0, 1]})


def test_masked_fit_uses_only_selected_columns():
    X, y = _toy()
    s = {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 15, "min_samples_leaf": 5,
         "l2_regularization": 0.0, "n_classes": 4, "features": [0, 1]}
    predict, _ = masked(FIT["hgb"])(s, X, y, X, y, 0)
    Z = X.copy()
    Z[:, 2:] = 123.0                                  # unselected columns must not matter
    assert np.allclose(predict(X), predict(Z))
    assert (predict(X).argmax(1) == y).mean() > 0.95


def test_joint_fitness_cache_resume(tmp_path):
    X, y = _toy()
    cache = tmp_path / "trials.jsonl"
    fit = JointFitness("hgb", 6, HP, {"n_classes": 4}, X, y, X, y, seed=0, cache_path=cache, classes=CLASSES)
    pts = np.random.default_rng(1).random((3, fit.dim))
    first = fit(pts)
    again = JointFitness("hgb", 6, HP, {"n_classes": 4}, X, y, X, y, seed=0, cache_path=cache, classes=CLASSES)
    assert np.allclose(again(pts), first)
    assert all(t["cached"] for t in again.trials) and len(cache.read_text().splitlines()) == 3
    assert all(isinstance(t["features"], list) for t in again.trials)
    with pytest.raises(ValueError):
        JointFitness("hgb", 5, HP, {}, X, y, X, y, seed=0)


@pytest.mark.parametrize("tuner", ["random", "sboa", "hsboa"])
def test_tune_joint_runs_and_refits(tuner, tmp_path):
    X, y = _toy()
    r = tune_joint("hgb", X, y, X, y, seed=0, tuner=tuner, hp_space=HP, fixed={}, pop_size=4, max_evals=8,
                   train_samples=400, min_features=2, cache_path=tmp_path / "t.jsonl", classes=CLASSES)
    assert r["predict"](X).shape == (len(X), 4)
    assert 1 <= r["distinct_trials"] <= 8 and len(r["trials"]) == 8 and r["search_rows"] <= 410
    assert r["settings"]["n_selected"] >= 2 and r["settings"]["n_classes"] == 4
    assert 0 <= r["search_best_val_macro_f1"] <= 1


def test_default_arm():
    X, y = _toy()
    fixed = {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 15, "min_samples_leaf": 5,
             "l2_regularization": 0.0, "class_weight_power": 0.5}
    assert default_settings(fixed, 3)["features"] == [0, 1, 2]
    r = fit_default("hgb", X, y, X, y, 0, fixed=fixed, n_selected=2, classes=CLASSES)
    assert (r["predict"](X).argmax(1) == y).mean() > 0.95 and r["distinct_trials"] == 0


def test_optional_models():
    assert available("hgb") and available("rf")
    X, y = _toy()
    s = {"n_estimators": 20, "learning_rate": 0.2, "max_depth": 3, "min_child_weight": 1.0, "reg_lambda": 1.0,
         "subsample": 1.0, "colsample_bytree": 1.0, "num_leaves": 7, "min_child_samples": 5,
         "class_weight_power": 0.5, "n_classes": 4}
    for model in ("xgb", "lgbm"):
        if not available(model):
            continue                                  # library not installed
        predict, _ = FIT[model](s, X, y, X, y, 0)
        assert (predict(X).argmax(1) == y).mean() > 0.9
    assert sample_weight(y, 0) is None
