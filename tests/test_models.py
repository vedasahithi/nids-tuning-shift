"""Tests for the QRNN model, the fitness functions and both training modes."""

import numpy as np
import pytest

from hsboa.fitness import (HyperparamFitness, WeightFitness, decode_hparams, one_hot,
                           stratified_subsample)
from hsboa.models import (QRNNSpec, build_keras_model, forward, get_flat_weights, init_weights,
                          set_flat_weights)
from hsboa.train import train_gradient, train_hpo, train_swarm


def _toy(n=1500, d=4, seed=0):
    """Five well-separated classes with imbalance, features in [0, 1]."""
    rng = np.random.default_rng(seed)
    y = rng.choice(5, size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    centres = np.linspace(0.1, 0.9, 5)
    X = np.clip(centres[y][:, None] + 0.04 * rng.standard_normal((n, d)), 0, 1)
    return X.astype(np.float32), y


# --- model ------------------------------------------------------------------

def test_parameter_count_and_pack_roundtrip():
    spec = QRNNSpec(d=10, units=16, window=2, n_classes=5)
    assert spec.n_params == 2 * 48 + 48 + 16 * 5 + 5          # conv kernel + bias + dense
    att = QRNNSpec(d=10, units=16, window=2, pooling="attention", attention_units=8, n_classes=5)
    assert att.n_params == spec.n_params + 16 * 8 + 8 + 8
    w = np.arange(spec.n_params, dtype=float)
    assert np.array_equal(spec.pack(spec.unpack(w)), w)
    with pytest.raises(ValueError):
        spec.unpack(w[:-1])
    with pytest.raises(ValueError):
        QRNNSpec(d=10, pooling="max")


def test_fo_pooling_matches_hand_computation():
    # One unit, window 1: z = tanh(a x), f = sigmoid(b x), o = sigmoid(c x).
    spec = QRNNSpec(d=3, units=1, window=1, n_classes=2)
    a, b, c = 0.7, -1.2, 2.0
    w = spec.pack({"conv_kernel": np.array([[[a, b, c]]]), "conv_bias": np.zeros(3),
                   "dense_kernel": np.array([[1.0, -1.0]]), "dense_bias": np.zeros(2)})
    x = np.array([[0.2, 0.5, 0.9]])
    sig = lambda v: 1 / (1 + np.exp(-v))
    cell = 0.0
    for xt in x[0]:
        cell = sig(b * xt) * cell + (1 - sig(b * xt)) * np.tanh(a * xt)
    h = sig(c * x[0, -1]) * cell
    expected = np.exp([h, -h]) / np.exp([h, -h]).sum()
    proba, H = forward(spec, w, x, return_hidden=True, dtype=np.float64)
    assert np.allclose(proba[0], expected, rtol=1e-12)
    assert H[0, -1, 0] == pytest.approx(h, rel=1e-12)


def test_causal_convolution_ignores_future_features():
    spec = QRNNSpec(d=6, units=4, window=3)
    w = init_weights(spec, 0)
    X = np.random.default_rng(1).random((5, 6))
    X2 = X.copy()
    X2[:, 4:] = 0.0                                             # change only the last two steps
    _, H1 = forward(spec, w, X, return_hidden=True, dtype=np.float64)
    _, H2 = forward(spec, w, X2, return_hidden=True, dtype=np.float64)
    assert np.allclose(H1[:, :4], H2[:, :4]) and not np.allclose(H1[:, 4:], H2[:, 4:])


@pytest.mark.parametrize("pooling", ["last", "attention"])
def test_numpy_and_keras_models_agree(pooling):
    spec = QRNNSpec(d=10, units=8, window=3, pooling=pooling, attention_units=6)
    model = build_keras_model(spec)
    w = np.random.default_rng(0).normal(0, 0.5, spec.n_params)
    set_flat_weights(spec, model, w)
    X = np.random.default_rng(1).random((32, 10)).astype(np.float32)
    assert np.allclose(model.predict(X, verbose=0), forward(spec, w, X), atol=1e-5)
    assert np.allclose(get_flat_weights(spec, model), w, atol=1e-6)


def test_init_weights_zero_biases_and_bounded_kernels():
    spec = QRNNSpec(d=10, units=16, window=2)
    p = spec.unpack(init_weights(spec, 0))
    assert not p["conv_bias"].any() and not p["dense_bias"].any()
    assert np.abs(p["dense_kernel"]).max() <= np.sqrt(6 / (16 + 5))


# --- fitness ----------------------------------------------------------------

def test_mse_of_uniform_prediction_is_0_16():
    spec = QRNNSpec(d=4, units=4, window=2, n_classes=5)
    X, y = _toy(200)
    f = WeightFitness(spec, X, y)
    # zero weights -> uniform softmax -> per row ((1-0.2)^2 + 4*0.2^2) / 5 = 0.16
    assert np.allclose(f(np.zeros((2, spec.n_params))), 0.16, rtol=1e-6)
    assert WeightFitness(spec, X, y, kind="ce").one(np.zeros(spec.n_params)) == pytest.approx(np.log(5), rel=1e-6)


def test_one_hot_and_stratified_subsample():
    assert one_hot(np.array([0, 4]), 5).tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
    y = np.repeat([0, 1, 2, 3, 4], [9000, 800, 150, 45, 5])
    idx = stratified_subsample(y, 1000, 0)
    assert abs(len(idx) - 1000) <= 5 and set(y[idx]) == {0, 1, 2, 3, 4}
    assert len(np.unique(idx)) == len(idx)
    assert np.array_equal(idx, stratified_subsample(y, 1000, 0))
    assert len(stratified_subsample(y, None, 0)) == len(y)


def test_decode_hparams():
    space = {"log10_learning_rate": [-4, -2], "log2_units": [3, 6], "window": [1, 4],
             "log2_batch_size": [6, 9]}
    lo = decode_hparams(np.zeros(4), space)
    hi = decode_hparams(np.ones(4), space)
    assert lo == {"learning_rate": pytest.approx(1e-4), "units": 8, "window": 1, "batch_size": 64}
    assert hi == {"learning_rate": pytest.approx(1e-2), "units": 64, "window": 4, "batch_size": 512}
    with pytest.raises(ValueError):
        decode_hparams(np.zeros(3), space)


# --- training ---------------------------------------------------------------

def test_swarm_training_learns_toy_problem():
    X, y = _toy()
    spec = QRNNSpec(d=4, units=4, window=2)
    res = train_swarm(spec, X, y, optimizer="hsboa", pop_size=20, max_evals=3000,
                      bound=3.0, fitness_samples=None, seed=0)
    assert res.n_fitness_evals == 3000 and len(res.history) == 3000
    assert res.history["best_fitness"].iloc[-1] < 0.08            # uniform guess = 0.16
    assert res.evaluate(X, y).summary["accuracy"] > 0.6            # majority class = 0.4


def test_gradient_training_learns_and_is_deterministic():
    X, y = _toy()
    Xv, yv = _toy(500, seed=1)
    spec = QRNNSpec(d=4, units=8, window=2, n_classes=5)
    a = train_gradient(spec, X, y, Xv, yv, learning_rate=0.01, epochs=15, batch_size=64, seed=3)
    b = train_gradient(spec, X, y, Xv, yv, learning_rate=0.01, epochs=15, batch_size=64, seed=3)
    assert list(a.history.columns) == ["epoch", "train_loss", "val_loss"]
    assert a.evaluate(Xv, yv).summary["macro_f1_present"] > 0.9   # the toy uses 5 of the 10 classes
    assert np.array_equal(a.weights, b.weights)
    assert a.n_gradient_steps == len(a.history) * int(np.ceil(len(X) / 64))


def test_hpo_runs_logs_trials_and_retrains():
    X, y = _toy(600)
    Xv, yv = _toy(300, seed=1)
    space = {"log10_learning_rate": [-3, -1.5], "log2_units": [2, 3]}
    base = {"epochs": 3, "patience": None, "window": 2, "batch_size": 64}
    res = train_hpo(4, X, y, Xv, yv, space=space, base=base, optimizer="sboa",
                    pop_size=4, max_evals=4, train_samples=None, seed=0)
    trials = res.extra["trials"]
    assert len(trials) == 4 and res.n_fitness_evals == 4
    assert res.extra["search_best_fitness"] == pytest.approx(trials["fitness"].min())
    assert 4 <= res.spec.units <= 8 and res.mode == "hpo"


def test_hpo_resumes_from_trial_checkpoint(tmp_path):
    X, y = _toy(600)
    Xv, yv = _toy(300, seed=1)
    kw = dict(space={"log10_learning_rate": [-3, -1.5], "log2_units": [2, 3]},
              base={"epochs": 2, "patience": None, "window": 2, "batch_size": 64},
              optimizer="hsboa", pop_size=4, max_evals=8, train_samples=None, seed=0,
              cache_path=tmp_path / "trials.jsonl")
    a = train_hpo(4, X, y, Xv, yv, **kw)
    lines = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert len(lines) == len(set(lines)) == a.extra["trials"]["cached"].eq(False).sum()
    b = train_hpo(4, X, y, Xv, yv, **kw)                      # rerun: every trial replayed
    assert b.extra["trials"]["cached"].all()
    assert b.extra["best_settings"] == a.extra["best_settings"]
    assert np.array_equal(a.weights, b.weights)


def test_class_weight_power_interpolates():
    from hsboa.train import _class_weight
    y = np.repeat([0, 1], [90, 10])
    assert _class_weight(y, "none") is None and _class_weight(y, 0.0) is None
    bal = _class_weight(y, "balanced")
    assert np.allclose([bal[0], bal[1]], [100 / 180, 100 / 20])
    half = _class_weight(y, 0.5)
    assert half[1] == pytest.approx(np.sqrt(5.0)) and half[0] == pytest.approx(np.sqrt(100 / 180))
    with pytest.raises(ValueError):
        _class_weight(y, 1.5)
    assert decode_hparams(np.array([0.25]), {"class_weight_power": [0, 1]}) == {"class_weight": 0.25}


def test_hpo_final_budget_and_distinct_work(tmp_path):
    X, y = _toy(600)
    Xv, yv = _toy(300, seed=1)
    res = train_hpo(4, X, y, Xv, yv, space={"log10_learning_rate": [-3, -1.5]},
                    base={"epochs": 2, "patience": None, "units": 4, "window": 2, "batch_size": 64},
                    optimizer="hsboa", pop_size=4, max_evals=12, train_samples=None, seed=0,
                    final_epochs=5, final_patience=None)
    assert len(res.history) == 5                                 # final retrain used its own budget
    trials = res.extra["trials"]
    assert res.extra["distinct_trials"] == trials["learning_rate"].nunique() <= len(trials)
    steps = 5 * int(np.ceil(600 / 64)) + res.extra["distinct_trials"] * 2 * int(np.ceil(600 / 64))
    assert res.n_gradient_steps == steps


def test_hyperparam_fitness_dim():
    f = HyperparamFitness(4, None, None, None, None, {"dropout": [0, 0.5], "window": [1, 4]}, {}, 0)
    assert f.dim == 2
