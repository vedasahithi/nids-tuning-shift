"""Training the QRNN classifier in two modes (weight search or gradient training).

* :func:`train_swarm` (option a): a population optimiser from
  hsboa.optimizers searches the flat weight vector, minimising Eq. (40).
* :func:`train_gradient`: Adam on categorical cross-entropy with one-hot
  targets on the full training part, early stopping on validation loss.
* :func:`train_hpo` (option b): an optimiser tunes the hyper-parameters of
  :func:`train_gradient` (fitness: 1 - validation macro-F1), then the final
  model is retrained with the best settings.

Every result carries the loss history, wall-clock time and the amount of work done (fitness
evaluations, gradient steps). Models are always scored with the
numpy forward pass, so both modes are evaluated by identical code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hsboa.fitness import HyperparamFitness, WeightFitness, one_hot, stratified_subsample
from hsboa.metrics import Report, evaluate, predict_labels
from hsboa.models import QRNNSpec, build_keras_model, forward, get_flat_weights
from hsboa.optimizers import get_optimizer


@dataclass
class TrainResult:
    mode: str
    spec: QRNNSpec
    weights: np.ndarray
    history: pd.DataFrame                    # per epoch (gradient) or per evaluation (swarm)
    wall_s: float
    n_fitness_evals: int = 0
    n_gradient_steps: int = 0
    extra: dict = field(default_factory=dict)

    def predict_proba(self, X) -> np.ndarray:
        return forward(self.spec, self.weights, X)

    def evaluate(self, X, y) -> Report:
        return evaluate(y, predict_labels(self.predict_proba(X)))


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and TensorFlow and make TF ops deterministic."""
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _class_weight(y, kind):
    """Per-class loss weights w_c = (n / (K n_c)) ** alpha.

    ``kind`` is "none" (alpha = 0), "balanced" (alpha = 1) or a number alpha
    in [0, 1], which interpolates between the two (searchable as
    ``class_weight_power``).
    """
    if kind in (None, "none"):
        return None
    alpha = 1.0 if kind == "balanced" else float(kind)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("class_weight must be 'none', 'balanced' or a power in [0, 1]")
    if alpha == 0.0:
        return None
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    w = (len(y) / (len(counts) * np.maximum(counts, 1))) ** alpha
    return {i: float(v) for i, v in enumerate(w)}


def train_gradient(spec: QRNNSpec, X, y, X_val, y_val, *, learning_rate: float = 1e-3,
                   batch_size: int = 256, epochs: int = 30, patience: int | None = 5,
                   dropout: float = 0.0, l2: float = 0.0, class_weight: str | float = "none",
                   seed: int = 0, verbose: int = 0) -> TrainResult:
    """Adam + cross-entropy on one-hot targets; early stopping on validation loss (best weights kept)."""
    import keras

    keras.backend.clear_session()
    seed_everything(seed)
    model = build_keras_model(spec, dropout=dropout, l2=l2)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy")
    cbs = []
    if patience:
        cbs.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                 restore_best_weights=True))
    t0 = time.perf_counter()
    h = model.fit(np.asarray(X, np.float32), one_hot(y, spec.n_classes),
                  validation_data=(np.asarray(X_val, np.float32), one_hot(y_val, spec.n_classes)),
                  batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose,
                  class_weight=_class_weight(np.asarray(y), class_weight), callbacks=cbs)
    wall = time.perf_counter() - t0
    hist = pd.DataFrame({"epoch": np.arange(1, len(h.history["loss"]) + 1),
                         "train_loss": h.history["loss"], "val_loss": h.history["val_loss"]})
    steps = len(hist) * int(np.ceil(len(X) / batch_size))
    w = get_flat_weights(spec, model)
    keras.backend.clear_session()
    return TrainResult("gradient", spec, w, hist, wall, n_gradient_steps=steps,
                       extra={"learning_rate": learning_rate, "batch_size": batch_size,
                              "dropout": dropout, "l2": l2, "class_weight": class_weight,
                              "best_epoch": int(hist["val_loss"].idxmin()) + 1})


def train_swarm(spec: QRNNSpec, X, y, *, optimizer: str = "hsboa", optimizer_params: dict | None = None,
                pop_size: int = 30, max_evals: int = 6000, bound: float = 2.0,
                fitness_samples: int | None = 10000, fitness: str = "mse", seed: int = 0) -> TrainResult:
    """Weight-space training: minimise Eq. (40) over the flat weight vector in [-bound, bound]^n."""
    idx = stratified_subsample(y, fitness_samples, np.random.SeedSequence([seed, 1]))
    f = WeightFitness(spec, np.asarray(X)[idx], np.asarray(y)[idx], kind=fitness)
    lb, ub = np.full(spec.n_params, -bound), np.full(spec.n_params, bound)
    opt = get_optimizer(optimizer, **(optimizer_params or {}))
    t0 = time.perf_counter()
    res = opt(f, lb, ub, pop_size, max_evals, seed)
    wall = time.perf_counter() - t0
    hist = pd.DataFrame({"eval": np.arange(1, res.n_evals + 1), "best_fitness": res.history})
    return TrainResult("swarm", spec, res.best_x, hist, wall, n_fitness_evals=res.n_evals,
                       extra={"optimizer": res.name, "fitness": fitness, "fitness_rows": len(idx),
                              "n_params": spec.n_params, "iterations": res.iterations,
                              "diversity": res.diversity, **res.extra})


def train_hpo(d: int, X, y, X_val, y_val, *, space: dict, base: dict, optimizer: str = "hsboa",
              optimizer_params: dict | None = None, pop_size: int = 8, max_evals: int = 48,
              train_samples: int | None = 30000, seed: int = 0,
              cache_path=None, final_epochs: int | None = None,
              final_patience: int | None = None,
              classes: tuple[str, ...] | None = None) -> TrainResult:
    """Tune hyper-parameters with a population optimiser, then retrain on the full training part.

    ``base`` must contain ``epochs`` and may set patience, class_weight, pooling,
    attention_units, and values for any dimension not in ``space``.
    ``cache_path`` checkpoints every trial (see HyperparamFitness), so an
    interrupted search resumes without retraining finished trials.
    ``final_epochs`` / ``final_patience`` set the budget of the final retrain
    (default: the trial budget). ``classes`` names the classes (default UNSW-NB15's ten).
    """
    idx = stratified_subsample(y, train_samples, np.random.SeedSequence([seed, 2]))
    fit = HyperparamFitness(d, np.asarray(X)[idx], np.asarray(y)[idx], X_val, y_val,
                            space, base, seed, cache_path=cache_path, classes=classes)
    opt = get_optimizer(optimizer, **(optimizer_params or {}))
    t0 = time.perf_counter()
    res = opt(fit, np.zeros(fit.dim), np.ones(fit.dim), pop_size, max_evals, seed)
    best = fit.settings(res.best_x)
    spec = QRNNSpec(d, best["units"], best["window"], best.get("pooling", "last"),
                    best.get("attention_units", 16), n_classes=len(fit.classes))
    final = train_gradient(spec, X, y, X_val, y_val, learning_rate=best["learning_rate"],
                           batch_size=best["batch_size"], epochs=final_epochs or best["epochs"],
                           patience=final_patience if final_epochs else best.get("patience"),
                           dropout=best.get("dropout", 0.0), l2=best.get("l2", 0.0),
                           class_weight=best.get("class_weight", "none"), seed=seed)
    trials = pd.DataFrame(fit.trials)
    final.mode = "hpo"
    final.wall_s = time.perf_counter() - t0
    final.n_fitness_evals = res.n_evals
    # Work actually done: each distinct candidate is trained once (repeats are replayed from the cache).
    trained = trials.drop_duplicates(subset=[c for c in trials.columns if c in space or c in (
        "learning_rate", "units", "window", "dropout", "l2", "batch_size", "class_weight")])
    final.n_gradient_steps += int((trained["epochs_run"] * np.ceil(len(idx) / trained["batch_size"])).sum())
    final.extra["distinct_trials"] = len(trained)
    final.extra.update(optimizer=res.name, best_settings=best, trials=trials,
                       search_best_fitness=res.best_f, search_rows=len(idx))
    return final
