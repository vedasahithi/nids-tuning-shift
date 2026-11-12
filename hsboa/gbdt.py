"""Joint feature and hyper-parameter selection for gradient boosting (E1b).

A candidate is a point in [0, 1]^(n + k): the first n coordinates are a
feature mask over the n MI-ordered input columns (column i is used if
x_i > 0.5), the remaining k are the classifier settings, decoded by key
prefix exactly like :func:`hsboa.baselines.decode`. If fewer than
``min_features`` bits are on, the highest-MI columns are added
(deterministic repair), so every candidate is a valid model.

Fitness is 1 - validation macro-F1 after fitting on the training subsample,
with every trial checkpointed (:class:`hsboa.baselines.BaselineFitness`).
The best setting is refitted on the full training part. The tuner is any
:mod:`hsboa.optimizers` method (HSBOA, SBOA, random search).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hsboa.baselines import FIT, BaselineFitness, decode
from hsboa.fitness import stratified_subsample


def joint_space(n_features: int, hp_space: dict) -> dict:
    """Search space: one [0, 1] mask coordinate per input column, then the classifier settings."""
    mask = {f"mask_{i:02d}": [0.0, 1.0] for i in range(n_features)}
    clash = set(mask) & set(hp_space)
    if clash:
        raise ValueError(f"hyper-parameter names clash with mask names: {sorted(clash)}")
    return {**mask, **hp_space}


def decode_mask(m, min_features: int = 3) -> list[int]:
    """Selected column indices (ascending = MI order) from mask coordinates, repaired to >= min_features."""
    m = np.asarray(m, dtype=float)
    if not 0 < min_features <= len(m):
        raise ValueError(f"min_features={min_features} not in 1..{len(m)}")
    on = m > 0.5
    for i in range(len(m)):                 # columns are MI-ordered: add the most informative first
        if on.sum() >= min_features:
            break
        on[i] = True
    return np.flatnonzero(on).tolist()


def decode_joint(x, n_features: int, hp_space: dict, min_features: int = 3) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) != n_features + len(hp_space):
        raise ValueError(f"expected {n_features + len(hp_space)} coordinates, got {len(x)}")
    feats = decode_mask(x[:n_features], min_features)
    return {"features": feats, "n_selected": len(feats), **decode(x[n_features:], hp_space)}


def masked(fit):
    """Wrap a hsboa.baselines fit so that it uses only the columns in settings['features']."""
    def fit_masked(s, X, y, X_val, y_val, seed):
        cols = list(s["features"])
        predict, info = fit(s, np.asarray(X)[:, cols], y, np.asarray(X_val)[:, cols], y_val, seed)
        return (lambda Z: predict(np.asarray(Z)[:, cols])), info
    return fit_masked


class JointFitness(BaselineFitness):
    """1 - validation macro-F1 of ``model`` fitted on the masked columns with the decoded settings."""

    def __init__(self, model: str, n_features: int, hp_space: dict, fixed: dict, X, y, X_val, y_val,
                 seed: int, min_features: int = 3, cache_path: str | Path | None = None,
                 classes: tuple[str, ...] | None = None):
        if np.asarray(X).shape[1] != n_features:
            raise ValueError(f"X has {np.asarray(X).shape[1]} columns, mask has {n_features}")
        super().__init__(model, joint_space(n_features, hp_space), fixed, X, y, X_val, y_val, seed,
                         cache_path=cache_path, classes=classes)
        self.n_features, self.hp_space, self.min_features = n_features, hp_space, min_features
        self.fit_fn = masked(FIT[model])

    @property
    def dim(self) -> int:
        return len(self.space)

    def settings(self, x) -> dict:
        return {**self.fixed, **decode_joint(x, self.n_features, self.hp_space, self.min_features)}


def default_settings(fixed: dict, n_selected: int) -> dict:
    """The untuned arm: the top ``n_selected`` MI columns and fixed settings."""
    return {**fixed, "features": list(range(n_selected)), "n_selected": n_selected}


def tune_joint(model: str, X, y, X_val, y_val, seed: int, *, tuner: str, hp_space: dict, fixed: dict,
               pop_size: int, max_evals: int, train_samples: int | None, min_features: int = 3,
               optimizer_params: dict | None = None, cache_path: str | Path | None = None,
               classes: tuple[str, ...] | None = None) -> dict:
    """Search on the stratified training subsample, refit the best setting on the full training part.

    Returns {"predict", "settings", "trials", "distinct_trials", "search_best_val_macro_f1", "wall_s"}.
    """
    import pandas as pd

    from hsboa.optimizers import get_optimizer

    t0 = time.perf_counter()
    if classes:
        fixed = {**fixed, "n_classes": len(classes)}
    idx = stratified_subsample(y, train_samples, np.random.SeedSequence([seed, 2]))
    fit = JointFitness(model, np.asarray(X).shape[1], hp_space, fixed, np.asarray(X)[idx], np.asarray(y)[idx],
                       X_val, y_val, seed, min_features=min_features, cache_path=cache_path, classes=classes)
    res = get_optimizer(tuner, **(optimizer_params or {}))(fit, np.zeros(fit.dim), np.ones(fit.dim),
                                                          pop_size, max_evals, seed)
    best = fit.settings(res.best_x)
    predict, _ = fit.fit_fn(best, X, y, X_val, y_val, seed)
    trials = pd.DataFrame(fit.trials)
    return {"predict": predict, "settings": best, "trials": trials,
            "distinct_trials": int((~trials["cached"]).sum()) if not trials.empty else 0,
            "search_best_val_macro_f1": 1.0 - float(res.best_f), "search_rows": len(idx),
            "wall_s": time.perf_counter() - t0}


def fit_default(model: str, X, y, X_val, y_val, seed: int, *, fixed: dict, n_selected: int,
                classes: tuple[str, ...] | None = None) -> dict:
    """The untuned arm, fitted on the full training part."""
    t0 = time.perf_counter()
    s = default_settings({**fixed, **({"n_classes": len(classes)} if classes else {})}, n_selected)
    predict, _ = masked(FIT[model])(s, X, y, X_val, y_val, seed)
    return {"predict": predict, "settings": s, "trials": None, "distinct_trials": 0,
            "search_best_val_macro_f1": None, "search_rows": 0, "wall_s": time.perf_counter() - t0}
