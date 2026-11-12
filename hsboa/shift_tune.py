"""Shift-aware joint feature and hyper-parameter selection (E6).

Unsupervised domain adaptation: the test partition's features may be used without its labels. A domain
classifier separates a seed's training part from the test features; its odds give
importance weights w(x) = p_test(x) / p_train(x) (hsboa.shift.density_ratio_weights,
capped at the 99th percentile, mean 1). Two uses:

* fitness  1 - importance-weighted validation macro-F1 (importance-weighted
           validation, Sugiyama, Krauledat & Mueller 2007): the optimizer is
           rewarded for settings that do well on test-like validation rows.
* training the training rows' weights multiply the class weights of the fit
           (the standard covariate-shift correction), for the iwtrain arm.

Validation-row weights come from a domain classifier that never saw those rows;
training-row weights are cross-fitted (5 folds), so no row is weighted by a model
fitted on it. The search space and decoding are E1b's (hsboa.gbdt), here for
LightGBM. Kept apart from hsboa.gbdt and hsboa.baselines so the E1b/E2 code hashes
do not change.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hsboa.baselines import _k, _proba_full, sample_weight
from hsboa.gbdt import JointFitness, default_settings, masked
from hsboa.shift import density_ratio_weights, weighted_macro_f1

CLIP = 0.99


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

def _domain_model(seed: int, max_iter: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                          early_stopping=False, random_state=seed)


def importance_weights(A_train: np.ndarray, A_val: np.ndarray, B_test: np.ndarray, seed: int, *,
                       max_iter: int = 200, folds: int = 5) -> dict:
    """Weights for the validation rows and (cross-fitted) for the training rows, both capped and mean 1.

    A_* are the raw encoded features of the seed's training and validation parts, B_test the
    test partition's features (no labels).
    """
    from sklearn.model_selection import StratifiedKFold
    n_tr, n_te = len(A_train), len(B_test)
    X = np.vstack([A_train, B_test])
    d = np.r_[np.zeros(n_tr, int), np.ones(n_te, int)]
    p_val = _domain_model(seed, max_iter).fit(X, d).predict_proba(A_val)[:, 1]
    p_tr = np.zeros(n_tr)
    for k, (fit_idx, held) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d)):
        m = _domain_model(seed + 1000 * (k + 1), max_iter).fit(X[fit_idx], d[fit_idx])
        rows = held[held < n_tr]                # held-out training rows of this fold
        p_tr[rows] = m.predict_proba(X[rows])[:, 1]
    # each training row's held-out model saw (folds-1)/folds of both domains, so the prior ratio is unchanged
    return {"val": density_ratio_weights(p_val, n_tr, n_te, CLIP),
            "train": density_ratio_weights(p_tr, n_tr, n_te, CLIP)}


# ---------------------------------------------------------------------------
# LightGBM with extra sample weights
# ---------------------------------------------------------------------------

def fit_lgbm_weighted(w_extra: np.ndarray | None):
    """hsboa.baselines.fit_lgbm with its class weights multiplied by per-row weights ``w_extra``."""
    def fit(s, X, y, X_val, y_val, seed):
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=s["n_estimators"], learning_rate=s["learning_rate"],
                           num_leaves=s["num_leaves"], min_child_samples=s["min_child_samples"],
                           reg_lambda=s["reg_lambda"], subsample=s["subsample"], subsample_freq=1,
                           colsample_bytree=s["colsample_bytree"], n_jobs=1, random_state=seed, verbose=-1)
        sw = sample_weight(y, s.get("class_weight_power", 0), _k(s))
        if w_extra is not None:
            sw = np.asarray(w_extra, float) if sw is None else sw * np.asarray(w_extra, float)
        m.fit(X, y, sample_weight=sw)
        return lambda Z: _proba_full(m, Z, _k(s)), {}
    return fit


def fit_hgb_weighted(w_extra: np.ndarray | None):
    """hsboa.baselines.fit_hgb with extra per-row weights (stand-in for LightGBM in tests)."""
    def fit(s, X, y, X_val, y_val, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(learning_rate=s["learning_rate"], max_iter=s["max_iter"],
                                           max_leaf_nodes=s["max_leaf_nodes"], min_samples_leaf=s["min_samples_leaf"],
                                           l2_regularization=s["l2_regularization"], early_stopping=False,
                                           random_state=seed)
        sw = sample_weight(y, s.get("class_weight_power", 0), _k(s))
        if w_extra is not None:
            sw = np.asarray(w_extra, float) if sw is None else sw * np.asarray(w_extra, float)
        m.fit(X, y, sample_weight=sw)
        return lambda Z: _proba_full(m, Z, _k(s)), {}
    return fit


FIT_WEIGHTED = {"lgbm": fit_lgbm_weighted, "hgb": fit_hgb_weighted}


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

class ShiftFitness(JointFitness):
    """1 - (importance-weighted) validation macro-F1 of ``model`` (lgbm | hgb) on the masked columns.

    ``w_val`` None gives the standard fitness; ``w_train`` (aligned with X) weights the fit.
    Every trial records both the weighted and the unweighted validation macro-F1.
    """

    def __init__(self, model: str, n_features: int, hp_space: dict, fixed: dict, X, y, X_val, y_val, seed: int, *,
                 w_val: np.ndarray | None, w_train: np.ndarray | None, min_features: int = 3,
                 cache_path: str | Path | None = None, classes: tuple[str, ...] | None = None):
        super().__init__(model, n_features, hp_space, fixed, X, y, X_val, y_val, seed, min_features=min_features,
                         cache_path=cache_path, classes=classes)
        self.w_val = None if w_val is None else np.asarray(w_val, float)
        self.fit_fn = masked(FIT_WEIGHTED[model](w_train))

    def one(self, x) -> float:
        key = json.dumps([float(v) for v in x])
        if key in self._cache:
            rec = {k: v for k, v in self._cache[key].items() if k != "key"}
            self.trials.append({**rec, "trial": len(self.trials), "cached": True})
            return rec["fitness"]
        s = self.settings(x)
        t0 = time.perf_counter()
        predict, _ = self.fit_fn(s, self.X, self.y, self.X_val, self.y_val, self.seed)
        pred = np.asarray(predict(self.X_val)).argmax(axis=1)
        k = len(self.classes)
        f1 = weighted_macro_f1(self.y_val, pred, k)
        f1_w = weighted_macro_f1(self.y_val, pred, k, self.w_val) if self.w_val is not None else f1
        rec = {**s, "val_macro_f1": f1, "val_macro_f1_weighted": f1_w, "fitness": 1.0 - f1_w,
               "wall_s": time.perf_counter() - t0, "gradient_steps": 0}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, **rec}) + "\n")
            self._cache[key] = {"key": key, **rec}
        self.trials.append({**rec, "trial": len(self.trials), "cached": False})
        return 1.0 - f1_w


def tune_shift(model: str, X, y, X_val, y_val, seed: int, *, tuner: str, hp_space: dict, pop_size: int,
               max_evals: int, train_samples: int | None, w_val: np.ndarray | None, w_train: np.ndarray | None,
               min_features: int = 3, optimizer_params: dict | None = None, cache_path=None,
               classes: tuple[str, ...] | None = None) -> dict:
    """Search on the stratified training subsample (E1b's), refit the best setting on the full training part.

    ``w_train`` is aligned with the full training part; the search uses its subsample rows.
    """
    import pandas as pd

    from hsboa.fitness import stratified_subsample
    from hsboa.optimizers import get_optimizer

    t0 = time.perf_counter()
    fixed = {"n_classes": len(classes)} if classes else {}
    idx = stratified_subsample(y, train_samples, np.random.SeedSequence([seed, 2]))
    X, y = np.asarray(X), np.asarray(y)
    fit = ShiftFitness(model, X.shape[1], hp_space, fixed, X[idx], y[idx], X_val, y_val, seed, w_val=w_val,
                       w_train=None if w_train is None else np.asarray(w_train)[idx], min_features=min_features,
                       cache_path=cache_path, classes=classes)
    res = get_optimizer(tuner, **(optimizer_params or {}))(fit, np.zeros(fit.dim), np.ones(fit.dim),
                                                          pop_size, max_evals, seed)
    best = fit.settings(res.best_x)
    final = masked(FIT_WEIGHTED[model](w_train))
    predict, _ = final(best, X, y, X_val, y_val, seed)
    trials = pd.DataFrame(fit.trials)
    return {"predict": predict, "settings": best, "trials": trials,
            "distinct_trials": int((~trials["cached"]).sum()) if not trials.empty else 0,
            "search_best_fitness_f1": 1.0 - float(res.best_f), "search_rows": len(idx),
            "wall_s": time.perf_counter() - t0}


def fit_default(model: str, X, y, X_val, y_val, seed: int, *, fixed: dict, n_selected: int,
                classes: tuple[str, ...] | None = None) -> dict:
    """The untuned arm: top ``n_selected`` MI columns, fixed settings, full training part."""
    t0 = time.perf_counter()
    s = default_settings({**fixed, **({"n_classes": len(classes)} if classes else {})}, n_selected)
    predict, _ = masked(FIT_WEIGHTED[model](None))(s, X, y, X_val, y_val, seed)
    return {"predict": predict, "settings": s, "trials": None, "distinct_trials": 0,
            "search_best_fitness_f1": None, "search_rows": 0, "wall_s": time.perf_counter() - t0}


def refit(model: str, settings: dict, X, y, X_val, y_val, seed: int, w_train: np.ndarray | None):
    """Deterministic refit of a finished run's final model (for the final test evaluation)."""
    predict, _ = masked(FIT_WEIGHTED[model](w_train))(settings, X, y, X_val, y_val, seed)
    return predict
