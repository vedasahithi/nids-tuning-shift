"""Baseline detectors for E2, the model comparison.

Every baseline follows the QRNN protocol exactly: the same split, the same
training-only preprocessing and d features, model selection on validation,
and a final fit on the full training part. Tuned baselines search on the same
stratified training subsample as the QRNN, with fitness 1 - validation
macro-F1, random search, and a per-trial checkpoint.

Models:
    qrnn_untuned  QRNN with Adam defaults (gradient-trained control)
    logreg        multinomial logistic regression (grid over C and class weighting)
    rf            random forest (random search)
    hgb           histogram gradient boosting (sklearn; stands in for XGBoost/LightGBM,
                  which cannot be installed on the development machine)
    mlp           Keras MLP (random search; order-insensitive neural control)
    cnn_bilstm    Conv1D + bidirectional LSTM on the feature sequence (architecture family of
                  Sinha & Manollas 2020; Jouhari et al. 2024), random search like the MLP.
    xgb, lgbm     XGBoost and LightGBM (random search). Their libraries are
                  optional dependencies; :func:`available` reports whether they can run.

Search spaces map [0, 1]^k to settings by key prefix: ``log10_x`` -> 10**v,
``log2_x`` -> round(2**v), ``int_x`` -> round(v), anything else -> v.
This module must not modify the E1 modules (it only imports them).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hsboa.fitness import one_hot, stratified_subsample
from hsboa.metrics import evaluate, predict_labels
from hsboa.models import N_CLASSES

NEURAL = ("mlp", "cnn_bilstm", "qrnn_untuned")


def decode(x, space: dict) -> dict:
    """Settings from a point in [0, 1]^k; ``space`` maps prefixed names to [lo, hi]."""
    keys = list(space)
    if len(x) != len(keys):
        raise ValueError(f"expected {len(keys)} coordinates for {keys}")
    out = {}
    for k, v in zip(keys, x):
        lo, hi = space[k]
        raw = lo + float(np.clip(v, 0, 1)) * (hi - lo)
        if k.startswith("log10_"):
            out[k[6:]] = 10 ** raw
        elif k.startswith("log2_"):
            out[k[5:]] = int(round(2 ** raw))
        elif k.startswith("int_"):
            out[k[4:]] = int(round(raw))
        else:
            out[k] = raw
    return out


def sample_weight(y, power: float, n_classes: int = N_CLASSES) -> np.ndarray | None:
    """w_c = (n / (K n_c)) ** power per row (0 = unweighted, 1 = balanced)."""
    if not power:
        return None
    counts = np.bincount(y, minlength=n_classes)
    w = (len(y) / (n_classes * np.maximum(counts, 1))) ** power
    return w[y]


def _class_weight_dict(y, power: float, n_classes: int = N_CLASSES):
    if not power:
        return None
    counts = np.bincount(y, minlength=n_classes)
    w = (len(y) / (n_classes * np.maximum(counts, 1))) ** power
    return {i: float(v) for i, v in enumerate(w)}


def _k(s: dict) -> int:
    """Number of classes for a fit (settings key ``n_classes``, set by tune_and_fit; default UNSW-NB15's 10)."""
    return int(s.get("n_classes", N_CLASSES))


# ---------------------------------------------------------------------------
# Model builders: fit(settings, X, y, X_val, y_val, seed) -> predict_proba function
# ---------------------------------------------------------------------------

def _proba_full(model, X, n_classes: int = N_CLASSES):
    """predict_proba with all class columns, even if a class was absent in training."""
    p = model.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    full = np.zeros((len(X), n_classes))
    full[:, model.classes_] = p
    return full


def fit_logreg(s, X, y, X_val, y_val, seed):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=s["C"], max_iter=3000, random_state=seed)
    m.fit(X, y, sample_weight=sample_weight(y, s.get("class_weight_power", 0), _k(s)))
    return lambda Z: _proba_full(m, Z, _k(s)), {}


def fit_rf(s, X, y, X_val, y_val, seed, n_jobs: int = 2):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=s["n_estimators"], max_depth=s["max_depth"],
                               min_samples_leaf=s["min_samples_leaf"], max_features=s["max_features"],
                               random_state=seed, n_jobs=n_jobs)
    m.fit(X, y, sample_weight=sample_weight(y, s.get("class_weight_power", 0), _k(s)))
    return lambda Z: _proba_full(m, Z, _k(s)), {}


def fit_hgb(s, X, y, X_val, y_val, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(learning_rate=s["learning_rate"], max_iter=s["max_iter"],
                                       max_leaf_nodes=s["max_leaf_nodes"],
                                       min_samples_leaf=s["min_samples_leaf"],
                                       l2_regularization=s["l2_regularization"],
                                       early_stopping=False, random_state=seed)
    m.fit(X, y, sample_weight=sample_weight(y, s.get("class_weight_power", 0), _k(s)))
    return lambda Z: _proba_full(m, Z, _k(s)), {}


def _fit_keras(model, s, X, y, X_val, y_val):
    import keras
    model.compile(optimizer=keras.optimizers.Adam(s.get("learning_rate", 1e-3)),
                  loss="categorical_crossentropy")
    k = _k(s)
    h = model.fit(np.asarray(X, np.float32), one_hot(y, k), batch_size=s.get("batch_size", 256),
                  epochs=s["epochs"], shuffle=True, verbose=0,
                  validation_data=(np.asarray(X_val, np.float32), one_hot(y_val, k)),
                  class_weight=_class_weight_dict(np.asarray(y), s.get("class_weight_power", 0), k))
    hist = {"train_loss": h.history["loss"], "val_loss": h.history["val_loss"]}
    predict = lambda Z: model.predict(np.asarray(Z, np.float32), batch_size=4096, verbose=0)
    return predict, {"history": hist, "n_params": int(model.count_params()),
                     "gradient_steps": len(hist["train_loss"]) * int(np.ceil(len(X) / s.get("batch_size", 256)))}


def fit_mlp(s, X, y, X_val, y_val, seed):
    import keras
    from hsboa.train import seed_everything
    keras.backend.clear_session()
    seed_everything(seed)
    reg = keras.regularizers.L2(s["l2"]) if s.get("l2", 0) > 0 else None
    inp = keras.Input((X.shape[1],))
    h = inp
    for i in range(s["layers"]):
        h = keras.layers.Dense(s["units"], activation="relu", kernel_regularizer=reg, name=f"dense_{i}")(h)
        if s.get("dropout", 0) > 0:
            h = keras.layers.Dropout(s["dropout"])(h)
    out = keras.layers.Dense(_k(s), activation="softmax", kernel_regularizer=reg)(h)
    return _fit_keras(keras.Model(inp, out), s, X, y, X_val, y_val)


def fit_cnn_bilstm(s, X, y, X_val, y_val, seed):
    """Conv1D(filters, k) -> MaxPool(2) -> BiLSTM(units) -> Dropout -> softmax."""
    import keras
    from hsboa.train import seed_everything
    keras.backend.clear_session()
    seed_everything(seed)
    d = X.shape[1]
    inp = keras.Input((d,))
    h = keras.layers.Reshape((d, 1))(inp)
    h = keras.layers.Conv1D(s["filters"], s["kernel"], padding="same", activation="relu")(h)
    h = keras.layers.MaxPooling1D(2)(h)
    h = keras.layers.Bidirectional(keras.layers.LSTM(s["lstm_units"]))(h)
    h = keras.layers.Dropout(s["dropout"])(h)
    out = keras.layers.Dense(_k(s), activation="softmax")(h)
    return _fit_keras(keras.Model(inp, out), s, X, y, X_val, y_val)


def fit_qrnn_untuned(s, X, y, X_val, y_val, seed):
    from hsboa.models import QRNNSpec, forward
    from hsboa.train import train_gradient
    spec = QRNNSpec(X.shape[1], s["units"], s["window"], s.get("pooling", "last"), n_classes=_k(s))
    r = train_gradient(spec, X, y, X_val, y_val, learning_rate=s["learning_rate"],
                       batch_size=s["batch_size"], epochs=s["epochs"], patience=None,
                       class_weight=s.get("class_weight", "none"), seed=seed)
    hist = {"train_loss": r.history["train_loss"].tolist(), "val_loss": r.history["val_loss"].tolist()}
    return (lambda Z: forward(spec, r.weights, Z)), {"history": hist, "n_params": spec.n_params,
                                                     "gradient_steps": r.n_gradient_steps}


def fit_xgb(s, X, y, X_val, y_val, seed):
    """XGBoost (optional dependency, see OPTIONAL)."""
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=s["n_estimators"], learning_rate=s["learning_rate"], max_depth=s["max_depth"],
                      min_child_weight=s["min_child_weight"], reg_lambda=s["reg_lambda"],
                      subsample=s["subsample"], colsample_bytree=s["colsample_bytree"],
                      tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    m.fit(X, y, sample_weight=sample_weight(y, s.get("class_weight_power", 0), _k(s)))
    return lambda Z: _proba_full(m, Z, _k(s)), {}


def fit_lgbm(s, X, y, X_val, y_val, seed):
    """LightGBM (optional dependency, see OPTIONAL)."""
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(n_estimators=s["n_estimators"], learning_rate=s["learning_rate"],
                       num_leaves=s["num_leaves"], min_child_samples=s["min_child_samples"],
                       reg_lambda=s["reg_lambda"], subsample=s["subsample"], subsample_freq=1,
                       colsample_bytree=s["colsample_bytree"], n_jobs=1, random_state=seed, verbose=-1)
    m.fit(X, y, sample_weight=sample_weight(y, s.get("class_weight_power", 0), _k(s)))
    return lambda Z: _proba_full(m, Z, _k(s)), {}


FIT = {"logreg": fit_logreg, "rf": fit_rf, "hgb": fit_hgb, "mlp": fit_mlp,
       "cnn_bilstm": fit_cnn_bilstm, "qrnn_untuned": fit_qrnn_untuned, "xgb": fit_xgb, "lgbm": fit_lgbm}
# Models whose library may be missing: {model: import name}.
OPTIONAL = {"xgb": "xgboost", "lgbm": "lightgbm"}


def available(model: str) -> bool:
    """False if the model needs an optional library that is not installed."""
    import importlib.util
    return model not in OPTIONAL or importlib.util.find_spec(OPTIONAL[model]) is not None


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

class BaselineFitness:
    """1 - validation macro-F1 of a baseline fitted with decoded settings; trials checkpointed."""

    def __init__(self, model: str, space: dict, fixed: dict, X, y, X_val, y_val, seed: int,
                 cache_path: str | Path | None = None, classes: tuple[str, ...] | None = None):
        from hsboa.data import CLASSES
        self.classes = tuple(classes) if classes else CLASSES
        self.model, self.space, self.fixed = model, space, dict(fixed)
        self.fit_fn = FIT[model]          # hsboa.gbdt swaps in a feature-masked fit
        self.X, self.y, self.X_val, self.y_val, self.seed = X, y, X_val, y_val, seed
        self.trials: list[dict] = []
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec

    def settings(self, x) -> dict:
        return {**self.fixed, **decode(x, self.space)}

    def one(self, x) -> float:
        key = json.dumps([float(v) for v in x])
        if key in self._cache:
            rec = {k: v for k, v in self._cache[key].items() if k != "key"}
            self.trials.append({**rec, "trial": len(self.trials), "cached": True})
            return rec["fitness"]
        s = self.settings(x)
        t0 = time.perf_counter()
        predict, info = self.fit_fn(s, self.X, self.y, self.X_val, self.y_val, self.seed)
        f1 = evaluate(self.y_val, predict_labels(predict(self.X_val)), classes=self.classes).summary["macro_f1"]
        rec = {**s, "val_macro_f1": f1, "fitness": 1.0 - f1, "wall_s": time.perf_counter() - t0,
               "gradient_steps": info.get("gradient_steps", 0)}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, **rec}) + "\n")
            self._cache[key] = {"key": key, **rec}
        self.trials.append({**rec, "trial": len(self.trials), "cached": False})
        return 1.0 - f1

    def __call__(self, Xc):
        return np.array([self.one(x) for x in np.atleast_2d(Xc)])


def grid_points(space: dict, grid: dict) -> np.ndarray:
    """Points in [0, 1]^k for an explicit grid {name: [raw values]} over ``space``."""
    import itertools
    axes = []
    for k in space:
        lo, hi = space[k]
        vals = grid[k]
        axes.append([(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals])
    return np.array(list(itertools.product(*axes)), dtype=float)


def tune_and_fit(model: str, mcfg: dict, X, y, X_val, y_val, seed: int, *, train_samples: int | None,
                 pop_size: int, run_dir: Path | None = None, classes: tuple[str, ...] | None = None) -> dict:
    """Tune (random search or grid) on the training subsample, then fit on the full training part.

    ``mcfg`` is the model's config entry: ``tune`` (none | grid | random), ``fixed``,
    ``space``, ``grid`` (for tune: grid) and ``max_evals`` (for tune: random).
    Returns {"predict", "settings", "trials", "info", "wall_s"}.
    """
    from hsboa.optimizers import random_search
    import pandas as pd

    t0 = time.perf_counter()
    fixed, tune = dict(mcfg.get("fixed", {})), mcfg.get("tune", "none")
    if classes:
        fixed["n_classes"] = len(classes)
    trials = pd.DataFrame()
    if tune == "none":
        best = fixed
    else:
        idx = stratified_subsample(y, train_samples, np.random.SeedSequence([seed, 2]))
        fit = BaselineFitness(model, mcfg["space"], fixed, np.asarray(X)[idx], np.asarray(y)[idx],
                              X_val, y_val, seed, cache_path=run_dir / "trials.jsonl" if run_dir else None,
                              classes=classes)
        if tune == "grid":
            pts = grid_points(mcfg["space"], mcfg["grid"])
            vals = fit(pts)
            best = fit.settings(pts[int(np.argmin(vals))])
        elif tune == "random":
            dim = len(mcfg["space"])
            res = random_search(fit, np.zeros(dim), np.ones(dim), pop_size, mcfg["max_evals"], seed)
            best = fit.settings(res.best_x)
        else:
            raise ValueError(f"tune must be none, grid or random, got {tune!r}")
        trials = pd.DataFrame(fit.trials)
        if model in NEURAL and "final_epochs" in mcfg:
            best = {**best, "epochs": mcfg["final_epochs"]}
    predict, info = FIT[model](best, X, y, X_val, y_val, seed)
    return {"predict": predict, "settings": best, "trials": trials, "info": info,
            "wall_s": time.perf_counter() - t0}
