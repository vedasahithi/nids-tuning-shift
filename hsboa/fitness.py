"""Fitness functions for the two QRNN training modes.

* :class:`WeightFitness` (option a): a candidate is the flat QRNN weight
  vector; fitness is the mean squared error between
  one-hot targets and softmax outputs, on a fixed stratified subsample of
  the training part. The subsample is the same for every candidate, so the
  fitness is deterministic and greedy selection is meaningful.
* :class:`HyperparamFitness` (option b): a candidate is a point in [0, 1]^k
  decoded into hyper-parameters; fitness is 1 - macro-F1 on the validation
  split after Adam training with those settings. Every trial is logged.

Both are batch functions X (n, D) -> (n,), matching hsboa.optimizers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hsboa.models import N_CLASSES, QRNNSpec, forward

HPARAM_ORDER = ("log10_learning_rate", "log2_units", "window", "dropout", "log10_l2",
                "log2_batch_size", "class_weight_power")


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def stratified_subsample(y: np.ndarray, n: int | None, rng) -> np.ndarray:
    """Indices of a class-stratified subsample of size about n (every class keeps >= 1 row)."""
    y = np.asarray(y)
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(rng)
    idx = []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        k = max(1, int(round(n * len(rows) / len(y))))
        idx.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return np.sort(np.concatenate(idx))


class WeightFitness:
    """Eq. (40) MSE (or cross-entropy) of a population of flat weight vectors."""

    def __init__(self, spec: QRNNSpec, X: np.ndarray, y: np.ndarray, kind: str = "mse"):
        if kind not in ("mse", "ce"):
            raise ValueError("kind must be 'mse' or 'ce'")
        self.spec, self.kind = spec, kind
        self.X = np.asarray(X, dtype=np.float32)
        self.Y = one_hot(y, spec.n_classes)

    def one(self, w: np.ndarray) -> float:
        P = forward(self.spec, w, self.X)
        if self.kind == "mse":
            return float(np.mean((self.Y - P) ** 2))
        return float(-np.mean(np.sum(self.Y * np.log(np.clip(P, 1e-7, 1.0)), axis=1)))

    def __call__(self, W: np.ndarray) -> np.ndarray:
        return np.array([self.one(w) for w in np.atleast_2d(W)])


def decode_hparams(x: np.ndarray, space: dict) -> dict:
    """Map a point in [0, 1]^k (order HPARAM_ORDER, only keys in ``space``) to hyper-parameters."""
    keys = [k for k in HPARAM_ORDER if k in space]
    if len(x) != len(keys):
        raise ValueError(f"expected {len(keys)} coordinates for {keys}")
    raw = {k: space[k][0] + float(np.clip(v, 0, 1)) * (space[k][1] - space[k][0])
           for k, v in zip(keys, x)}
    out = {}
    if "log10_learning_rate" in raw:
        out["learning_rate"] = 10 ** raw["log10_learning_rate"]
    if "log2_units" in raw:
        out["units"] = int(round(2 ** raw["log2_units"]))
    if "window" in raw:
        out["window"] = int(round(raw["window"]))
    if "dropout" in raw:
        out["dropout"] = raw["dropout"]
    if "log10_l2" in raw:
        out["l2"] = 10 ** raw["log10_l2"]
    if "log2_batch_size" in raw:
        out["batch_size"] = int(round(2 ** raw["log2_batch_size"]))
    if "class_weight_power" in raw:
        out["class_weight"] = raw["class_weight_power"]          # 0 = none, 1 = balanced
    return out


class HyperparamFitness:
    """1 - validation macro-F1 after gradient training with decoded hyper-parameters.

    ``base`` holds the training settings that are not searched (epochs,
    patience, class_weight, pooling, ...). The training seed is the same for
    every candidate, so candidates differ only in their hyper-parameters.

    ``cache_path`` (JSON lines) checkpoints every finished trial. A rerun with
    the same seed asks for the same candidates in the same order, so finished
    trials are replayed from the file instead of retrained, and an
    interruption loses at most the trial in progress.
    """

    def __init__(self, d: int, X, y, X_val, y_val, space: dict, base: dict, seed: int,
                 cache_path: str | Path | None = None, classes: tuple[str, ...] | None = None):
        from hsboa.data import CLASSES
        self.classes = tuple(classes) if classes else CLASSES     # class names (default: UNSW-NB15's ten)
        self.d, self.X, self.y, self.X_val, self.y_val = d, X, y, X_val, y_val
        self.space, self.base, self.seed = space, dict(base), seed
        self.trials: list[dict] = []
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec

    @staticmethod
    def _key(x: np.ndarray) -> str:
        return json.dumps([float(v) for v in x])          # exact repr: same candidate, same key

    @property
    def dim(self) -> int:
        return sum(k in self.space for k in HPARAM_ORDER)

    def settings(self, x: np.ndarray) -> dict:
        return {**self.base, **decode_hparams(x, self.space)}

    def one(self, x: np.ndarray) -> float:
        from hsboa.train import train_gradient
        from hsboa.metrics import evaluate, predict_labels

        key = self._key(x)
        if key in self._cache:
            rec = {k: v for k, v in self._cache[key].items() if k != "key"}
            self.trials.append({**rec, "trial": len(self.trials), "cached": True})
            return rec["fitness"]
        s = self.settings(x)
        spec = QRNNSpec(self.d, s["units"], s["window"], s.get("pooling", "last"),
                        s.get("attention_units", 16), n_classes=len(self.classes))
        t0 = time.perf_counter()
        res = train_gradient(spec, self.X, self.y, self.X_val, self.y_val,
                             learning_rate=s["learning_rate"], batch_size=s["batch_size"],
                             epochs=s["epochs"], patience=s.get("patience"),
                             dropout=s.get("dropout", 0.0), l2=s.get("l2", 0.0),
                             class_weight=s.get("class_weight", "none"), seed=self.seed)
        f1 = evaluate(self.y_val, predict_labels(forward(spec, res.weights, self.X_val)),
                      classes=self.classes).summary["macro_f1"]
        rec = {**s, "val_macro_f1": f1, "fitness": 1.0 - f1, "epochs_run": len(res.history),
               "wall_s": time.perf_counter() - t0}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, **rec}) + "\n")
            self._cache[key] = {"key": key, **rec}
        self.trials.append({**rec, "trial": len(self.trials), "cached": False})
        return 1.0 - f1

    def __call__(self, Xc: np.ndarray) -> np.ndarray:
        return np.array([self.one(x) for x in np.atleast_2d(Xc)])
