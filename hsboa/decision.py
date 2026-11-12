"""Decision-layer tuning (direction B, 2026-10-28): per-class biases on predicted log-probabilities.

The trained classifier stays fixed. Its predicted class becomes
argmax_c (log p_c + b_c), and the bias vector b in [-bound, bound]^K is
chosen to maximise macro-F1 on held-out predictions. One evaluation is an
argmax over stored probabilities (milliseconds, no retraining), so every
search method can afford thousands of evaluations on the full data, unlike
hyper-parameter tuning (E1, E1b), where each evaluation is a training run.

Search methods: any :mod:`hsboa.optimizers` method (HSBOA, SBOA, PSO, DE,
random search) and ``coord``, a cyclic coordinate search over a grid of
bias values (the usual simple baseline for per-class threshold tuning).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
METHODS = ("hsboa", "sboa", "pso", "de", "random", "coord")


def log_proba(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(P, dtype=float), EPS, 1.0))


def apply_bias(L: np.ndarray, b) -> np.ndarray:
    """Predicted class indices from log-probabilities L (n, K) and biases b (K,)."""
    return (L + np.asarray(b, dtype=float)).argmax(axis=1)


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Plain macro-F1 over all k classes (absent classes count 0), as hsboa.metrics.evaluate."""
    cm = np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = (cm.sum(axis=1) + cm.sum(axis=0)).astype(float)          # 2tp + fp + fn
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)))


def per_class_f1(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = (cm.sum(axis=1) + cm.sum(axis=0)).astype(float)
    return np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)


class BiasObjective:
    """Batch objective for the optimisers: rows of X are bias vectors, value = 1 - macro-F1."""

    def __init__(self, L: np.ndarray, y: np.ndarray, k: int):
        self.L, self.y, self.k, self.n_evals = np.asarray(L, dtype=float), np.asarray(y), int(k), 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        self.n_evals += len(X)
        return np.array([1.0 - macro_f1(self.y, apply_bias(self.L, b), self.k) for b in X])


def coordinate_search(f, k: int, bound: float, max_evals: int, n_grid: int = 25) -> tuple[np.ndarray, float, int]:
    """Cyclic coordinate search from b = 0: per class, try n_grid values and keep the best.

    Stops when the budget is used or a full sweep brings no strict improvement.
    Returns (best b, best value, evaluations used).
    """
    b = np.zeros(k)
    best = float(f(b[None])[0])
    n, grid = 1, np.linspace(-bound, bound, n_grid)
    improved = True
    while improved and n < max_evals:
        improved = False
        for j in range(k):
            m = min(n_grid, max_evals - n)
            if m <= 0:
                break
            cand = np.repeat(b[None], m, axis=0)
            cand[:, j] = grid[:m]
            vals = np.asarray(f(cand), dtype=float)
            n += m
            i = int(np.argmin(vals))
            if vals[i] < best - 1e-12:
                best, b, improved = float(vals[i]), cand[i].copy(), True
    return b, best, n


def tune_bias(method: str, L: np.ndarray, y: np.ndarray, k: int, *, max_evals: int, pop_size: int,
              bound: float, seed, optimizer_params: dict | None = None) -> dict:
    """Search biases on (L, y). Returns {"b", "macro_f1", "n_evals"} (macro-F1 on the data searched)."""
    f = BiasObjective(L, y, k)
    if method == "coord":
        b, v, _ = coordinate_search(f, k, bound, max_evals)
    else:
        from hsboa.optimizers import get_optimizer
        res = get_optimizer(method, **(optimizer_params or {}))(f, np.full(k, -bound), np.full(k, bound),
                                                                pop_size, max_evals, seed)
        b, v = res.best_x, res.best_f
        base = float(f(np.zeros((1, k)))[0])               # never return worse than the plain argmax
        if base <= v:
            b, v = np.zeros(k), base
    return {"b": np.asarray(b, dtype=float), "macro_f1": 1.0 - float(v), "n_evals": f.n_evals}
