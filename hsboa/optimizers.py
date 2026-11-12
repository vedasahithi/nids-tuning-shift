"""Population optimisers with one shared interface.

Every optimiser has the signature

    optimize(f, lb, ub, pop_size, max_evals, rng, callback=None, **params) -> Result

* ``f`` takes a batch X of shape (n, D) and returns n fitness values (minimised).
  Use :func:`batched` to wrap a function of one vector.
* ``lb``, ``ub`` are arrays of length D (they define the dimension).
* ``max_evals`` is a hard budget. Initialisation counts. The last iteration
  is truncated if needed, so every method uses exactly the same number of
  evaluations; unevaluated trial points get +inf and are rejected.
* All randomness comes from ``rng`` (a ``numpy.random.Generator`` or a seed).
* Candidates are clipped to the bounds before they are evaluated.
* ``callback(t, X, fit, best_f)`` is called after each iteration.

Selection follows each method's source. SBOA, HSBOA, POA and DE use greedy
(one-to-one) selection. PSO and CSA keep greedy personal-best / memory
archives and move the particles or crows unconditionally. GWO is canonical
(no greedy step, leaders are the three best points found so far).

Sources and default parameters:
    SBOA  Fu, Liu, Liu & Wang 2024, Artif. Intell. Rev. 57:123 (ref. [4]);
          stages follow the authors' MATLAB code.
    HSBOA SBOA + harmonic adjustment H in the escape step (see
          hsboa.harmonic). Ablation variants: "noise", "momentum",
          "shuffled" (see :func:`hsboa`).
    PSO   Clerc & Kennedy 2002 constriction, w = 0.7298, c1 = c2 = 1.49618.
    DE    Storn & Price 1997, DE/rand/1/bin, F = 0.5, CR = 0.9.
    GWO   Mirjalili, Mirjalili & Lewis 2014, a: 2 -> 0 linearly.
    POA   Trojovsky & Dehghani 2022 (pelican), R = 0.2.
    CSA   Askarzadeh 2016 (crow search), AP = 0.1, fl = 2.
    random  Uniform random search: the standard control for hyper-parameter tuning.

SBOA notes. The iteration count T used in t/T is the number of iterations
the budget allows (two evaluations per bird per iteration). As in the
authors' code, the escape branch (r < 0.5) and the random bird are drawn
once per iteration, and K = round(1 + rand) per bird. Updates are
synchronous: the best bird is refreshed after each phase rather than after
each bird. Several SBOA, POA and HSBOA moves multiply by x itself, which
pulls the search toward the origin. Benchmarks must therefore use shifted
functions (hsboa.benchmarks) or the comparison favours these methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from math import gamma, pi, sin
from typing import Callable

import numpy as np
from scipy.spatial.distance import pdist

from hsboa.harmonic import HarmonicConfig, HistoryBuffer, harmonic_adjustment, kappa_schedule

HSBOA_VARIANTS = ("harmonic", "noise", "momentum", "shuffled")


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    best_x: np.ndarray
    best_f: float
    n_evals: int
    iterations: int
    history: np.ndarray                 # best-so-far after every evaluation, length n_evals
    diversity: np.ndarray               # mean pairwise distance (normalised), per iteration incl. t = 0
    spread: np.ndarray                  # mean per-dimension std (normalised), per iteration incl. t = 0
    extra: dict = field(default_factory=dict)


class Budget:
    """Counts evaluations, enforces ``max_evals`` and records the best-so-far trace."""

    def __init__(self, f: Callable, max_evals: int):
        self.f, self.max_evals, self.n = f, int(max_evals), 0
        self.trace = np.empty(self.max_evals)
        self.best_f, self.best_x = np.inf, None

    @property
    def exhausted(self) -> bool:
        return self.n >= self.max_evals

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        k = min(len(X), self.max_evals - self.n)
        out = np.full(len(X), np.inf)
        if k <= 0:
            return out
        vals = np.asarray(self.f(X[:k]), dtype=float).reshape(k)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        out[:k] = vals
        self.trace[self.n:self.n + k] = np.minimum.accumulate(np.r_[self.best_f, vals])[1:]
        j = int(np.argmin(vals))
        if vals[j] < self.best_f:
            self.best_f, self.best_x = float(vals[j]), X[j].copy()
        self.n += k
        return out


def batched(f: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], np.ndarray]:
    """Turn a function of one vector into a batch function of shape (n, D) -> (n,)."""
    return lambda X: np.array([f(x) for x in X], dtype=float)


def diversity(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> tuple[float, float]:
    """(mean pairwise distance / sqrt(D), mean per-dimension std), both on [0, 1]-scaled coordinates."""
    Z = (X - lb) / (ub - lb)
    pair = pdist(Z).mean() / np.sqrt(Z.shape[1]) if len(Z) > 1 else 0.0
    return float(pair), float(Z.std(axis=0).mean())


def levy(rng: np.random.Generator, shape, beta: float = 1.5, scale: float = 0.01) -> np.ndarray:
    """Mantegna's Levy-flight step, as in SBOA Eq. (11)."""
    sigma = (gamma(1 + beta) * sin(pi * beta / 2)
             / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(shape) * sigma
    v = rng.standard_normal(shape)
    return scale * u / np.abs(v) ** (1 / beta)


def _iterations(max_evals: int, pop_size: int, evals_per_bird: int) -> int:
    """Iterations the budget allows after initialisation (the last one may be partial)."""
    return max(1, -(-(max_evals - pop_size) // (evals_per_bird * pop_size)))


class _Run:
    """Set-up and bookkeeping common to all optimisers."""

    def __init__(self, name, f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird, x0=None):
        self.name = name
        self.lb = np.asarray(lb, dtype=float).ravel()
        self.ub = np.asarray(ub, dtype=float).ravel()
        if self.lb.shape != self.ub.shape or np.any(self.ub <= self.lb):
            raise ValueError("lb and ub must have the same length and lb < ub")
        if max_evals < pop_size:
            raise ValueError("max_evals must cover at least the initial population")
        self.D, self.N = len(self.lb), int(pop_size)
        self.rng = np.random.default_rng(rng)
        self.budget = Budget(f, max_evals)
        self.callback = callback
        self.T = _iterations(max_evals, self.N, evals_per_bird)
        self.div, self.spr = [], []
        self.t = 0
        self.x0 = None if x0 is None else np.clip(np.asarray(x0, dtype=float).ravel(), self.lb, self.ub)
        if self.x0 is not None and self.x0.shape != self.lb.shape:
            raise ValueError("x0 must have the same length as lb")

    def init(self) -> tuple[np.ndarray, np.ndarray]:
        X = self.rng.uniform(self.lb, self.ub, (self.N, self.D))
        if self.x0 is not None:                 # optional known candidate (for example the default settings)
            X[0] = self.x0
        fit = self.budget(X)
        self.log(X, fit)
        return X, fit

    def clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lb, self.ub)

    def greedy(self, X, fit, X_new):
        f_new = self.budget(X_new)
        better = f_new <= fit
        X, fit = X.copy(), fit.copy()
        X[better], fit[better] = X_new[better], f_new[better]
        return X, fit

    def log(self, X, fit) -> None:
        d, s = diversity(X, self.lb, self.ub)
        self.div.append(d)
        self.spr.append(s)
        if self.callback is not None and self.t > 0:
            self.callback(self.t, X, fit, self.budget.best_f)

    def iterations(self):
        for t in range(1, self.T + 1):
            if self.budget.exhausted:
                return
            self.t = t
            yield t

    def result(self, **extra) -> Result:
        b = self.budget
        return Result(self.name, b.best_x, b.best_f, b.n, self.t, b.trace[:b.n].copy(),
                      np.array(self.div), np.array(self.spr), extra)


# ---------------------------------------------------------------------------
# SBOA and HSBOA
# ---------------------------------------------------------------------------

class _Windows:
    """Read-only stand-in for HistoryBuffer that serves pre-computed windows."""

    def __init__(self, ready: np.ndarray, windows: np.ndarray):
        self.ready, self._w = ready, windows

    def windows(self) -> np.ndarray:
        return self._w


class _Perturbation:
    """The term added to the SBOA escape move: harmonic H or an ablation control."""

    def __init__(self, variant: str, cfg: HarmonicConfig, N: int, D: int):
        if variant not in HSBOA_VARIANTS:
            raise ValueError(f"variant must be one of {HSBOA_VARIANTS}")
        self.variant, self.cfg = variant, cfg
        self.buffer = HistoryBuffer(N, D, cfg.L)

    def __call__(self, X, t, T, rng) -> np.ndarray:
        buf, cfg = self.buffer, self.cfg
        if not buf.ready.any():
            return np.zeros_like(X)        # warm-up: no rng draws, so every variant equals SBOA here
        if self.variant == "harmonic":
            return harmonic_adjustment(buf, X, t, T, cfg)
        if self.variant == "noise":
            # Same per-bird norm as H, random direction: tests whether any kick of that size helps.
            H = harmonic_adjustment(buf, X, t, T, cfg)
            Z = rng.standard_normal(X.shape)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            return Z * np.linalg.norm(H, axis=1, keepdims=True)
        if self.variant == "momentum":
            # Two-lag term kappa * (x(t-1) - x(t-2)), active under the same warm-up as H.
            H = np.zeros_like(X)
            w, ready = buf.windows(), buf.ready
            H[ready] = kappa_schedule(t, T, cfg.kappa0, cfg.schedule) * (w[ready, -1] - w[ready, -2])
            return H
        # "shuffled": the same harmonic fit on a time-permuted history (destroys temporal order).
        w = buf.windows()
        perm = np.argsort(rng.random((len(w), cfg.L)), axis=1)
        w = np.take_along_axis(w, perm[:, :, None], axis=1)
        return harmonic_adjustment(_Windows(buf.ready, w), X, t, T, cfg)


def _sboa_core(name, f, lb, ub, pop_size, max_evals, rng, callback, perturbation=None, x0=None):
    run = _Run(name, f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=2, x0=x0)
    N, D, T, r = run.N, run.D, run.T, run.rng
    X, fit = run.init()
    pert = None
    if perturbation is not None:
        pert = _Perturbation(*perturbation, N, D)
        pert.buffer.push(X)
    h_norm, move_norm = [], []

    for t in run.iterations():
        # Hunting (exploration): three stages over the run.
        best = X[np.argmin(fit)]
        if t < T / 3:                                                   # searching for prey
            r1, r2 = r.integers(N, size=N), r.integers(N, size=N)
            X1 = X + (X[r1] - X[r2]) * r.random((N, D))
        elif t < 2 * T / 3:                                             # consuming prey
            X1 = best + np.exp((t / T) ** 4) * (r.standard_normal((N, D)) - 0.5) * (best - X)
        else:                                                           # attacking prey
            CF = (1 - t / T) ** (2 * t / T)
            X1 = best + CF * X * (0.5 * levy(r, (N, D)))
        X, fit = run.greedy(X, fit, run.clip(X1))

        # Escape (exploitation): camouflage or run away.
        best = X[np.argmin(fit)]
        x_rand = X[r.integers(N)]
        if r.random() < 0.5:
            X2 = best + (1 - t / T) ** 2 * (2 * r.random((N, D)) - 1) * X
        else:
            K = np.round(1 + r.random((N, 1)))
            X2 = X + r.random((N, D)) * (x_rand - K * X)
        if pert is not None:
            H = pert(X, t, T, r)
            h_norm.append(float(np.linalg.norm(H, axis=1).mean()))
            move_norm.append(float(np.linalg.norm(X2 - X, axis=1).mean()))
            X2 = X2 + H
        X, fit = run.greedy(X, fit, run.clip(X2))
        if pert is not None:
            pert.buffer.push(X)
        run.log(X, fit)

    extra = {} if pert is None else {"h_norm": np.array(h_norm), "move_norm": np.array(move_norm)}
    return run.result(**extra)


def sboa(f, lb, ub, pop_size, max_evals, rng, callback=None, x0=None) -> Result:
    """Secretary Bird Optimisation Algorithm (Fu et al. 2024); ``x0`` optionally replaces the first bird."""
    return _sboa_core("SBOA", f, lb, ub, pop_size, max_evals, rng, callback, x0=x0)


def hsboa(f, lb, ub, pop_size, max_evals, rng, callback=None,
          harmonic: HarmonicConfig | dict | None = None, variant: str = "harmonic") -> Result:
    """SBOA plus a history-based term added to the escape move before clipping.

    variant:
        "harmonic"  H from the causal harmonic forecast (the proposed method).
        "noise"     random direction with the same per-bird norm as H.
        "momentum"  kappa(t) * (x(t-1) - x(t-2)).
        "shuffled"  harmonic H computed on a time-shuffled history.
    All variants use the same kappa schedule and the same warm-up (L
    positions), so they differ only in the form of the term. ``extra``
    holds the mean norms of the term and of the SBOA escape move per
    iteration, for reporting their relative size.
    """
    cfg = harmonic if isinstance(harmonic, HarmonicConfig) else HarmonicConfig(**(harmonic or {}))
    name = "HSBOA" if variant == "harmonic" else f"HSBOA-{variant}"
    return _sboa_core(name, f, lb, ub, pop_size, max_evals, rng, callback, (variant, cfg))


# ---------------------------------------------------------------------------
# Comparison optimisers
# ---------------------------------------------------------------------------

def pso(f, lb, ub, pop_size, max_evals, rng, callback=None,
        w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618, vmax: float = 0.2) -> Result:
    """Global-best PSO with constriction weights; |v| <= vmax * (ub - lb)."""
    run = _Run("PSO", f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=1)
    N, D, r = run.N, run.D, run.rng
    X, fit = run.init()
    P, pf = X.copy(), fit.copy()
    vlim = vmax * (run.ub - run.lb)
    V = r.uniform(-vlim, vlim, (N, D))
    for _ in run.iterations():
        g = P[np.argmin(pf)]
        V = w * V + c1 * r.random((N, D)) * (P - X) + c2 * r.random((N, D)) * (g - X)
        V = np.clip(V, -vlim, vlim)
        X_new = X + V
        X = run.clip(X_new)
        V[X != X_new] = 0.0                                          # stop at the wall instead of sticking to it
        fx = run.budget(X)
        better = fx < pf
        P[better], pf[better] = X[better], fx[better]
        run.log(X, fx)
    return run.result()


def de(f, lb, ub, pop_size, max_evals, rng, callback=None,
       F: float = 0.5, CR: float = 0.9) -> Result:
    """Differential evolution, DE/rand/1/bin."""
    run = _Run("DE", f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=1)
    N, D, r = run.N, run.D, run.rng
    if N < 4:
        raise ValueError("DE/rand/1 needs pop_size >= 4")
    X, fit = run.init()
    rows = np.arange(N)[:, None]
    for _ in run.iterations():
        idx = np.argsort(r.random((N, N - 1)), axis=1)[:, :3]
        idx = idx + (idx >= rows)                                    # skip the target vector i
        V = X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]])
        cross = r.random((N, D)) < CR
        cross[np.arange(N), r.integers(D, size=N)] = True            # at least one gene from V
        X, fit = run.greedy(X, fit, run.clip(np.where(cross, V, X)))
        run.log(X, fit)
    return run.result()


def gwo(f, lb, ub, pop_size, max_evals, rng, callback=None) -> Result:
    """Grey wolf optimiser (canonical: no greedy step)."""
    run = _Run("GWO", f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=1)
    N, D, T, r = run.N, run.D, run.T, run.rng
    X, fit = run.init()
    order = np.argsort(fit)[:3]
    leaders, lf = X[order].copy(), fit[order].copy()
    for t in run.iterations():
        a = 2 * (1 - (t - 1) / T)
        moves = []
        for k in range(3):
            A = a * (2 * r.random((N, D)) - 1)
            C = 2 * r.random((N, D))
            moves.append(leaders[k] - A * np.abs(C * leaders[k] - X))
        X = run.clip(np.mean(moves, axis=0))
        fit = run.budget(X)
        pool, pf = np.vstack([leaders, X]), np.r_[lf, fit]
        order = np.argsort(pf, kind="stable")[:3]
        leaders, lf = pool[order].copy(), pf[order].copy()
        run.log(X, fit)
    return run.result()


def poa(f, lb, ub, pop_size, max_evals, rng, callback=None, R: float = 0.2) -> Result:
    """Pelican optimisation algorithm; the prey is a random pelican, as in the authors' code."""
    run = _Run("POA", f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=2)
    N, D, T, r = run.N, run.D, run.T, run.rng
    X, fit = run.init()
    for t in run.iterations():
        k = r.integers(N)
        prey, f_prey = X[k].copy(), fit[k]
        I = np.round(1 + r.random((N, 1)))
        step = r.random((N, 1))
        toward = (fit > f_prey)[:, None]
        X1 = np.where(toward, X + step * (prey - I * X), X + step * (X - prey))   # moving toward prey
        X, fit = run.greedy(X, fit, run.clip(X1))
        X2 = X + R * (1 - t / T) * (2 * r.random((N, D)) - 1) * X                # winging on the surface
        X, fit = run.greedy(X, fit, run.clip(X2))
        run.log(X, fit)
    return run.result()


def csa(f, lb, ub, pop_size, max_evals, rng, callback=None,
        AP: float = 0.1, fl: float = 2.0) -> Result:
    """Crow search algorithm with a greedy memory per crow."""
    run = _Run("CSA", f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=1)
    N, D, r = run.N, run.D, run.rng
    X, fit = run.init()
    M, mf = X.copy(), fit.copy()
    for _ in run.iterations():
        j = r.integers(N, size=N)
        follow = (r.random(N) >= AP)[:, None]
        chase = X + r.random((N, 1)) * fl * (M[j] - X)
        X = run.clip(np.where(follow, chase, r.uniform(run.lb, run.ub, (N, D))))
        fx = run.budget(X)
        better = fx < mf
        M[better], mf[better] = X[better], fx[better]
        run.log(X, fx)
    return run.result()


def random_search(f, lb, ub, pop_size, max_evals, rng, callback=None, x0=None) -> Result:
    """Uniform random sampling in batches of pop_size (one batch = one iteration); ``x0`` replaces the first point."""
    run = _Run("Random", f, lb, ub, pop_size, max_evals, rng, callback, evals_per_bird=1, x0=x0)
    X, fit = run.init()
    for _ in run.iterations():
        X = run.rng.uniform(run.lb, run.ub, (run.N, run.D))
        fit = run.budget(X)
        run.log(X, fit)
    return run.result()


OPTIMIZERS: dict[str, Callable[..., Result]] = {
    "sboa": sboa, "hsboa": hsboa, "pso": pso, "de": de, "gwo": gwo, "poa": poa, "csa": csa,
    "random": random_search,
}
# Population methods proper (excludes the random-search control).
METAHEURISTICS = tuple(k for k in OPTIMIZERS if k != "random")


def get_optimizer(name: str, **params) -> Callable[..., Result]:
    """Optimiser by name with fixed parameters, e.g. ``get_optimizer("hsboa", variant="noise")``."""
    try:
        return partial(OPTIMIZERS[name.lower()], **params)
    except KeyError:
        raise ValueError(f"unknown optimiser {name!r}; choose from {sorted(OPTIMIZERS)}") from None
