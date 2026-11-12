"""Benchmark functions for the optimiser comparison and the ablation study.

All functions take a batch X of shape (n, D) and return n values, with
minimum 0. Several SBOA/POA/HSBOA moves multiply by x, which biases the
search toward the origin. That is exactly where the unshifted optimum of
most classical functions lies, so the ablation uses CEC-style variants:
f(M (x - o)) with a random shift o (inside 80% of the box) and, optionally,
a random rotation M. The official CEC 2017 data files are not bundled; the
shifted/rotated classical functions follow the same construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import special_ortho_group


def sphere(X):
    return np.sum(X ** 2, axis=1)


def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)


def ackley(X):
    D = X.shape[1]
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, axis=1) / D))
            - np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / D) + 20 + np.e)


def rosenbrock(X):
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2, axis=1)


def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return 1 + np.sum(X ** 2, axis=1) / 4000 - np.prod(np.cos(X / np.sqrt(i)), axis=1)


def schwefel_222(X):
    A = np.abs(X)
    return np.sum(A, axis=1) + np.prod(A, axis=1)


def schwefel_226(X):
    """Schwefel 2.26; optimum near x_i = 420.9687, away from the origin even unshifted."""
    return 418.9828872724338 * X.shape[1] - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


@dataclass(frozen=True)
class Benchmark:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    bound: float                        # search box [-bound, bound]^D
    optimum: np.ndarray | None = None   # location of the minimum in the transformed space

    def bounds(self, D: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(D, -self.bound), np.full(D, self.bound)

    def __call__(self, X):
        return self.f(np.atleast_2d(X))


# Functions that make_benchmark never shifts or rotates (their optimum is already off-origin).
UNTRANSFORMED = ("schwefel_226",)

# name -> (function, box half-width, unshifted optimum value of x_i)
BASE = {
    "sphere": (sphere, 100.0, 0.0),
    "schwefel_222": (schwefel_222, 10.0, 0.0),
    "rosenbrock": (rosenbrock, 30.0, 1.0),
    "rastrigin": (rastrigin, 5.12, 0.0),
    "ackley": (ackley, 32.0, 0.0),
    "griewank": (griewank, 600.0, 0.0),
    "schwefel_226": (schwefel_226, 500.0, 420.9687463),
}


def make_benchmark(name: str, D: int, rng=None, shift: bool = True,
                   rotate: bool = False) -> Benchmark:
    """Benchmark ``name`` in D dimensions, optionally shifted and rotated (seeded by ``rng``).

    Schwefel 2.26 is never shifted or rotated: its optimum already lies near the box edge.
    """
    f, bound, x_opt = BASE[name]
    rng = np.random.default_rng(rng)
    if name in UNTRANSFORMED or not (shift or rotate):
        return Benchmark(name, f, bound, np.full(D, x_opt))
    o = rng.uniform(-0.8 * bound, 0.8 * bound, D) if shift else np.zeros(D)
    M = special_ortho_group.rvs(D, random_state=rng) if rotate and D > 1 else np.eye(D)
    # Rosenbrock's minimum is at 1; shifting by o - 1 puts it at o.
    base_shift = o - x_opt if shift else np.zeros(D)

    def g(X, f=f, o=o, M=M, s=base_shift, x_opt=x_opt):
        return f((X - o) @ M.T + x_opt) if rotate else f(X - s)

    tag = ("shifted_" if shift else "") + ("rotated_" if rotate else "")
    return Benchmark(tag + name, g, bound, o)


def suite(D: int, rng=None, shift: bool = True, rotate: bool = False) -> list[Benchmark]:
    """All benchmarks in BASE, each with its own transform drawn from ``rng``."""
    rng = np.random.default_rng(rng)
    return [make_benchmark(n, D, rng, shift, rotate) for n in BASE]
