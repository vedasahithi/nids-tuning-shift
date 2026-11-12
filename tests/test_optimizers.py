"""Tests for the shared optimiser interface."""

import numpy as np
import pytest

from hsboa.benchmarks import BASE, make_benchmark, rastrigin, sphere, suite
from hsboa.harmonic import HarmonicConfig
from hsboa.optimizers import (HSBOA_VARIANTS, METAHEURISTICS, OPTIMIZERS, Budget, batched, get_optimizer,
                              hsboa, levy, sboa)

D, N, BUDGET = 10, 20, 6000


def _random_search(f, lb, ub, n, seed=0):
    X = np.random.default_rng(seed).uniform(lb, ub, (n, len(lb)))
    return f(X).min()


# --- interface: budget, bounds, reproducibility -----------------------------

@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_exact_budget_and_trace(name):
    lb, ub = np.full(D, -5.0), np.full(D, 5.0)
    budget = 1013                                       # not a multiple of the evals per iteration
    res = OPTIMIZERS[name](sphere, lb, ub, N, budget, 0)
    assert res.n_evals == budget
    assert len(res.history) == budget
    assert np.all(np.diff(res.history) <= 0)            # best-so-far never gets worse
    assert res.best_f == pytest.approx(res.history[-1])
    assert sphere(res.best_x[None])[0] == pytest.approx(res.best_f)
    assert len(res.diversity) == res.iterations + 1 == len(res.spread)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_evaluated_point_is_inside_the_bounds(name):
    lb, ub = np.full(D, -1.0), np.full(D, 2.0)
    seen = []

    def f(X):
        seen.append(X.copy())
        return sphere(X - 1.9)                          # optimum near a bound to provoke violations

    OPTIMIZERS[name](f, lb, ub, N, 2000, 1)
    allX = np.vstack(seen)
    assert np.all(allX >= lb) and np.all(allX <= ub)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_same_seed_same_result(name):
    lb, ub = np.full(D, -5.12), np.full(D, 5.12)
    a = OPTIMIZERS[name](rastrigin, lb, ub, N, 1500, 7)
    b = OPTIMIZERS[name](rastrigin, lb, ub, N, 1500, 7)
    c = OPTIMIZERS[name](rastrigin, lb, ub, N, 1500, 8)
    assert np.array_equal(a.history, b.history) and np.array_equal(a.best_x, b.best_x)
    assert not np.array_equal(a.history, c.history)


def test_budget_truncates_and_rejects_unevaluated_points():
    b = Budget(sphere, 5)
    out = b(np.ones((8, 2)))
    assert b.n == 5 and b.exhausted
    assert np.all(np.isfinite(out[:5])) and np.all(np.isinf(out[5:]))
    assert np.all(np.isinf(b(np.zeros((3, 2)))))        # nothing more is evaluated
    assert b.best_f == 2.0


def test_nan_fitness_is_treated_as_worst():
    b = Budget(lambda X: np.array([np.nan, 1.0]), 2)
    assert b(np.zeros((2, 1))).tolist() == [np.inf, 1.0]


def test_batched_wrapper_and_registry():
    f = batched(lambda x: float(np.sum(x ** 2)))
    assert f(np.array([[1.0, 2.0], [0.0, 0.0]])).tolist() == [5.0, 0.0]
    with pytest.raises(ValueError, match="unknown optimiser"):
        get_optimizer("abc")
    with pytest.raises(ValueError):
        sboa(sphere, np.zeros(2), np.ones(2), 10, 5, 0)   # budget below the initial population


def test_levy_is_heavy_tailed():
    s = levy(np.random.default_rng(0), 200_000)
    assert np.median(np.abs(s)) < 0.02                    # scale 0.01
    assert np.abs(s).max() > 100 * np.median(np.abs(s))   # occasional long jumps


# --- optimisation quality: Sphere and Rastrigin, shifted -------------------
# Sanity bounds that catch broken code, not weak algorithms: POA's moves scale
# with x and stall on shifted functions (about 0.55 x random search on shifted
# Rastrigin at this budget). Relative performance belongs in the benchmark study.

@pytest.mark.parametrize("name", sorted(METAHEURISTICS))
def test_shifted_sphere_beats_random_search(name):
    bm = make_benchmark("sphere", D, rng=3)
    lb, ub = bm.bounds(D)
    res = OPTIMIZERS[name](bm, lb, ub, N, BUDGET, 0)
    rs = _random_search(bm, lb, ub, BUDGET)
    assert res.best_f < 0.1 * rs, (res.best_f, rs)


@pytest.mark.parametrize("name", sorted(METAHEURISTICS))
def test_shifted_rastrigin_beats_random_search(name):
    bm = make_benchmark("rastrigin", D, rng=3)
    lb, ub = bm.bounds(D)
    res = OPTIMIZERS[name](bm, lb, ub, N, BUDGET, 0)
    assert res.best_f < 0.75 * _random_search(bm, lb, ub, BUDGET)


def test_methods_solve_shifted_sphere():
    bm = make_benchmark("sphere", D, rng=3)
    lb, ub = bm.bounds(D)
    for name in ("sboa", "hsboa", "pso"):
        assert OPTIMIZERS[name](bm, lb, ub, N, BUDGET, 0).best_f < 1e-3, name


def test_origin_bias_of_x_scaled_moves():
    # Documents the bias: GWO and POA are many orders of magnitude better when the
    # optimum is at the origin. This is why benchmarks must be shifted.
    for name in ("gwo", "poa"):
        plain = make_benchmark("sphere", D, shift=False)
        moved = make_benchmark("sphere", D, rng=3)
        lb, ub = plain.bounds(D)
        a = OPTIMIZERS[name](plain, lb, ub, N, BUDGET, 0).best_f
        b = OPTIMIZERS[name](moved, lb, ub, N, BUDGET, 0).best_f
        assert a < 1e-10 * b, (name, a, b)


# --- HSBOA vs SBOA ----------------------------------------------------------

def test_hsboa_with_zero_kappa_is_exactly_sboa():
    bm = make_benchmark("rastrigin", D, rng=1)
    lb, ub = bm.bounds(D)
    a = sboa(bm, lb, ub, N, 2000, 5)
    b = hsboa(bm, lb, ub, N, 2000, 5, harmonic=HarmonicConfig(kappa0=0.0))
    assert np.array_equal(a.history, b.history) and np.array_equal(a.best_x, b.best_x)


def test_hsboa_equals_sboa_during_warm_up():
    # With L = 10 the harmonic term is zero for the first 9 iterations (2 * N evals each).
    bm = make_benchmark("sphere", D, rng=1)
    lb, ub = bm.bounds(D)
    a = sboa(bm, lb, ub, N, 4000, 2)
    b = hsboa(bm, lb, ub, N, 4000, 2, harmonic=HarmonicConfig(L=10, F=2, kappa0=1.0))
    warm = N + 2 * N * 9
    assert np.array_equal(a.history[:warm], b.history[:warm])
    assert not np.array_equal(a.history, b.history)
    assert np.all(b.extra["h_norm"][:9] == 0) and np.any(b.extra["h_norm"][9:] > 0)


@pytest.mark.parametrize("variant", HSBOA_VARIANTS)
def test_ablation_variants_share_warm_up_and_size_logging(variant):
    bm = make_benchmark("sphere", D, rng=1)
    lb, ub = bm.bounds(D)
    res = hsboa(bm, lb, ub, N, 3000, 0, harmonic={"kappa0": 0.5}, variant=variant)
    assert res.n_evals == 3000
    assert len(res.extra["h_norm"]) == len(res.extra["move_norm"]) == res.iterations
    assert np.all(res.extra["h_norm"][:9] == 0)


def test_noise_variant_matches_harmonic_norm():
    # Same seed and history up to the first active iteration -> the same per-bird norm.
    bm = make_benchmark("sphere", D, rng=1)
    lb, ub = bm.bounds(D)
    cfg = HarmonicConfig(kappa0=0.5)
    h = hsboa(bm, lb, ub, N, N + 2 * N * 10, 0, harmonic=cfg)
    n = hsboa(bm, lb, ub, N, N + 2 * N * 10, 0, harmonic=cfg, variant="noise")
    assert h.extra["h_norm"][9] == pytest.approx(n.extra["h_norm"][9])


def test_invalid_variant():
    with pytest.raises(ValueError, match="variant"):
        hsboa(sphere, np.zeros(2), np.ones(2), 10, 100, 0, variant="xyz")


# --- benchmarks -------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(BASE))
def test_benchmark_minimum_is_zero_at_reported_optimum(name):
    for shift, rotate in ((False, False), (True, False), (True, True)):
        bm = make_benchmark(name, 5, rng=0, shift=shift, rotate=rotate)
        lb, ub = bm.bounds(5)
        assert np.all(bm.optimum >= lb) and np.all(bm.optimum <= ub)
        assert bm(bm.optimum)[0] == pytest.approx(0.0, abs=1e-4)
        X = np.random.default_rng(1).uniform(lb, ub, (50, 5))
        assert np.all(bm(X) >= -1e-4)


def test_suite_shifts_differ_between_functions():
    s = suite(5, rng=0)
    assert len(s) == len(BASE)
    shifted = [b for b in s if b.name.startswith("shifted_")]
    assert not np.allclose(shifted[0].optimum, shifted[1].optimum)


def test_x0_seeds_the_first_candidate_and_default_is_unchanged():
    f = lambda X: np.sum((np.atleast_2d(X) - 0.3) ** 2, axis=1)
    lb, ub = np.zeros(4), np.ones(4)
    for name in ("sboa", "random"):
        plain = get_optimizer(name)(f, lb, ub, 6, 30, 5)
        again = get_optimizer(name)(f, lb, ub, 6, 30, 5)
        assert np.array_equal(plain.history, again.history)
        seeded = get_optimizer(name, x0=np.full(4, 0.3))(f, lb, ub, 6, 30, 5)
        assert seeded.history[0] == 0.0 and seeded.history[-1] == 0.0
