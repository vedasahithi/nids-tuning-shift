"""Step-by-step numerical examples of the harmonic term.

Usage (from the repository root):
    python scripts/harmonic_example.py

Part A  Two-point example g = (0, 2): a 2/i vs the correct 1/i Nyquist coefficient.
Part B  The corrected two-point model is a replay of x(t-1) (1-D escape step).
Part C  The causal L-point update used by HSBOA, one bird and one dimension.
Part C uses an illustrative trajectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hsboa.harmonic import (HarmonicConfig, forecast_weights, harmonic_fit,  # noqa: E402
                            kappa_schedule, trig_interpolate,
                            trig_interpolation_coefficients)


def part_a() -> None:
    print("A. Two-point example, g = (0, 2), i = 2")
    g1, g2 = 0.0, 2.0
    z0 = (g1 + g2) / 2
    z1_naive = g2 - g1
    print(f"   2/i       z0={z0:.3f} z1=(2/i)*sum = {z1_naive:.3f} -> reconstruct "
          f"{[round(float(z0 + z1_naive * np.cos(np.pi * y)), 3) for y in (1, 2)]}")
    a, b = trig_interpolation_coefficients([g1, g2])
    print(f"   1/i       z0={a[0]:.3f} z1=(1/i)*sum = {a[1]:.3f} -> reconstruct "
          f"{trig_interpolate(a, b, 2, [1, 2]).round(3).tolist()}\n")


def part_b() -> None:
    print("B. Two-point model inside the SBOA escape step (1-D)")
    x_prev, x_t, x_rand, omega2, gamma, kappa = 0.20, 0.60, -0.30, 0.4, 1.0, 0.1
    x_sboa = x_t + omega2 * (x_rand - gamma * x_t)
    a, b = trig_interpolation_coefficients([x_prev, x_t])
    g_hat = trig_interpolate(a, b, 2, [3]).item()
    H = kappa * (g_hat - x_t)
    print(f"   SBOA move: {x_t} + {omega2}*({x_rand} - {gamma}*{x_t}) = {x_sboa:.3f}")
    print(f"   two-point forecast g(t+1) = {g_hat:.3f}  (= x(t-1): a replay, no periodic structure)")
    print(f"   H = {kappa}*({g_hat:.3f} - {x_t}) = {H:.3f};  x_new = {x_sboa + H:.3f}\n")


def part_c() -> None:
    cfg = HarmonicConfig(L=10, F=2, kappa0=0.1, schedule="linear")
    t, T = 40, 100
    # Illustrative accepted positions x(t-9)..x(t) of one bird in one dimension.
    history = np.array([0.52, 0.47, 0.55, 0.49, 0.58, 0.51, 0.60, 0.54, 0.63, 0.57])
    x_t = history[-1]
    x_rand, omega2, gamma = 0.20, 0.4, 1.0
    x_sboa = x_t + omega2 * (x_rand - gamma * x_t)
    P = cfg.period or 2 * cfg.L
    w = forecast_weights(cfg.L, cfg.F, P)
    g_hat, c0 = harmonic_fit(history, cfg.F, P)
    kappa = kappa_schedule(t, T, cfg.kappa0, cfg.schedule)
    H = kappa * (g_hat - x_t)
    print(f"C. Causal L-point update: L={cfg.L}, F={cfg.F}, P={P:g}, kappa0={cfg.kappa0}, "
          f"{cfg.schedule} schedule, t={t}/{T}")
    print(f"   history x(t-9..t): {history.tolist()}")
    print(f"   forecast weights w (oldest..newest, sum={w.sum():.3f}): {w.round(3).tolist()}")
    print(f"   g_hat(t+1) = w @ history = {g_hat:.4f}   (fitted mean c0 = {c0:.4f})")
    print(f"   kappa(t) = {cfg.kappa0}*(1 - {t}/{T}) = {kappa:.4f}")
    print(f"   SBOA move: {x_t} + {omega2}*({x_rand} - {gamma}*{x_t}) = {x_sboa:.4f}")
    print(f"   H = {kappa:.4f}*({g_hat:.4f} - {x_t}) = {H:.4f}")
    print(f"   x_new = clip(x_SBOA + H) = {x_sboa + H:.4f}  (then greedy selection)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
