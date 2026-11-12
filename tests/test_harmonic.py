"""Tests of the harmonic term: (a) reconstruction of the input points, (b) exact recovery
of a known sinusoid, (c) the two-point example g = (0, 2), and the causal update.
"""

import numpy as np
import pytest

from hsboa.harmonic import (HarmonicConfig, HistoryBuffer, forecast_weights,
                            harmonic_adjustment, harmonic_fit, harmonic_forecast,
                            kappa_schedule, trig_interpolate,
                            trig_interpolation_coefficients)


# --- (a) reconstruction of the input points, even and odd lengths -----

@pytest.mark.parametrize("i", [1, 2, 3, 4, 5, 7, 8, 12])
def test_interpolation_reconstructs_input_points(i):
    g = np.random.default_rng(i).normal(size=i)
    a, b = trig_interpolation_coefficients(g)
    assert np.allclose(trig_interpolate(a, b, i, np.arange(1, i + 1)), g)


# --- (c) the two-point example g = (0, 2) -----------------------------

def test_two_point_example_g_0_2():
    a, b = trig_interpolation_coefficients([0.0, 2.0])
    assert a.tolist() == pytest.approx([1.0, 1.0])      # z0 = (g1+g2)/2, z1 = (g2-g1)/2
    assert trig_interpolate(a, b, 2, [1, 2]).tolist() == pytest.approx([0.0, 2.0])
    # A 2/i Nyquist coefficient (z1 = g2 - g1) gives -1 and 3.
    z0, z1_naive = 1.0, 2.0
    assert [z0 + z1_naive * np.cos(np.pi * y) for y in (1, 2)] == pytest.approx([-1.0, 3.0])


def test_two_point_model_only_replays_the_previous_value():
    # Two-point model: g(y+1) = g(y-1). This is the triviality behind
    # the move to an L-point regression.
    g1, g2 = 0.2, 0.6
    a, b = trig_interpolation_coefficients([g1, g2])
    assert trig_interpolate(a, b, 2, [3]).item() == pytest.approx(g1)


# --- (b) exact recovery of a known sinusoid ---------------------------

def test_fixed_frequency_regression_recovers_sinusoid_and_forecasts_exactly():
    L, P = 12, 24
    s = np.arange(1, L + 2)
    signal = 0.5 + 0.8 * np.cos(2 * np.pi * s / P + 0.3)
    g_hat, c0 = harmonic_fit(signal[:L], F=1, period=P)
    assert c0 == pytest.approx(0.5)
    assert g_hat == pytest.approx(signal[L])


def test_periodogram_regression_recovers_unknown_frequency():
    L = 32
    s = np.arange(1, L + 2)
    signal = -1.0 + 0.7 * np.sin(2 * np.pi * s / 6.4)   # period not tied to L
    g_hat = harmonic_forecast(signal[:L], F=1, frequencies="periodogram")
    assert g_hat == pytest.approx(signal[L], abs=0.05)


# --- causal regression properties --------------------------------------------

def test_forecast_is_a_linear_filter_whose_weights_sum_to_one():
    L, F = 10, 2
    w = forecast_weights(L, F)
    assert w.sum() == pytest.approx(1.0)
    g = np.random.default_rng(0).normal(size=L)
    assert harmonic_forecast(g, F) == pytest.approx(w @ g)
    assert harmonic_forecast(np.full(L, 3.3), F) == pytest.approx(3.3)


def test_period_equal_to_window_wraps_around_to_the_oldest_point():
    # Why the default period is 2L: with P = L the forecast for s = L+1 equals
    # the fitted value at s = 1.
    L, F = 10, 2
    g = np.random.default_rng(1).normal(size=L)
    from hsboa.harmonic import design_matrix
    nu = np.arange(1, F + 1) / L
    A = design_matrix(np.arange(1, L + 1), nu)
    fitted_first = (A @ np.linalg.lstsq(A, g, rcond=None)[0])[0]
    assert harmonic_forecast(g, F, period=L) == pytest.approx(fitted_first)
    assert harmonic_forecast(g, F) != pytest.approx(fitted_first)


def test_columns_are_forecast_independently():
    rng = np.random.default_rng(2)
    win = rng.normal(size=(10, 4))
    batch = harmonic_forecast(win, F=2)
    loop = [harmonic_forecast(win[:, j], F=2) for j in range(4)]
    assert batch.tolist() == pytest.approx(loop)


def test_smoothing_constraint_enforced():
    with pytest.raises(ValueError, match="2F\\+1 < L"):
        harmonic_forecast(np.zeros(5), F=2)
    with pytest.raises(ValueError):
        HarmonicConfig(L=4, F=2)


# --- HSBOA adjustment ----------------------------------------------------------

def test_kappa_schedules():
    assert kappa_schedule(0, 100, 0.1) == pytest.approx(0.1)
    assert kappa_schedule(50, 100, 0.1) == pytest.approx(0.05)
    assert kappa_schedule(100, 100, 0.1) == 0.0
    assert kappa_schedule(70, 100, 0.1, "constant") == 0.1
    assert kappa_schedule(50, 100, 0.1, "cosine") == pytest.approx(0.05)


def test_history_buffer_order_and_warmup():
    buf = HistoryBuffer(n_birds=2, dim=1, length=3)
    for t in range(1, 3):
        buf.push(np.array([[t], [10 * t]]))
    assert not buf.ready.any()
    buf.push(np.array([[3], [30]]))
    assert buf.ready.all()
    assert buf.windows()[1, :, 0].tolist() == [10, 20, 30]   # oldest first
    buf.push(np.array([[4], [40]]))
    assert buf.windows()[0, :, 0].tolist() == [2, 3, 4]
    buf.reset([0])
    assert buf.ready.tolist() == [False, True]


def test_adjustment_is_zero_in_warmup_then_kappa_times_forecast_gap():
    cfg = HarmonicConfig(L=6, F=2, kappa0=0.2, schedule="constant")
    rng = np.random.default_rng(3)
    traj = rng.normal(size=(7, 3, 2))                 # (iterations, birds, dims)
    buf = HistoryBuffer(3, 2, cfg.L)
    for t in range(cfg.L - 1):
        buf.push(traj[t])
    assert np.all(harmonic_adjustment(buf, traj[cfg.L - 1], t=5, T=10, cfg=cfg) == 0)
    buf.push(traj[cfg.L - 1])
    X = traj[cfg.L - 1]
    H = harmonic_adjustment(buf, X, t=6, T=10, cfg=cfg)
    expected = 0.2 * (harmonic_forecast(traj[:cfg.L, 1, 0], cfg.F) - X[1, 0])
    assert H.shape == X.shape
    assert H[1, 0] == pytest.approx(expected)


def test_config_file_matches_dataclass():
    import yaml
    from hsboa.folds import PROJECT_ROOT
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "hsboa.yaml").read_text())
    cfg = HarmonicConfig(**raw["harmonic"])
    assert (cfg.L, cfg.F, cfg.period) == (10, 2, None)


def test_oscillatory_mode_uses_zero_mean_component():
    cfg = HarmonicConfig(L=8, F=1, kappa0=1.0, schedule="constant", mode="oscillatory")
    buf = HistoryBuffer(1, 1, cfg.L)
    for v in np.full(cfg.L, 2.5):
        buf.push(np.array([[v]]))
    # A constant history has no oscillation, so H = 0 whatever the current position.
    assert harmonic_adjustment(buf, np.array([[9.0]]), 0, 10, cfg).item() == pytest.approx(0.0)
