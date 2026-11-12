"""Harmonic analysis of a bird's search history.

Three layers:

1. **Exact trigonometric interpolation** of i points. For even i, the Nyquist term x = i/2 has
   coefficient 1/i, not 2/i. With this coefficient the input
   points are reproduced exactly, so g = (0, 2) gives z0 = 1, z1 = 1 rather
   than the (-1, 3) a 2/i coefficient gives. For i = 2 the model is a
   pure parity term (cos(pi*y) = (-1)^y), so its one-step extrapolation
   g(y+1) equals g(y-1). This layer is therefore not used for
   the update; the regression below is.

2. **Causal harmonic regression**. Fit
       g(s) = c0 + sum_{f=1..F} [a_f cos(2 pi nu_f s) + b_f sin(2 pi nu_f s)]
   by least squares to the last L positions (s = 1..L), with 2F+1 < L so
   the fit smooths instead of interpolating, and forecast g(L+1). Only past
   values are used, so the update is causal.

   The base period matters. With nu_f = f / P and P = L, every basis
   function repeats with period L, so the forecast wraps around to the
   fitted value at s = 1. That is a smoothed replay of the oldest point,
   the same defect as the two-point model with a longer lag. The default is
   therefore P = 2L. With fixed frequencies the forecast is a fixed linear
   filter of the history, g_hat(L+1) = w @ g (see :func:`forecast_weights`).
   That makes it directly comparable with momentum or lag terms in the
   ablation. ``frequencies="periodogram"`` instead estimates the F
   dominant frequencies from the data (Damsleth & Spjotvoll 1982).

3. **The HSBOA adjustment**: X_new = X_SBOA + H with
       H = kappa(t) * (g_hat(t+1) - x(t))      mode "forecast" (default)
       H = kappa(t) * (g_hat(t+1) - c0)        mode "oscillatory" (zero-mean part)
   computed per dimension and vectorised over birds. H is zero during
   warm-up (fewer than L recorded positions), so HSBOA reduces to plain
   SBOA then. The history holds each bird's position after greedy
   selection, recorded once per iteration. Time steps are therefore
   uniform, and a rejected move repeats the previous position.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

FREQUENCY_MODES = ("fixed", "periodogram")
ADJUSTMENT_MODES = ("forecast", "oscillatory")
KAPPA_SCHEDULES = ("linear", "constant", "cosine")


# ---------------------------------------------------------------------------
# 1. Exact trigonometric interpolation (corrected Eqs. 13-28)
# ---------------------------------------------------------------------------

def trig_interpolation_coefficients(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients (a, b) of the trigonometric polynomial through g_1..g_i.

    g_y = a_0 + sum_{x=1..floor(i/2)} [a_x cos(2 pi x y / i) + b_x sin(2 pi x y / i)],
    with y = 1..i. a_0 is the mean. a_x and b_x use the factor 2/i, except
    for the Nyquist term x = i/2 when i is even, which uses 1/i and has
    b = 0.
    """
    g = np.asarray(g, dtype=float)
    i = len(g)
    if i < 1:
        raise ValueError("need at least one point")
    y = np.arange(1, i + 1)
    m = i // 2
    a = np.zeros(m + 1)
    b = np.zeros(m + 1)
    a[0] = g.mean()
    for x in range(1, m + 1):
        c = np.cos(2 * np.pi * x * y / i)
        s = np.sin(2 * np.pi * x * y / i)
        if i % 2 == 0 and x == m:
            a[x] = (g * c).sum() / i
        else:
            a[x] = 2 * (g * c).sum() / i
            b[x] = 2 * (g * s).sum() / i
    return a, b


def trig_interpolate(a: np.ndarray, b: np.ndarray, i: int, y) -> np.ndarray:
    """Evaluate the interpolating polynomial of :func:`trig_interpolation_coefficients` at y."""
    y = np.asarray(y, dtype=float)
    out = np.full(y.shape, a[0])
    for x in range(1, len(a)):
        out = out + a[x] * np.cos(2 * np.pi * x * y / i) + b[x] * np.sin(2 * np.pi * x * y / i)
    return out


# ---------------------------------------------------------------------------
# 2. Causal harmonic regression
# ---------------------------------------------------------------------------

def _check(L: int, F: int) -> None:
    if F < 1 or L < 2 * F + 2:
        raise ValueError(f"need F >= 1 and 2F+1 < L (smoothing, not interpolation); got L={L}, F={F}")


def design_matrix(s: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Columns [1, cos(2 pi nu_1 s), sin(2 pi nu_1 s), ...] for frequencies nu (cycles per step)."""
    s = np.asarray(s, dtype=float)[:, None]
    nu = np.asarray(frequencies, dtype=float)[None, :]
    ang = 2 * np.pi * nu * s
    cols = [np.ones((len(s), 1))]
    for k in range(nu.shape[1]):
        cols += [np.cos(ang[:, [k]]), np.sin(ang[:, [k]])]
    return np.hstack(cols)


@lru_cache(maxsize=64)
def _projection(L: int, F: int, period: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """(forecast weights w, intercept weights w0) for fixed frequencies f / period."""
    nu = np.arange(1, F + 1) / period
    pinv = np.linalg.pinv(design_matrix(np.arange(1, L + 1), nu))
    row = design_matrix(np.array([L + horizon]), nu)
    return (row @ pinv).ravel(), pinv[0].copy()


def forecast_weights(L: int, F: int, period: float | None = None,
                     horizon: int = 1) -> np.ndarray:
    """Linear filter w such that g_hat(L + horizon) = w @ g for fixed frequencies.

    The weights sum to 1 (a constant history is forecast exactly).
    """
    _check(L, F)
    return _projection(L, F, float(period or 2 * L), horizon)[0].copy()


def _periodogram_frequencies(g: np.ndarray, F: int, pad: int = 8) -> np.ndarray:
    """F strongest local maxima of the zero-padded periodogram (zero frequency excluded)."""
    L = len(g)
    n = pad * L
    power = np.abs(np.fft.rfft(g - g.mean(), n=n)) ** 2
    freqs = np.fft.rfftfreq(n)
    peaks = [k for k in range(1, len(power) - 1)
             if power[k] >= power[k - 1] and power[k] >= power[k + 1]]
    if len(power) > 1 and power[-1] > power[-2]:
        peaks.append(len(power) - 1)
    peaks = sorted(peaks, key=lambda k: -power[k])[:F]
    if len(peaks) < F:                                   # flat series: fall back to a regular grid
        return np.arange(1, F + 1) / (2 * L)
    return np.sort(freqs[peaks])


def harmonic_fit(window: np.ndarray, F: int, period: float | None = None,
                 frequencies: str = "fixed", horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Forecast and intercept for each column of ``window`` (shape (L,) or (L, k)).

    Returns (g_hat, c0), each of shape (k,), or scalars for a 1-D window.
    """
    w = np.asarray(window, dtype=float)
    one_d = w.ndim == 1
    if one_d:
        w = w[:, None]
    L, k = w.shape
    _check(L, F)
    if frequencies == "fixed":
        wf, w0 = _projection(L, F, float(period or 2 * L), horizon)
        g_hat, c0 = wf @ w, w0 @ w
    elif frequencies == "periodogram":
        g_hat, c0 = np.empty(k), np.empty(k)
        s = np.arange(1, L + 1)
        for j in range(k):
            nu = _periodogram_frequencies(w[:, j], F)
            coef, *_ = np.linalg.lstsq(design_matrix(s, nu), w[:, j], rcond=None)
            g_hat[j] = (design_matrix(np.array([L + horizon]), nu) @ coef).item()
            c0[j] = coef[0]
    else:
        raise ValueError(f"frequencies must be one of {FREQUENCY_MODES}")
    return (g_hat[0], c0[0]) if one_d else (g_hat, c0)


def harmonic_forecast(window: np.ndarray, F: int, period: float | None = None,
                      frequencies: str = "fixed", horizon: int = 1) -> np.ndarray:
    """One-step-ahead (or ``horizon``-step) forecast for each column of ``window``."""
    return harmonic_fit(window, F, period, frequencies, horizon)[0]


# ---------------------------------------------------------------------------
# 3. HSBOA adjustment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarmonicConfig:
    """All harmonic settings; the values used in the study are in configs/hsboa.yaml."""

    L: int = 10                      # history length (positions)
    F: int = 2                       # number of harmonics, 2F+1 < L
    period: float | None = None      # base period P for fixed frequencies; None -> 2L
    frequencies: str = "fixed"       # fixed | periodogram
    kappa0: float = 0.1              # adjustment scale at t = 0
    schedule: str = "linear"         # linear | constant | cosine decay of kappa
    mode: str = "forecast"           # forecast | oscillatory

    def __post_init__(self):
        _check(self.L, self.F)
        if self.frequencies not in FREQUENCY_MODES:
            raise ValueError(f"frequencies must be one of {FREQUENCY_MODES}")
        if self.mode not in ADJUSTMENT_MODES:
            raise ValueError(f"mode must be one of {ADJUSTMENT_MODES}")
        if self.schedule not in KAPPA_SCHEDULES:
            raise ValueError(f"schedule must be one of {KAPPA_SCHEDULES}")


def kappa_schedule(t: int, T: int, kappa0: float, kind: str = "linear") -> float:
    """Adjustment scale at iteration t of T (t = 0..T)."""
    frac = min(max(t / T, 0.0), 1.0)
    if kind == "linear":
        return kappa0 * (1 - frac)
    if kind == "constant":
        return kappa0
    if kind == "cosine":
        return kappa0 * 0.5 * (1 + np.cos(np.pi * frac))
    raise ValueError(f"schedule must be one of {KAPPA_SCHEDULES}")


class HistoryBuffer:
    """Last L positions of every bird, oldest first. Call :meth:`push` once per iteration."""

    def __init__(self, n_birds: int, dim: int, length: int):
        self.length = length
        self._buf = np.zeros((n_birds, length, dim))
        self._count = np.zeros(n_birds, dtype=int)

    def push(self, X: np.ndarray) -> None:
        """Record every bird's position after greedy selection (shape (n_birds, dim))."""
        self._buf[:, :-1] = self._buf[:, 1:]
        self._buf[:, -1] = X
        self._count = np.minimum(self._count + 1, self.length)

    def reset(self, birds) -> None:
        """Forget the history of the given birds (for example after re-initialisation)."""
        self._count[birds] = 0

    @property
    def ready(self) -> np.ndarray:
        return self._count >= self.length

    @property
    def counts(self) -> np.ndarray:
        """Number of recorded positions per bird (at most L)."""
        return self._count.copy()

    def windows(self) -> np.ndarray:
        """Shape (n_birds, L, dim). Only rows where :attr:`ready` is True are complete."""
        return self._buf.copy()


def harmonic_adjustment(buffer: HistoryBuffer, X: np.ndarray, t: int, T: int,
                        cfg: HarmonicConfig) -> np.ndarray:
    """H for every bird (shape like X). Zero for birds still in warm-up."""
    X = np.asarray(X, dtype=float)
    H = np.zeros_like(X)
    ready = buffer.ready
    if not ready.any():
        return H
    win = buffer.windows()[ready]                          # (m, L, D)
    m, L, D = win.shape
    cols = win.transpose(1, 0, 2).reshape(L, m * D)        # each column = one (bird, dim) series
    g_hat, c0 = harmonic_fit(cols, cfg.F, cfg.period, cfg.frequencies)
    g_hat, c0 = g_hat.reshape(m, D), c0.reshape(m, D)
    target = X[ready] if cfg.mode == "forecast" else c0
    H[ready] = kappa_schedule(t, T, cfg.kappa0, cfg.schedule) * (g_hat - target)
    return H
